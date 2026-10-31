"""Stream extraction artifacts without holding a dataset's masks in memory."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pickle
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from . import PROCEDURE_VERSION, __version__
from .inputs import AnnotationRecord, prepare_input
from .selection import select_candidate


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_metadata(record: AnnotationRecord) -> dict[str, Any]:
    return {
        "ann_id": record.ann_id,
        "image_key": record.image_key,
        "image_path": str(record.image_path),
        "dataset": record.dataset,
        "split": record.split,
        "identity": record.identity,
        "input_kind": record.input_kind,
        "bbox_xywh": record.bbox_xywh,
        "prompts": record.prompts,
        "source": record.source,
        "source_index": record.source_index,
        "provenance": record.provenance,
    }


def inspect_records(records: list[AnnotationRecord], *, check_images: bool = True) -> dict:
    """Preflight input geometry and prompts without importing or loading SAM3."""
    items = []
    errors = []
    for record in records:
        item = record_metadata(record)
        if check_images:
            try:
                prepared = prepare_input(record)
                item.update(
                    original_size=list(prepared.original_size),
                    crop_xyxy=list(prepared.crop_xyxy),
                    input_size=list(prepared.image.size),
                )
                prepared.image.close()
            except (OSError, ValueError) as exc:
                item["error"] = str(exc)
                errors.append({"ann_id": record.ann_id, "error": str(exc)})
        items.append(item)
    return {
        "procedure_version": PROCEDURE_VERSION,
        "annotations": len(records),
        "input_kinds": dict(Counter(record.input_kind for record in records)),
        "errors": errors,
        "records": items,
    }


class ArtifactWriter:
    """Write audit JSONL and CFR-compatible COCO RLE in input order."""

    def __init__(self, output_dir: Path, *, save_candidate_masks: bool, legacy_pickle: bool):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise RuntimeError("Install the base requirements.txt for COCO RLE export.") from exc
        self.mask_utils = mask_utils
        self.output_dir = output_dir
        self.save_candidate_masks = save_candidate_masks
        self.legacy: dict[str, np.ndarray | None] | None = {} if legacy_pickle else None
        self.audit_path = output_dir / "annotation_match_metadata.jsonl"
        self.rle_path = output_dir / "annotation_masks_rle.json.gz"
        self._resources = ExitStack()
        self._closed = False
        self.count = 0
        try:
            self.audit = self._resources.enter_context(
                self.audit_path.with_suffix(".jsonl.partial").open("w", encoding="utf-8")
            )
            self.rle_raw = self._resources.enter_context(
                self.rle_path.with_suffix(".gz.partial").open("wb")
            )
            self.rle = self._resources.enter_context(io.TextIOWrapper(
                gzip.GzipFile(filename="", mode="wb", fileobj=self.rle_raw, mtime=0),
                encoding="utf-8",
            ))
            header = {
                "format": "cfr-mask-rle-v1",
                "procedure_version": PROCEDURE_VERSION,
                "coordinate_frame": "SAM input image (crop or full image), shape [height, width]; see per-annotation audit",
                "mask_semantics": "selected unchanged SAM3 binary mask; null means rejected or input error",
            }
            self.rle.write(json.dumps(header, sort_keys=True)[:-1] + ', "annotations": [\n')
            if save_candidate_masks:
                (output_dir / "candidate_masks").mkdir()
        except BaseException:
            self.close(complete=False)
            raise

    def write(self, metadata: dict, selected_mask: np.ndarray | None, output: dict | None = None):
        ann_id = metadata["ann_id"]
        if self.save_candidate_masks and output is not None:
            arrays = np.asarray(output["masks"])
            if arrays.ndim == 4 and arrays.shape[1] == 1:
                arrays = arrays[:, 0]
            name = hashlib.sha256(ann_id.encode("utf-8")).hexdigest() + ".npz"
            relative = Path("candidate_masks") / name
            # Full input-sized candidates; no cropping or cleanup changes their pixels.
            np.savez_compressed(self.output_dir / relative, masks=arrays)
            metadata["candidate_npz"] = relative.as_posix()
        else:
            metadata["candidate_npz"] = None
        self.audit.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")

        item = {
            "id": self.count,
            "annotation_id": ann_id,
            "present": selected_mask is not None,
            "segmentation": None,
            "area": 0,
            "bbox": None,
        }
        if selected_mask is not None:
            rle = self.mask_utils.encode(np.asfortranarray(selected_mask, dtype=np.uint8))
            item.update(
                segmentation={"size": list(selected_mask.shape), "counts": rle["counts"].decode("ascii")},
                area=int(self.mask_utils.area(rle)),
                bbox=self.mask_utils.toBbox(rle).tolist(),
            )
        if self.count:
            self.rle.write(",\n")
        self.rle.write(json.dumps(item, sort_keys=True, allow_nan=False))
        self.count += 1
        if self.legacy is not None:
            self.legacy[ann_id] = selected_mask

    def close(self, *, complete: bool):
        """Close once, publishing files only after every export finishes writing."""
        if self._closed:
            return
        self._closed = True
        try:
            if complete:
                self.rle.write("\n]}\n")
        finally:
            self._resources.close()
        if not complete:
            return
        pending = [
            (self.audit_path.with_suffix(".jsonl.partial"), self.audit_path),
            (self.rle_path.with_suffix(".gz.partial"), self.rle_path),
        ]
        if self.legacy is not None:
            pickle_path = self.output_dir / "annotation_masks.pkl"
            partial_pickle = pickle_path.with_suffix(".pkl.partial")
            with partial_pickle.open("wb") as handle:
                pickle.dump(self.legacy, handle, protocol=4)
            pending.append((partial_pickle, pickle_path))
        published = []
        try:
            for partial, final in pending:
                partial.rename(final)
                published.append((partial, final))
        except BaseException:
            # A later publish failure must not leave earlier files looking complete.
            for partial, final in reversed(published):
                final.rename(partial)
            raise


def run_extraction(
    records: list[AnnotationRecord],
    output_dir: Path,
    backend: Any,
    *,
    batch_size: int = 1,
    save_candidate_masks: bool = False,
    legacy_pickle: bool = False,
    run_config: dict | None = None,
) -> dict:
    """Use an injected predict(images, prompts) backend; never add a score filter."""
    if not records:
        raise ValueError("No annotation records to process.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if len({record.ann_id for record in records}) != len(records):
        raise ValueError("Annotation IDs must be unique in each extraction run.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use a new run directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "procedure_version": PROCEDURE_VERSION,
        "package_version": __version__,
        "status": "running",
        "records_total": len(records),
        "annotations_attempted": 0,
        "annotations_accepted": 0,
        "candidate_count": 0,
        "errors": 0,
        "status_counts": {},
        "backend": backend.metadata,
        "mask_cleanup": "none",
        "centrality": "clipped SAM box contains input center (W/2,H/2), half-open XYXY bounds",
        "ranking": ["foreground pixels descending", "SAM confidence descending", "candidate_index ascending"],
        "coverage_denominator": "entire input image pixel count H*W (mask_coverage)",
        "minimum_coverage": None,
        "additional_score_filter": None,
        "coordinate_frame": "full SAM input dimensions; selected mask is not cropped to its predicted box",
        "batch_size": batch_size,
        "save_candidate_masks": save_candidate_masks,
        "legacy_pickle": legacy_pickle,
        "run_config": run_config or {},
        "implementation_sha256": {
            path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "input_metadata_sha256": {
            source: file_sha256(Path(source)) for source in sorted({record.source for record in records})
            if Path(source).is_file()
        },
    }
    dump_json(output_dir / "extraction_summary.json", summary)
    counts: Counter = Counter()
    writer = None
    try:
        writer = ArtifactWriter(output_dir, save_candidate_masks=save_candidate_masks, legacy_pickle=legacy_pickle)
        for start in range(0, len(records), batch_size):
            prepared_items = []
            ordered_items = []
            for record in records[start:start + batch_size]:
                metadata = record_metadata(record)
                try:
                    prepared = prepare_input(record)
                except (OSError, ValueError) as exc:
                    status = "image_missing" if isinstance(exc, FileNotFoundError) else "input_error"
                    metadata.update(status=status, accepted=False, error=str(exc), candidates=[], selected_index=None)
                    ordered_items.append((metadata, None))
                    summary["errors"] += 1
                    continue
                metadata.update(
                    original_size=list(prepared.original_size),
                    crop_xyxy=list(prepared.crop_xyxy),
                    input_size=list(prepared.image.size),
                    input_rgb_sha256=hashlib.sha256(prepared.image.tobytes()).hexdigest(),
                    mask_coordinate_frame="original_image" if prepared.record.input_kind == "full_image" else "annotation_input_crop",
                    selection_policy="input_center",
                )
                ordered_items.append((metadata, prepared))
                prepared_items.append(prepared)

            try:
                outputs = backend.predict(
                    [item.image for item in prepared_items],
                    [item.record.prompts for item in prepared_items],
                ) if prepared_items else []
                if len(outputs) != len(prepared_items):
                    raise ValueError("SAM backend returned a different number of outputs than inputs.")
                output_iter = iter(outputs)
                for metadata, prepared in ordered_items:
                    summary["annotations_attempted"] += 1
                    if prepared is None:
                        counts[metadata["status"]] += 1
                        writer.write(metadata, None)
                        continue
                    output = next(output_iter)
                    selected = select_candidate(output, prepared.image.size)
                    metadata.update(
                        status=selected.status,
                        accepted=selected.selected_index is not None,
                        selected_index=selected.selected_index,
                        candidates=selected.candidates,
                        eligible_candidate_count=sum(c["eligible"] for c in selected.candidates),
                    )
                    counts[selected.status] += 1
                    summary["candidate_count"] += len(selected.candidates)
                    summary["annotations_accepted"] += int(selected.selected_index is not None)
                    writer.write(metadata, selected.selected_mask, output)
            finally:
                for prepared in prepared_items:
                    prepared.image.close()
        writer.close(complete=True)
    except BaseException as exc:
        if writer is not None:
            try:
                writer.close(complete=False)
            except Exception as cleanup_exc:
                summary["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", status_counts=dict(counts))
        dump_json(output_dir / "extraction_summary.json", summary)
        raise

    summary.update(
        status="completed_with_input_errors" if summary["errors"] else "complete",
        status_counts=dict(counts),
        annotation_masks_rle="annotation_masks_rle.json.gz",
        annotation_match_metadata="annotation_match_metadata.jsonl",
    )
    dump_json(output_dir / "extraction_summary.json", summary)
    return summary
