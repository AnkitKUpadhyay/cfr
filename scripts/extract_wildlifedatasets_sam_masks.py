#!/usr/bin/env python3
"""Extract SAM3 masks while preserving all low-IoU candidates for review.

This script is a dataset-agnostic replacement for the older per-species SAM3
wrappers.  It writes the legacy annotation_masks.pkl file for compatibility,
but also stores every SAM candidate mask and detailed matching metadata so
humans can inspect cases that do not pass the usual IoU threshold.

Supported inputs:
  - COCO/GZCD-style JSON files with images/annotations.
  - WildlifeDatasets/MegaDescriptor CSV files with image_id, identity, path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_SM_ROOT = os.environ.get("SEMANTIC_MATCHING_ROOT", "")
if _SM_ROOT:
    SEMANTIC_MATCHING_ROOT = Path(_SM_ROOT)
    sys.path.insert(0, str(SEMANTIC_MATCHING_ROOT))
    sys.path.insert(0, str(SEMANTIC_MATCHING_ROOT / "src"))


@dataclass
class AnnotationRecord:
    ann_id: str
    image_key: str
    image_path: Path
    dataset: str
    split: str | None
    identity: str | None
    bbox_xywh: list[float] | None
    source_index: int
    source: str
    raw: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def safe_name(value: str, max_len: int = 96) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    if len(clean) > max_len:
        clean = clean[:max_len]
    return f"{clean}_{digest}"


def bbox_xywh_from_mapping(row: dict[str, Any]) -> list[float] | None:
    if "bbox" in row and row["bbox"] not in (None, ""):
        value = row["bbox"]
        if isinstance(value, str):
            stripped = value.strip().strip("[]()")
            parts = [p for p in stripped.replace(";", ",").split(",") if p.strip()]
            if len(parts) >= 4:
                vals = [float(p) for p in parts[:4]]
                return vals if vals[2] > 0 and vals[3] > 0 else None
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            vals = [float(v) for v in value[:4]]
            return vals if vals[2] > 0 and vals[3] > 0 else None
    keys = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    if all(k in row and row[k] not in (None, "") for k in keys):
        vals = [float(row[k]) for k in keys]
        return vals if vals[2] > 0 and vals[3] > 0 else None
    return None


def bbox_xyxy(bbox_xywh: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    img_w, img_h = image_size
    x, y, w, h = [float(v) for v in bbox_xywh]
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img_w, int(round(x + w)))
    y2 = min(img_h, int(round(y + h)))
    return x1, y1, x2, y2


def full_image_bbox(image_size: tuple[int, int]) -> list[float]:
    img_w, img_h = image_size
    return [0.0, 0.0, float(img_w), float(img_h)]


def is_full_image_bbox(
    bbox_xywh: list[float] | None,
    image_size: tuple[int, int],
    tol: float = 2.0,
) -> bool:
    if bbox_xywh is None:
        return True
    img_w, img_h = image_size
    x, y, w, h = [float(v) for v in bbox_xywh]
    return x <= tol and y <= tol and w >= img_w - tol and h >= img_h - tol


def box_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def resolve_image_path(rel_or_abs: str, image_root: Path, dataset: str | None = None) -> Path:
    raw = Path(str(rel_or_abs))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(image_root / raw)
    if dataset:
        candidates.append(image_root / dataset / raw)
        if len(raw.parts) > 1 and raw.parts[0] == dataset:
            candidates.append(image_root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def image_relpath_from_json(image: dict[str, Any]) -> str | None:
    for key in ("image_path", "file_name", "path", "filepath"):
        value = image.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def ann_key(ann: dict[str, Any], fallback: int) -> str:
    for key in ("annot_uuid", "uuid", "id", "annotation_id"):
        value = ann.get(key)
        if value not in (None, ""):
            return str(value)
    return f"ann_{fallback}"


def image_key(image: dict[str, Any], fallback: int) -> str:
    for key in ("uuid", "image_uuid", "id", "image_id"):
        value = image.get(key)
        if value not in (None, ""):
            return str(value)
    return f"image_{fallback}"


def image_aliases(image: dict[str, Any], fallback: int) -> list[str]:
    aliases = [image_key(image, fallback)]
    for key in ("uuid", "image_uuid", "id", "image_id"):
        value = image.get(key)
        if value not in (None, ""):
            aliases.append(str(value))
    deduped = []
    seen = set()
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            deduped.append(alias)
    return deduped


def load_json_records(
    path: Path,
    image_root: Path,
    dataset: str,
    split: str | None,
) -> list[AnnotationRecord]:
    data = read_json(path)
    images = {}
    for idx, image in enumerate(data.get("images", [])):
        for alias in image_aliases(image, idx):
            images[alias] = image
    records: list[AnnotationRecord] = []
    for idx, ann in enumerate(data.get("annotations", [])):
        img_ref = ann.get("image_uuid", ann.get("image_id", ann.get("image")))
        image = images.get(str(img_ref))
        if image is None:
            continue
        rel = image_relpath_from_json(image)
        if not rel:
            continue
        row = dict(ann)
        bbox = bbox_xywh_from_mapping(row)
        records.append(
            AnnotationRecord(
                ann_id=ann_key(ann, idx),
                image_key=str(img_ref),
                image_path=resolve_image_path(rel, image_root),
                dataset=dataset,
                split=split,
                identity=str(
                    ann.get("identity")
                    or ann.get("name_uuid")
                    or ann.get("individual_id")
                    or ann.get("name")
                    or ""
                )
                or None,
                bbox_xywh=bbox,
                source_index=idx,
                source=str(path),
                raw=row,
            )
        )
    return records


def load_csv_records(
    path: Path,
    image_root: Path,
    dataset: str | None,
    split: str | None,
    dataset_filter: str | None,
    split_filter: str | None,
) -> list[AnnotationRecord]:
    records: list[AnnotationRecord] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            row_dataset = str(row.get("dataset") or dataset or path.parent.name)
            row_split = str(row.get("split") or split or "")
            if dataset_filter and row_dataset != dataset_filter:
                continue
            if split_filter and row_split != split_filter:
                continue
            rel = row.get("path") or row.get("image_path") or row.get("file_name")
            if not rel:
                continue
            image_id = str(row.get("image_id") or row.get("id") or idx)
            ann_id_value = str(row.get("annotation_id") or row.get("annot_uuid") or image_id)
            records.append(
                AnnotationRecord(
                    ann_id=ann_id_value,
                    image_key=image_id,
                    image_path=resolve_image_path(str(rel), image_root, row_dataset),
                    dataset=row_dataset,
                    split=row_split or split,
                    identity=str(row.get("identity") or row.get("name") or "") or None,
                    bbox_xywh=bbox_xywh_from_mapping(row),
                    source_index=idx,
                    source=str(path),
                    raw=dict(row),
                )
            )
    return records


def group_by_image(records: Iterable[AnnotationRecord]) -> dict[str, list[AnnotationRecord]]:
    grouped: dict[str, list[AnnotationRecord]] = {}
    for record in records:
        grouped.setdefault(record.image_key, []).append(record)
    return grouped


def to_numpy(output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")
    if masks is None:
        return (
            np.empty((0, 1, 1), dtype=bool),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    return masks_np, boxes_np, scores_np


def crop_mask(mask: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=np.uint8)
    return (mask[y1:y2, x1:x2] > 0.5).astype(np.uint8)


def write_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, default=json_default, sort_keys=True) + "\n")


def process_records(args: argparse.Namespace, records: list[AnnotationRecord]) -> dict[str, Any]:
    from src.config.config import SAMConfig
    from src.segmentation.sam3_segmenter import SAM3Segmenter

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidate_masks"
    if args.save_candidate_masks:
        candidate_dir.mkdir(parents=True, exist_ok=True)

    sam_config = SAMConfig(
        device=args.device,
        min_score=args.min_sam_score,
        species_prompts={args.species_key: args.prompts},
    )
    segmenter = SAM3Segmenter(sam_config)

    grouped = group_by_image(records)
    image_items = list(grouped.items())
    if args.limit:
        image_items = image_items[: args.limit]

    accepted_masks: dict[str, np.ndarray | None] = {}
    best_any_masks: dict[str, np.ndarray | None] = {}
    stats = {
        "records_total": len(records),
        "images_total": len(grouped),
        "images_attempted": 0,
        "images_missing": 0,
        "images_failed": 0,
        "annotations_attempted": 0,
        "annotations_accepted": 0,
        "annotations_with_any_candidate": 0,
        "annotations_low_iou_best": 0,
        "annotations_no_candidate": 0,
        "candidate_count": 0,
        "accept_iou": args.accept_iou,
        "min_sam_score": args.min_sam_score,
        "candidate_mask_mode": args.candidate_mask_mode,
    }

    ann_meta_path = args.output_dir / "annotation_match_metadata.jsonl"
    image_meta_path = args.output_dir / "image_candidate_metadata.jsonl"

    with ann_meta_path.open("w") as ann_meta, image_meta_path.open("w") as image_meta:
        for start in tqdm(range(0, len(image_items), args.batch_size), desc="SAM3 batches"):
            batch_items = image_items[start : start + args.batch_size]
            images: list[Image.Image] = []
            usable_items = []
            for img_key, ann_records in batch_items:
                image_path = ann_records[0].image_path
                if not image_path.exists():
                    stats["images_missing"] += 1
                    for record in ann_records:
                        accepted_masks[record.ann_id] = None
                        best_any_masks[record.ann_id] = None
                        write_jsonl(
                            ann_meta,
                            {
                                "ann_id": record.ann_id,
                                "image_key": record.image_key,
                                "image_path": str(record.image_path),
                                "dataset": record.dataset,
                                "split": record.split,
                                "status": "image_missing",
                            },
                        )
                    continue
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as exc:
                    stats["images_failed"] += 1
                    for record in ann_records:
                        accepted_masks[record.ann_id] = None
                        best_any_masks[record.ann_id] = None
                        write_jsonl(
                            ann_meta,
                            {
                                "ann_id": record.ann_id,
                                "image_key": record.image_key,
                                "image_path": str(record.image_path),
                                "dataset": record.dataset,
                                "split": record.split,
                                "status": "image_open_failed",
                                "error": str(exc),
                            },
                        )
                    continue
                images.append(image)
                usable_items.append((img_key, ann_records, image))

            if not usable_items:
                continue

            try:
                outputs = segmenter.segment_images_batch(images, [args.species_key] * len(images))
                if args.min_sam_score > 0:
                    outputs = [
                        segmenter.filter_masks_by_score(output, min_score=args.min_sam_score)
                        for output in outputs
                    ]
            except Exception as exc:
                stats["images_failed"] += len(usable_items)
                for img_key, ann_records, _image in usable_items:
                    for record in ann_records:
                        accepted_masks[record.ann_id] = None
                        best_any_masks[record.ann_id] = None
                        write_jsonl(
                            ann_meta,
                            {
                                "ann_id": record.ann_id,
                                "image_key": record.image_key,
                                "image_path": str(record.image_path),
                                "dataset": record.dataset,
                                "split": record.split,
                                "status": "sam_failed",
                                "error": str(exc),
                            },
                        )
                continue

            for (img_key, ann_records, image), output in zip(usable_items, outputs):
                stats["images_attempted"] += 1
                masks, boxes, scores = to_numpy(output)
                img_w, img_h = image.size
                candidate_entries = []
                candidate_arrays: dict[str, np.ndarray] = {}
                for cand_idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                    x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    if x2 <= x1 or y2 <= y1:
                        x1, y1, x2, y2 = 0, 0, img_w, img_h
                    if args.candidate_mask_mode == "full":
                        stored_mask = (mask > 0.5).astype(np.uint8)
                        crop_xyxy = [0, 0, img_w, img_h]
                    elif args.candidate_mask_mode == "none":
                        stored_mask = None
                        crop_xyxy = [x1, y1, x2, y2]
                    else:
                        stored_mask = crop_mask(mask, (x1, y1, x2, y2))
                        crop_xyxy = [x1, y1, x2, y2]
                    mask_key = f"mask_{cand_idx:03d}"
                    if args.save_candidate_masks and stored_mask is not None:
                        candidate_arrays[mask_key] = stored_mask
                    candidate_entries.append(
                        {
                            "candidate_index": cand_idx,
                            "score": float(score),
                            "box_xyxy": [x1, y1, x2, y2],
                            "crop_xyxy": crop_xyxy,
                            "stored_key": mask_key if stored_mask is not None else None,
                            "stored_shape": list(stored_mask.shape) if stored_mask is not None else None,
                            "area_frac_in_stored_crop": float(stored_mask.mean()) if stored_mask is not None and stored_mask.size else None,
                        }
                    )
                stats["candidate_count"] += len(candidate_entries)

                candidate_npz_rel = None
                if args.save_candidate_masks and candidate_arrays:
                    candidate_npz = candidate_dir / f"{safe_name(img_key)}.npz"
                    np.savez_compressed(candidate_npz, **candidate_arrays)
                    candidate_npz_rel = str(candidate_npz.relative_to(args.output_dir))

                write_jsonl(
                    image_meta,
                    {
                        "image_key": img_key,
                        "image_path": str(ann_records[0].image_path),
                        "dataset": ann_records[0].dataset,
                        "split": ann_records[0].split,
                        "image_size": [img_w, img_h],
                        "candidate_npz": candidate_npz_rel,
                        "candidates": candidate_entries,
                    },
                )

                for record in ann_records:
                    stats["annotations_attempted"] += 1
                    full_image_mode = is_full_image_bbox(record.bbox_xywh, (img_w, img_h))
                    bbox = record.bbox_xywh or full_image_bbox((img_w, img_h))
                    ann_xyxy = bbox_xyxy(bbox, (img_w, img_h))
                    matches = []
                    for cand in candidate_entries:
                        iou = box_iou_xyxy(tuple(ann_xyxy), tuple(cand["box_xyxy"]))
                        matches.append(
                            {
                                "candidate_index": cand["candidate_index"],
                                "bbox_iou": iou,
                                "score": cand["score"],
                                "box_xyxy": cand["box_xyxy"],
                                "stored_key": cand["stored_key"],
                            }
                        )
                    if full_image_mode:
                        matches.sort(key=lambda item: (item["score"], item["bbox_iou"]), reverse=True)
                    else:
                        matches.sort(key=lambda item: (item["bbox_iou"], item["score"]), reverse=True)
                    best = matches[0] if matches else None
                    accepted = bool(best is not None and (full_image_mode or best["bbox_iou"] >= args.accept_iou))
                    if best is None:
                        accepted_masks[record.ann_id] = None
                        best_any_masks[record.ann_id] = None
                        stats["annotations_no_candidate"] += 1
                        status = "no_candidate"
                    else:
                        best_idx = int(best["candidate_index"])
                        best_full_mask = masks[best_idx]
                        best_crop = crop_mask(best_full_mask, ann_xyxy)
                        best_any_masks[record.ann_id] = best_crop
                        stats["annotations_with_any_candidate"] += 1
                        if accepted:
                            accepted_masks[record.ann_id] = best_crop
                            stats["annotations_accepted"] += 1
                            status = "accepted_full_image_best_score" if full_image_mode else "accepted"
                        else:
                            accepted_masks[record.ann_id] = None
                            stats["annotations_low_iou_best"] += 1
                            status = "best_below_accept_iou"

                    write_jsonl(
                        ann_meta,
                        {
                            "ann_id": record.ann_id,
                            "image_key": record.image_key,
                            "image_path": str(record.image_path),
                            "dataset": record.dataset,
                            "split": record.split,
                            "identity": record.identity,
                            "bbox_xywh": bbox,
                            "bbox_xyxy": list(ann_xyxy),
                            "full_image_mode": full_image_mode,
                            "source": record.source,
                            "source_index": record.source_index,
                            "status": status,
                            "accepted": accepted,
                            "best_candidate": best,
                            "all_candidate_matches": matches,
                            "candidate_npz": candidate_npz_rel,
                        },
                    )

    with (args.output_dir / "annotation_masks.pkl").open("wb") as handle:
        pickle.dump(accepted_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.output_dir / "annotation_best_any_iou_masks.pkl").open("wb") as handle:
        pickle.dump(best_any_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        **stats,
        "output_dir": str(args.output_dir),
        "annotation_masks_pkl": str(args.output_dir / "annotation_masks.pkl"),
        "annotation_best_any_iou_masks_pkl": str(args.output_dir / "annotation_best_any_iou_masks.pkl"),
        "annotation_match_metadata_jsonl": str(ann_meta_path),
        "image_candidate_metadata_jsonl": str(image_meta_path),
        "prompts": args.prompts,
        "input_json": str(args.input_json) if args.input_json else None,
        "input_csv": str(args.input_csv) if args.input_csv else None,
        "dataset_filter": args.dataset_filter,
        "split_filter": args.split_filter,
    }
    with (args.output_dir / "extraction_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=json_default)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_json", type=Path)
    source.add_argument("--input_csv", type=Path)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--dataset_filter", default=None)
    parser.add_argument("--split_filter", default=None)
    parser.add_argument("--species_key", default="target")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["animal"],
        help="Text prompts passed to SAM3 for the species_key.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_sam_score", type=float, default=0.0)
    parser.add_argument("--accept_iou", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit in images; 0 means all.")
    parser.add_argument(
        "--candidate_mask_mode",
        choices=("box_crop", "full", "none"),
        default="box_crop",
    )
    parser.add_argument("--save_candidate_masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_json:
        dataset = args.dataset or args.input_json.stem
        records = load_json_records(args.input_json, args.image_root, dataset, args.split)
    else:
        records = load_csv_records(
            args.input_csv,
            args.image_root,
            args.dataset,
            args.split,
            args.dataset_filter,
            args.split_filter,
        )
    if not records:
        raise SystemExit("No records found for requested input.")
    summary = process_records(args, records)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
