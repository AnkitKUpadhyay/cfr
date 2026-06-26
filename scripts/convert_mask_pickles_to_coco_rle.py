#!/usr/bin/env python3
"""Convert CFR mask pickle files to COCO-style compressed RLE JSON.

The source pickle files map annotation ids to 2-D binary mask arrays, or to
None when no mask is available. This script preserves that mapping but writes
standard pycocotools compressed RLE in JSON so the masks can be inspected and
loaded without Python pickle.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


PRIVATE_MARKERS = tuple(
    marker.strip()
    for marker in os.environ.get("CFR_PRIVATE_MARKERS", "").split(",")
    if marker.strip()
)

MASK_FILES = (
    (
        "annotation_masks.pkl",
        "annotation_masks_rle.json.gz",
        "accepted masks only; null means no accepted mask for this annotation",
    ),
    (
        "annotation_best_any_iou_masks.pkl",
        "annotation_best_any_iou_masks_rle.json.gz",
        "best available candidate mask; null means no candidate mask was available",
    ),
)


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rel_mask_dir(row: dict[str, str]) -> Path:
    group = row["group"]
    dataset = row["dataset"]
    return Path("sam_masks_rle") / group / dataset


def check_public_string(value: Any, *, field: str) -> str:
    text = str(value)
    if text.startswith("/") or text.startswith("~"):
        raise ValueError(f"Absolute/private-looking path found in {field}: {text!r}")
    for marker in PRIVATE_MARKERS:
        if marker in text:
            raise ValueError(f"Private marker {marker!r} found in {field}: {text!r}")
    return text


def mask_to_coco_rle(mask: Any) -> dict[str, Any] | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {arr.shape}")
    arr = np.asfortranarray((arr > 0).astype(np.uint8))
    rle = mask_utils.encode(arr)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "segmentation": {"size": [int(arr.shape[0]), int(arr.shape[1])], "counts": counts},
        "area": int(mask_utils.area(rle)),
        "bbox": [float(x) for x in mask_utils.toBbox(rle).tolist()],
    }


def iter_annotation_records(mask_dict: dict[Any, Any]) -> tuple[int, int, Any]:
    present = 0
    for i, (ann_id, mask) in enumerate(mask_dict.items()):
        ann_id_text = check_public_string(ann_id, field="annotation_id")
        encoded = mask_to_coco_rle(mask)
        if encoded is None:
            record = {
                "id": i,
                "annotation_id": ann_id_text,
                "present": False,
                "segmentation": None,
                "area": 0,
                "bbox": None,
            }
        else:
            present += 1
            record = {
                "id": i,
                "annotation_id": ann_id_text,
                "present": True,
                **encoded,
            }
        yield present, i + 1, record


def write_rle_json(
    pickle_path: Path,
    output_path: Path,
    *,
    row: dict[str, str],
    source_name: str,
    mask_semantics: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        return {
            "output": str(output_path),
            "status": "exists",
            "bytes": output_path.stat().st_size,
        }

    with pickle_path.open("rb") as f:
        mask_dict = pickle.load(f)
    if not isinstance(mask_dict, dict):
        raise TypeError(f"{pickle_path} did not contain a dict")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "cfr-mask-rle-v1",
        "rle_encoding": "COCO compressed RLE from pycocotools.mask.encode",
        "coordinate_frame": (
            "same as source pickle: annotation-box coordinates for tight boxes, "
            "image coordinates for full-image/crop annotations"
        ),
        "dataset": check_public_string(row["dataset"], field="dataset"),
        "group": check_public_string(row["group"], field="group"),
        "source_pickle_name": source_name,
        "mask_semantics": mask_semantics,
        "records_total": int(row["records_total"]),
        "annotations_accepted": int(row["annotations_accepted"]),
        "manifest_prompts": row.get("manifest_prompts", ""),
    }

    present = 0
    total = 0
    with gzip.open(output_path, "wt", encoding="utf-8") as out:
        out.write("{\n")
        for key, value in metadata.items():
            out.write(f"  {json.dumps(key)}: ")
            json.dump(value, out, separators=(",", ":"))
            out.write(",\n")
        out.write('  "annotations": [\n')
        first = True
        for present_so_far, total_so_far, record in iter_annotation_records(mask_dict):
            present = present_so_far
            total = total_so_far
            if not first:
                out.write(",\n")
            json.dump(record, out, separators=(",", ":"))
            first = False
        out.write("\n  ]\n")
        out.write("}\n")

    return {
        "output": str(output_path),
        "status": "written",
        "records_total": total,
        "present": present,
        "null": total - present,
        "bytes": output_path.stat().st_size,
    }


def write_release_index(rows: list[dict[str, str]], output_root: Path) -> None:
    path = output_root / "central_sam_mask_dataset_index_34_rle.tsv"
    fields = [
        "dataset",
        "group",
        "records_total",
        "annotations_accepted",
        "accepted_percent",
        "manifest_prompts",
        "mask_dir",
        "annotation_masks_rle",
        "annotation_best_any_iou_masks_rle",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            mask_dir = rel_mask_dir(row)
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "group": row["group"],
                    "records_total": row["records_total"],
                    "annotations_accepted": row["annotations_accepted"],
                    "accepted_percent": row["accepted_percent"],
                    "manifest_prompts": row.get("manifest_prompts", ""),
                    "mask_dir": mask_dir.as_posix(),
                    "annotation_masks_rle": (mask_dir / "annotation_masks_rle.json.gz").as_posix(),
                    "annotation_best_any_iou_masks_rle": (
                        mask_dir / "annotation_best_any_iou_masks_rle.json.gz"
                    ).as_posix(),
                }
            )


def write_release_readme(rows: list[dict[str, str]], output_root: Path) -> None:
    total = sum(int(r["records_total"]) for r in rows)
    accepted = sum(int(r["annotations_accepted"]) for r in rows)
    coverage = accepted / total * 100 if total else 0.0
    text = f"""# CFR SAM Mask RLE Artifact

This anonymous artifact contains the derived foreground-mask resource used for
the 34-dataset CFR evaluation. The masks are converted from the internal pickle
files to JSON files containing COCO-style compressed RLE segmentations from
`pycocotools.mask.encode`.

Raw source images are not included here unless allowed by the original dataset
license. The released masks contain no absolute source paths.

## Coverage

Accepted masks cover {accepted:,} of {total:,} eligible annotation records
({coverage:.1f}%).

## Layout

```text
cfr_sam_mask_rle_release/
  README_SAM_MASK_RLE_RELEASE.md
  central_sam_mask_dataset_index_34_rle.tsv
  sam_masks_rle/
    <group>/<dataset>/
      annotation_masks_rle.json.gz
      annotation_best_any_iou_masks_rle.json.gz
```

Each dataset has exactly two RLE mask files:

- `annotation_masks_rle.json.gz`: accepted masks only. Entries with no accepted
  mask have `present=false` and `segmentation=null`.
- `annotation_best_any_iou_masks_rle.json.gz`: best candidate per annotation,
  including candidates that failed the acceptance rule. Entries with no
  candidate have `present=false` and `segmentation=null`.

The JSON schema is:

```json
{{
  "format": "cfr-mask-rle-v1",
  "rle_encoding": "COCO compressed RLE from pycocotools.mask.encode",
  "coordinate_frame": "same as source pickle...",
  "dataset": "<dataset>",
  "group": "<group>",
  "annotations": [
    {{
      "id": 0,
      "annotation_id": "<source annotation id>",
      "present": true,
      "segmentation": {{"size": [height, width], "counts": "<compressed RLE>"}},
      "area": 123,
      "bbox": [x, y, width, height]
    }}
  ]
}}
```

The coordinate frame is unchanged from the source pickle files: masks are in
annotation-box coordinates for tight-box annotations and image coordinates for
full-image or crop-level annotations.

Use `pycocotools.mask.decode(record["segmentation"])` to recover a binary mask
for any record with `present=true`.
"""
    (output_root / "README_SAM_MASK_RLE_RELEASE.md").write_text(text)


def select_rows(
    rows: list[dict[str, str]], datasets: list[str] | None, row_index: int | None
) -> list[dict[str, str]]:
    selected = rows
    if datasets:
        wanted = set(datasets)
        selected = [r for r in selected if r["dataset"] in wanted]
    if row_index is not None:
        selected = [selected[row_index]]
    return selected


def convert_rows(rows: list[dict[str, str]], output_root: Path, overwrite: bool) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        in_dir = Path(row["central_mask_dir"])
        out_dir = output_root / rel_mask_dir(row)
        for source_name, output_name, semantics in MASK_FILES:
            result = write_rle_json(
                in_dir / source_name,
                out_dir / output_name,
                row=row,
                source_name=source_name,
                mask_semantics=semantics,
                overwrite=overwrite,
            )
            result.update({"dataset": row["dataset"], "group": row["group"], "source": source_name})
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--row_index", type=int, default=None)
    parser.add_argument("--prepare_release", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_index(args.index)
    if args.prepare_release:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_release_index(rows, args.output_root)
        write_release_readme(rows, args.output_root)
    if args.convert:
        selected = select_rows(rows, args.dataset, args.row_index)
        results = convert_rows(selected, args.output_root, args.overwrite)
        print(json.dumps(results, indent=2))
    if not args.prepare_release and not args.convert:
        raise SystemExit("Nothing to do. Use --prepare_release and/or --convert.")


if __name__ == "__main__":
    main()
