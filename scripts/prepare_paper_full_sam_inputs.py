#!/usr/bin/env python3
"""Prepare full-dataset SAM inputs for non-WildlifeDatasets paper datasets."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT = Path(os.environ.get("CFR_CODE_ROOT", str(Path(__file__).resolve().parents[1])))
INPUT_DIR = PROJECT / "configs" / "sam_full_dataset_inputs"
MANIFEST = PROJECT / "configs" / "sam_mask_jobs_paper_full.jsonl"
SCRATCH_ROOT = Path(os.environ.get("CFR_MASK_OUTPUT_ROOT", "outputs/sam_masks"))
SOURCE_IMAGE_ROOT = Path(os.environ.get("SOURCE_IMAGE_ROOT", "/path/to/source_images"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle)


def add_split_tag(items: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    tagged = []
    for item in items:
        copied = dict(item)
        copied.setdefault("split", split)
        tagged.append(copied)
    return tagged


def combine_seal() -> Path:
    train = read_json(PROJECT / "seal_train_census.json")
    test = read_json(PROJECT / "seal_test_census.json")
    combined = {
        "info": {
            "description": "Combined local Saimaa seal census train+test metadata for full-dataset SAM extraction.",
            "source_files": ["seal_train_census.json", "seal_test_census.json"],
        },
        "categories": train.get("categories", test.get("categories", [])),
        "images": add_split_tag(train.get("images", []), "train")
        + add_split_tag(test.get("images", []), "test"),
        "annotations": add_split_tag(train.get("annotations", []), "train")
        + add_split_tag(test.get("annotations", []), "test"),
    }

    image_keys = [str(img.get("uuid", img.get("id"))) for img in combined["images"]]
    ann_keys = [
        str(ann.get("annot_uuid", ann.get("uuid", ann.get("id"))))
        for ann in combined["annotations"]
    ]
    if len(image_keys) != len(set(image_keys)):
        raise ValueError("Combined seal input has duplicate image keys.")
    if len(ann_keys) != len(set(ann_keys)):
        raise ValueError("Combined seal input has duplicate annotation keys.")

    output = INPUT_DIR / "saimaa_seal_full_gzcd_compat.json"
    write_json(output, combined)
    return output


def copy_existing_full_inputs() -> tuple[Path, Path]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    whale_src = PROJECT / "instances_train2020_whaleshark_gzcd_compat.json"
    leopard_src = PROJECT / "leopard_train_test_gzcd_compat.json"
    whale_dst = INPUT_DIR / "whaleshark_full_gzcd_compat.json"
    leopard_dst = INPUT_DIR / "leopard_full_gzcd_compat.json"
    shutil.copy2(whale_src, whale_dst)
    shutil.copy2(leopard_src, leopard_dst)
    return whale_dst, leopard_dst


def write_manifest(whale_json: Path, leopard_json: Path, seal_json: Path) -> None:
    rows = [
        {
            "dataset": "whaleshark",
            "image_root": str(SOURCE_IMAGE_ROOT / "whaleshark" / "images"),
            "input_path": str(whale_json),
            "input_type": "json",
            "job_name": "whaleshark_full_20260620",
            "output_dir": str(SCRATCH_ROOT / "paper_full_datasets" / "whaleshark_full_20260620"),
            "prompts": ["whale shark"],
            "split": None,
        },
        {
            "dataset": "leopard",
            "image_root": str(SOURCE_IMAGE_ROOT / "leopard" / "images"),
            "input_path": str(leopard_json),
            "input_type": "json",
            "job_name": "leopard_full_20260620",
            "output_dir": str(SCRATCH_ROOT / "paper_full_datasets" / "leopard_full_20260620"),
            "prompts": ["leopard"],
            "split": None,
        },
        {
            "dataset": "saimaa_seal",
            "image_root": str(SOURCE_IMAGE_ROOT / "saimaa_seal"),
            "input_path": str(seal_json),
            "input_type": "json",
            "job_name": "saimaa_seal_full_20260620",
            "output_dir": str(SCRATCH_ROOT / "paper_full_datasets" / "saimaa_seal_full_20260620"),
            "prompts": ["seal"],
            "split": None,
        },
    ]
    with MANIFEST.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(path: Path) -> tuple[int, int]:
    data = read_json(path)
    return len(data.get("images", [])), len(data.get("annotations", []))


def main() -> None:
    whale_json, leopard_json = copy_existing_full_inputs()
    seal_json = combine_seal()
    write_manifest(whale_json, leopard_json, seal_json)
    for label, path in [
        ("whaleshark", whale_json),
        ("leopard", leopard_json),
        ("saimaa_seal", seal_json),
    ]:
        images, annotations = summarize(path)
        print(f"{label}: {images} images, {annotations} annotations -> {path}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
