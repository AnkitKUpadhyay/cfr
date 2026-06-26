#!/usr/bin/env python3
"""Build JSONL job manifests for WildlifeDatasets SAM mask extraction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT = Path(os.environ.get("CFR_CODE_ROOT", str(Path(__file__).resolve().parents[1])))
WILDLIFE_ROOT = Path(os.environ.get("WILDLIFEDATASETS_ROOT", "/path/to/WildlifeDatasets"))
MEGADESC_ROOT = Path(
    os.environ.get("MEGADESCRIPTOR_DATASETS_ROOT", "/path/to/megadescriptor-train-datasets")
)
SOURCE_IMAGE_ROOT = Path(os.environ.get("SOURCE_IMAGE_ROOT", "/path/to/source_images"))
PAPER_METADATA_ROOT = Path(os.environ.get("PAPER_METADATA_ROOT", "/path/to/cfr_metadata"))


CROSS_VIEW_DATASETS = {
    "seaturtle_head": {
        "image_root": SOURCE_IMAGE_ROOT / "SeaTurtleID2022",
        "input_json": PAPER_METADATA_ROOT / "seaturtle_head_gzcd_compat.json",
        "prompt": ["sea turtle head", "sea turtle"],
    },
    "stripespotter": {
        "image_root": SOURCE_IMAGE_ROOT / "StripeSpotter",
        "input_json": PAPER_METADATA_ROOT / "stripespotter_gzcd_compat.json",
        "prompt": ["zebra"],
    },
    "hyenaid2022": {
        "image_root": WILDLIFE_ROOT / "HyenaID2022",
        "input_json": PAPER_METADATA_ROOT / "hyenaid2022_gzcd_compat.json",
        "prompt": ["hyena"],
    },
    "nyaladata": {
        "image_root": WILDLIFE_ROOT / "NyalaData",
        "input_json": PAPER_METADATA_ROOT / "nyaladata_gzcd_compat.json",
        # The "nyala" prompt produced no detections in the original SAM3
        # mask-control run; the paper masks used the broader "animal" prompt.
        "prompt": ["animal"],
    },
    "reunionturtles": {
        "image_root": WILDLIFE_ROOT / "ReunionTurtles",
        "input_json": PROJECT / "wildlifedatasets_cross_view" / "jsons" / "reunionturtles_gzcd_compat.json",
        "prompt": ["sea turtle", "turtle"],
    },
    "amvrakikosturtles": {
        "image_root": WILDLIFE_ROOT / "AmvrakikosTurtles",
        "input_json": PROJECT / "wildlifedatasets_cross_view" / "jsons" / "amvrakikosturtles_gzcd_compat.json",
        "prompt": ["sea turtle head", "sea turtle", "turtle"],
    },
    "zakynthosturtles": {
        "image_root": WILDLIFE_ROOT / "ZakynthosTurtles",
        "input_json": PROJECT / "wildlifedatasets_cross_view" / "jsons" / "zakynthosturtles_gzcd_compat.json",
        "prompt": ["sea turtle", "turtle"],
    },
    "melops": {
        "image_root": WILDLIFE_ROOT / "Melops",
        "input_json": PROJECT / "wildlifedatasets_cross_view" / "jsons" / "melops_gzcd_compat.json",
        "prompt": ["fish"],
    },
}


MEGADESC_PROMPTS = {
    "AAUZebraFish": ["zebrafish", "fish"],
    "ATRW": ["tiger"],
    "AerialCattle2017": ["cow", "cattle"],
    "BelugaID": ["beluga whale", "whale"],
    "BirdIndividualID": ["bird"],
    "CTai": ["chimpanzee"],
    "CZoo": ["chimpanzee"],
    "Cows2021": ["cow", "cattle"],
    "FriesianCattle2015": ["cow", "cattle"],
    "FriesianCattle2017": ["cow", "cattle"],
    "GiraffeZebraID": ["giraffe", "zebra"],
    "Giraffes": ["giraffe"],
    "HappyWhale": ["whale"],
    "HumpbackWhaleID": ["humpback whale", "whale"],
    "IPanda50": ["giant panda", "panda"],
    "LionData": ["lion"],
    "MacaqueFaces": ["macaque", "monkey"],
    "NDD20": ["dolphin"],
    "NOAARightWhale": ["right whale", "whale"],
    "OpenCows2020": ["cow", "cattle"],
    "SMALST": ["zebra"],
    "ZindiTurtleRecall": ["sea turtle", "turtle"],
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def cross_view_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset, cfg in CROSS_VIEW_DATASETS.items():
        input_json = cfg["input_json"]
        if not input_json.exists():
            continue
        rows.append(
            {
                "job_name": dataset,
                "input_type": "json",
                "input_path": str(input_json),
                "image_root": str(cfg["image_root"]),
                "output_dir": str(output_root / "paper_wildlifedatasets" / dataset),
                "dataset": dataset,
                "split": None,
                "prompts": cfg["prompt"],
            }
        )
    return rows


def megadescriptor_rows(output_root: Path, datasets: list[str] | None) -> list[dict[str, Any]]:
    rows = []
    metadata_root = MEGADESC_ROOT / "metadata" / "datasets"
    image_root = MEGADESC_ROOT / "images" / "size-518"
    for metadata_csv in sorted(metadata_root.glob("*/metadata.csv")):
        dataset = metadata_csv.parent.name
        if datasets and dataset not in datasets:
            continue
        rows.append(
            {
                "job_name": f"megadescriptor_{dataset}",
                "input_type": "csv",
                "input_path": str(metadata_csv),
                "image_root": str(image_root),
                "output_dir": str(output_root / "megadescriptor_train" / dataset),
                "dataset": dataset,
                "split": None,
                "prompts": MEGADESC_PROMPTS.get(dataset, ["animal"]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_manifest",
        type=Path,
        default=PROJECT / "wildlifedatasets_sam_mask_jobs.jsonl",
    )
    parser.add_argument(
        "--mask_output_root",
        type=Path,
        default=Path(os.environ.get("CFR_MASK_OUTPUT_ROOT", "outputs/sam_masks")),
    )
    parser.add_argument("--include_cross_view", action="store_true")
    parser.add_argument("--include_megadescriptor", action="store_true")
    parser.add_argument("--megadescriptor_datasets", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    if args.include_cross_view:
        rows.extend(cross_view_rows(args.mask_output_root))
    if args.include_megadescriptor:
        rows.extend(megadescriptor_rows(args.mask_output_root, args.megadescriptor_datasets))
    if not rows:
        raise SystemExit("No jobs requested. Use --include_cross_view and/or --include_megadescriptor.")
    write_jsonl(args.output_manifest, rows)
    print(f"Wrote {len(rows)} jobs to {args.output_manifest}")


if __name__ == "__main__":
    main()
