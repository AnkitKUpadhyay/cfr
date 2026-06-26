#!/usr/bin/env python3
"""Download selected WildlifeDatasets datasets and export catalogues.

This script is intentionally separate from the embedding environments.  The
base Python environment already has wildlife-datasets installed, while the
model extraction environment does not need it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from wildlife_datasets import datasets
from wildlife_datasets.datasets.datasets import WildlifeDataset
from wildlife_datasets.datasets.downloads import DownloadKaggle


PACKAGE_DATASETS = {
    "CzechLynx": "CzechLynxv2",
    "ReunionTurtles": "ReunionTurtles",
    "AmvrakikosTurtles": "AmvrakikosTurtles",
    "ZakynthosTurtles": "ZakynthosTurtles",
    "BrownBearHeads": "BrownBearHeads",
}


class BrownBearHeads(DownloadKaggle, WildlifeDataset):
    """Temporary local wrapper until the PyPI package exposes BrownBearHeads."""

    summary = {
        "licenses": "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        "licenses_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "url": "https://www.kaggle.com/datasets/picekl/brown-bear-heads",
        "publication_url": "https://doi.org/10.1016/j.cub.2025.12.022",
        "cite": "rosenberg2026individual",
        "animals": {"brown bear"},
        "animals_simple": "bears",
        "real_animals": True,
        "year": 2026,
        "reported_n_total": 72939,
        "reported_n_individuals": 109,
        "wild": True,
        "clear_photos": True,
        "pose": "multiple",
        "unique_pattern": False,
        "from_video": False,
        "cropped": True,
        "span": "5.1 years",
        "size": 6740,
    }
    kaggle_url = "picekl/brown-bear-heads"
    kaggle_type = "datasets"

    def create_catalogue(self) -> pd.DataFrame:
        metadata_path = self.find_metadata_path()
        df = pd.read_csv(metadata_path, low_memory=False)
        if "image_id" not in df.columns:
            df["image_id"] = range(len(df))
        if "species" not in df.columns:
            df["species"] = "brown bear"
        return self.finalize_catalogue(df)

    def find_metadata_path(self) -> str:
        assert self.root is not None
        candidates = [
            Path(self.root) / "metadata.csv",
            Path(self.root) / "BrownBearHeads" / "metadata.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError("Could not find BrownBearHeads metadata.csv in the provided root.")


LOCAL_DATASET_CLASSES = {
    "BrownBearHeads": BrownBearHeads,
}

MELops_ZENODO = {
    "metadata": "https://zenodo.org/records/17404087/files/Melops_metadata.txt?download=1",
    "head": "https://zenodo.org/records/17404087/files/Melops_head.zip?download=1",
}

SIDE_HINT_COLUMNS = (
    "viewpoint",
    "orientation",
    "side",
    "position",
    "view",
    "image_side",
)
DATE_HINT_COLUMNS = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "year",
    "encounter",
    "encounter_id",
    "sighting",
    "sightingnumb",
    "filename_year",
    "dayseq",
)


def json_default(value: Any) -> str:
    return str(value)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_write_json(data: dict[str, Any], path: Path) -> None:
    mkdir(path.parent)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def summarize_dataframe(df: pd.DataFrame, dataset_name: str, root: Path) -> dict[str, Any]:
    lowered = {col.lower(): col for col in df.columns}
    side_cols = [lowered[key] for key in SIDE_HINT_COLUMNS if key in lowered]
    date_cols = [lowered[key] for key in DATE_HINT_COLUMNS if key in lowered]

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "root": str(root),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "side_candidate_columns": side_cols,
        "date_or_encounter_candidate_columns": date_cols,
    }
    for col in ("identity", "ID", "id", "name"):
        if col in df.columns:
            summary[f"unique_{col}"] = int(df[col].nunique(dropna=True))
    for col in side_cols + date_cols:
        values = df[col].dropna()
        value_strings = values.map(str)
        summary[f"{col}_nunique"] = int(value_strings.nunique())
        summary[f"{col}_sample_values"] = [str(x) for x in value_strings.unique()[:12]]
    if "path" in df.columns:
        summary["path_examples"] = [str(x) for x in df["path"].dropna().astype(str).head(5)]
    return summary


def get_dataset_class(class_name: str):
    cls = getattr(datasets, class_name, None)
    if cls is not None:
        return cls
    cls = LOCAL_DATASET_CLASSES.get(class_name)
    if cls is not None:
        return cls
    raise AttributeError(f"wildlife_datasets.datasets has no class {class_name}")


def export_package_dataset(dataset_name: str, root: Path, stats_root: Path, force: bool) -> None:
    cls_name = PACKAGE_DATASETS[dataset_name]
    cls = get_dataset_class(cls_name)
    print(f"[download] {dataset_name} -> {root}", flush=True)
    cls.get_data(str(root), force=force)

    print(f"[catalogue] {dataset_name}", flush=True)
    dataset = cls(str(root), remove_unknown=False, check_files=True)
    df = dataset.df.copy()
    if dataset_name == "CzechLynx" and "real_animal" in df.columns:
        df = df[df["real_animal"] == True].reset_index(drop=True)

    mkdir(stats_root)
    catalogue_path = stats_root / f"{dataset_name}_catalogue.csv"
    summary_path = stats_root / f"{dataset_name}_catalogue_summary.json"
    dataset_summary_path = stats_root / f"{dataset_name}_wildlife_summary.json"

    df.to_csv(catalogue_path, index=False)
    safe_write_json(summarize_dataframe(df, dataset_name, root), summary_path)
    safe_write_json({"dataset": dataset_name, "summary": dataset.summary}, dataset_summary_path)

    print(f"[wrote] {catalogue_path}", flush=True)
    print(f"[wrote] {summary_path}", flush=True)


def download_file(url: str, output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"[skip] {output_path} exists", flush=True)
        return
    mkdir(output_path.parent)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[download] {url} -> {output_path}", flush=True)
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp_path.rename(output_path)


def read_melops_metadata(path: Path) -> pd.DataFrame:
    # The Zenodo record describes a text table.  sep=None lets pandas infer tab,
    # semicolon, or comma delimiters after download.
    return pd.read_csv(path, sep=None, engine="python")


def find_image_paths(root: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            paths.setdefault(path.stem, path)
            paths.setdefault(path.name, path)
    return paths


def export_melops(root: Path, stats_root: Path, force: bool, extract: bool) -> None:
    mkdir(root)
    metadata_path = root / "Melops_metadata.txt"
    head_zip_path = root / "Melops_head.zip"
    head_extract_root = root / "Melops_head"

    download_file(MELops_ZENODO["metadata"], metadata_path, force=force)
    download_file(MELops_ZENODO["head"], head_zip_path, force=force)

    if extract and (force or not head_extract_root.exists()):
        print(f"[extract] {head_zip_path} -> {head_extract_root}", flush=True)
        mkdir(head_extract_root)
        with zipfile.ZipFile(head_zip_path) as zf:
            zf.extractall(head_extract_root)

    df = read_melops_metadata(metadata_path)
    if "ID" in df.columns and "identity" not in df.columns:
        df["identity"] = df["ID"].astype(str)
    if "side" in df.columns and "orientation" not in df.columns:
        df["orientation"] = df["side"].astype(str)

    if extract and head_extract_root.exists():
        image_lookup = find_image_paths(head_extract_root)
        paths = []
        for _, row in df.iterrows():
            candidates = []
            for col in ("filename_year", "filename", "image", "path"):
                if col in df.columns and pd.notna(row.get(col)):
                    candidates.append(str(row[col]))
            match = None
            for candidate in candidates:
                key = Path(candidate).name
                match = image_lookup.get(key) or image_lookup.get(Path(key).stem)
                if match is not None:
                    break
            paths.append(str(match.relative_to(root)) if match is not None else "")
        df["path"] = paths
        df["matched_head_image"] = df["path"].astype(bool)

    mkdir(stats_root)
    catalogue_path = stats_root / "Melops_catalogue.csv"
    summary_path = stats_root / "Melops_catalogue_summary.json"
    df.to_csv(catalogue_path, index=False)
    summary = summarize_dataframe(df, "Melops", root)
    if "matched_head_image" in df.columns:
        summary["matched_head_images"] = int(df["matched_head_image"].sum())
    safe_write_json(summary, summary_path)

    print(f"[wrote] {catalogue_path}", flush=True)
    print(f"[wrote] {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("WILDLIFEDATASETS_ROOT", "data/WildlifeDatasets")),
        help="Dataset storage root.",
    )
    parser.add_argument(
        "--stats_root",
        type=Path,
        default=Path(os.environ.get("WILDLIFEDATASETS_STATS_ROOT", "outputs/catalogues")),
        help="Where exported catalogues and audit summaries are written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CzechLynx", "ReunionTurtles", "AmvrakikosTurtles", "ZakynthosTurtles", "Melops"],
        help="Dataset names to download/export.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing archives/data.")
    parser.add_argument(
        "--no_extract_melops",
        action="store_true",
        help="Download Melops metadata/head zip but do not extract the head images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mkdir(args.root)
    mkdir(args.stats_root)

    for dataset_name in args.datasets:
        dataset_root = args.root / dataset_name
        if dataset_name in PACKAGE_DATASETS:
            export_package_dataset(dataset_name, dataset_root, args.stats_root, args.force)
        elif dataset_name == "Melops":
            if hasattr(datasets, "Melops"):
                export_package_dataset(dataset_name, dataset_root, args.stats_root, args.force)
            else:
                print("[info] wildlife_datasets.datasets.Melops is unavailable; using Zenodo files.", flush=True)
                export_melops(
                    dataset_root,
                    args.stats_root,
                    args.force,
                    extract=not args.no_extract_melops,
                )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    print("[done]", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
