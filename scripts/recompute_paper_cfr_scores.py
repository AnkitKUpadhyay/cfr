#!/usr/bin/env python3
"""Recompute paper SSI and CFR columns from saved CFR retrieval summaries.

This script does not rerun embedding extraction or retrieval. It consumes a CSV
written by run_cfr_evaluation.py or summarize_cfr_results.py and recomputes the
paper definitions:

  S_T = max(0, R_T - R_rand)
  CFR = min(S_full, S_fg) / (1 - R_rand)
  SSI = min(1, S_shortcut / S_full)

By default, silhouette is treated as a shortcut view except for datasets where
contour/outline can be intended identity evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_SILHOUETTE_IDENTITY_DATASETS = {
    "belugaid",
    "happywhale",
    "humpbackwhaleid",
    "noaarightwhale",
    "whale shark",
    "whaleshark",
}


def clean_key(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def parse_float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                parsed = float(value)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return parsed
    return None


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def recompute_row(
    row: dict[str, str],
    silhouette_identity_keys: set[str],
) -> dict[str, Any]:
    dataset = row.get("dataset_label") or row.get("dataset") or ""
    model = row.get("model_label") or row.get("model") or ""
    dataset_key = row.get("dataset") or dataset
    merged_dataset_key = clean_key(dataset_key)
    merged_dataset_label = clean_key(dataset)

    random_rate = parse_float(row, "random", "random_at1", "Random")
    full = parse_float(row, "full", "full_at1", "Full")
    foreground = parse_float(row, "foreground", "foreground_at1", "Fg", "Frg.")
    background = parse_float(row, "background", "background_at1", "Bg", "Bgr+Sil")
    silhouette = parse_float(row, "silhouette", "silhouette_at1", "Sil")

    if random_rate is None or full is None or foreground is None or background is None:
        raise ValueError(f"Missing required retrieval columns for {dataset}/{model}")

    signals = {
        "full": max(0.0, full - random_rate),
        "foreground": max(0.0, foreground - random_rate),
        "background": max(0.0, background - random_rate),
    }
    if silhouette is not None:
        signals["silhouette"] = max(0.0, silhouette - random_rate)

    use_silhouette = (
        silhouette is not None
        and merged_dataset_key not in silhouette_identity_keys
        and merged_dataset_label not in silhouette_identity_keys
    )
    shortcut_view = "background"
    shortcut_signal = signals["background"]
    if use_silhouette and signals.get("silhouette", 0.0) > shortcut_signal:
        shortcut_view = "silhouette"
        shortcut_signal = signals["silhouette"]

    headroom = max(0.0, 1.0 - random_rate)
    cfr = min(signals["full"], signals["foreground"]) / headroom if headroom > 0 else None
    ssi = min(1.0, shortcut_signal / signals["full"]) if signals["full"] > 0 else None

    return {
        "dataset": dataset,
        "model": model,
        "random_at1": random_rate,
        "full_at1": full,
        "foreground_at1": foreground,
        "background_at1": background,
        "silhouette_at1": silhouette,
        "shortcut_view": shortcut_view,
        "ssi": ssi,
        "cfr": cfr,
    }


def read_rows(path: Path, k: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if k is None or "k" not in rows[0]:
        return rows
    return [row for row in rows if str(row.get("k")) == str(k)]


def write_compact_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "random_at1",
        "full_at1",
        "foreground_at1",
        "background_at1",
        "silhouette_at1",
        "shortcut_view",
        "ssi",
        "cfr",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format_float(row[key])
                    if isinstance(row.get(key), float)
                    else row.get(key, "")
                    for key in fieldnames
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument(
        "--silhouette_identity_datasets",
        nargs="*",
        default=sorted(DEFAULT_SILHOUETTE_IDENTITY_DATASETS),
        help="Datasets where silhouette should not be counted as shortcut evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity_keys = {clean_key(value) for value in args.silhouette_identity_datasets}
    rows = [
        recompute_row(row, identity_keys)
        for row in read_rows(args.input_csv, args.k)
    ]
    write_compact_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
