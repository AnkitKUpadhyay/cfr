#!/usr/bin/env python3
"""Summarize evidence quadrants for the MegaDescriptor mixing ablation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


BASELINE_MODEL = "megad_full22"
MODEL_ORDER = [
    BASELINE_MODEL,
    "megad_fg25_fallback",
    "megad_fg50_fallback",
    "megad_fg75_fallback",
]


MODEL_SHORT_LABELS = {
    BASELINE_MODEL: "Full image",
    "megad_fg25_fallback": "Foreground 25%",
    "megad_fg50_fallback": "Foreground 50%",
    "megad_fg75_fallback": "Foreground 75%",
}

SCOPE_ORDER = ["Training overlap", "OOD"]

QUADRANT_ORDER = [
    "Foreground-driven",
    "Dataset leakage",
    "Shortcut-driven",
    "Weak retrieval",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["ssi"] = float(row["ssi"])
        row["cfr"] = float(row["cfr"])
        row["full"] = float(row["full"])
        row["foreground"] = float(row["foreground"])
        row["model_order"] = int(row["model_order"])
    return rows


def quadrant(row: dict[str, Any]) -> str:
    high_cfr = row["cfr"] >= 0.5
    low_ssi = row["ssi"] <= 0.5
    if high_cfr and low_ssi:
        return "Foreground-driven"
    if high_cfr:
        return "Dataset leakage"
    if not low_ssi:
        return "Shortcut-driven"
    return "Weak retrieval"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_quadrants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        row["dataset"]: quadrant(row)
        for row in rows
        if row["model"] == BASELINE_MODEL
    }
    output: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        for model in MODEL_ORDER:
            selected = [
                row
                for row in rows
                if row["evaluation_set"] == scope and row["model"] == model
            ]
            counts = Counter(quadrant(row) for row in selected)
            entered = sum(
                baseline[row["dataset"]] != "Foreground-driven"
                and quadrant(row) == "Foreground-driven"
                for row in selected
            )
            left = sum(
                baseline[row["dataset"]] == "Foreground-driven"
                and quadrant(row) != "Foreground-driven"
                for row in selected
            )
            output.append(
                {
                    "evaluation_set": scope,
                    "model": model,
                    "model_label": selected[0]["model_label"],
                    "num_datasets": len(selected),
                    "foreground_driven": counts["Foreground-driven"],
                    "dataset_leakage": counts["Dataset leakage"],
                    "shortcut_driven": counts["Shortcut-driven"],
                    "weak_retrieval": counts["Weak retrieval"],
                    "entered_foreground_driven": entered,
                    "left_foreground_driven": left,
                }
            )
    return output


def transition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {
        row["dataset"]: row
        for row in rows
        if row["model"] == BASELINE_MODEL
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "evaluation_set": row["evaluation_set"],
                "model": row["model"],
                "model_label": row["model_label"],
                "baseline_quadrant": quadrant(baseline[row["dataset"]]),
                "model_quadrant": quadrant(row),
                "baseline_ssi": f"{baseline[row['dataset']]['ssi']:.9f}",
                "baseline_cfr": f"{baseline[row['dataset']]['cfr']:.9f}",
                "ssi": f"{row['ssi']:.9f}",
                "cfr": f"{row['cfr']:.9f}",
                "delta_ssi": f"{row['ssi'] - baseline[row['dataset']]['ssi']:.9f}",
                "delta_cfr": f"{row['cfr'] - baseline[row['dataset']]['cfr']:.9f}",
            }
        )
    return output


def macro_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        for model in MODEL_ORDER:
            selected = [
                row
                for row in rows
                if row["evaluation_set"] == scope and row["model"] == model
            ]
            full = fmean(row["full"] for row in selected)
            foreground = fmean(row["foreground"] for row in selected)
            output.append(
                {
                    "evaluation_set": scope,
                    "model": model,
                    "training_input": MODEL_SHORT_LABELS[model],
                    "full": full,
                    "foreground": foreground,
                    "full_foreground_gap": full - foreground,
                    "ssi": fmean(row["ssi"] for row in selected),
                    "cfr": fmean(row["cfr"] for row in selected),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = read_rows(args.detailed_csv)
    rows = [row for row in all_rows if row["model"] in MODEL_ORDER]
    for row in rows:
        row["model_label"] = MODEL_SHORT_LABELS[row["model"]]
    if len(rows) != 34 * len(MODEL_ORDER):
        raise SystemExit(
            f"Expected {34 * len(MODEL_ORDER)} rows for the four reported "
            f"checkpoints, found {len(rows)}"
        )
    counts_by_model = Counter(row["model"] for row in rows)
    incomplete = {
        model: counts_by_model.get(model, 0)
        for model in MODEL_ORDER
        if counts_by_model.get(model, 0) != 34
    }
    if incomplete:
        raise SystemExit(f"Incomplete model counts: {incomplete}")
    quadrants = summarize_quadrants(rows)
    write_csv(args.output_dir / "megadescriptor_mix_ablation_summary_k1.csv", macro_summary(rows))
    write_csv(
        args.output_dir / "megadescriptor_mix_ablation_quadrant_counts_k1.csv",
        quadrants,
    )
    write_csv(
        args.output_dir / "megadescriptor_mix_ablation_quadrant_transitions_k1.csv",
        transition_rows(rows),
    )


if __name__ == "__main__":
    main()
