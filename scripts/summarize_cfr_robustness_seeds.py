#!/usr/bin/env python3
"""Split CFR robustness rows by seed and summarize variation across seeds."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SEVERITIES = ("low", "medium", "high")
DATASET_GROUP_FIELDS = (
    "dataset",
    "dataset_label",
    "model",
    "model_label",
    "condition",
    "family",
    "severity",
    "severity_value",
    "scope",
    "k",
    "num_embeddings",
    "valid_queries",
)
DATASET_METRICS = (
    "random",
    "full",
    "foreground",
    "background",
    "silhouette",
    "ssi",
    "cfr",
    "delta_full",
    "delta_foreground",
    "delta_background",
    "delta_silhouette",
    "delta_ssi",
    "delta_cfr",
    "mean_mask_changed_fraction",
    "mean_image_changed_fraction",
)
MACRO_GROUP_FIELDS = (
    "condition",
    "severity",
    "severity_value",
    "scope",
    "model",
    "k",
    "datasets",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_groups(
    rows: list[dict[str, str]],
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        result: dict[str, Any] = dict(zip(group_fields, key))
        seeds = sorted({int(row["seed"]) for row in group})
        result["seed_count"] = len(seeds)
        result["seeds"] = ";".join(map(str, seeds))
        shortcut_views = sorted(
            {row.get("shortcut_view", "") for row in group if row.get("shortcut_view")}
        )
        if shortcut_views:
            result["shortcut_views"] = ";".join(shortcut_views)
        for field in metric_fields:
            values = finite_values(group, field)
            result[f"{field}_mean"] = statistics.fmean(values) if values else ""
            result[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
            )
        output.append(result)

    fields = [*group_fields, "seed_count", "seeds"]
    if any("shortcut_views" in row for row in output):
        fields.append("shortcut_views")
    for field in metric_fields:
        fields.extend((f"{field}_mean", f"{field}_std"))
    return output, fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir

    severity_rows: list[dict[str, str]] = []
    for severity in SEVERITIES:
        path = args.input_dir / f"cfr_robustness_{severity}_all_datasets.csv"
        rows = read_rows(path)
        severity_rows.extend(rows)
        fields = list(rows[0]) if rows else []
        for seed in (0, 1, 2):
            selected = [row for row in rows if int(row["seed"]) == seed]
            write_rows(
                output_dir / f"cfr_robustness_{severity}_seed{seed}.csv",
                selected,
                fields,
            )

    dataset_summary, dataset_fields = summarize_groups(
        severity_rows, DATASET_GROUP_FIELDS, DATASET_METRICS
    )
    write_rows(
        output_dir / "cfr_robustness_dataset_seed_mean_std.csv",
        dataset_summary,
        dataset_fields,
    )

    macro_rows = read_rows(args.input_dir / "cfr_robustness_macro.csv")
    macro_metrics = tuple(
        field
        for field in macro_rows[0]
        if field not in {*MACRO_GROUP_FIELDS, "seed"}
    )
    macro_summary, macro_fields = summarize_groups(
        macro_rows, MACRO_GROUP_FIELDS, macro_metrics
    )
    write_rows(
        output_dir / "cfr_robustness_macro_seed_mean_std.csv",
        macro_summary,
        macro_fields,
    )

    print(
        f"Wrote 9 seed-specific CSVs, {len(dataset_summary)} dataset summaries, "
        f"and {len(macro_summary)} macro summaries under {output_dir}"
    )


if __name__ == "__main__":
    main()
