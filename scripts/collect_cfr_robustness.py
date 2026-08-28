#!/usr/bin/env python3
"""Collect CFR robustness JSON files and compare them with Table S3."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

import run_cfr_evaluation as base


METRICS = ("full", "foreground", "background", "silhouette", "ssi", "cfr")
SUMMARY_METRICS = ("random", *METRICS)
DISPLAY_NAMES = {
    "mask_erode": "Mask erosion",
    "mask_dilate": "Mask dilation",
    "mask_interior_false_negative": "Interior false negative",
    "mask_exterior_false_positive": "Exterior false positive",
    "gaussian_noise": "Gaussian noise",
    "gaussian_blur": "Gaussian blur",
    "jpeg_compression": "JPEG compression",
    "foreground_cutout": "Foreground CutOut",
    "background_cutout": "Background CutOut",
}
SEVERITY_ORDER = ("low", "medium", "high")
SEVERITY_CAPTIONS = {
    "low": (
        r"Mask and CutOut area $5\%$, Gaussian noise $\sigma=0.02$, "
        r"blur radius 1, and JPEG quality 80"
    ),
    "medium": (
        r"Mask and CutOut area $10\%$, Gaussian noise $\sigma=0.05$, "
        r"blur radius 2, and JPEG quality 50"
    ),
    "high": (
        r"Mask and CutOut area $20\%$, Gaussian noise $\sigma=0.10$, "
        r"blur radius 4, and JPEG quality 20"
    ),
}
EXPORT_FIELDS = (
    "dataset",
    "dataset_label",
    "model",
    "model_label",
    "condition",
    "condition_tag",
    "family",
    "severity",
    "severity_value",
    "seed",
    "scope",
    "k",
    "num_embeddings",
    "valid_queries",
    "random",
    "full",
    "foreground",
    "background",
    "silhouette",
    "shortcut_view",
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


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def final_scores(
    dataset: str,
    random_rate: float,
    full: float,
    foreground: float,
    background: float,
    silhouette: float,
) -> tuple[float | None, float | None, str]:
    signals = {
        "full": max(0.0, full - random_rate),
        "foreground": max(0.0, foreground - random_rate),
        "background": max(0.0, background - random_rate),
        "silhouette": max(0.0, silhouette - random_rate),
    }
    silhouette_identity = {
        base.clean_dataset_key(value)
        for value in base.DEFAULT_SILHOUETTE_IDENTITY_DATASETS
    }
    candidates = {"background": signals["background"]}
    if base.clean_dataset_key(dataset) not in silhouette_identity:
        candidates["silhouette"] = signals["silhouette"]
    shortcut_view = max(candidates, key=candidates.get)
    full_signal = signals["full"]
    headroom = 1.0 - random_rate
    cfr = (
        min(full_signal, signals["foreground"]) / headroom
        if headroom > 0
        else None
    )
    ssi = min(1.0, candidates[shortcut_view] / full_signal) if full_signal > 0 else None
    return cfr, ssi, shortcut_view


def baseline_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            random_rate = float(raw["random"])
            full = float(raw["full"])
            foreground = float(raw["foreground"])
            background = float(raw["background"])
            silhouette = float(raw["silhouette"])
            cfr, ssi, shortcut_view = final_scores(
                raw["dataset"],
                random_rate,
                full,
                foreground,
                background,
                silhouette,
            )
            rows.append(
                {
                    "dataset": raw["dataset"],
                    "dataset_label": raw.get("dataset_label", raw["dataset"]),
                    "model": raw["model"],
                    "model_label": raw.get("model_label", raw["model"]),
                    "k": int(raw.get("k") or 1),
                    "condition": "clean",
                    "condition_tag": "clean",
                    "source": "table_s3",
                    "family": "clean",
                    "severity": "clean",
                    "severity_value": 0.0,
                    "seed": 0,
                    "scope": "both",
                    "num_embeddings": int(raw["num_embeddings"]),
                    "valid_queries": int(raw["valid_queries"]),
                    "random": random_rate,
                    "full": full,
                    "foreground": foreground,
                    "background": background,
                    "silhouette": silhouette,
                    "shortcut_view": shortcut_view,
                    "ssi": ssi,
                    "cfr": cfr,
                }
            )
    return rows


def robustness_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("conditions/*/results/*/*_robustness.json")):
        result = json.loads(path.read_text())
        condition = result["condition"]
        summary_path = (
            path.parents[2]
            / "extract_summaries"
            / result["model"]["key"]
            / (
                f"{result['dataset']['key']}_{result['model']['key']}_"
                f"{result['condition_tag']}.json"
            )
        )
        extraction = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        for k_text, scores in result["cfr_by_k"].items():
            retrieval = scores["retrieval"]
            rows.append(
                {
                    "dataset": result["dataset"]["key"],
                    "dataset_label": result["dataset"]["label"],
                    "model": result["model"]["key"],
                    "model_label": result["model"]["label"],
                    "k": int(k_text),
                    "condition": condition["name"],
                    "condition_tag": result["condition_tag"],
                    "source": "robustness_run",
                    "family": condition["family"],
                    "severity": condition["level"],
                    "severity_value": condition["value"],
                    "seed": condition["seed"],
                    "scope": condition["scope"],
                    "num_embeddings": result["counts"]["num_embeddings"],
                    "valid_queries": result["counts"]["valid_queries"],
                    "random": scores["random"],
                    "full": retrieval.get("full"),
                    "foreground": retrieval.get("foreground"),
                    "background": retrieval.get("background"),
                    "silhouette": retrieval.get("silhouette"),
                    "shortcut_view": scores.get("shortcut_view"),
                    "ssi": scores.get("ssi"),
                    "cfr": scores.get("cfr"),
                    "mean_mask_changed_fraction": extraction.get(
                        "mean_mask_changed_fraction"
                    ),
                    "mean_image_changed_fraction": extraction.get(
                        "mean_image_changed_fraction"
                    ),
                }
            )
    return rows


def write_rows(
    rows: list[dict[str, Any]],
    path: Path,
    fields: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = fields or tuple(sorted({key for row in rows for key in row}))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def add_clean_deltas(rows: list[dict[str, Any]]) -> None:
    clean = {
        (row["dataset"], row["model"], row["k"]): row
        for row in rows
        if row.get("source") == "table_s3"
    }
    for row in rows:
        baseline = clean.get((row["dataset"], row["model"], row["k"]))
        for metric in SUMMARY_METRICS:
            value = as_float(row.get(metric))
            reference = as_float(baseline.get(metric)) if baseline else None
            row[f"delta_{metric}"] = (
                value - reference if value is not None and reference is not None else None
            )


def macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["condition"],
            row["severity"],
            row["severity_value"],
            row["seed"],
            row["scope"],
            row["model"],
            row["k"],
        )
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        condition, severity, value, seed, scope, model, k = key
        result: dict[str, Any] = {
            "condition": condition,
            "severity": severity,
            "severity_value": value,
            "seed": seed,
            "scope": scope,
            "model": model,
            "k": k,
            "datasets": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [as_float(row.get(metric)) for row in group]
            finite = [value for value in values if value is not None]
            result[f"mean_{metric}"] = sum(finite) / len(finite) if finite else None
            deltas = [as_float(row.get(f"delta_{metric}")) for row in group]
            finite_deltas = [value for value in deltas if value is not None]
            result[f"mean_delta_{metric}"] = (
                sum(finite_deltas) / len(finite_deltas) if finite_deltas else None
            )
            result[f"mean_abs_delta_{metric}"] = (
                sum(abs(item) for item in finite_deltas) / len(finite_deltas)
                if finite_deltas
                else None
            )
            paired = [
                (current - delta, current)
                for current, delta in zip(values, deltas)
                if current is not None and delta is not None
            ]
            if len(paired) >= 2:
                clean_values, perturbed_values = zip(*paired)
                rho = spearmanr(clean_values, perturbed_values).statistic
                result[f"spearman_{metric}"] = (
                    float(rho) if np.isfinite(rho) else None
                )
            else:
                result[f"spearman_{metric}"] = None
        output.append(result)
    return output


def latex_score(value: Any, signed: bool = False) -> str:
    number = as_float(value)
    if number is None:
        return "--"
    if round(number, 3) == 0:
        return "0.000"
    return f"{number:+.3f}" if signed else f"{number:.3f}"


def average_table_rows_across_seeds(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"])].append(row)

    fields = (
        "mean_delta_full",
        "mean_delta_cfr",
        "mean_abs_delta_cfr",
        "mean_delta_ssi",
        "mean_abs_delta_ssi",
        "spearman_cfr",
        "spearman_ssi",
    )
    output: list[dict[str, Any]] = []
    for group in groups.values():
        result = dict(group[0])
        result["seed_count"] = len(group)
        for field in fields:
            values = [as_float(row.get(field)) for row in group]
            finite = [value for value in values if value is not None]
            result[field] = sum(finite) / len(finite) if finite else None
        output.append(result)
    return output


def severity_table_lines(rows: list[dict[str, Any]], severity: str) -> list[str]:
    selected = average_table_rows_across_seeds(
        [
            row
            for row in rows
            if row["severity"] == severity and row["condition"] in DISPLAY_NAMES
        ]
    )
    condition_order = {
        name: index
        for index, name in enumerate(DISPLAY_NAMES)
    }
    model_order = {"miewid": 0, "megadescriptor": 1}
    selected.sort(
        key=lambda row: (
            condition_order[row["condition"]],
            model_order.get(row["model"], 99),
        )
    )
    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Perturbation} & \textbf{Model} & $\Delta$\textbf{Full} & $\Delta$\textbf{CFR} & $|\Delta\mathrm{CFR}|$ & $\Delta$\textbf{SSI} & $|\Delta\mathrm{SSI}|$ & $\rho_{\mathrm{CFR}}$ & $\rho_{\mathrm{SSI}}$ \\",
        r"\midrule",
    ]
    last_condition = None
    for row in selected:
        condition = row["condition"]
        if last_condition is not None and condition != last_condition:
            lines.append(r"\midrule")
        label = DISPLAY_NAMES[condition] if condition != last_condition else ""
        model = "MiewID" if row["model"] == "miewid" else "MegaDescriptor"
        lines.append(
            f"{label} & {model} & "
            f"{latex_score(row.get('mean_delta_full'), signed=True)} & "
            f"{latex_score(row.get('mean_delta_cfr'), signed=True)} & "
            f"{latex_score(row.get('mean_abs_delta_cfr'))} & "
            f"{latex_score(row.get('mean_delta_ssi'), signed=True)} & "
            f"{latex_score(row.get('mean_abs_delta_ssi'))} & "
            f"{latex_score(row.get('spearman_cfr'))} & "
            f"{latex_score(row.get('spearman_ssi'))} \\\\"
        )
        last_condition = condition
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                rf"\caption{{\textbf{{Sensitivity at {severity} perturbation severity.}} "
                + SEVERITY_CAPTIONS[severity]
                + r". Each row macro-averages paired changes over 34 datasets. "
                r"$\Delta$ is the perturbed score minus the clean score, "
                r"$|\Delta|$ is the mean absolute paired change, and $\rho$ is "
                r"the Spearman correlation between clean and perturbed "
                r"dataset-level scores. Stochastic conditions are averaged "
                r"over three seeds; deterministic conditions use one run.}"
            ),
            rf"\label{{tab:cfr-robustness-{severity}}}",
            r"\end{table*}",
            "",
        ]
    )
    return lines


def write_all_severity_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for severity in SEVERITY_ORDER:
        lines.extend(severity_table_lines(rows, severity))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--baseline_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_root / "summaries"
    rows = baseline_rows(args.baseline_csv) + robustness_rows(args.results_root)
    add_clean_deltas(rows)
    write_rows(
        rows,
        output_dir / "cfr_robustness_all_rows.csv",
        fields=EXPORT_FIELDS,
    )
    for severity in SEVERITY_ORDER:
        write_rows(
            [row for row in rows if row.get("severity") == severity],
            output_dir / f"cfr_robustness_{severity}_all_datasets.csv",
            fields=EXPORT_FIELDS,
        )
    macro = macro_rows(rows)
    write_rows(macro, output_dir / "cfr_robustness_macro.csv")
    write_all_severity_latex(
        macro, output_dir / "cfr_robustness_all_severities.tex"
    )
    print(f"Collected {len(rows)} rows under {output_dir}")


if __name__ == "__main__":
    main()
