#!/usr/bin/env python3
"""Summarize evidence quadrants for the MegaDescriptor mixing ablation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


BASELINE_MODEL = "megad_full22"
MODEL_ORDER = [
    BASELINE_MODEL,
    "megad_fg25_fallback",
    "megad_fg50_fallback",
    "megad_fg75_fallback",
]

VARIANT_ORDER = [model for model in MODEL_ORDER if model != BASELINE_MODEL]

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


def macro_points(rows: list[dict[str, Any]], scope: str) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for model in MODEL_ORDER:
        selected = [
            row
            for row in rows
            if row["evaluation_set"] == scope and row["model"] == model
        ]
        points[model] = (
            fmean(row["cfr"] for row in selected),
            1.0 - fmean(row["ssi"] for row in selected),
        )
    return points


def add_quadrants(axis: Any, label_size: float = 8) -> None:
    regions = [
        (0.0, 0.5, 0.5, 0.5, "#dce9f6", "Weak retrieval", 0.25, 0.94),
        (0.5, 0.5, 0.5, 0.5, "#dcefdc", "Foreground-driven", 0.75, 0.94),
        (0.0, 0.0, 0.5, 0.5, "#f4dddd", "Shortcut-driven", 0.25, 0.06),
        (0.5, 0.0, 0.5, 0.5, "#f5ebc9", "Dataset leakage", 0.75, 0.06),
    ]
    for x, y, width, height, color, label, text_x, text_y in regions:
        axis.add_patch(
            Rectangle((x, y), width, height, facecolor=color, edgecolor="none", zorder=0)
        )
        axis.text(
            text_x,
            text_y,
            label,
            ha="center",
            va="center",
            fontsize=label_size,
            color="#4a4a4a",
        )
    axis.axvline(0.5, color="#777777", linewidth=0.8)
    axis.axhline(0.5, color="#777777", linewidth=0.8)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="white", linewidth=0.6, alpha=0.8)


def plot_macro_figure(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.55), constrained_layout=True)
    color = "#c45d2d"
    for axis, scope in zip(axes, SCOPE_ORDER):
        add_quadrants(axis)
        points = macro_points(rows, scope)
        xs = [points[model][0] for model in MODEL_ORDER]
        ys = [points[model][1] for model in MODEL_ORDER]
        axis.plot(
            xs,
            ys,
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            label="Foreground-mixed training",
            zorder=3,
        )
        axis.annotate(
            "",
            xy=(xs[-1], ys[-1]),
            xytext=(xs[-2], ys[-2]),
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.4},
            zorder=4,
        )
        baseline_x, baseline_y = points[BASELINE_MODEL]
        axis.scatter(
            [baseline_x],
            [baseline_y],
            marker="*",
            s=70,
            color="#222222",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.annotate(
            "Full image",
            (baseline_x, baseline_y),
            xytext=(6, -11 if scope == "Training overlap" else -10),
            textcoords="offset points",
            fontsize=7,
            color="#222222",
        )
        axis.set_title(f"{scope} ({22 if scope == 'Training overlap' else 12})", fontsize=10)
        axis.set_xlabel("CFR@1")
    axes[0].set_ylabel("1 - SSI@1")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=1,
        frameon=False,
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


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


def latex_score(value: float) -> str:
    return f"{value:.3f}"


def write_summary_latex(
    macro: list[dict[str, Any]],
    quadrants: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"\textbf{Evaluation set} & \textbf{Training input} & \textbf{Full} & \textbf{Fg} & \textbf{Full$-$Fg} & \textbf{SSI} & \textbf{CFR} \\",
        r"\midrule",
    ]
    previous_scope = None
    for row in macro:
        scope = row["evaluation_set"]
        if previous_scope is not None and scope != previous_scope:
            lines.append(r"\midrule")
        scope_label = (
            f"{scope} ({22 if scope == 'Training overlap' else 12})"
            if scope != previous_scope
            else ""
        )
        lines.append(
            f"{scope_label} & {row['training_input']} & "
            f"{latex_score(row['full'])} & "
            f"{latex_score(row['foreground'])} & "
            f"{latex_score(row['full_foreground_gap'])} & "
            f"{latex_score(row['ssi'])} & "
            f"{latex_score(row['cfr'])} \\\\"
        )
        previous_scope = scope
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\textbf{Macro-average foreground-mixing ablation for MegaDescriptor.} Full image is MegaD$^C$, trained on the same 22-dataset collection with no foreground-only inputs. Foreground 25\%, 50\%, and 75\% denote the probability of selecting a foreground-only training input. Samples without an accepted mask use the full image. Full$-$Fg is the raw retrieval gap.}",
            r"\label{tab:megad-mix-ablation-summary}",
            r"\end{table*}",
            "",
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.6pt}",
            r"\begin{tabular}{@{}llrrrrrr@{}}",
            r"\toprule",
            r"\textbf{Evaluation set} & \textbf{Training input} & \textbf{Foreground-driven} & \textbf{Leakage} & \textbf{Shortcut} & \textbf{Weak} & \textbf{Enter} & \textbf{Leave} \\",
            r"\midrule",
        ]
    )
    previous_scope = None
    for row in quadrants:
        scope = row["evaluation_set"]
        if previous_scope is not None and scope != previous_scope:
            lines.append(r"\midrule")
        scope_label = (
            f"{scope} ({row['num_datasets']})" if scope != previous_scope else ""
        )
        baseline = row["model"] == BASELINE_MODEL
        enter = "--" if baseline else str(row["entered_foreground_driven"])
        leave = "--" if baseline else str(row["left_foreground_driven"])
        lines.append(
            f"{scope_label} & {MODEL_SHORT_LABELS[row['model']]} & "
            f"{row['foreground_driven']} & {row['dataset_leakage']} & "
            f"{row['shortcut_driven']} & {row['weak_retrieval']} & "
            f"{enter} & {leave} \\\\"
        )
        previous_scope = scope
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\textbf{Evidence-quadrant distribution under foreground-mixed training.} Foreground-driven uses $\mathrm{CFR}\geq0.5$ and $\mathrm{SSI}\leq0.5$. Leakage, Shortcut, and Weak denote the other quadrants in Fig.~\ref{fig:dataset_evidence}. Enter and Leave count transitions into and out of the foreground-driven quadrant relative to the full-image MegaD$^C$ baseline. The $0.5$ boundaries are descriptive.}",
            r"\label{tab:megad-mix-ablation-quadrants}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def plot_dataset_movements(rows: list[dict[str, Any]], output: Path) -> None:
    panel_models = [
        "megad_fg25_fallback",
        "megad_fg50_fallback",
        "megad_fg75_fallback",
    ]
    panel_titles = {
        "megad_fg25_fallback": "Foreground 25%",
        "megad_fg50_fallback": "Foreground 50%",
        "megad_fg75_fallback": "Foreground 75%",
    }
    colors = {"Training overlap": "#147a66", "OOD": "#b4553d"}
    markers = {"Training overlap": "o", "OOD": "^"}
    lookup = {(row["model"], row["dataset"]): row for row in rows}
    datasets = sorted({row["dataset"] for row in rows})

    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), constrained_layout=True)
    for panel_index, (axis, model) in enumerate(zip(axes, panel_models)):
        add_quadrants(axis, label_size=5.7)
        emphasized: list[dict[str, Any]] = []
        for dataset in datasets:
            baseline = lookup[(BASELINE_MODEL, dataset)]
            variant = lookup[(model, dataset)]
            scope = baseline["evaluation_set"]
            start = (baseline["cfr"], 1.0 - baseline["ssi"])
            end = (variant["cfr"], 1.0 - variant["ssi"])
            changes_quadrant = quadrant(baseline) != quadrant(variant)
            axis.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": colors[scope],
                    "linewidth": 1.15 if changes_quadrant else 0.45,
                    "alpha": 0.9 if changes_quadrant else 0.28,
                    "shrinkA": 1.5,
                    "shrinkB": 1.5,
                },
                zorder=3 if changes_quadrant else 2,
            )
            if changes_quadrant:
                emphasized.append(variant)
        baseline_rows = [lookup[(BASELINE_MODEL, dataset)] for dataset in datasets]
        axis.scatter(
            [row["cfr"] for row in baseline_rows],
            [1.0 - row["ssi"] for row in baseline_rows],
            marker="x",
            s=11,
            linewidth=0.55,
            color="#555555",
            alpha=0.55,
            zorder=3,
        )
        for scope in SCOPE_ORDER:
            selected = [
                lookup[(model, dataset)]
                for dataset in datasets
                if lookup[(model, dataset)]["evaluation_set"] == scope
            ]
            axis.scatter(
                [row["cfr"] for row in selected],
                [1.0 - row["ssi"] for row in selected],
                marker=markers[scope],
                s=16,
                facecolor=colors[scope],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                zorder=4,
            )
        for scope in SCOPE_ORDER:
            selected = [
                row for row in emphasized if row["evaluation_set"] == scope
            ]
            axis.scatter(
                [row["cfr"] for row in selected],
                [1.0 - row["ssi"] for row in selected],
                marker=markers[scope],
                s=32,
                facecolor="none",
                edgecolor="#202020",
                linewidth=0.75,
                zorder=5,
            )
        axis.set_title(panel_titles[model], fontsize=9)
        axis.tick_params(labelsize=6.5)
        axis.set_xlabel("CFR@1", fontsize=8)
        if panel_index == 0:
            axis.set_ylabel("1 - SSI@1", fontsize=8)

    legend = [
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color="#555555",
            label="Full-image baseline",
        ),
        Line2D(
            [0],
            [0],
            marker=markers["Training overlap"],
            linestyle="none",
            markerfacecolor=colors["Training overlap"],
            markeredgecolor="white",
            label="Represented datasets",
        ),
        Line2D(
            [0],
            [0],
            marker=markers["OOD"],
            linestyle="none",
            markerfacecolor=colors["OOD"],
            markeredgecolor="white",
            label="Unseen datasets",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


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
    write_csv(
        args.output_dir / "megadescriptor_mix_ablation_quadrant_counts_k1.csv",
        quadrants,
    )
    write_csv(
        args.output_dir / "megadescriptor_mix_ablation_quadrant_transitions_k1.csv",
        transition_rows(rows),
    )
    write_summary_latex(
        macro_summary(rows),
        quadrants,
        args.output_dir / "megadescriptor_mix_ablation_summary.tex",
    )
    plot_macro_figure(
        rows,
        args.output_dir / "megadescriptor_mix_ablation_macro_paths.pdf",
    )
    plot_dataset_movements(
        rows,
        args.output_dir / "megadescriptor_mix_ablation_quadrant_movements.pdf",
    )


if __name__ == "__main__":
    main()
