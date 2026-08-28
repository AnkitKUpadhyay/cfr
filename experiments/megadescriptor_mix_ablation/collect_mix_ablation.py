#!/usr/bin/env python3
"""Collect full-image-control and foreground-mixed MegaDescriptor CFR results.

The collector recomputes SSI and CFR from raw retrieval rates using the final
paper formulas, writes analysis-ready CSVs, and creates supplementary LaTeX
tables. It never copies checkpoint paths into the generated artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


TRAINING_DATASETS = {
    "AAUZebraFish",
    "ATRW",
    "AerialCattle2017",
    "BelugaID",
    "BirdIndividualID",
    "CTai",
    "CZoo",
    "Cows2021",
    "FriesianCattle2015",
    "FriesianCattle2017",
    "GiraffeZebraID",
    "Giraffes",
    "HappyWhale",
    "HumpbackWhaleID",
    "IPanda50",
    "LionData",
    "MacaqueFaces",
    "NDD20",
    "NOAARightWhale",
    "OpenCows2020",
    "SMALST",
    "ZindiTurtleRecall",
}

SILHOUETTE_IDENTITY_DATASETS = {
    "belugaid",
    "happywhale",
    "humpbackwhaleid",
    "noaarightwhale",
    "whaleshark",
    "whale shark",
}

DATASET_LABELS = {
    "whaleshark": "Whale shark",
    "leopard": "Leopard",
    "saimaa_seal": "Saimaa seal",
    "seaturtle_head": "SeaTurtle head",
    "stripespotter": "StripeSpotter",
    "hyenaid2022": "HyenaID2022",
    "BrownBearHeads": "BrownBearHeads",
    "czechlynx": "CzechLynx",
    "nyaladata": "NyalaData",
    "reunionturtles": "Reunion turtles",
    "amvrakikosturtles": "Amvrakikos turtles",
    "melops": "Melops",
}

METRICS = ("full", "foreground", "background", "silhouette", "ssi", "cfr")
BASELINE_MODEL = "megad_full22"


def normalized(text: str) -> str:
    return text.strip().lower().replace("_", " ")


def latex_escape(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def read_dataset_order(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [
            row["dataset"]
            for row in csv.DictReader(handle, delimiter="\t")
            if int(float(row.get("annotations_accepted") or 0)) > 0
        ]


def read_registry(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
            if not (row.get("key") or "").strip():
                continue
            rows.append(
                {
                    "key": row["key"].strip(),
                    "label": row["label"].strip(),
                    "foreground_fraction": float(row["foreground_fraction"]),
                    "missing_foreground_policy": row["missing_foreground_policy"].strip(),
                    "order": index,
                }
            )
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_result(root: Path, dataset: str, model: str) -> dict[str, Any] | None:
    path = root / "results" / model / f"{dataset}_{model}_cfr.json"
    return load_json(path) if path.exists() else None


def compute_metrics(result: dict[str, Any], dataset: str, k: int) -> dict[str, float | None]:
    block = result["cfr_by_k"][str(k)]
    random_rate = float(block["random"])
    retrieval = {key: float(value) for key, value in block["retrieval"].items()}
    signals = {key: max(0.0, value - random_rate) for key, value in retrieval.items()}

    shortcut_views = ["background"]
    if normalized(dataset) not in SILHOUETTE_IDENTITY_DATASETS:
        shortcut_views.append("silhouette")
    shortcut_view = max(shortcut_views, key=lambda key: signals.get(key, 0.0))
    shortcut_signal = signals.get(shortcut_view, 0.0)
    full_signal = signals.get("full", 0.0)
    foreground_signal = signals.get("foreground", 0.0)
    headroom = max(0.0, 1.0 - random_rate)

    return {
        "random": random_rate,
        "full": retrieval["full"],
        "foreground": retrieval["foreground"],
        "background": retrieval["background"],
        "silhouette": retrieval["silhouette"],
        "shortcut_view": shortcut_view,
        "ssi": min(1.0, shortcut_signal / full_signal) if full_signal > 0 else None,
        "cfr": min(full_signal, foreground_signal) / headroom if headroom > 0 else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return fmean(values) if values else None


def fmt(value: Any) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def short_model_label(row: dict[str, Any]) -> str:
    if row["model"] == BASELINE_MODEL:
        return "Full"
    fraction = int(round(float(row["foreground_fraction"]) * 100))
    return f"Fg{fraction}"


def tabular_lines(datasets: list[str], rows_by_dataset: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        r"\begin{tabular}[t]{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Rand.} & \textbf{Full} & \textbf{Fg} & \textbf{Bg+Sil} & \textbf{Sil} & \textbf{SSI} & \textbf{CFR} \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(datasets):
        rows = rows_by_dataset[dataset]
        for model_index, row in enumerate(rows):
            dataset_text = latex_escape(row["dataset_label"]) if model_index == 0 else ""
            values = [
                dataset_text,
                short_model_label(row),
                fmt(row["random"]),
                fmt(row["full"]),
                fmt(row["foreground"]),
                fmt(row["background"]),
                fmt(row["silhouette"]),
                fmt(row["ssi"]),
                fmt(row["cfr"]),
            ]
            lines.append(" & ".join(values) + r" \\")
        if dataset_index != len(datasets) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def write_detailed_tables(
    path: Path,
    datasets: list[str],
    detailed_rows: list[dict[str, Any]],
    label_prefix: str,
    scope_text: str,
) -> None:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detailed_rows:
        rows_by_dataset[row["dataset"]].append(row)

    blocks: list[str] = []
    for start in range(0, len(datasets), 4):
        chunk = datasets[start : start + 4]
        left = chunk[:2]
        right = chunk[2:]
        table_number = start // 4 + 1
        blocks.extend(
            [
                r"\begin{table*}[!t]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.2pt}",
                r"\resizebox{\textwidth}{!}{%",
                *tabular_lines(left, rows_by_dataset),
            ]
        )
        if right:
            blocks.extend([r"\hfill", *tabular_lines(right, rows_by_dataset)])
        blocks.extend(
            [
                r"}",
                (
                    r"\caption{\textbf{MegaDescriptor foreground-mixing ablation ("
                    + latex_escape(scope_text)
                    + r").} Full is MegaD$^C$, the full-image-only baseline trained on the "
                    r"same 22-dataset collection with no foreground-only inputs. Fg25, Fg50, "
                    r"and Fg75 denote foreground-input probability; selected samples without "
                    r"an accepted mask use the full image. Values are top-1 retrieval rates "
                    r"except SSI and CFR.}"
                ),
                rf"\label{{tab:{label_prefix}-{table_number}}}",
                r"\end{table*}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks))


def write_macro_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Evaluation set} & \textbf{Model} & \textbf{Full} & \textbf{Fg} & \textbf{SSI} & \textbf{CFR} & $\Delta$\textbf{Full} & $\Delta$\textbf{Fg} & $\Delta$\textbf{SSI} & $\Delta$\textbf{CFR} \\",
        r"\midrule",
    ]
    previous_subset = None
    for row in rows:
        if previous_subset is not None and row["evaluation_set"] != previous_subset:
            lines.append(r"\midrule")
        previous_subset = row["evaluation_set"]
        values = [
            latex_escape(row["evaluation_set"]),
            latex_escape(row["model_label"]),
            fmt(row["full"]),
            fmt(row["foreground"]),
            fmt(row["ssi"]),
            fmt(row["cfr"]),
            fmt(row["delta_full"]),
            fmt(row["delta_foreground"]),
            fmt(row["delta_ssi"]),
            fmt(row["delta_cfr"]),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{\textbf{Macro-average MegaDescriptor foreground-mixing ablation.} "
            r"OOD contains the 12 datasets absent from the 22-dataset retraining collection; "
            r"training overlap contains those 22 datasets. Deltas are relative to the "
            r"full-image MegaD$^C$ baseline within each evaluation set.}",
            r"\label{tab:megad-mix-ablation-macro}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central_index", type=Path, required=True)
    parser.add_argument("--baseline_registry", type=Path, required=True)
    parser.add_argument("--model_registry", type=Path, required=True)
    parser.add_argument("--ablation_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.ablation_root / "tables"
    datasets = read_dataset_order(args.central_index)
    baseline_registry = read_registry(args.baseline_registry)
    if len(baseline_registry) != 1 or baseline_registry[0]["key"] != BASELINE_MODEL:
        raise SystemExit(
            f"--baseline_registry must contain only {BASELINE_MODEL!r}; "
            f"found {[row['key'] for row in baseline_registry]}"
        )
    registry = read_registry(args.model_registry)
    models = [baseline_registry[0], *registry]
    for order, model in enumerate(models):
        model["order"] = order

    detailed: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in datasets:
        for model in models:
            result = load_result(args.ablation_root, dataset, model["key"])
            if result is None:
                missing.append(f"{dataset}/{model['key']}")
                continue
            metrics = compute_metrics(result, dataset, args.k)
            detailed.append(
                {
                    "dataset": dataset,
                    "dataset_label": result.get("dataset", {}).get("label")
                    or DATASET_LABELS.get(dataset, dataset),
                    "evaluation_set": "Training overlap" if dataset in TRAINING_DATASETS else "OOD",
                    "model": model["key"],
                    "model_label": model["label"],
                    "model_order": model["order"],
                    "foreground_fraction": model["foreground_fraction"],
                    "missing_foreground_policy": model["missing_foreground_policy"],
                    **metrics,
                }
            )

    expected = len(datasets) * len(models)
    if missing and not args.allow_missing:
        raise SystemExit(
            f"Loaded {len(detailed)}/{expected} expected rows. Missing:\n" + "\n".join(missing)
        )

    baseline_by_dataset = {
        row["dataset"]: row for row in detailed if row["model"] == BASELINE_MODEL
    }
    for row in detailed:
        baseline = baseline_by_dataset.get(row["dataset"])
        for metric in METRICS:
            current = row.get(metric)
            reference = baseline.get(metric) if baseline else None
            row[f"delta_{metric}"] = (
                float(current) - float(reference)
                if current is not None and reference is not None
                else None
            )

    model_order = {model["key"]: model["order"] for model in models}
    dataset_order = {dataset: index for index, dataset in enumerate(datasets)}
    detailed.sort(key=lambda row: (dataset_order[row["dataset"]], model_order[row["model"]]))

    subset_definitions = [
        ("OOD (12)", [dataset for dataset in datasets if dataset not in TRAINING_DATASETS]),
        ("Training overlap (22)", [dataset for dataset in datasets if dataset in TRAINING_DATASETS]),
        ("All (34)", datasets),
    ]
    macro: list[dict[str, Any]] = []
    for subset_label, subset_datasets in subset_definitions:
        for model in models:
            selected = [
                row
                for row in detailed
                if row["dataset"] in subset_datasets and row["model"] == model["key"]
            ]
            summary: dict[str, Any] = {
                "evaluation_set": subset_label,
                "num_datasets": len(selected),
                "model": model["key"],
                "model_label": model["label"],
                "foreground_fraction": model["foreground_fraction"],
                "missing_foreground_policy": model["missing_foreground_policy"],
            }
            for metric in METRICS:
                summary[metric] = finite_mean(selected, metric)
                summary[f"delta_{metric}"] = finite_mean(selected, f"delta_{metric}")
            macro.append(summary)

    write_csv(output_dir / "megadescriptor_mix_ablation_detailed_k1.csv", detailed)
    write_csv(output_dir / "megadescriptor_mix_ablation_macro_k1.csv", macro)
    write_macro_table(output_dir / "table_megadescriptor_mix_ablation_macro_k1.tex", macro)

    complete_datasets = [
        dataset
        for dataset in datasets
        if sum(row["dataset"] == dataset for row in detailed) == len(models)
    ]
    ood_datasets = [dataset for dataset in complete_datasets if dataset not in TRAINING_DATASETS]
    write_detailed_tables(
        output_dir / "table_megadescriptor_mix_ablation_ood_detailed_k1.tex",
        ood_datasets,
        detailed,
        "megad-mix-ood",
        "OOD datasets",
    )
    write_detailed_tables(
        output_dir / "table_megadescriptor_mix_ablation_all_detailed_k1.tex",
        complete_datasets,
        detailed,
        "megad-mix-all",
        "all datasets",
    )

    status = {
        "datasets": len(datasets),
        "training_overlap_datasets": sum(dataset in TRAINING_DATASETS for dataset in datasets),
        "ood_datasets": sum(dataset not in TRAINING_DATASETS for dataset in datasets),
        "baseline_model": BASELINE_MODEL,
        "models_including_full_image_baseline": len(models),
        "expected_rows": expected,
        "loaded_rows": len(detailed),
        "missing": missing,
    }
    (output_dir / "megadescriptor_mix_ablation_status.json").write_text(
        json.dumps(status, indent=2)
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
