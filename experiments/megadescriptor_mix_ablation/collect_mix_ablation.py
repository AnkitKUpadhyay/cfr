#!/usr/bin/env python3
"""Collect full-image-control and foreground-mixed MegaDescriptor CFR results.

The collector recomputes SSI and CFR from raw retrieval rates using the final
paper formulas and writes analysis-ready CSVs. It never copies checkpoint paths into the generated artifacts.
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
    "seaturtle_head": "SeaTurtleID2022",
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
    output_dir = args.output_dir or args.ablation_root / "summaries"
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
