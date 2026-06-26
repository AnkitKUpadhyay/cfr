#!/usr/bin/env python3
"""Collect saved CFR JSON results into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_cfr_evaluation import (
    CENTRAL_INDEX,
    HEADLINE_DATASETS,
    dataset_specs_from_index,
    flatten_result_for_k,
    get_model_specs,
    json_default,
    write_dict_rows,
    write_latex_table,
)


def default_datasets(suite: str, central_index: Path) -> list[str]:
    if suite == "headline":
        return list(HEADLINE_DATASETS)
    if suite == "all":
        specs = dataset_specs_from_index(central_index)
        return [key for key, spec in specs.items() if spec.annotations_accepted > 0]
    raise SystemExit("--datasets is required for --suite custom")


def default_models(suite: str) -> list[str]:
    return ["miewid", "megadescriptor"]


def load_result(results_root: Path, dataset: str, model: str) -> dict[str, Any] | None:
    path = results_root / "results" / model / f"{dataset}_{model}_cfr.json"
    if not path.exists():
        return None
    with path.open("r") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--central_index", type=Path, default=CENTRAL_INDEX)
    parser.add_argument("--suite", choices=["headline", "all", "custom"], default="headline")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["miewid", "megadescriptor"],
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--output_name", help="Default: cfr_normal_<suite>")
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.datasets or default_datasets(args.suite, args.central_index)
    models = args.models or default_models(args.suite)
    model_specs = get_model_specs()

    missing: list[str] = []
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        for model in models:
            result = load_result(args.results_root, dataset, model)
            if result is None:
                missing.append(f"{dataset}/{model}")
                continue
            results.append(result)

    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing CFR result JSONs:\n"
            + "\n".join(missing)
            + "\nUse --allow_missing to write a partial summary."
        )

    rows = []
    for result in results:
        for k in args.ks:
            if str(k) in result.get("cfr_by_k", {}):
                rows.append(flatten_result_for_k(result, k))

    output_name = args.output_name or f"cfr_normal_{args.suite}"
    summary_csv = args.results_root / f"{output_name}.csv"
    write_dict_rows(rows, summary_csv)
    for k in args.ks:
        rows_k = [row for row in rows if row["k"] == k]
        write_dict_rows(rows_k, args.results_root / f"{output_name}_k{k}.csv")
        model_labels = ", ".join(model_specs[m].label for m in models)
        write_latex_table(
            rows_k,
            args.results_root / f"{output_name}_k{k}.tex",
            caption=(
                f"Counterfactual Foreground Retrieval components at rank {k} "
                f"for {args.suite} datasets using {model_labels}. "
                "Gallery is all accepted-mask annotations except the query annotation itself; "
                "same-image and same-encounter exclusions are not applied."
            ),
            label=f"tab:{output_name.replace('_', '-')}-k{k}",
        )

    status = {
        "results_root": str(args.results_root),
        "suite": args.suite,
        "datasets_requested": datasets,
        "models_requested": models,
        "results_loaded": len(results),
        "rows_written": len(rows),
        "missing": missing,
        "summary_csv": str(summary_csv),
    }
    status_path = args.results_root / f"{output_name}_status.json"
    with status_path.open("w") as handle:
        json.dump(status, handle, indent=2, default=json_default)
    print(json.dumps(status, indent=2, default=json_default))


if __name__ == "__main__":
    main()
