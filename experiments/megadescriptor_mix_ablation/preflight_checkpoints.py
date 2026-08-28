#!/usr/bin/env python3
"""Verify that every ablation checkpoint exactly matches MegaDescriptor-L-384."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm

import run_mix_ablation_evaluation as evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_registry", type=Path, required=True)
    args = parser.parse_args()

    specs = evaluation.get_model_specs(args.model_registry)
    first = next(iter(specs.values()))
    model = timm.create_model(first.model_tag, pretrained=False, num_classes=0)
    reports = []
    for spec in specs.values():
        report = evaluation.load_checkpoint(model, spec.checkpoint_path)
        reports.append(
            {
                "key": spec.key,
                "label": spec.label,
                "num_keys": report["num_keys"],
                "missing_count": report["missing_count"],
                "unexpected_count": report["unexpected_count"],
            }
        )
    print(json.dumps({"models_checked": len(reports), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
