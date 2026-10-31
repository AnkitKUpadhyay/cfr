"""Command line entry point for the standardized annotation-crop procedure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inputs import load_records
from .pipeline import dump_json, inspect_records, run_extraction


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", type=Path, help="CSV or COCO/GZCD JSON annotation metadata.")
    source.add_argument("--manifest", type=Path, help="JSONL jobs; relative job paths resolve beside this file.")
    parser.add_argument("--index", type=int, default=0, help="Zero-based manifest job index.")
    parser.add_argument("--image-root", type=Path, help="Root for relative image paths; defaults beside metadata.")
    parser.add_argument("--output-dir", type=Path, help="New or empty directory; existing artifacts are never overwritten.")
    parser.add_argument("--dataset", help="Dataset label recorded in the output; does not change selection.")
    parser.add_argument("--split", help="Default split label for annotations without one.")
    parser.add_argument("--prompt", action="append", help="Default target prompt; per-record prompts/category map take precedence.")
    parser.add_argument("--prompt-map", type=Path, help="JSON mapping category names/IDs to prompts.")
    parser.add_argument("--model", help="Hugging Face model ID or local saved model directory (default facebook/sam3).")
    parser.add_argument("--revision", help="Pin the Hugging Face model/processor commit or revision.")
    parser.add_argument("--device", help="Torch device (default cuda).")
    parser.add_argument("--confidence-threshold", type=float, help="SAM's existing instance filter (default score > 0.3).")
    parser.add_argument("--mask-threshold", type=float, help="SAM pixel binarization threshold (default probability > 0.5).")
    parser.add_argument("--seed", type=int, help="Reproducible inference seed (default 0).")
    parser.add_argument("--batch-size", type=int, default=1, help="Prepared inputs per group; inference remains one image/prompt at a time.")
    parser.add_argument("--limit", type=int, default=0, help="First N annotations, in metadata order; 0 means all.")
    parser.add_argument("--local-files-only", action="store_true", help="Load only cached/local model and processor files.")
    parser.add_argument("--save-candidate-masks", action="store_true", help="Save unchanged full-crop candidate masks in NPZ for audit.")
    parser.add_argument("--legacy-pickle", action="store_true", help="Also save annotation_masks.pkl; keeps selected masks in RAM.")
    parser.add_argument("--dry-run", action="store_true", help="Validate prompts and image/crop geometry without loading SAM3.")
    parser.add_argument("--report", type=Path, help="Write the complete dry-run report to this new JSON file.")
    args = parser.parse_args(argv)
    if args.index < 0 or args.limit < 0 or args.batch_size < 1:
        parser.error("index/limit must be nonnegative and batch-size must be positive")
    if args.report and not args.dry_run:
        parser.error("--report is only used with --dry-run")
    return args


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config(args) -> dict:
    job = {}
    base = Path.cwd()
    if args.manifest:
        manifest = args.manifest.expanduser().resolve()
        base = manifest.parent
        with manifest.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if args.index >= len(rows):
            raise ValueError(f"Manifest has {len(rows)} jobs, no index {args.index}.")
        job = rows[args.index]
        if not isinstance(job, dict):
            raise ValueError("Manifest job must be a JSON object.")
        # Old options must not look active in a new-procedure run.
        obsolete = {"accept_iou", "min_sam_score", "candidate_mask_mode", "species_key"} & job.keys()
        if obsolete:
            raise ValueError(f"Remove obsolete manifest settings: {', '.join(sorted(obsolete))}. This selector uses no box IoU or second confidence filter.")

    def path_value(arg, *keys, default=None):
        if arg is not None:
            return str(Path(arg).expanduser().resolve())
        value = next((job[key] for key in keys if job.get(key) is not None), default)
        if value is None:
            return None
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else base / path).resolve())

    metadata = path_value(args.metadata, "metadata", "input_path")
    if not metadata:
        raise ValueError("Manifest job needs metadata (or input_path).")
    image_root = path_value(args.image_root, "image_root", default=str(Path(metadata).parent))
    output_dir = path_value(args.output_dir, "output_dir")
    if not args.dry_run and output_dir is None:
        raise ValueError("Specify --output-dir or a manifest output_dir.")
    mapping = args.prompt_map
    if mapping:
        prompt_map = _read_json(mapping.expanduser().resolve())
    else:
        prompt_map = job.get("prompt_map", {})
        if isinstance(prompt_map, str):
            mapping_path = Path(prompt_map).expanduser()
            prompt_map = _read_json(mapping_path if mapping_path.is_absolute() else base / mapping_path)
    if not isinstance(prompt_map, dict):
        raise ValueError("prompt_map must be a JSON object.")
    prompts = args.prompt if args.prompt is not None else job.get("prompts", job.get("prompt"))
    if isinstance(prompts, str):
        prompts = [prompts]
    if prompts is not None and (not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts)):
        raise ValueError("prompts must be a nonempty list of nonempty strings.")

    def setting(name, default):
        value = getattr(args, name)
        return value if value is not None else job.get(name, default)

    confidence = float(setting("confidence_threshold", 0.3))
    mask_threshold = float(setting("mask_threshold", 0.5))
    if not 0 <= confidence <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("Confidence and mask thresholds must be finite numbers in [0,1].")
    model_id = setting("model", "facebook/sam3")
    model_path = Path(model_id).expanduser()
    model_base = Path.cwd() if args.model is not None else base
    possible_local = model_path if model_path.is_absolute() else model_base / model_path
    if possible_local.is_dir() or str(model_id).startswith((".", "~", "/")):
        model_id = str(possible_local.resolve())
    local_only = job.get("local_files_only", False)
    if not isinstance(local_only, bool):
        raise ValueError("Manifest local_files_only must be a JSON boolean.")
    return {
        "metadata": metadata,
        "image_root": image_root,
        "output_dir": output_dir,
        "dataset": args.dataset or job.get("dataset"),
        "split": args.split or job.get("split"),
        "prompts": prompts,
        "prompt_map": prompt_map,
        "model_id": model_id,
        "revision": setting("revision", None),
        "device": setting("device", "cuda"),
        "confidence_threshold": confidence,
        "mask_threshold": mask_threshold,
        "seed": int(setting("seed", 0)),
        "local_files_only": args.local_files_only or local_only,
        "limit": args.limit,
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        config = resolve_config(args)
        records = load_records(
            Path(config["metadata"]), Path(config["image_root"]),
            dataset=config["dataset"], split=config["split"],
            default_prompts=config["prompts"], prompt_map=config["prompt_map"],
        )
        if args.limit:
            records = records[:args.limit]
        if not records:
            raise ValueError("No annotation records found.")
        if args.dry_run:
            report = inspect_records(records)
            report["configuration"] = config
            if args.report:
                report_path = args.report.expanduser().resolve()
                if report_path.exists():
                    raise FileExistsError(f"Report already exists: {report_path}")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                dump_json(report_path, report)
            preview = {k: v for k, v in report.items() if k != "records"}
            preview["preview"] = report["records"][:5]
            print(json.dumps(preview, indent=2, sort_keys=True, allow_nan=False))
            if report["errors"]:
                raise SystemExit(2)
            return

        output_dir = Path(config["output_dir"])
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError(f"Use a new or empty output directory: {output_dir}")
        # Lazy import: dry runs/tests do not require Transformers, Torch, or model access.
        from .backend import Sam3Backend

        backend = Sam3Backend(**{key: config[key] for key in (
            "model_id", "revision", "device", "confidence_threshold", "mask_threshold", "seed", "local_files_only"
        )})
        summary = run_extraction(
            records, output_dir, backend, batch_size=args.batch_size,
            save_candidate_masks=args.save_candidate_masks, legacy_pickle=args.legacy_pickle,
            run_config=config,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        if summary["errors"]:
            raise SystemExit(2)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
