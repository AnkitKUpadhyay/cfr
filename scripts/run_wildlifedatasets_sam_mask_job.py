#!/usr/bin/env python3
"""Run one SAM mask-extraction job from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(os.environ.get("CFR_CODE_ROOT", str(Path(__file__).resolve().parents[1])))


def read_job(manifest: Path, index: int) -> dict:
    with manifest.open("r") as handle:
        for i, line in enumerate(handle):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Manifest {manifest} has no zero-based index {index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT_IMAGES", "0")))
    parser.add_argument("--min_sam_score", type=float, default=float(os.environ.get("MIN_SAM_SCORE", "0.0")))
    parser.add_argument("--accept_iou", type=float, default=float(os.environ.get("ACCEPT_IOU", "0.5")))
    parser.add_argument(
        "--candidate_mask_mode",
        choices=("box_crop", "full", "none"),
        default=os.environ.get("CANDIDATE_MASK_MODE", "box_crop"),
    )
    parser.add_argument(
        "--save_candidate_masks",
        action="store_true",
        default=os.environ.get("SAVE_CANDIDATE_MASKS", "1") != "0",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = args.index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("MASK_JOB_INDEX", "0"))
        index = int(raw)

    job = read_job(args.manifest, index)
    cmd = [
        sys.executable,
        str(PROJECT / "scripts" / "extract_wildlifedatasets_sam_masks.py"),
        "--image_root",
        job["image_root"],
        "--output_dir",
        job["output_dir"],
        "--dataset",
        str(job.get("dataset") or job["job_name"]),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--min_sam_score",
        str(args.min_sam_score),
        "--accept_iou",
        str(args.accept_iou),
        "--candidate_mask_mode",
        args.candidate_mask_mode,
        "--species_key",
        "target",
        "--prompts",
        *[str(prompt) for prompt in job.get("prompts", ["animal"])],
    ]
    if job["input_type"] == "json":
        cmd.extend(["--input_json", job["input_path"]])
        if job.get("split"):
            cmd.extend(["--split", str(job["split"])])
    elif job["input_type"] == "csv":
        cmd.extend(["--input_csv", job["input_path"]])
    else:
        raise ValueError(f"Unknown input_type: {job['input_type']}")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.save_candidate_masks:
        cmd.append("--save_candidate_masks")

    print(json.dumps({"index": index, "job": job, "cmd": cmd}, indent=2), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
