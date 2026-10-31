#!/usr/bin/env python3
"""Replay saved candidates for center-only rejections; never run or change SAM.

Keep existing acceptances and other rejections unchanged. For no_central_candidate
records, choose a valid candidate by confidence, mask pixels, then stable index.
Write a separate compatible result directory, referencing the original candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


FALLBACK_VERSION = "confidence-fallback-v1"
RANKING = ["SAM confidence descending", "foreground pixels descending", "candidate_index ascending"]
ARTIFACTS = ("extraction_summary.json", "annotation_match_metadata.jsonl", "annotation_masks_rle.json.gz")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def choose_fallback(record):
    """Return a candidate only for a consistent center-only rejection."""
    if record["status"] != "no_central_candidate":
        return None
    require(not record["accepted"] and record["selected_index"] is None,
            "Center rejection already has a selected mask")
    require(record.get("selection_region_xyxy") is None,
            "Confidence fallback for target-region inputs requires a separate policy")
    valid = [c for c in record["candidates"] if c["valid"]]
    require(bool(valid), "Center rejection has no valid candidates")
    for c in valid:
        require(not c["contains_center"] and not c["eligible"], "Center rejection has a central candidate")
        require(math.isfinite(c["score"]) and 0 <= c["score"] <= 1,
                "Valid candidate must have finite confidence in [0,1]")
        require(type(c["foreground_pixels"]) is int and c["foreground_pixels"] > 0,
                "Valid candidate has no foreground pixels")
    return min(valid, key=lambda c: (-c["score"], -c["foreground_pixels"], c["index"]))


def apply_fallback(source: Path, output: Path):
    source, output = source.resolve(), output.absolute()
    require(source != output.resolve(), "Use a separate output directory")
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise FileExistsError(staging)
    hashes = {name: sha(source / name) for name in ARTIFACTS}
    summary = json.loads((source / ARTIFACTS[0]).read_text())
    require(summary["status"] == "complete" and summary["errors"] == 0,
            "Source must be complete and have no processing errors")
    require("fallback" not in summary, "Do not apply fallback twice")
    with (source / ARTIFACTS[1]).open() as stream:
        records = [json.loads(line) for line in stream]
    with gzip.open(source / ARTIFACTS[2], "rt") as stream:
        archive = json.load(stream)
    mask_records = archive["annotations"]
    ids = [r["ann_id"] for r in records]
    require(len(ids) == len(set(ids)) == summary["records_total"] == summary["annotations_attempted"],
            "Source IDs or record totals disagree")
    require(ids == [r["annotation_id"] for r in mask_records], "Source audit and RLE order disagree")
    require(dict(Counter(r["status"] for r in records)) == summary["status_counts"],
            "Source status totals disagree")
    require(sum(r["accepted"] for r in records) == summary["annotations_accepted"],
            "Source acceptance total disagrees")
    require((source / "candidate_masks").is_dir(), "Saved candidate masks are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    (staging / "candidate_masks").symlink_to(
        os.path.relpath(source / "candidate_masks", staging), target_is_directory=True)

    selected_rows = []
    counts = Counter()
    procedure = summary["procedure_version"] + "+" + FALLBACK_VERSION
    archive["source_procedure_version"] = archive["procedure_version"]
    archive["procedure_version"] = procedure
    with (staging / ARTIFACTS[1]).open("w") as audit:
        for record, item in zip(records, mask_records):
            require(item["present"] == record["accepted"] == (item["segmentation"] is not None),
                    "Source acceptance and RLE disagree")
            chosen = choose_fallback(record)
            if chosen is not None:
                require(item["area"] == 0 and item["bbox"] is None, "Rejected source RLE is not empty")
                index = chosen["index"]
                require([c["index"] for c in record["candidates"]] == list(range(len(record["candidates"]))),
                        "Candidate indexes do not match archive positions")
                candidate_path = (source / record["candidate_npz"]).resolve()
                require(candidate_path.is_relative_to((source / "candidate_masks").resolve()),
                        "Candidate archive must be inside source candidate_masks")
                candidate_hash = sha(candidate_path)
                with np.load(candidate_path, allow_pickle=False) as arrays:
                    masks = arrays["masks"]
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                width, height = record["input_size"]
                require(masks.dtype == np.bool_ and masks.shape == (len(record["candidates"]), height, width),
                        "Candidate archive dtype or input dimensions disagree")
                mask = masks[index]
                pixels = int(np.count_nonzero(mask))
                require(pixels == chosen["foreground_pixels"] and pixels > 0,
                        "Chosen candidate pixel count disagrees")
                require(math.isclose(pixels / mask.size, chosen["mask_coverage"], abs_tol=1e-12),
                        "Chosen candidate coverage disagrees")
                encoded = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
                require(np.array_equal(mask_utils.decode(encoded).astype(bool), mask), "RLE round-trip failed")
                item.update(present=True,
                            segmentation={"size": [height, width], "counts": encoded["counts"].decode("ascii")},
                            area=pixels, bbox=mask_utils.toBbox(encoded).tolist())
                valid_count = sum(c["valid"] for c in record["candidates"])
                record.update(base_status=record["status"], base_selection_policy=record["selection_policy"],
                              status="accepted", accepted=True, selected_index=index,
                              selection_policy="highest_confidence_fallback", fallback_applied=True,
                              fallback_valid_candidate_count=valid_count, fallback_ranking=RANKING,
                              eligibility_semantics="candidate eligible flags and eligible_candidate_count describe the original center filter")
                selected_rows.append({"annotation_id": record["ann_id"], "valid_candidates": valid_count,
                                      "selected_index": index, "selected_score": chosen["score"],
                                      "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                                      "candidate_npz": record["candidate_npz"], "candidate_npz_sha256": candidate_hash})
                if len(selected_rows) % 500 == 0:
                    print(json.dumps({"fallback_masks_written": len(selected_rows)}), flush=True)
            counts[record["status"]] += 1
            audit.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    with (staging / ARTIFACTS[2]).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                json.dump(archive, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
    require({name: sha(source / name) for name in ARTIFACTS} == hashes, "Source artifacts changed during replay")
    manifest = {
        "source_dir": str(source), "source_artifact_sha256": hashes, "output_dir": str(output),
        "fallback_version": FALLBACK_VERSION, "scope": "no_central_candidate only", "ranking": RANKING,
        "additional_score_filter": None, "mask_cleanup": "none", "new_inference_performed": False,
        "source_accepted_retained": summary["annotations_accepted"], "newly_accepted": len(selected_rows),
        "newly_accepted_single_candidate": sum(r["valid_candidates"] == 1 for r in selected_rows),
        "newly_accepted_multiple_candidates": sum(r["valid_candidates"] > 1 for r in selected_rows),
        "candidate_storage": "Relative symlink to the original candidate_masks directory; retain the source run",
        "script_sha256": sha(Path(__file__)), "human_mask_audit_completed": False,
        "selected_candidates": selected_rows,
    }
    updated = deepcopy(summary)
    updated.update(procedure_version=procedure, source_procedure_version=summary["procedure_version"],
                   annotations_accepted=counts["accepted"], status_counts=dict(counts),
                   centrality="Required in original selection; waived only for no_central_candidate fallback",
                   ranking={"original_selection": summary["ranking"], "confidence_fallback": RANKING},
                   fallback={k: v for k, v in manifest.items() if k != "selected_candidates"})
    updated["run_config"]["output_dir"] = str(output)
    updated["run_config"]["source_run"] = str(source)
    updated["run_config"]["operation"] = "replay_saved_candidates"
    write_json(staging / ARTIFACTS[0], updated)
    manifest["output_artifact_sha256"] = {name: sha(staging / name) for name in ARTIFACTS}
    write_json(staging / "fallback_manifest.json", manifest)
    staging.rename(output)
    print(json.dumps({"output_dir": str(output), "records": len(records),
                      "newly_accepted": len(selected_rows), "status_counts": dict(counts)}, indent=2), flush=True)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    apply_fallback(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
