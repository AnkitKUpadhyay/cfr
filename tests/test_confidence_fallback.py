"""Verify fallback scope, confidence priority, tie resolution, and lossless export."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from sam3_masks.inputs import load_records
from sam3_masks.pipeline import run_extraction
from scripts.apply_confidence_fallback import apply_fallback, choose_fallback, sha


def load_prediction(directory, annotation_id):
    with (directory / "annotation_match_metadata.jsonl").open() as stream:
        audit = next(row for row in map(json.loads, stream) if row["ann_id"] == annotation_id)
    with gzip.open(directory / "annotation_masks_rle.json.gz", "rt") as stream:
        record = next(row for row in json.load(stream)["annotations"]
                      if row["annotation_id"] == annotation_id)
    if not record["present"]:
        return audit, None
    rle = dict(record["segmentation"])
    rle["counts"] = rle["counts"].encode("ascii")
    return audit, mask_utils.decode(rle)


class FallbackTests(unittest.TestCase):
    def rejection(self):
        return {"status": "no_central_candidate", "accepted": False, "selected_index": None,
                "candidates": [{"index": 0, "valid": True, "contains_center": False,
                                "eligible": False, "score": .8, "foreground_pixels": 10},
                               {"index": 1, "valid": True, "contains_center": False,
                                "eligible": False, "score": .9, "foreground_pixels": 2}]}

    def test_confidence_then_area_then_stable_index(self):
        row = self.rejection()
        self.assertEqual(choose_fallback(row)["index"], 1)
        row["candidates"][0]["score"] = .9
        self.assertEqual(choose_fallback(row)["index"], 0)
        row["candidates"][1]["foreground_pixels"] = 10
        row["candidates"].reverse()
        self.assertEqual(choose_fallback(row)["index"], 0)

    def test_invalid_high_score_is_excluded_and_inconsistency_fails(self):
        row = self.rejection()
        row["candidates"][1]["valid"] = False
        self.assertEqual(choose_fallback(row)["index"], 0)
        row["candidates"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            choose_fallback(row)
        row = self.rejection()
        row["candidates"][0]["contains_center"] = True
        with self.assertRaises(ValueError):
            choose_fallback(row)

    def test_replay_preserves_primary_choice_empty_rejection_and_mask_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys = ("accepted", "fallback", "empty")
            for key in keys:
                Image.new("RGB", (12, 8)).save(root / f"{key}.png")
            metadata = root / "inputs.json"
            metadata.write_text(json.dumps({"images": [{"id": key, "file_name": f"{key}.png"} for key in keys],
                                           "annotations": [{"id": key, "image_id": key} for key in keys]}))
            records = load_records(metadata, root, dataset="Test", default_prompts=["whale"])
            masks = np.zeros((2, 8, 12), dtype=bool)
            masks[0, :3, :3] = True
            masks[1, :2, :2] = True
            outputs = [
                {"masks": masks, "boxes": [[0, 0, 12, 8]] * 2, "scores": [.8, .9],
                 "prompt_texts": ["whale"] * 2},
                {"masks": masks, "boxes": [[0, 0, 3, 3], [0, 0, 2, 2]], "scores": [.8, .9],
                 "prompt_texts": ["whale"] * 2},
                {"masks": np.zeros((0, 8, 12), dtype=bool), "boxes": np.empty((0, 4)),
                 "scores": [], "prompt_texts": []},
            ]

            class Backend:
                metadata = {"backend": "synthetic_test"}

                def predict(self, images, prompts):
                    return [outputs.pop(0) for _ in images]

            source, destination = root / "source", root / "derived"
            run_extraction(records, source, Backend(), save_candidate_masks=True)
            source_files = {name: sha(source / name) for name in
                            ("extraction_summary.json", "annotation_match_metadata.jsonl", "annotation_masks_rle.json.gz")}
            before = json.load(gzip.open(source / "annotation_masks_rle.json.gz", "rt"))["annotations"]
            result = apply_fallback(source, destination)
            after = json.load(gzip.open(destination / "annotation_masks_rle.json.gz", "rt"))["annotations"]
            self.assertEqual(result["status_counts"], {"accepted": 2, "no_candidate": 1})
            self.assertEqual(before[0], after[0])
            self.assertEqual(before[2], after[2])
            self.assertEqual(source_files, {name: sha(source / name) for name in source_files})
            accepted, original = load_prediction(destination, "accepted")
            fallback, recovered = load_prediction(destination, "fallback")
            rejected, missing = load_prediction(destination, "empty")
            self.assertEqual(accepted["selected_index"], 0)  # Primary area choice stays intact.
            self.assertEqual(fallback["selected_index"], 1)  # Higher confidence wins despite smaller area.
            np.testing.assert_array_equal(original, masks[0])
            np.testing.assert_array_equal(recovered, masks[1])
            self.assertFalse(fallback["candidates"][1]["eligible"])
            self.assertEqual(fallback["selection_policy"], "highest_confidence_fallback")
            self.assertIsNone(missing)
            self.assertTrue((destination / fallback["candidate_npz"]).is_file())
            with self.assertRaises(FileExistsError):
                apply_fallback(source, destination)


if __name__ == "__main__":
    unittest.main()
