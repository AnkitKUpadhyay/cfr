"""Synthetic golden cases for the agreed mask-selection policy."""

import json
import unittest

import numpy as np

from sam3_masks.selection import select_candidate


class SelectionTests(unittest.TestCase):
    def candidates(self, masks, boxes=None, scores=None):
        masks = np.asarray(masks)
        n, h, w = masks.shape
        return {
            "masks": masks,
            "boxes": np.asarray(boxes if boxes is not None else [[0, 0, w, h]] * n,
                                dtype=float).reshape(n, 4),
            "scores": np.asarray(scores if scores is not None else [0.8] * n,
                                 dtype=float),
        }

    def test_overlapping_equal_boxes_choose_more_visible_mask_pixels(self):
        background = np.zeros((6, 8), dtype=bool)
        background[0, :] = True
        foreground = np.zeros_like(background)
        foreground[1:5, 2:6] = True
        result = select_candidate(
            self.candidates([background, foreground], scores=[0.99, 0.31]), (8, 6)
        )
        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.candidates[1]["foreground_pixels"], 16)
        self.assertAlmostEqual(result.candidates[1]["mask_coverage"], 16 / 48)

    def test_mask_gap_at_center_does_not_disqualify_candidate(self):
        mask = np.ones((6, 8), dtype=bool)
        mask[3:, 3:5] = False
        result = select_candidate(self.candidates([mask]), (8, 6))
        self.assertFalse(mask[3, 4])
        self.assertTrue(result.candidates[0]["contains_center"])
        self.assertEqual(result.status, "accepted")

    def test_rejects_noncentral_candidates_without_score_fallback(self):
        masks = np.ones((2, 6, 8), dtype=bool)
        data = self.candidates(masks, [[0, 0, 3, 6], [5, 0, 8, 6]], [0.999, 1.0])
        result = select_candidate(data, (8, 6))
        self.assertEqual(result.status, "no_central_candidate")
        self.assertIsNone(result.selected_index)
        self.assertIsNone(result.selected_mask)

    def test_equal_area_uses_confidence_then_original_order(self):
        masks = np.ones((3, 6, 8), dtype=bool)
        data = self.candidates(masks, scores=[0.5, 0.9, 0.9])
        data["prompt_texts"] = ["first", "second", "third"]
        result = select_candidate(data, (8, 6))
        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.candidates[1]["prompt_text"], "second")
        self.assertEqual([r["index"] for r in result.candidates], [0, 1, 2])






    def test_selector_does_not_add_a_confidence_cutoff(self):
        result = select_candidate(
            self.candidates(np.ones((1, 6, 8)), scores=[0.01]), (8, 6)
        )
        self.assertEqual(result.status, "accepted")

    def test_box_clipping_never_replaces_invalid_boxes(self):
        masks = np.ones((5, 6, 8), dtype=bool)
        data = self.candidates(masks, [
            [float("nan"), 0, 8, 6],
            [5, 0, 2, 6],
            [10, 0, 20, 6],
            [4, 0, 4, 6],
            [-2, -3, 12, 9],
        ])
        result = select_candidate(data, (8, 6))
        self.assertEqual(result.selected_index, 4)
        self.assertEqual(result.candidates[4]["clipped_box"], [0, 0, 8, 6])
        self.assertEqual([r["invalid_reason"] for r in result.candidates[:4]], [
            "non_finite_box", "invalid_box", "empty_box_after_clipping", "invalid_box"
        ])
        self.assertEqual(result.candidates[0]["raw_box"], [None, 0, 8, 6])
        self.assertIsNone(result.candidates[0]["clipped_box"])
        json.dumps(result.candidates, allow_nan=False)

    def test_empty_masks_and_nonfinite_scores_are_invalid(self):
        masks = np.ones((3, 6, 8), dtype=bool)
        masks[0] = False
        result = select_candidate(
            self.candidates(masks, scores=[0.8, np.inf, np.nan]), (8, 6)
        )
        self.assertEqual(result.status, "no_valid_candidate")
        self.assertEqual([r["invalid_reason"] for r in result.candidates], [
            "empty_mask", "non_finite_score", "non_finite_score"
        ])
        self.assertEqual(result.candidates[0]["foreground_pixels"], 0)
        json.dumps(result.candidates, allow_nan=False)

    def test_nonbinary_or_nonfinite_masks_are_discarded_individually(self):
        masks = np.ones((3, 6, 8), dtype=float)
        masks[0, 0, 0] = np.nan
        masks[1, 0, 0] = 0.75
        result = select_candidate(self.candidates(masks), (8, 6))
        self.assertEqual(result.selected_index, 2)
        self.assertEqual(result.candidates[0]["invalid_reason"], "non_finite_mask")
        self.assertEqual(result.candidates[1]["invalid_reason"], "non_binary_mask")
        self.assertIsNone(result.candidates[0]["foreground_pixels"])
        json.dumps(result.candidates, allow_nan=False)

    def test_specks_and_holes_preserved_without_mutation(self):
        mask = np.zeros((7, 9), dtype=np.uint8)
        mask[1:6, 2:7] = 1
        mask[3, 4] = 0  # Internal hole is kept.
        mask[0, 0] = 1  # Isolated foreground speck is kept.
        data = self.candidates([mask], [[-1, 0, 10, 7]])
        original_masks = data["masks"].copy()
        original_boxes = data["boxes"].copy()
        result = select_candidate(data, (9, 7))
        np.testing.assert_array_equal(result.selected_mask, mask)
        self.assertEqual(result.selected_mask.dtype, np.uint8)
        np.testing.assert_array_equal(data["boxes"], original_boxes)
        result.selected_mask[:] = 0
        np.testing.assert_array_equal(data["masks"], original_masks)

    def test_coverage_denominator_is_crop_not_candidate_box(self):
        mask = np.zeros((10, 12), dtype=bool)
        mask[4:6, 5:7] = True
        result = select_candidate(self.candidates([mask], [[5, 4, 7, 6]]), (12, 10))
        self.assertEqual(result.candidates[0]["mask_coverage"], 4 / 120)

    def test_center_uses_geometric_coordinates_and_half_open_boxes(self):
        masks = np.ones((3, 7, 9), dtype=bool)
        boxes = [[0, 0, 4.5, 7], [4.5, 0, 9, 7], [4.1, 3.1, 4.9, 3.9]]
        result = select_candidate(self.candidates(masks, boxes, [0.99, 0.8, 0.7]), (9, 7))
        self.assertEqual([r["contains_center"] for r in result.candidates], [False, True, True])
        self.assertEqual(result.selected_index, 1)

    def test_single_channel_masks_are_supported(self):
        data = self.candidates(np.ones((2, 6, 8), dtype=bool))
        data["masks"] = data["masks"][:, None]
        result = select_candidate(data, (8, 6))
        self.assertEqual(result.selected_mask.shape, (6, 8))
        self.assertEqual(result.selected_index, 0)

    def test_empty_prediction_has_distinct_status(self):
        result = select_candidate(self.candidates(np.zeros((0, 6, 8))), (8, 6))
        self.assertEqual(result.status, "no_candidate")
        self.assertEqual(result.candidates, [])
        self.assertIsNone(result.selected_mask)

    def test_malformed_candidate_contract_raises(self):
        valid = self.candidates(np.ones((2, 6, 8)))
        invalid = [
            {**valid, "masks": np.ones((2, 8, 6))},
            {**valid, "masks": np.ones((2, 2, 6, 8))},
            {**valid, "boxes": np.ones((1, 4))},
            {**valid, "scores": np.ones((2, 1))},
            {**valid, "prompt_texts": ["antelope"]},
            {**valid, "prompt_texts": "antelope"},
            {**valid, "prompt_texts": ["antelope", 3]},
            {"masks": valid["masks"], "boxes": valid["boxes"]},
        ]
        for data in invalid:
            with self.subTest(keys=list(data), shapes={k: np.shape(v) for k, v in data.items()}):
                with self.assertRaises(ValueError):
                    select_candidate(data, (8, 6))

    def test_invalid_image_size_raises(self):
        data = self.candidates(np.ones((1, 6, 8)))
        for size in [(0, 6), (-8, 6), (8.0, 6), (True, 6), (8,), "8,6"]:
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    select_candidate(data, size)


if __name__ == "__main__":
    unittest.main()
