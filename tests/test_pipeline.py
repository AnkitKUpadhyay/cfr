"""End-to-end crop, selection, and streamed export tests without SAM or a GPU."""

import gzip
import json
import pickle
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from sam3_masks.inputs import load_records
from sam3_masks.pipeline import ArtifactWriter, inspect_records, run_extraction


class FakeBackend:
    metadata = {"implementation": "test-only fake", "confidence_filter": "backend-owned"}

    def __init__(self, predict_one=None, fail_call=None):
        self.predict_one = predict_one
        self.fail_call = fail_call
        self.calls = []
        self.seen = []

    def predict(self, images, prompts):
        self.calls.append((list(images), list(prompts)))
        if len(self.calls) == self.fail_call:
            raise RuntimeError("simulated inference failure")
        outputs = []
        for image, prompt in zip(images, prompts):
            self.seen.append(np.array(image))
            if self.predict_one:
                outputs.append(self.predict_one(image, prompt))
            else:
                w, h = image.size
                outputs.append({"masks": np.ones((1, h, w), dtype=bool),
                                "boxes": np.array([[0, 0, w, h]], dtype=float),
                                "scores": np.array([0.8]), "prompt_texts": prompt})
        return outputs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        pixels = np.zeros((6, 12, 3), dtype=np.uint8)
        pixels[:, :6] = [240, 30, 20]
        pixels[:, 6:] = [20, 30, 240]
        self.pixels = pixels
        Image.fromarray(pixels).save(self.root / "source.png")

    def tearDown(self):
        self.temp.cleanup()

    def records(self, annotations=None, images=None):
        data = {
            "images": images or [{"id": "source", "file_name": "source.png"}],
            "annotations": annotations or [{"id": "native", "image_id": "source"}],
        }
        path = self.root / "annotations.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return load_records(path, self.root, dataset="TestAnimals", default_prompts=["antelope"])

    def artifacts(self, destination):
        with gzip.open(destination / "annotation_masks_rle.json.gz", "rt") as handle:
            rle = json.load(handle)
        audit = [json.loads(line) for line in
                 (destination / "annotation_match_metadata.jsonl").read_text().splitlines()]
        return rle["annotations"], audit

    def decoded(self, item):
        rle = dict(item["segmentation"])
        rle["counts"] = rle["counts"].encode("ascii")
        return mask_utils.decode(rle)

    def test_two_annotations_of_same_source_are_cropped_before_inference(self):
        records = self.records([
            {"id": "left", "image_id": "source", "bbox": [0, 0, 6, 6]},
            {"id": "right", "image_id": "source", "bbox": [6, 1, 6, 4]},
        ])
        backend = FakeBackend()
        destination = self.root / "run"
        summary = run_extraction(records, destination, backend, batch_size=2)
        self.assertEqual(summary["annotations_accepted"], 2)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][1], [["antelope"], ["antelope"]])
        np.testing.assert_array_equal(backend.seen[0], self.pixels[:, :6])
        np.testing.assert_array_equal(backend.seen[1], self.pixels[1:5, 6:])
        exported, audit = self.artifacts(destination)
        self.assertEqual([a["annotation_id"] for a in exported], ["left", "right"])
        self.assertEqual([a["crop_xyxy"] for a in audit], [[0, 0, 6, 6], [6, 1, 12, 5]])
        self.assertEqual([self.decoded(a).shape for a in exported], [(6, 6), (4, 6)])

    def test_largest_visible_mask_and_gap_survive_rle_candidate_and_pickle_exports(self):
        masks = np.zeros((2, 6, 12), dtype=bool)
        masks[0, 0, :] = True
        masks[1, 1:5, 2:10] = True
        masks[1, 3, 6] = False  # Crop center falls in a real gap/hole.
        masks[1, 5, 11] = True  # Isolated speck remains unchanged.
        def predict(image, prompt):
            return {"masks": masks[:, None], "boxes": [[0, 0, 12, 6]] * 2,
                    "scores": [0.99, 0.31], "prompt_texts": prompt * 2}
        destination = self.root / "run"
        summary = run_extraction(self.records(), destination, FakeBackend(predict),
                                 save_candidate_masks=True, legacy_pickle=True)
        exported, audit = self.artifacts(destination)
        expected = masks[1].astype(np.uint8)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(audit[0]["selected_index"], 1)
        self.assertEqual(audit[0]["eligible_candidate_count"], 2)
        np.testing.assert_array_equal(self.decoded(exported[0]), expected)
        self.assertEqual(exported[0]["area"], int(expected.sum()))
        self.assertEqual(audit[0]["candidates"][1]["mask_coverage"], expected.sum() / 72)
        with np.load(destination / audit[0]["candidate_npz"]) as data:
            np.testing.assert_array_equal(data["masks"], masks)
        with (destination / "annotation_masks.pkl").open("rb") as handle:
            legacy = pickle.load(handle)
        np.testing.assert_array_equal(legacy["native"], expected)

    def test_rejections_export_null_masks_with_distinct_reasons(self):
        for expected_status, masks, boxes in [
            ("no_valid_candidate", np.zeros((1, 6, 12), dtype=bool), [[0, 0, 12, 6]]),
            ("no_central_candidate", np.ones((1, 6, 12), dtype=bool), [[0, 0, 2, 6]]),
            ("no_candidate", np.zeros((0, 6, 12), dtype=bool), np.empty((0, 4))),
        ]:
            with self.subTest(status=expected_status):
                def predict(image, prompt):
                    return {"masks": masks, "boxes": boxes, "scores": [0.999] * len(masks)}
                destination = self.root / expected_status
                summary = run_extraction(self.records(), destination, FakeBackend(predict), legacy_pickle=True)
                exported, audit = self.artifacts(destination)
                self.assertEqual(summary["annotations_accepted"], 0)
                self.assertEqual(audit[0]["status"], expected_status)
                self.assertFalse(exported[0]["present"])
                self.assertIsNone(exported[0]["segmentation"])
                self.assertIsNone(exported[0]["bbox"])
                with (destination / "annotation_masks.pkl").open("rb") as handle:
                    self.assertIsNone(pickle.load(handle)["native"])

    def test_prepared_crop_preserves_source_box_provenance_without_double_crop(self):
        records = self.records([{"id": "prepared", "image_id": "source",
                                 "bbox": [100, 150, 200, 300], "input_kind": "prepared_crop"}])
        backend = FakeBackend()
        destination = self.root / "run"
        run_extraction(records, destination, backend)
        exported, audit = self.artifacts(destination)
        np.testing.assert_array_equal(backend.seen[0], self.pixels)
        self.assertEqual(audit[0]["input_size"], [12, 6])
        self.assertEqual(audit[0]["crop_xyxy"], [0, 0, 12, 6])
        self.assertEqual(audit[0]["provenance"]["source_bbox"], [100, 150, 200, 300])
        self.assertEqual(self.decoded(exported[0]).shape, (6, 12))

    def test_full_image_uses_common_center_and_preserves_full_selected_mask(self):
        records = self.records([{
            "id": "full", "image_id": "source", "input_kind": "full_image",
            "source_bbox_xywh": [0, 1, 4, 4], "sam_prompt": "zebra",
        }])
        masks = np.zeros((2, 6, 12), dtype=bool)
        masks[0, :, 3:] = True
        masks[0, 3, 6] = False  # Genuine gap at the whole-input center.
        masks[1, :, :5] = True
        def predict(image, prompt):
            return {"masks": masks, "boxes": [[2, 0, 12, 6], [0, 0, 5, 6]],
                    "scores": [0.31, 0.99], "prompt_texts": prompt * 2}
        backend = FakeBackend(predict)
        destination = self.root / "full_image_run"
        summary = run_extraction(records, destination, backend, save_candidate_masks=True)
        exported, audit = self.artifacts(destination)
        self.assertEqual(summary["annotations_accepted"], 1)
        np.testing.assert_array_equal(backend.seen[0], self.pixels)
        self.assertEqual(audit[0]["selected_index"], 0)
        self.assertEqual(audit[0]["selection_policy"], "input_center")
        self.assertEqual(audit[0]["crop_xyxy"], [0, 0, 12, 6])
        self.assertEqual(audit[0]["provenance"]["source_bbox"], [0, 1, 4, 4])
        self.assertEqual(audit[0]["eligible_candidate_count"], 1)
        np.testing.assert_array_equal(self.decoded(exported[0]), masks[0])
        self.assertEqual(inspect_records(records)["errors"], [])


    def test_input_errors_are_counted_in_order_and_inference_continues(self):
        records = self.records(
            [{"id": "missing", "image_id": "missing"},
             {"id": "bad-box", "image_id": "source", "bbox": [30, 30, 5, 5]},
             {"id": "valid", "image_id": "source", "bbox": [0, 0, 6, 6]}],
            [{"id": "missing", "file_name": "absent.png"},
             {"id": "source", "file_name": "source.png"}],
        )
        destination = self.root / "run"
        backend = FakeBackend()
        summary = run_extraction(records, destination, backend, batch_size=3)
        exported, audit = self.artifacts(destination)
        self.assertEqual(summary["status"], "completed_with_input_errors")
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["annotations_attempted"], 3)
        self.assertEqual(summary["annotations_accepted"], 1)
        self.assertEqual([a["annotation_id"] for a in exported], ["missing", "bad-box", "valid"])
        self.assertEqual([a["status"] for a in audit], ["image_missing", "input_error", "accepted"])
        self.assertEqual(len(backend.seen), 1)
        inspection = inspect_records(records)
        self.assertEqual(len(inspection["errors"]), 2)

    def test_existing_output_is_never_overwritten(self):
        destination = self.root / "run"
        destination.mkdir()
        marker = destination / "user-file.txt"
        marker.write_text("retain me")
        backend = FakeBackend()
        with self.assertRaises(FileExistsError):
            run_extraction(self.records(), destination, backend)
        self.assertEqual(marker.read_text(), "retain me")
        self.assertEqual(backend.calls, [])
        self.assertEqual(list(destination.iterdir()), [marker])

    def test_backend_failure_keeps_partial_outputs_and_failed_summary(self):
        records = self.records([
            {"id": "first", "image_id": "source", "bbox": [0, 0, 6, 6]},
            {"id": "second", "image_id": "source", "bbox": [6, 0, 6, 6]},
        ])
        destination = self.root / "run"
        with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
            run_extraction(records, destination, FakeBackend(fail_call=2),
                           batch_size=1, legacy_pickle=True)
        summary = json.loads((destination / "extraction_summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["annotations_accepted"], 1)
        self.assertTrue((destination / "annotation_match_metadata.jsonl.partial").is_file())
        self.assertTrue((destination / "annotation_masks_rle.json.gz.partial").is_file())
        self.assertFalse((destination / "annotation_match_metadata.jsonl").exists())
        self.assertFalse((destination / "annotation_masks_rle.json.gz").exists())
        self.assertFalse((destination / "annotation_masks.pkl").exists())
        partial = (destination / "annotation_match_metadata.jsonl.partial").read_text().splitlines()
        self.assertEqual(len(partial), 1)
        self.assertEqual(json.loads(partial[0])["ann_id"], "first")

    def test_writer_constructor_failure_marks_run_failed(self):
        destination = self.root / "run"
        with mock.patch("sam3_masks.pipeline.gzip.GzipFile", side_effect=RuntimeError("cannot open compressor")):
            with self.assertRaisesRegex(RuntimeError, "cannot open compressor"):
                run_extraction(self.records(), destination, FakeBackend())
        summary = json.loads((destination / "extraction_summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["annotations_attempted"], 0)
        self.assertFalse((destination / "annotation_match_metadata.jsonl").exists())
        self.assertFalse((destination / "annotation_masks_rle.json.gz").exists())

    def test_pickle_finalization_failure_preserves_partial_exports(self):
        destination = self.root / "run"
        with mock.patch("sam3_masks.pipeline.pickle.dump", side_effect=OSError("cannot finish pickle")):
            with self.assertRaisesRegex(OSError, "cannot finish pickle"):
                run_extraction(self.records(), destination, FakeBackend(), legacy_pickle=True)
        summary = json.loads((destination / "extraction_summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        for name in ("annotation_match_metadata.jsonl", "annotation_masks_rle.json.gz", "annotation_masks.pkl"):
            self.assertFalse((destination / name).exists())
            self.assertTrue((destination / (name + ".partial")).is_file())

    def test_failed_publish_rolls_back_previously_published_files(self):
        destination = self.root / "run"
        original_rename = Path.rename
        attempts = []
        def fail_second(source, target):
            attempts.append((source, target))
            if len(attempts) == 2:
                raise OSError("simulated publish failure")
            return original_rename(source, target)
        with mock.patch.object(Path, "rename", new=fail_second):
            with self.assertRaisesRegex(OSError, "simulated publish failure"):
                run_extraction(self.records(), destination, FakeBackend())
        self.assertEqual(len(attempts), 3)  # First publish, second failure, first rollback.
        self.assertFalse((destination / "annotation_match_metadata.jsonl").exists())
        self.assertFalse((destination / "annotation_masks_rle.json.gz").exists())
        self.assertTrue((destination / "annotation_match_metadata.jsonl.partial").exists())
        self.assertTrue((destination / "annotation_masks_rle.json.gz.partial").exists())
        summary = json.loads((destination / "extraction_summary.json").read_text())
        self.assertEqual(summary["status"], "failed")

    def test_writer_close_is_idempotent_for_success_and_failure(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                destination = self.root / str(complete)
                destination.mkdir()
                writer = ArtifactWriter(destination, save_candidate_masks=False, legacy_pickle=False)
                writer.close(complete=complete)
                writer.close(complete=False)
                writer.close(complete=True)
                self.assertTrue(writer.audit.closed)
                self.assertTrue(writer.rle.closed)
                self.assertTrue(writer.rle_raw.closed)
                if complete:
                    exported, audit = self.artifacts(destination)
                    self.assertEqual(exported, [])
                    self.assertEqual(audit, [])
                else:
                    self.assertFalse((destination / "annotation_masks_rle.json.gz").exists())

    def test_fixed_predictions_produce_identical_rle_bytes(self):
        records = self.records()
        for name in ("first", "second"):
            run_extraction(records, self.root / name, FakeBackend())
        first = (self.root / "first" / "annotation_masks_rle.json.gz").read_bytes()
        second = (self.root / "second" / "annotation_masks_rle.json.gz").read_bytes()
        self.assertEqual(first, second)
        self.assertFalse((self.root / "first" / "annotation_masks.pkl").exists())


if __name__ == "__main__":
    unittest.main()
