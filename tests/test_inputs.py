"""Tests of annotation provenance, coordinate handling, and prompt safety."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sam3_masks.inputs import load_records, prepare_input


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = Image.new("RGB", (10, 8))
        image.putdata([(x * 20, y * 20, 50) for y in range(8) for x in range(10)])
        image.save(self.root / "animal.png")
        image.save(self.root / "second.png")

    def csv(self, rows):
        path = self.root / "metadata.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def json(self, images, annotations, categories=None):
        path = self.root / "metadata.json"
        path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories or []}))
        return path

    def load(self, path, **kwargs):
        kwargs.setdefault("default_prompts", ["animal"])
        return load_records(path, self.root, **kwargs)

    def test_source_crop_pixels_and_rounding(self):
        path = self.csv([{"annotation_id": "target", "path": "animal.png", "bbox": "[2.4, 1.6, 4.2, 3.8]"}])
        record = self.load(path)[0]
        prepared = prepare_input(record)
        self.assertEqual(record.input_kind, "source_box")
        self.assertEqual(prepared.original_size, (10, 8))
        self.assertEqual(prepared.crop_xyxy, (2, 2, 7, 5))
        self.assertEqual(prepared.image.size, (5, 3))
        self.assertEqual(prepared.image.getpixel((0, 0)), (40, 40, 50))
        self.assertEqual(prepared.image.mode, "RGB")

    def test_negative_origin_clips_without_padding(self):
        path = self.csv([{"id": 0, "path": "animal.png", "bbox": "-2,-1,5,4"}])
        prepared = prepare_input(self.load(path)[0])
        self.assertEqual(prepared.crop_xyxy, (0, 0, 3, 3))
        self.assertEqual(prepared.image.size, (3, 3))

    def test_empty_clipped_or_rounded_box_rejected(self):
        for bbox in ("20,20,3,3", "1,1,0.1,0.1", "-10,-10,1,1"):
            with self.subTest(bbox=bbox):
                path = self.csv([{"id": "a", "path": "animal.png", "bbox": bbox}])
                record = self.load(path)[0]
                with self.assertRaisesRegex(ValueError, "empty after rounding"):
                    prepare_input(record)

    def test_prepared_crop_ignores_original_box_retains_provenance(self):
        path = self.csv([{"id": "a", "path": "animal.png", "bbox": "100,100,200,300", "is_cropped": "true", "source_image": "original.jpg"}])
        record = self.load(path)[0]
        self.assertEqual(record.input_kind, "prepared_crop")
        self.assertIsNone(record.bbox_xywh)
        self.assertEqual(record.provenance["source_bbox"], [100, 100, 200, 300])
        self.assertEqual(record.provenance["source_image"], "original.jpg")
        self.assertEqual(prepare_input(record).image.size, (10, 8))

    def test_source_bbox_alone_is_provenance_not_current_crop(self):
        path = self.csv([{"id": "a", "path": "animal.png", "source_bbox": "100,100,200,300", "input_kind": "prepared_crop"}])
        record = self.load(path)[0]
        self.assertIsNone(record.bbox_xywh)
        self.assertEqual(prepare_input(record).crop_xyxy, (0, 0, 10, 8))


    def test_full_image_current_crop_bbox_is_a_conflict(self):
        path = self.csv([{
            "id": "flank", "path": "animal.png", "input_kind": "full_image",
            "bbox": "1,1,4,4",
        }])
        with self.assertRaises(ValueError):
            self.load(path)

    def test_full_image_retains_source_region_without_cropping(self):
        path = self.csv([{
            "id": "target", "path": "animal.png", "input_kind": "full_image",
            "source_bbox_xywh": "1,1,4,4",
        }])
        record = self.load(path)[0]
        prepared = prepare_input(record)
        self.assertIsNone(record.bbox_xywh)
        self.assertEqual(record.provenance["source_bbox"], "1,1,4,4")
        self.assertEqual(prepared.crop_xyxy, (0, 0, 10, 8))
        with Image.open(self.root / "animal.png") as original:
            self.assertEqual(prepared.image.tobytes(), original.tobytes())

    def test_superseded_selection_bbox_is_rejected_instead_of_ignored(self):
        for box in ("1,1,4,4", "bad", "1,1,0,4"):
            with self.subTest(box=box):
                path = self.csv([{
                    "id": "target", "path": "animal.png", "input_kind": "full_image",
                    "selection_bbox": box,
                }])
                with self.assertRaisesRegex(ValueError, "selection_bbox is unsupported"):
                    self.load(path)


    def test_false_cropped_flag_is_not_truthy_string(self):
        for value in ("false", "0", "no"):
            with self.subTest(value=value):
                path = self.csv([{"id": "a", "path": "animal.png", "bbox": "2,1,4,3", "is_cropped": value}])
                record = self.load(path)[0]
                self.assertEqual(record.input_kind, "source_box")
                self.assertEqual(prepare_input(record).image.size, (4, 3))

    def test_multiple_boxed_annotations_share_original_image(self):
        path = self.csv([
            {"annotation_id": "b", "image_id": "source", "path": "animal.png", "bbox": "5,0,5,8"},
            {"annotation_id": "a", "image_id": "source", "path": "animal.png", "bbox": "0,0,5,8"},
        ])
        records = self.load(path)
        self.assertEqual([record.ann_id for record in records], ["b", "a"])
        self.assertEqual([record.source_index for record in records], [0, 1])
        self.assertEqual([prepare_input(record).image.getpixel((0, 0)) for record in records], [(100, 0, 50), (0, 0, 50)])

    def test_multiple_no_box_annotations_rejected_even_with_path_alias(self):
        path = self.csv([
            {"annotation_id": "a", "path": "animal.png"},
            {"annotation_id": "b", "path": "./animal.png"},
        ])
        with self.assertRaisesRegex(ValueError, "multiple annotations"):
            self.load(path)

    def test_duplicate_ids_rejected(self):
        path = self.csv([{"id": "same", "path": "animal.png"}, {"id": "same", "path": "second.png"}])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            self.load(path)

    def test_invalid_bbox_never_becomes_native_crop(self):
        for bbox in ("1,2,3", "1,2,0,4", "1,2,-1,4", "NaN,2,3,4", "1,2,Infinity,4", "bad", "[]", "[1,2,3,4,5]"):
            with self.subTest(bbox=bbox):
                path = self.csv([{"id": "a", "path": "animal.png", "bbox": bbox}])
                with self.assertRaises(ValueError):
                    self.load(path)

    def test_partial_bbox_columns_rejected(self):
        path = self.csv([{"id": "a", "path": "animal.png", "bbox_x": 1, "bbox_y": 2, "bbox_w": 3}])
        with self.assertRaisesRegex(ValueError, "must all be supplied"):
            self.load(path)

    def test_source_kind_requires_current_box(self):
        path = self.csv([{"id": "a", "path": "animal.png", "input_kind": "source_box", "source_bbox": "1,2,3,4"}])
        with self.assertRaisesRegex(ValueError, "requires bbox"):
            self.load(path)

    def test_json_category_prompts_and_gzcd_references(self):
        path = self.json(
            [{"uuid": "im", "id": 7, "file_name": "animal.png"}],
            [{"annot_uuid": "g", "image_uuid": "im", "category_id": 1, "bbox": [0, 0, 4, 8]},
             {"annotation_id": "z", "image_id": 7, "category_id": 2, "bbox": [4, 0, 6, 8]}],
            [{"id": 1, "name": "giraffe"}, {"id": 2, "name": "zebra"}],
        )
        records = self.load(path, dataset="MixedSpecies", default_prompts=None,
                            prompt_map={"giraffe": "giraffe", "zebra": "zebra"})
        self.assertEqual([record.prompts for record in records], [["giraffe"], ["zebra"]])
        self.assertEqual(records[0].provenance["category"], {"id": 1, "name": "giraffe"})

    def test_prompt_priority_explicit_then_map_then_default(self):
        path = self.csv([
            {"id": "a", "path": "animal.png", "species": "nyala", "sam_prompt": "nyala antelope"},
            {"id": "b", "path": "second.png", "species": "nyala"},
        ])
        records = self.load(path, prompt_map={"nyala": ["antelope"]})
        self.assertEqual([record.prompts for record in records], [["nyala antelope"], ["antelope"]])
        self.assertEqual(self.load(path)[1].prompts, ["animal"])

    def test_category_id_prompt_mapping(self):
        path = self.csv([{"id": "a", "path": "animal.png", "category_id": 3}])
        self.assertEqual(self.load(path, prompt_map={"3": ["zebra"]})[0].prompts, ["zebra"])


    def test_gz_explicit_prompt_without_category_allowed(self):
        path = self.csv([{"id": "a", "path": "animal.png", "sam_prompt": "zebra"}])
        self.assertEqual(self.load(path, dataset="GiraffeZebraID")[0].prompts, ["zebra"])

    def test_dataset_name_does_not_change_prompt_routing(self):
        path = self.csv([{"id": "a", "path": "animal.png", "species": "giraffe"}])
        for dataset in ("GiraffeZebraID", "UnseenDataset"):
            with self.subTest(dataset=dataset):
                self.assertEqual(self.load(path, dataset=dataset,
                    prompt_map={"giraffe": "giraffe"})[0].prompts, ["giraffe"])
                self.assertEqual(self.load(path, dataset=dataset)[0].prompts, ["animal"])
                with self.assertRaisesRegex(ValueError, "No prompt"):
                    self.load(path, dataset=dataset, default_prompts=None)



    def test_ordered_same_target_prompts_preserved(self):
        path = self.csv([{"id": "a", "path": "animal.png", "species": "nyala"}])
        self.assertEqual(self.load(path, default_prompts=["antelope", "nyala antelope"])[0].prompts, ["antelope", "nyala antelope"])
        self.assertEqual(self.load(path, prompt_map={"nyala": ["nyala antelope", "antelope"]})[0].prompts, ["nyala antelope", "antelope"])

    def test_metadata_split_takes_precedence_over_default(self):
        path = self.csv([{"id": "a", "path": "animal.png", "split": "test"}, {"id": "b", "path": "second.png"}])
        self.assertEqual([record.split for record in self.load(path, split="train")], ["test", "train"])

    def test_missing_json_reference_not_silently_dropped(self):
        path = self.json([], [{"id": "a", "image_id": "missing"}])
        with self.assertRaisesRegex(ValueError, "missing image"):
            self.load(path)

    def test_missing_json_image_path_not_silently_dropped(self):
        path = self.json([{"id": 1}], [{"id": "a", "image_id": 1}])
        with self.assertRaisesRegex(ValueError, "missing path"):
            self.load(path)

    def test_missing_physical_image_retained_for_runner_io_report(self):
        path = self.csv([{"id": "a", "path": "missing.png"}])
        record = self.load(path)[0]
        with self.assertRaises(FileNotFoundError):
            prepare_input(record)

    def test_native_image_is_loaded_without_exif_transpose(self):
        image = Image.new("RGB", (10, 8))
        exif = Image.Exif()
        exif[274] = 6
        image.save(self.root / "oriented.jpg", exif=exif)
        path = self.csv([{"id": "a", "path": "oriented.jpg"}])
        prepared = prepare_input(self.load(path)[0])
        self.assertEqual(prepared.image.size, (10, 8))
        self.assertEqual(prepared.original_size, (10, 8))


if __name__ == "__main__":
    unittest.main()
