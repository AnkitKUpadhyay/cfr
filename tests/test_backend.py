"""Backend contract tests; no model download, GPU, or PyTorch installation needed."""

from __future__ import annotations

import contextlib
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from sam3_masks.backend import Sam3Backend


class _Inputs(dict):
    def to(self, device):
        self["device"] = str(device)
        return self


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.results = {}
        self.model_loads = []
        self.processor_loads = []
        self.processor_calls = []
        self.postprocess_calls = []
        self.in_inference = False

        @contextlib.contextmanager
        def inference_mode():
            self.in_inference = True
            try:
                yield
            finally:
                self.in_inference = False

        class FakeModel:
            config = SimpleNamespace(_commit_hash="checkpoint-sha")

            def to(inner, device):
                self.events.append(("model_device", str(device)))
                return inner

            def eval(inner):
                self.events.append(("eval",))
                return inner

            def __call__(inner, **inputs):
                self.assertTrue(self.in_inference)
                self.assertIn(("eval",), self.events)
                return inputs["prompt"]

        class FakeProcessor:
            def __call__(inner, **kwargs):
                self.processor_calls.append(kwargs)
                return _Inputs(prompt=kwargs["text"])

            def post_process_instance_segmentation(inner, outputs, **kwargs):
                self.postprocess_calls.append(kwargs)
                return [self.results[outputs]]

        self.fake_model = FakeModel()
        self.fake_processor = FakeProcessor()

        def load_model(model_id, **kwargs):
            self.model_loads.append((model_id, kwargs))
            return self.fake_model

        def load_processor(model_id, **kwargs):
            self.processor_loads.append((model_id, kwargs))
            return self.fake_processor

        class Device:
            def __init__(inner, value):
                inner.value = value
                inner.type = value.split(":")[0]

            def __str__(inner):
                return inner.value

        self.fake_torch = SimpleNamespace(
            device=Device,
            cuda=SimpleNamespace(
                is_available=lambda: False,
                is_initialized=lambda: False,
            ),
            manual_seed=lambda seed: self.events.append(("seed", seed)),
            use_deterministic_algorithms=lambda *args, **kwargs: self.events.append(("deterministic", args, kwargs)),
            set_float32_matmul_precision=lambda value: self.events.append(("precision", value)),
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True, version=lambda: None),
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            ),
            float32="float32",
            version=SimpleNamespace(cuda=None),
            get_num_threads=lambda: 1,
            inference_mode=inference_mode,
        )
        self.fake_transformers = SimpleNamespace(
            Sam3Model=SimpleNamespace(from_pretrained=load_model),
            Sam3Processor=SimpleNamespace(from_pretrained=load_processor),
        )
        self.environment_patch = patch.dict(os.environ)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.dependencies_patch = patch("sam3_masks.backend._dependencies", return_value=(self.fake_torch, self.fake_transformers))
        self.dependencies_patch.start()
        self.addCleanup(self.dependencies_patch.stop)
        self.metadata_patch = patch("sam3_masks.backend._package_metadata", return_value={"version": "test"})
        self.metadata_patch.start()
        self.addCleanup(self.metadata_patch.stop)

    @staticmethod
    def result(masks, scores):
        return {
            "masks": np.asarray(masks, dtype=np.int64),
            "scores": np.asarray(scores, dtype=np.float32),
            "boxes": np.tile(np.asarray([[0, 0, 5, 5]], dtype=np.float32), (len(scores), 1)),
        }

    def test_explicit_settings_eval_and_checkpoint_revision(self):
        backend = Sam3Backend(device="cpu", revision="main", local_files_only=True, seed=19)
        self.assertEqual(self.model_loads, [("facebook/sam3", {
            "attn_implementation": "eager", "dtype": "float32",
            "local_files_only": True, "revision": "main",
        })])
        self.assertEqual(self.processor_loads, [("facebook/sam3", {
            "local_files_only": True, "revision": "checkpoint-sha",
        })])
        self.assertIn(("eval",), self.events)
        self.assertIn(("seed", 19), self.events)
        self.assertIn(("deterministic", (True,), {"warn_only": False}), self.events)
        self.assertFalse(self.fake_torch.backends.cudnn.benchmark)
        self.assertFalse(self.fake_torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(backend.metadata["resolved_revision"], "checkpoint-sha")
        self.assertEqual(backend.metadata["mask_cleanup"], "none")

    def test_threshold_arguments_order_and_unmodified_masks(self):
        masks = np.zeros((2, 5, 5), dtype=np.int64)
        masks[0, 1:4, 1:4] = 1
        masks[0, 2, 2] = 0  # Preserve this enclosed hole.
        masks[0, 0, 0] = 1  # Preserve this isolated speck.
        masks[1, 2, 2] = 1
        self.results["antelope"] = self.result(masks, [0.4, 0.8])
        self.results["animal"] = self.result(masks[1:], [0.9])
        backend = Sam3Backend(device="cpu")
        output = backend.predict([Image.new("RGB", (5, 5))], [["antelope", "animal"]])[0]
        self.assertEqual(self.postprocess_calls, [{
            "threshold": 0.3, "mask_threshold": 0.5, "target_sizes": [(5, 5)],
        }] * 2)
        np.testing.assert_array_equal(output["masks"][:2], masks.astype(bool))
        self.assertEqual(output["masks"].dtype, np.bool_)
        self.assertEqual(output["prompt_texts"], ["antelope", "antelope", "animal"])
        self.assertEqual(output["prompt_indices"].tolist(), [0, 0, 1])
        self.assertEqual(output["prompt_candidate_indices"].tolist(), [0, 1, 0])
        self.assertEqual(output["candidate_indices"].tolist(), [0, 1, 2])
        np.testing.assert_allclose(output["scores"], [0.4, 0.8, 0.9])

    def test_no_additional_downstream_score_filter(self):
        # The processor owns filtering: an adapter must not silently filter twice.
        self.results["animal"] = self.result(np.ones((1, 5, 5)), [0.1])
        backend = Sam3Backend(device="cpu")
        output = backend.predict([Image.new("RGB", (5, 5))], [["animal"]])[0]
        self.assertEqual(len(output["masks"]), 1)

    def test_empty_masks_are_normalized_to_input_dimensions(self):
        self.results["animal"] = self.result(np.empty((0, 16, 16)), [])
        backend = Sam3Backend(device="cpu")
        output = backend.predict([Image.new("RGB", (7, 5))], [["animal"]])[0]
        self.assertEqual(output["masks"].shape, (0, 5, 7))
        self.assertEqual(output["boxes"].shape, (0, 4))
        self.assertEqual(output["scores"].shape, (0,))

    def test_image_grouping_does_not_change_calls_or_predictions(self):
        self.results["animal"] = self.result(np.ones((1, 5, 5)), [0.8])
        backend = Sam3Backend(device="cpu")
        image = Image.new("RGB", (5, 5))
        combined = backend.predict([image, image], [["animal"], ["animal"]])
        separate = backend.predict([image], [["animal"]]) + backend.predict([image], [["animal"]])
        for left, right in zip(combined, separate):
            for key in ("masks", "boxes", "scores", "candidate_indices"):
                np.testing.assert_array_equal(left[key], right[key])
        self.assertTrue(all(isinstance(call["images"], Image.Image) for call in self.processor_calls))

    def test_malformed_predictions_raise(self):
        self.results["animal"] = self.result(np.ones((1, 4, 4)), [0.8])
        backend = Sam3Backend(device="cpu")
        with self.assertRaisesRegex(RuntimeError, "input dimensions"):
            backend.predict([Image.new("RGB", (5, 5))], [["animal"]])

    def test_nonbinary_and_nonfinite_masks_are_not_coerced_to_foreground(self):
        backend = Sam3Backend(device="cpu")
        image = Image.new("RGB", (5, 5))
        for invalid_value in (0.5, 2, -1, float("nan"), float("inf")):
            with self.subTest(invalid_value=invalid_value):
                result = self.result(np.ones((1, 5, 5)), [0.8])
                result["masks"] = result["masks"].astype(np.float32)
                result["masks"][0, 2, 2] = invalid_value
                self.results["animal"] = result
                with self.assertRaisesRegex(RuntimeError, "nonbinary or nonfinite"):
                    backend.predict([image], [["animal"]])

    def test_configuration_and_prompt_errors(self):
        for kwargs in ({"confidence_threshold": float("nan")}, {"mask_threshold": -1}, {"seed": -1}):
            with self.assertRaises(ValueError):
                Sam3Backend(device="cpu", **kwargs)
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            Sam3Backend(device="cuda")
        backend = Sam3Backend(device="cpu")
        image = Image.new("RGB", (5, 5))
        for images, prompts in (([image], []), ([image], [[]]), ([image], [[" "]]), ([image], ["animal"])):
            with self.assertRaises(ValueError):
                backend.predict(images, prompts)

    def test_local_directory_revision_is_not_claimed_as_a_snapshot_pin(self):
        self.fake_model.config = SimpleNamespace(_commit_hash=None)
        with tempfile.TemporaryDirectory() as directory:
            backend = Sam3Backend(model_id=directory, revision="ignored-branch", device="cpu")
        self.assertNotIn("revision", self.model_loads[0][1])
        self.assertNotIn("revision", self.processor_loads[0][1])
        self.assertEqual(backend.metadata["model_source"], "local_directory")
        self.assertFalse(backend.metadata["revision_applies_to_source"])
        self.assertIsNone(backend.metadata["resolved_revision"])
        self.assertIsNone(backend.metadata["processor_revision"])

    def test_checkpoint_errors_preserve_the_original_revision_error(self):
        self.fake_transformers.Sam3Model.from_pretrained = lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("missing-sha is not a valid git identifier")
        )
        with self.assertRaisesRegex(RuntimeError, "missing-sha is not a valid git identifier"):
            Sam3Backend(device="cpu", revision="missing-sha")

    def test_preinitialized_cuda_rejected_consistently_on_retry(self):
        self.fake_torch.cuda.is_available = lambda: True
        self.fake_torch.cuda.is_initialized = lambda: True
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "CUDA was initialized before"):
                Sam3Backend(device="cuda")
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)


if __name__ == "__main__":
    unittest.main()
