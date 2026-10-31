"""CLI preflight and configuration tests require no model dependencies."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sam3_masks.cli import parse_args, resolve_config


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = Path(__file__).resolve().parents[1]
        (self.root / "images").mkdir()
        Image.new("RGB", (12, 10), "white").save(self.root / "images" / "target.png")
        self.metadata = self.root / "annotations.csv"
        self.metadata.write_text("annotation_id,path,bbox\ntarget,target.png,\n")

    def config(self, *args):
        return resolve_config(parse_args(list(args)))

    def test_dry_run_does_not_import_torch_or_transformers(self):
        # Explicitly block those imports even if the test environment has them.
        script = """
import importlib.abc
import sys
class BlockModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('dry-run imported model dependency: ' + fullname)
        return None
sys.meta_path.insert(0, BlockModelImports())
from sam3_masks.cli import main
main(sys.argv[1:])
"""
        report = self.root / "preflight.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.project)
        result = subprocess.run(
            [sys.executable, "-c", script, "--metadata", str(self.metadata), "--image-root", str(self.root / "images"), "--prompt", "antelope", "--dry-run", "--report", str(report)],
            cwd=self.root, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = json.loads(report.read_text())
        self.assertEqual(saved["annotations"], 1)
        self.assertEqual(saved["errors"], [])
        self.assertEqual(saved["records"][0]["input_size"], [12, 10])
        self.assertEqual(saved["records"][0]["prompts"], ["antelope"])
        self.assertEqual(saved["records"][0]["input_kind"], "native_crop")

    def test_manifest_relative_paths_and_prompt_map(self):
        manifest_dir = self.root / "jobs"
        manifest_dir.mkdir()
        (manifest_dir / "prompts.json").write_text(json.dumps({"nyala": "antelope"}))
        manifest = manifest_dir / "manifest.jsonl"
        manifest.write_text(json.dumps({"metadata": "../annotations.csv", "image_root": "../images", "output_dir": "outputs", "prompt_map": "prompts.json", "prompts": ["animal"], "dataset": "NyalaData"}) + "\n")
        config = self.config("--manifest", str(manifest), "--dry-run")
        self.assertEqual(config["metadata"], str(self.metadata))
        self.assertEqual(config["image_root"], str(self.root / "images"))
        self.assertEqual(config["output_dir"], str(manifest_dir / "outputs"))
        self.assertEqual(config["prompt_map"], {"nyala": "antelope"})
        self.assertEqual(config["prompts"], ["animal"])

    def test_cli_overrides_manifest(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps({"metadata": "annotations.csv", "image_root": "images", "dataset": "old", "prompt": "animal", "seed": 13, "device": "cuda", "confidence_threshold": 0.6}) + "\n")
        config = self.config("--manifest", str(manifest), "--dataset", "new", "--prompt", "antelope", "--seed", "17", "--device", "cpu", "--confidence-threshold", "0.3", "--dry-run")
        self.assertEqual(config["dataset"], "new")
        self.assertEqual(config["prompts"], ["antelope"])
        self.assertEqual(config["seed"], 17)
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["confidence_threshold"], 0.3)

    def test_manifest_old_selection_settings_rejected(self):
        for key in ("accept_iou", "min_sam_score", "candidate_mask_mode", "species_key"):
            with self.subTest(key=key):
                manifest = self.root / "manifest.jsonl"
                manifest.write_text(json.dumps({"metadata": "annotations.csv", key: 0}) + "\n")
                with self.assertRaisesRegex(ValueError, "obsolete"):
                    self.config("--manifest", str(manifest), "--dry-run")

    def test_cli_old_selection_flags_rejected(self):
        for flag in ("--accept-iou", "--min-sam-score", "--candidate-mask-mode"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(["--metadata", str(self.metadata), "--dry-run", flag, "0.5"])
                self.assertEqual(raised.exception.code, 2)

    def test_manifest_job_index_is_deterministic(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps({"metadata": "annotations.csv", "prompt": prompt}) for prompt in ("animal", "antelope")))
        self.assertEqual(self.config("--manifest", str(manifest), "--index", "1", "--dry-run")["prompts"], ["antelope"])
        with self.assertRaisesRegex(ValueError, "no index 2"):
            self.config("--manifest", str(manifest), "--index", "2", "--dry-run")

    def test_manifest_local_model_path_resolves_beside_manifest(self):
        (self.root / "weights").mkdir()
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps({"metadata": "annotations.csv", "model": "weights"}) + "\n")
        config = self.config("--manifest", str(manifest), "--dry-run")
        self.assertEqual(config["model_id"], str(self.root / "weights"))
        config = self.config("--metadata", str(self.metadata), "--dry-run")
        self.assertEqual(config["model_id"], "facebook/sam3")

    def test_manifest_boolean_is_not_parsed_as_truthy_string(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text(json.dumps({"metadata": "annotations.csv", "local_files_only": "false"}) + "\n")
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            self.config("--manifest", str(manifest), "--dry-run")


if __name__ == "__main__":
    unittest.main()
