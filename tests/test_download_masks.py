"""Check release isolation, integrity failures, and offline download planning."""
from __future__ import annotations

import builtins
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_masks as dm


class DownloadMasksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = dm.load_registry()

    def plan(self, release="paper", names=None, **kwargs):
        return dm.make_plan(self.registry, release, names or ["NyalaData"],
                            self.root / "output", **kwargs)

    def fake_hub(self, plan, *, corrupt=None, omit_from_manifest=None):
        cache = self.root / "cache" / plan.release
        cache.mkdir(parents=True, exist_ok=True)
        lines = []
        for name in plan.files:
            if name == "SHA256SUMS":
                continue
            payload = f"{plan.revision}\n{name}\n".encode()
            path = cache / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload if name != corrupt else b"corrupted content")
            if name != omit_from_manifest:
                lines.append(f"{hashlib.sha256(payload).hexdigest()}  {name}\n")
        (cache / "SHA256SUMS").write_text("".join(lines))
        calls = []

        def fetch(**kwargs):
            self.assertEqual(kwargs["repo_id"], "imageomics/cfr-wildlife-masks")
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(kwargs["revision"], plan.revision)
            calls.append(kwargs["filename"])
            return str(cache / kwargs["filename"])

        return fetch, calls

    def test_official_names_aliases_and_pinned_release_selection(self):
        paper = self.plan(names=["SeaTurtleID2022", "seaturtle_head", "NYALADATA"])
        self.assertEqual(paper.revision, "7655704d058af7fa4ed5e3b901cff1d521d7ede7")
        self.assertEqual(paper.datasets, ("NyalaData", "SeaTurtleID2022"))
        self.assertIn("Masks_data/seaturtle_head/annotation_masks_rle.json.gz", paper.files)
        self.assertFalse(any("best_any" in name for name in paper.files))
        standardized = self.plan("standardized", ["PolarBearVidID"])
        self.assertEqual(standardized.revision, "3baa6a58da04cc18bf5e21f77cf0d366efe1e5f7")
        self.assertIn("data/PolarBearVidID/metadata.parquet", standardized.files)
        self.assertNotEqual(paper.destination, standardized.destination)

    def test_candidate_files_are_opt_in_and_paper_only(self):
        plan = self.plan(include_candidates=True)
        self.assertIn("Masks_data/nyaladata/annotation_best_any_iou_masks_rle.json.gz", plan.files)
        with self.assertRaisesRegex(ValueError, "only to the paper"):
            self.plan("standardized", include_candidates=True)

    def test_unavailable_dataset_fails_before_download(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            self.plan("standardized", ["SeaTurtleID2022"])
        with self.assertRaisesRegex(ValueError, "Select --datasets"):
            dm.make_plan(self.registry, "paper", None, self.root)

    def test_offline_list_and_dry_run_do_not_import_hub(self):
        original_import = builtins.__import__

        def forbid_hub(name, *args, **kwargs):
            if name.startswith("huggingface_hub"):
                raise AssertionError("Offline planning imported Hugging Face")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=forbid_hub):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(dm.main(["--release", "paper", "--list"]), 0)
            self.assertIn("SeaTurtleID2022", output.getvalue())
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(dm.main(["--release", "standardized", "--datasets", "NyalaData", "--dry-run"]), 0)
            self.assertEqual(json.loads(output.getvalue())["destination"], "data/masks/standardized")

    def test_downloads_only_selected_files_and_reuses_verified_files(self):
        plan = self.plan()
        fetch, calls = self.fake_hub(plan)
        self.assertEqual(dm.download(plan, fetch), plan.destination)
        self.assertEqual(set(calls), set(plan.files))
        self.assertEqual(json.loads((plan.destination / dm.MARKER).read_text()), dm.release_identity(plan))
        actual = {str(p.relative_to(plan.destination)) for p in plan.destination.rglob("*") if p.is_file()}
        self.assertEqual(actual, {*plan.files, dm.MARKER})
        calls.clear()
        dm.download(plan, fetch)
        self.assertEqual(calls, ["SHA256SUMS"])

    def test_corrupt_download_does_not_publish_partial_files(self):
        plan = self.plan()
        fetch, _ = self.fake_hub(plan, corrupt=plan.files[-1])
        with self.assertRaisesRegex(ValueError, "SHA256 checksum mismatch"):
            dm.download(plan, fetch)
        self.assertFalse(plan.destination.exists())

    def test_can_add_another_dataset_from_the_same_snapshot(self):
        single = self.plan()
        expanded = self.plan(names=["NyalaData", "StripeSpotter"])
        fetch, calls = self.fake_hub(expanded)
        dm.download(single, fetch)
        calls.clear()
        dm.download(expanded, fetch)
        self.assertEqual(calls, ["SHA256SUMS", "Masks_data/stripespotter/annotation_masks_rle.json.gz"])
        self.assertTrue((single.destination / "Masks_data/nyaladata/annotation_masks_rle.json.gz").exists())

    def test_corrupt_existing_file_is_not_overwritten(self):
        plan = self.plan()
        fetch, calls = self.fake_hub(plan)
        dm.download(plan, fetch)
        target = plan.destination / plan.files[-1]
        target.write_bytes(b"locally altered")
        calls.clear()
        with self.assertRaisesRegex(ValueError, "Existing file has a checksum mismatch"):
            dm.download(plan, fetch)
        self.assertEqual(target.read_bytes(), b"locally altered")
        self.assertEqual(calls, ["SHA256SUMS"])

    def test_revision_mismatch_and_untracked_folder_are_rejected(self):
        plan = self.plan()
        plan.destination.mkdir(parents=True)
        marker = plan.destination / dm.MARKER
        marker.write_text(json.dumps({**dm.release_identity(plan), "revision": "a" * 40}))
        with self.assertRaisesRegex(ValueError, "Different release"):
            dm.download(plan, lambda **_: self.fail("must not download"))
        marker.unlink()
        (plan.destination / "existing.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "Nonempty untracked"):
            dm.download(plan, lambda **_: self.fail("must not download"))

    def test_missing_manifest_entry_is_rejected(self):
        plan = self.plan()
        fetch, _ = self.fake_hub(plan, omit_from_manifest=plan.files[-1])
        with self.assertRaisesRegex(ValueError, "Files missing from SHA256SUMS"):
            dm.download(plan, fetch)
        self.assertFalse(plan.destination.exists())

    def test_unsafe_or_duplicate_manifest_paths_are_rejected(self):
        manifest = self.root / "SHA256SUMS"
        manifest.write_text(f"{'a' * 64}  ../outside\n")
        with self.assertRaisesRegex(ValueError, "Invalid release file path"):
            dm.parse_checksums(manifest)
        manifest.write_text(f"{'a' * 64}  README.md\n{'b' * 64}  README.md\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            dm.parse_checksums(manifest)


if __name__ == "__main__":
    unittest.main()
