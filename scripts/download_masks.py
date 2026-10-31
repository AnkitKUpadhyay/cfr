#!/usr/bin/env python3
"""Download a fixed CFR mask release and verify its SHA256 checksums."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Callable

REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "mask_releases.json"
MARKER = ".cfr-mask-release.json"


@dataclass(frozen=True)
class DownloadPlan:
    release: str
    repo_id: str
    revision: str
    datasets: tuple[str, ...]
    files: tuple[str, ...]
    destination: Path


def load_registry(path: Path = REGISTRY) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def safe_relative_path(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Invalid release file path: {name!r}")
    if str(path) != name or name == ".":
        raise ValueError(f"Noncanonical release file path: {name!r}")
    return name


def make_plan(registry: dict, release: str, datasets: list[str] | None,
              output_dir: Path, *, include_candidates: bool = False,
              all_datasets: bool = False) -> DownloadPlan:
    config = registry["releases"][release]
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("The release must pin a full commit SHA")
    if include_candidates and "candidate_file" not in config:
        raise ValueError("--include-candidates applies only to the paper release")
    names = config["datasets"]
    aliases = {alias.casefold(): name for name, key in names.items() for alias in (name, key)}
    if all_datasets:
        chosen = sorted(names)
    else:
        if not datasets:
            raise ValueError("Select --datasets NAME [NAME ...] or --all")
        unknown = [name for name in datasets if name.casefold() not in aliases]
        if unknown:
            raise ValueError(f"Unknown dataset(s) for {release}: {', '.join(unknown)}; use --list")
        chosen = sorted({aliases[name.casefold()] for name in datasets})
    files = ["SHA256SUMS", *config["root_files"]]
    dataset_files = list(config["dataset_files"])
    if include_candidates:
        dataset_files.append(config["candidate_file"])
    for name in chosen:
        files.extend(f'{config["dataset_root"]}/{names[name]}/{filename}'
                     for filename in dataset_files)
    return DownloadPlan(release, registry["repo_id"], config["revision"], tuple(chosen),
                        tuple(safe_relative_path(name) for name in files), output_dir / release)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parse_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise ValueError(f"Invalid SHA256SUMS entry on line {number}")
        digest, name = match.groups()
        safe_relative_path(name)
        if name in checksums:
            raise ValueError(f"Duplicate SHA256SUMS entry: {name}")
        checksums[name] = digest.lower()
    return checksums


def release_identity(plan: DownloadPlan) -> dict[str, str]:
    return {"release": plan.release, "repo_id": plan.repo_id, "revision": plan.revision}


def check_destination(plan: DownloadPlan) -> None:
    marker = plan.destination / MARKER
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != release_identity(plan):
            raise ValueError(f"Different release already occupies {plan.destination}; choose another --output-dir")
    elif plan.destination.exists() and any(plan.destination.iterdir()):
        raise ValueError(f"Nonempty untracked destination: {plan.destination}; choose another --output-dir")


def download(plan: DownloadPlan, hub_download: Callable | None = None) -> Path:
    """Verify cached files, stage new files, then add them without overwriting."""
    check_destination(plan)
    if hub_download is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("Install the downloader dependency: pip install huggingface_hub") from error
        hub_download = hf_hub_download

    def fetch(name: str) -> Path:
        return Path(hub_download(repo_id=plan.repo_id, repo_type="dataset",
                                 revision=plan.revision, filename=name))

    manifest = fetch("SHA256SUMS")
    expected = parse_checksums(manifest)
    missing = [name for name in plan.files if name != "SHA256SUMS" and name not in expected]
    if missing:
        raise ValueError(f"Files missing from SHA256SUMS: {', '.join(missing)}")
    expected["SHA256SUMS"] = sha256(manifest)
    destination = plan.destination.resolve()
    pending = []
    for name in plan.files:
        path = plan.destination / name
        if not path.resolve().is_relative_to(destination):
            raise ValueError(f"Destination path escapes the release directory: {name}")
        if path.exists():
            if not path.is_file() or sha256(path) != expected[name]:
                raise ValueError(f"Existing file has a checksum mismatch: {path}; remove it explicitly or choose another --output-dir")
        else:
            pending.append(name)

    plan.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cfr-masks-", dir=plan.destination.parent) as temporary:
        stage = Path(temporary)
        for name in pending:
            cached = manifest if name == "SHA256SUMS" else fetch(name)
            staged = stage / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, staged)
            if sha256(staged) != expected[name]:
                raise ValueError(f"Downloaded file has a SHA256 checksum mismatch: {name}")
        # No destination files are changed until every requested file has passed.
        check_destination(plan)
        plan.destination.mkdir(parents=True, exist_ok=True)
        marker = plan.destination / MARKER
        if not marker.exists():
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(release_identity(plan), stream, indent=2)
                stream.write("\n")
        for name in pending:
            path = plan.destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # The staging directory shares a filesystem with the destination.
                # link() adds a complete file atomically and never replaces one.
                os.link(stage / name, path)
            except FileExistsError:
                if sha256(path) != expected[name]:
                    raise ValueError(f"Destination changed during download: {path}")
    return plan.destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=("paper", "standardized"), required=True)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--datasets", nargs="+", metavar="NAME")
    select.add_argument("--all", action="store_true", help="Download every dataset in this release")
    parser.add_argument("--output-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--include-candidates", action="store_true", help="Include paper-release audit candidates")
    parser.add_argument("--list", action="store_true", help="List dataset names without network access")
    parser.add_argument("--dry-run", action="store_true", help="Print the download plan without network access")
    args = parser.parse_args(argv)
    registry = load_registry()
    if args.list:
        print("\n".join(sorted(registry["releases"][args.release]["datasets"])))
        return 0
    try:
        plan = make_plan(registry, args.release, args.datasets, args.output_dir,
                         include_candidates=args.include_candidates, all_datasets=args.all)
        if args.dry_run:
            print(json.dumps({**release_identity(plan), "destination": str(plan.destination),
                              "datasets": plan.datasets, "files": plan.files}, indent=2))
        else:
            output = download(plan)
            print(f"Verified {len(plan.datasets)} dataset(s) from {plan.release} at {plan.revision}")
            print(output)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
