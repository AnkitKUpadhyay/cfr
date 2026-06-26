#!/usr/bin/env python3
"""Shared helpers for cross-view animal ReID metadata."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEFT_LABELS = {"left", "l", "left side", "left_view", "left-view"}
RIGHT_LABELS = {"right", "r", "right side", "right_view", "right-view"}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return json.load(handle)


def norm_viewpoint(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower().replace("-", "_")
    if text in LEFT_LABELS:
        return "left"
    if text in RIGHT_LABELS:
        return "right"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if text:
        return text
    return "unknown"


def get_identity(ann: dict[str, Any]) -> str:
    for key in (
        "name",
        "individual_id",
        "identity",
        "individual",
        "id_individual",
        "name_uuid",
    ):
        value = ann.get(key)
        if value not in (None, ""):
            return str(value)
    values = ann.get("individual_ids")
    if isinstance(values, list) and values:
        return "|".join(str(v) for v in values)
    return "unknown"


def get_annotation_key(ann: dict[str, Any], fallback_index: int | None = None) -> str:
    for key in ("id", "uuid", "annot_uuid"):
        value = ann.get(key)
        if value not in (None, ""):
            return str(value)
    if fallback_index is not None:
        return str(fallback_index)
    return "unknown"


def get_image_key_from_ann(ann: dict[str, Any]) -> str:
    for key in ("image_id", "image_uuid", "image_path"):
        value = ann.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def get_image_key(image: dict[str, Any]) -> str:
    for key in ("id", "uuid", "image_uuid", "image_path", "file_name"):
        value = image.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def annotation_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for idx, ann in enumerate(data.get("annotations", [])):
        records.append(
            {
                "index": idx,
                "ann": ann,
                "ann_key": get_annotation_key(ann, idx),
                "image_key": get_image_key_from_ann(ann),
                "identity": get_identity(ann),
                "viewpoint": norm_viewpoint(ann.get("viewpoint")),
            }
        )
    return records


def identity_view_counts(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        identity = record["identity"]
        if identity != "unknown":
            counts[identity][record["viewpoint"]] += 1
    return counts


def linked_identities(records: list[dict[str, Any]]) -> set[str]:
    counts = identity_view_counts(records)
    return {
        identity
        for identity, view_counts in counts.items()
        if view_counts["left"] > 0 and view_counts["right"] > 0
    }


def image_lookup(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {get_image_key(image): image for image in data.get("images", [])}
