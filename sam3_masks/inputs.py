"""Annotation-level input loading and preparation, independent of SAM or torch.

Boxes are XYWH in the coordinates of the current image unless ``input_kind``
is ``prepared_crop`` (or ``is_cropped`` is true). Original source metadata is
always retained as provenance and never used to crop a prepared crop again.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class AnnotationRecord:
    ann_id: str
    image_key: str
    image_path: Path
    dataset: str
    split: str | None
    identity: str | None
    bbox_xywh: list[float] | None
    input_kind: str
    prompts: list[str]
    source_index: int
    source: str
    provenance: dict[str, Any]


@dataclass
class PreparedInput:
    record: AnnotationRecord
    image: Image.Image
    crop_xyxy: tuple[int, int, int, int]
    original_size: tuple[int, int]


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if _present(value):
            return value
    return None


def _text(value: Any) -> str | None:
    return str(value) if _present(value) else None


def _bool(value: Any, field: str) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "1"):
            return True
        if normalized in ("false", "no", "0"):
            return False
    raise ValueError(f"{field} must be true/false, yes/no, or 1/0; got {value!r}")


def _bbox(row: dict[str, Any]) -> list[float] | None:
    raw = row.get("bbox")
    keys = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    if not _present(raw):
        supplied = [_present(row.get(key)) for key in keys]
        if not any(supplied):
            return None
        if not all(supplied):
            raise ValueError("bbox_x, bbox_y, bbox_w, bbox_h must all be supplied")
        raw = [row[key] for key in keys]
    if isinstance(raw, str):
        value = raw.strip()
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            raw = re.split(r"[,;\s]+", value.strip("[]() "))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"bbox must contain exactly four XYWH numbers; got {raw!r}")
    try:
        box = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox must contain four numeric XYWH values; got {raw!r}") from exc
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"bbox coordinates must be finite; got {raw!r}")
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"bbox width and height must be positive; got {raw!r}")
    if not math.isfinite(box[0] + box[2]) or not math.isfinite(box[1] + box[3]):
        raise ValueError(f"bbox endpoints must be finite; got {raw!r}")
    return box


def _prompts(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must supply a nonempty prompt string or ordered list")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in value):
        raise ValueError(f"{field} must contain only nonempty prompt strings")
    return [prompt.strip() for prompt in value]


def _category_info(row: dict[str, Any], categories: dict[str, dict[str, Any]]) -> tuple[str | None, str | None, str | None]:
    category_id = _text(row.get("category_id"))
    category = row.get("category")
    category_name = _text(row.get("category_name"))
    if isinstance(category, dict):
        category_name = category_name or _text(category.get("name"))
        category_id = category_id or _text(category.get("id"))
    elif _present(category):
        category_name = category_name or str(category)
    if category_id in categories:
        category_name = category_name or _text(categories[category_id].get("name"))
    species = _text(row.get("species"))
    return category_name, species, category_id


def _resolve_prompts(row: dict[str, Any], categories: dict[str, dict[str, Any]], default_prompts: list[str] | None, prompt_map: dict[str, list[str]] | None) -> tuple[list[str], str | None, str | None, str | None]:
    name, species, category_id = _category_info(row, categories)
    explicit = _first(row, "sam_prompt", "prompt")
    selected = None
    if explicit is not None:
        if not isinstance(explicit, str):
            raise ValueError("sam_prompt/prompt must be a single string")
        selected = _prompts(explicit, "sam_prompt/prompt")
    else:
        mapped = {str(key).strip().casefold(): value for key, value in (prompt_map or {}).items()}
        for key in (name, species, category_id):
            if key is not None and key.strip().casefold() in mapped:
                selected = _prompts(mapped[key.strip().casefold()], f"prompt_map[{key!r}]")
                break
    if selected is not None:
        return selected, name, species, category_id
    if default_prompts is None:
        raise ValueError("No prompt for annotation: supply sam_prompt/prompt, a matching prompt_map entry, or a default prompt")
    return _prompts(default_prompts, "default_prompts"), name, species, category_id


def _resolve_path(value: Any, image_root: Path, dataset: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Annotation image is missing path/image_path/file_name")
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [image_root / raw, image_root / dataset / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # Preserve missing image records so the runner can emit an explicit I/O error.
    return candidates[0].resolve()


def _record(row: dict[str, Any], *, image: dict[str, Any] | None, ann_id: str, image_key: str, source_index: int, source: Path, image_root: Path, dataset: str | None, split: str | None, categories: dict[str, dict[str, Any]], default_prompts: list[str] | None, prompt_map: dict[str, list[str]] | None) -> AnnotationRecord:
    combined = dict(image or {})
    combined.update(row)
    resolved_dataset = str(dataset or _first(combined, "dataset", "dataset_name") or source.stem)
    # In COCO, annotation.path is provenance at most; image entries own current paths.
    path_row = image if image is not None else row
    path_value = _first(path_row, "path", "image_path", "file_name", "filepath")
    image_path = _resolve_path(path_value, image_root, resolved_dataset)
    box = _bbox(combined)
    supplied_kind = _text(combined.get("input_kind"))
    if supplied_kind is not None:
        supplied_kind = supplied_kind.strip().lower()
        if supplied_kind not in ("source_box", "native_crop", "prepared_crop", "full_image"):
            raise ValueError(f"Unsupported input_kind {supplied_kind!r}")
    already_cropped = _bool(combined.get("is_cropped"), "is_cropped")
    if already_cropped and supplied_kind in ("source_box", "native_crop", "full_image"):
        raise ValueError("is_cropped=true conflicts with input_kind; use prepared_crop")
    kind = supplied_kind or ("prepared_crop" if already_cropped else "source_box" if box is not None else "native_crop")
    if kind == "source_box" and box is None:
        raise ValueError("input_kind=source_box requires bbox in the current image's coordinates")
    if kind == "native_crop" and box is not None:
        raise ValueError("input_kind=native_crop conflicts with bbox; use source_box or prepared_crop explicitly")
    if _present(combined.get("selection_bbox")):
        raise ValueError("selection_bbox is unsupported: selection always uses the whole input center and mask area; retain a source region as source_bbox_xywh provenance")
    if kind == "full_image" and box is not None:
        raise ValueError("full_image must not supply bbox; retain a source region as source_bbox_xywh provenance")
    prompts, category_name, species, category_id = _resolve_prompts(combined, categories, default_prompts, prompt_map)
    provenance = {
        "metadata_source": str(source),
        "source_index": source_index,
        "annotation_id": ann_id,
        "image_key": image_key,
        "annotation": dict(row),
        "category_id": category_id,
        "category_name": category_name,
        "species": species,
        "body_part": combined.get("body_part"),
        "source_image": _first(combined, "source_image", "source_image_path", "original_image", "original_image_path"),
        "source_bbox": _first(combined, "source_bbox", "source_bbox_xywh", "original_bbox"),
    }
    if image is not None:
        provenance["image"] = dict(image)
    if category_id in categories:
        provenance["category"] = dict(categories[category_id])
    elif "category" in combined:
        provenance["category"] = combined["category"]
    if box is not None:
        provenance["metadata_bbox_xywh"] = list(box)
        if provenance["source_bbox"] is None:
            provenance["source_bbox"] = list(box)
    if kind == "source_box" and provenance["source_image"] is None:
        provenance["source_image"] = str(image_path)
    return AnnotationRecord(
        ann_id=ann_id, image_key=image_key, image_path=image_path,
        dataset=resolved_dataset, split=_text(combined.get("split")) or split,
        identity=_text(_first(combined, "identity", "name_uuid", "individual_id", "name")),
        bbox_xywh=box if kind == "source_box" else None, input_kind=kind,
        prompts=prompts, source_index=source_index, source=str(source), provenance=provenance,
    )


def load_records(path: Path, image_root: Path, dataset: str | None = None, split: str | None = None, default_prompts: list[str] | None = None, prompt_map: dict[str, list[str]] | None = None) -> list[AnnotationRecord]:
    """Read CSV or COCO/GZCD JSON in stable source order; reject ambiguity.

    A no-box input must have one annotation per resolved current image path.
    This checks annotation ambiguity, not whether the animal is centered.
    Missing files are retained for the runner to report through its I/O status.
    """
    path = Path(path).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    records = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            rows = list(reader)
        for index, row in enumerate(rows):
            annotation_id = _first(row, "annotation_id", "annot_uuid", "image_id", "id")
            if annotation_id is None:
                annotation_id = f"ann_{index}"
            key = _first(row, "image_id", "image_uuid", "id")
            if key is None:
                key = f"image_{index}"
            try:
                records.append(_record(row, image=None, ann_id=str(annotation_id), image_key=str(key), source_index=index, source=path, image_root=image_root, dataset=dataset, split=split, categories={}, default_prompts=default_prompts, prompt_map=prompt_map))
            except ValueError as exc:
                raise ValueError(f"CSV row {index + 2} ({annotation_id}): {exc}") from exc
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):
            raise ValueError("JSON must contain images and annotations arrays")
        images: dict[str, dict[str, Any]] = {}
        for index, image in enumerate(data["images"]):
            if not isinstance(image, dict):
                raise ValueError(f"JSON image {index} must be an object")
            aliases = {str(image[key]) for key in ("uuid", "image_uuid", "id", "image_id") if _present(image.get(key))}
            if not aliases:
                aliases = {f"image_{index}"}
            for alias in aliases:
                if alias in images and images[alias] is not image:
                    raise ValueError(f"Duplicate/ambiguous JSON image reference {alias!r}")
                images[alias] = image
        categories: dict[str, dict[str, Any]] = {}
        for category in data.get("categories", []):
            if not isinstance(category, dict) or not _present(category.get("id")):
                raise ValueError("Each JSON category needs an id")
            key = str(category["id"])
            if key in categories:
                raise ValueError(f"Duplicate JSON category id {key!r}")
            categories[key] = category
        for index, row in enumerate(data["annotations"]):
            if not isinstance(row, dict):
                raise ValueError(f"JSON annotation {index} must be an object")
            annotation_id = _first(row, "annot_uuid", "uuid", "id", "annotation_id")
            if annotation_id is None:
                annotation_id = f"ann_{index}"
            reference = _first(row, "image_uuid", "image_id", "image")
            if reference is None or str(reference) not in images:
                raise ValueError(f"JSON annotation {annotation_id} refers to missing image {reference!r}")
            try:
                records.append(_record(row, image=images[str(reference)], ann_id=str(annotation_id), image_key=str(reference), source_index=index, source=path, image_root=image_root, dataset=dataset, split=split, categories=categories, default_prompts=default_prompts, prompt_map=prompt_map))
            except ValueError as exc:
                raise ValueError(f"JSON annotation {annotation_id}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported metadata format {path.suffix!r}; use .csv or .json")
    ids = Counter(record.ann_id for record in records)
    duplicates = [key for key, count in ids.items() if count > 1]
    if duplicates:
        raise ValueError(f"Annotation IDs must be unique to prevent overwritten masks; duplicates: {duplicates[:5]}")
    image_counts = Counter(record.image_path for record in records)
    for record in records:
        if record.bbox_xywh is None and image_counts[record.image_path] > 1:
            raise ValueError(f"Image {record.image_path} has multiple annotations including {record.ann_id} without a current-image bbox; supply source boxes or distinct prepared crops")
    return records


def prepare_input(record: AnnotationRecord) -> PreparedInput:
    """Load an RGB annotation crop without rotating the metadata coordinate frame."""
    with Image.open(record.image_path) as source:
        original_size = source.size
        image = source.convert("RGB")
    width, height = original_size
    crop = (0, 0, width, height)
    if record.input_kind == "source_box":
        if record.bbox_xywh is None:
            raise ValueError(f"Annotation {record.ann_id} has source_box input without bbox")
        box = _bbox({"bbox": record.bbox_xywh})
        assert box is not None
        x, y, box_width, box_height = box
        crop = (
            min(width, max(0, int(round(x)))),
            min(height, max(0, int(round(y)))),
            min(width, max(0, int(round(x + box_width)))),
            min(height, max(0, int(round(y + box_height)))),
        )
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise ValueError(f"Annotation {record.ann_id} bbox is empty after rounding and clipping to {original_size}")
        image = image.crop(crop)
    elif record.input_kind not in ("native_crop", "prepared_crop", "full_image"):
        raise ValueError(f"Unknown input_kind {record.input_kind!r}")
    elif record.bbox_xywh is not None:
        raise ValueError("Native/prepared input unexpectedly has a current-image bbox")
    return PreparedInput(record=record, image=image, crop_xyxy=crop, original_size=original_size)
