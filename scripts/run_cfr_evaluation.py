#!/usr/bin/env python3
"""Run Counterfactual Foreground Retrieval (CFR) evaluation.

This is the central, release-facing runner for the WACV CFR experiments.  It
reads the central SAM-mask dataset index, extracts/caches embeddings for each
requested model and counterfactual input view, evaluates all-vs-all retrieval on
the mask-valid full dataset, and writes detailed intermediate results plus
paper-ready summary tables.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import tempfile
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN_255 = np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0


# Silhouette fill policy (documented):
#   foreground silhouette = solid black, background = ImageNet-mean neutral fill.
# Black on the neutral background gives a high-contrast, non-textural shape cue
# so that R_sil@k measures shape/crop evidence rather than fill ambiguity.
CFR_SILHOUETTE_FILL_255 = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)


RELEASE_ROOT = Path(__file__).resolve().parents[1]
CENTRAL_INDEX = Path(
    os.environ.get(
        "CFR_CENTRAL_INDEX",
        str(RELEASE_ROOT / "configs" / "cfr_dataset_index.template.tsv"),
    )
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("CFR_OUTPUT_ROOT", str(RELEASE_ROOT / "outputs" / "cfr"))
)
VARIANTS = ("full", "foreground", "background", "silhouette")
MISSING_TEXT_VALUES = {"", "unknown", "nan", "none", "na", "n/a", "null", "-1"}
HEADLINE_DATASETS = (
    "whaleshark",
    "leopard",
    "saimaa_seal",
    "seaturtle_head",
    "stripespotter",
    "hyenaid2022",
    "BrownBearHeads",
    "czechlynx",
    "nyaladata",
)

DATASET_LABELS = {
    "whaleshark": "Whale shark",
    "leopard": "Leopard",
    "saimaa_seal": "Saimaa seal",
    "seaturtle_head": "SeaTurtle head",
    "stripespotter": "StripeSpotter",
    "hyenaid2022": "HyenaID2022",
    "BrownBearHeads": "BrownBearHeads",
    "czechlynx": "CzechLynx",
    "nyaladata": "NyalaData",
    "reunionturtles": "Reunion turtles",
    "amvrakikosturtles": "Amvrakikos turtles",
    "zakynthosturtles": "Zakynthos turtles",
    "melops": "Melops",
}

MODEL_LABELS = {
    "miewid": "MiewID",
    "megadescriptor": "MegaDescriptor",
}

DEFAULT_MODEL_TAGS = {
    "miewid": "conservationxlabs/miewid-msv3",
    "megadescriptor": "hf-hub:BVRA/MegaDescriptor-L-384",
}

DEFAULT_SILHOUETTE_IDENTITY_DATASETS = {
    "belugaid",
    "happywhale",
    "humpbackwhaleid",
    "noaarightwhale",
    "whaleshark",
    "whale shark",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    group: str
    metadata_path: Path
    image_root: Path
    mask_dir: Path
    mask_pkl: Path
    input_type: str
    records_total: int
    annotations_accepted: int
    notes: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str
    model_name: str | None = None
    model_tag: str | None = None
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class CfrRecord:
    ann_id: str
    image_key: str
    image_path: Path
    identity: str
    bbox_xywh: list[float] | None
    raw: dict[str, Any]
    source_index: int
    dataset: str
    encounter: str = ""


class CfrImageDataset(Dataset):
    def __init__(
        self,
        spec: DatasetSpec,
        transform: transforms.Compose,
        image_variant: str,
        crop_mode: str = "bbox",
        limit: int | None = None,
    ) -> None:
        self.spec = spec
        self.transform = transform
        self.image_variant = image_variant
        self.crop_mode = crop_mode
        self.masks = load_mask_pkl(spec.mask_pkl)
        records = load_records(spec)
        self.records_before_filter = len(records)
        self.skipped_unknown_identity = 0
        self.skipped_missing_masks = 0
        kept: list[CfrRecord] = []
        for record in records:
            if not is_valid_identity(record.identity):
                self.skipped_unknown_identity += 1
                continue
            if lookup_mask(record, self.masks) is None:
                self.skipped_missing_masks += 1
                continue
            kept.append(record)
        if limit is not None:
            kept = kept[:limit]
        self.records = kept

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        try:
            image = Image.open(record.image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image {record.image_path}") from exc
        crop = crop_image(image, record.bbox_xywh, self.crop_mode)
        mask = lookup_mask(record, self.masks)
        if mask is None:
            raise KeyError(f"Missing accepted mask for {record.ann_id}")
        mask_bool = mask_for_crop(mask, image, record.bbox_xywh, crop, self.crop_mode)
        image_variant = apply_image_variant(crop, mask_bool, self.image_variant)
        tensor = self.transform(image_variant)
        return {
            "image": tensor,
            "ann_id": record.ann_id,
            "identity": record.identity,
            "image_path": str(record.image_path),
            "encounter": record.encounter,
            "mask_area_frac": float(mask_bool.mean()) if mask_bool.size else 0.0,
        }


class LoadedFeatureModel:
    def __init__(
        self,
        model: nn.Module,
        resize_size: int,
        model_tag: str,
        checkpoint_report: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.resize_size = resize_size
        self.model_tag = model_tag
        self.checkpoint_report = checkpoint_report


class FeatureModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            for key in ("embeddings", "features", "last_hidden_state", "logits"):
                if key in out:
                    out = out[key]
                    break
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim > 2:
            out = out.mean(dim=tuple(range(1, out.ndim - 1)))
        return torch.nn.functional.normalize(out.float(), p=2, dim=1)


def as_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "backbone"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a state dict or checkpoint dict: {type(checkpoint)!r}")
    return checkpoint


def strip_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key[len(prefix) :] if key.startswith(prefix) else key, value)
        for key, value in state_dict.items()
    )


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = as_state_dict(checkpoint)
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = [("raw", state_dict)]
    for prefix in ("module.", "model.", "backbone."):
        if any(key.startswith(prefix) for key in state_dict):
            candidates.append((f"strip:{prefix}", strip_state_dict_prefix(state_dict, prefix)))

    best_name = ""
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_missing: list[str] = []
    best_unexpected: list[str] = []
    best_score: int | None = None
    errors: list[str] = []

    for name, candidate in candidates:
        try:
            result = model.load_state_dict(candidate, strict=False)
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
            continue
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        score = len(missing) + len(unexpected)
        if best_score is None or score < best_score:
            best_name = name
            best_state_dict = candidate
            best_missing = missing
            best_unexpected = unexpected
            best_score = score
        if score == 0:
            break

    if best_state_dict is None:
        details = "\n".join(errors) if errors else "No loadable state-dict candidates."
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}:\n{details}")

    model.load_state_dict(best_state_dict, strict=False)
    return {
        "path": str(checkpoint_path),
        "key_transform": best_name,
        "num_keys": len(best_state_dict),
        "missing_count": len(best_missing),
        "unexpected_count": len(best_unexpected),
        "missing_keys_sample": best_missing[:20],
        "unexpected_keys_sample": best_unexpected[:20],
    }


def load_animal_reid_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, int, str, dict[str, Any] | None]:
    model_tag = args.model_tag or DEFAULT_MODEL_TAGS[args.model]
    checkpoint_report = None
    if args.checkpoint_path and args.model != "megadescriptor":
        raise ValueError("--checkpoint_path is currently supported only for --model megadescriptor")
    if args.model == "miewid":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
        resize_size = args.resize_size or 440
    elif args.model == "megadescriptor":
        import timm

        model = timm.create_model(model_tag, pretrained=args.checkpoint_path is None, num_classes=0)
        if args.checkpoint_path:
            checkpoint_report = load_checkpoint(model, args.checkpoint_path)
        resize_size = args.resize_size or 384
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return FeatureModel(model).to(device).eval(), int(resize_size), model_tag, checkpoint_report


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def clean_dataset_key(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def resolve_existing_path(primary: str, fallback: str = "", *, base_dir: Path | None = None) -> Path:
    candidates = [Path(primary)] if primary else []
    if fallback:
        candidates.append(Path(fallback))
    if base_dir is not None:
        candidates = [candidate if candidate.is_absolute() else base_dir / candidate for candidate in candidates]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else Path("")


def dataset_specs_from_index(index_path: Path) -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    base_dir = index_path.resolve().parent
    for row in read_tsv(index_path):
        dataset = row["dataset"]
        metadata = resolve_existing_path(
            row.get("central_metadata_input", ""),
            row.get("metadata_input", ""),
            base_dir=base_dir,
        )
        image_root = Path(row["raw_image_root"])
        if not image_root.is_absolute():
            image_root = base_dir / image_root
        mask_dir_raw = row.get("central_mask_dir", "") or row.get("mask_dir", "")
        mask_dir = Path(mask_dir_raw)
        if not mask_dir.is_absolute():
            mask_dir = base_dir / mask_dir
        if row.get("annotation_masks_rle"):
            mask_pkl = Path(row["annotation_masks_rle"])
            if not mask_pkl.is_absolute():
                mask_pkl = base_dir / mask_pkl
        else:
            rle_path = mask_dir / "annotation_masks_rle.json.gz"
            mask_pkl = rle_path if rle_path.exists() else mask_dir / "annotation_masks.pkl"
        suffix = metadata.suffix.lower()
        input_type = "csv" if suffix == ".csv" else "json"
        specs[dataset] = DatasetSpec(
            key=dataset,
            label=DATASET_LABELS.get(dataset, dataset),
            group=row.get("group", ""),
            metadata_path=metadata,
            image_root=image_root,
            mask_dir=mask_dir,
            mask_pkl=mask_pkl,
            input_type=input_type,
            records_total=parse_int(row.get("records_total", "0")),
            annotations_accepted=parse_int(row.get("annotations_accepted", "0")),
            notes=row.get("notes", ""),
        )
    return specs


def get_model_specs() -> dict[str, ModelSpec]:
    return {
        "miewid": ModelSpec(
            "miewid",
            MODEL_LABELS["miewid"],
            "animal_reid",
            model_name="miewid",
        ),
        "megadescriptor": ModelSpec(
            "megadescriptor",
            MODEL_LABELS["megadescriptor"],
            "animal_reid",
            model_name="megadescriptor",
        ),
    }


def is_valid_identity(identity: str | None) -> bool:
    if identity is None:
        return False
    text = str(identity).strip()
    return text.lower() not in MISSING_TEXT_VALUES


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def get_identity(row: dict[str, Any]) -> str:
    for key in (
        "identity",
        "name",
        "individual_id",
        "individual",
        "id_individual",
        "name_uuid",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    values = row.get("individual_ids")
    if isinstance(values, list) and values:
        return "|".join(str(v) for v in values)
    return "unknown"


def get_ann_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("annotation_id", "annot_uuid", "uuid", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    value = row.get("image_id")
    if value not in (None, ""):
        return str(value)
    return str(fallback)


def clean_group_value(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in MISSING_TEXT_VALUES else text


def get_first_clean(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row:
            value = clean_group_value(row.get(key))
            if value:
                return value
    return ""


def get_encounter(row: dict[str, Any]) -> str:
    """Best-effort encounter/sighting key from metadata, else "".

    Used only by --exclude_same_encounter.  Explicit encounter, occurrence,
    sighting, video, or track identifiers take precedence.  Full timestamp
    fields are preferred over date-only fields.  If a dataset stores date and
    time separately, combine them into one encounter key.
    """
    explicit = get_first_clean(
        row,
        (
            "encounter",
            "encounter_id",
            "encounter_uuid",
            "occurrence",
            "occurrence_id",
            "sighting",
            "sighting_id",
            "sightingnumb",
            "sightings",
            "video",
            "video_id",
            "track",
            "track_id",
        ),
    )
    if explicit:
        return explicit

    timestamp = get_first_clean(
        row,
        (
            "datetime",
            "timestamp",
            "date_captured",
            "capture_datetime",
        ),
    )
    if timestamp:
        return timestamp

    date = get_first_clean(row, ("date", "date_iso", "capture_date"))
    time = get_first_clean(row, ("time", "time_iso", "capture_time"))
    if date and time:
        return f"{date} {time}"
    return date


def get_image_key_from_ann(row: dict[str, Any]) -> str:
    for key in ("image_id", "image_uuid", "image_path", "path", "file_name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def get_image_key(row: dict[str, Any]) -> str:
    for key in ("id", "uuid", "image_uuid", "image_id", "image_path", "path", "file_name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def image_relpath(row: dict[str, Any]) -> str | None:
    for key in ("image_path", "path", "file_name", "filepath"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def bbox_xywh(row: dict[str, Any]) -> list[float] | None:
    value = row.get("bbox")
    if value not in (None, ""):
        if isinstance(value, str):
            parts = [
                part
                for part in value.strip().strip("[]()").replace(";", ",").split(",")
                if part.strip()
            ]
            if len(parts) >= 4:
                vals = [float(part) for part in parts[:4]]
                return vals if vals[2] > 0 and vals[3] > 0 else None
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            vals = [float(v) for v in value[:4]]
            return vals if vals[2] > 0 and vals[3] > 0 else None
    keys = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    if all(row.get(key) not in (None, "") for key in keys):
        vals = [float(row[key]) for key in keys]
        return vals if vals[2] > 0 and vals[3] > 0 else None
    return None


def resolve_image_path(rel_or_abs: str, image_root: Path, dataset: str | None = None) -> Path:
    raw = Path(str(rel_or_abs))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(image_root / raw)
    if dataset:
        candidates.append(image_root / dataset / raw)
    if len(raw.parts) > 1 and dataset and raw.parts[0] == dataset:
        candidates.append(image_root / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_json_records(spec: DatasetSpec) -> list[CfrRecord]:
    data = load_json(spec.metadata_path)
    images: dict[str, dict[str, Any]] = {}
    for image in data.get("images", []):
        for key in ("id", "uuid", "image_uuid", "image_id", "image_path", "file_name", "path"):
            value = image.get(key)
            if value not in (None, ""):
                images[str(value)] = image
    records: list[CfrRecord] = []
    for idx, ann in enumerate(data.get("annotations", [])):
        img_key = get_image_key_from_ann(ann)
        image = images.get(img_key)
        if image is None:
            continue
        rel = image_relpath(image)
        if not rel:
            continue
        records.append(
            CfrRecord(
                ann_id=get_ann_id(ann, idx),
                image_key=img_key,
                image_path=resolve_image_path(rel, spec.image_root),
                identity=get_identity(ann),
                bbox_xywh=bbox_xywh(ann),
                raw=dict(ann),
                source_index=idx,
                dataset=spec.key,
                encounter=get_encounter(ann) or get_encounter(image),
            )
        )
    return records


def load_csv_records(spec: DatasetSpec) -> list[CfrRecord]:
    records: list[CfrRecord] = []
    with spec.metadata_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            rel = row.get("path") or row.get("image_path") or row.get("file_name")
            if rel in (None, ""):
                continue
            dataset = str(row.get("dataset") or spec.key)
            image_id = str(row.get("image_id") or row.get("id") or idx)
            records.append(
                CfrRecord(
                    ann_id=str(row.get("annotation_id") or row.get("annot_uuid") or image_id),
                    image_key=image_id,
                    image_path=resolve_image_path(str(rel), spec.image_root, dataset),
                    identity=get_identity(row),
                    bbox_xywh=bbox_xywh(row),
                    raw=dict(row),
                    source_index=idx,
                    dataset=dataset,
                    encounter=get_encounter(row),
                )
            )
    return records


def load_records(spec: DatasetSpec) -> list[CfrRecord]:
    if spec.input_type == "csv":
        return load_csv_records(spec)
    return load_json_records(spec)


def load_rle_json(path: Path) -> dict[Any, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        obj = json.load(handle)
    masks = {}
    for record in obj.get("annotations", []):
        ann_id = record["annotation_id"]
        segmentation = record.get("segmentation")
        masks[ann_id] = None if segmentation is None else mask_utils.decode(segmentation).astype(np.uint8)
    return masks


def load_mask_pkl(path: Path) -> dict[Any, Any]:
    if path.name.endswith(".json") or path.name.endswith(".json.gz"):
        return load_rle_json(path)
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Mask PKL must contain a dict, got {type(obj)} from {path}")
    return obj


def candidate_mask_keys(record: CfrRecord) -> list[Any]:
    candidates: list[Any] = [record.ann_id, str(record.ann_id)]
    for key in ("annotation_id", "annot_uuid", "uuid", "id", "image_id"):
        value = record.raw.get(key)
        if value not in (None, ""):
            candidates.extend([value, str(value)])
            if isinstance(value, str) and value.isdigit():
                candidates.append(int(value))
    deduped = []
    seen = set()
    for key in candidates:
        marker = (type(key).__name__, str(key))
        if marker not in seen:
            seen.add(marker)
            deduped.append(key)
    return deduped


def lookup_mask(record: CfrRecord, masks: dict[Any, Any]) -> np.ndarray | None:
    for key in candidate_mask_keys(record):
        if key in masks:
            value = masks[key]
            if value is None:
                return None
            return np.asarray(value)
    return None


def crop_box(
    image: Image.Image,
    bbox: list[float] | None,
    crop_mode: str,
) -> tuple[int, int, int, int]:
    if crop_mode == "full" or bbox is None:
        return (0, 0, image.width, image.height)
    x, y, w, h = bbox
    x1 = max(0, int(round(float(x))))
    y1 = max(0, int(round(float(y))))
    x2 = min(image.width, int(round(float(x + w))))
    y2 = min(image.height, int(round(float(y + h))))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, image.width, image.height)
    return (x1, y1, x2, y2)


def crop_image(image: Image.Image, bbox: list[float] | None, crop_mode: str) -> Image.Image:
    return image.crop(crop_box(image, bbox, crop_mode))


def mask_for_crop(
    mask: np.ndarray,
    image: Image.Image,
    bbox: list[float] | None,
    crop: Image.Image,
    crop_mode: str,
) -> np.ndarray:
    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim > 2:
        mask_bool = np.squeeze(mask_bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {mask_bool.shape}")
    if mask_bool.shape == (image.height, image.width):
        x1, y1, x2, y2 = crop_box(image, bbox, crop_mode)
        mask_bool = mask_bool[y1:y2, x1:x2]
    if mask_bool.shape != (crop.height, crop.width):
        mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
        resample = getattr(Image, "Resampling", Image).NEAREST
        mask_img = mask_img.resize((crop.width, crop.height), resample)
        mask_bool = np.asarray(mask_img) > 127
    return mask_bool


def apply_image_variant(crop: Image.Image, mask_bool: np.ndarray, image_variant: str) -> Image.Image:
    if image_variant == "full":
        return crop
    img_np = np.asarray(crop).astype(np.float32)
    neutral = np.zeros_like(img_np, dtype=np.float32)
    neutral[:, :] = IMAGENET_MEAN_255
    mask_3d = mask_bool[:, :, None]
    if image_variant == "foreground":
        out = np.where(mask_3d, img_np, neutral)
    elif image_variant == "background":
        out = np.where(mask_3d, neutral, img_np)
    elif image_variant == "silhouette":
        fg = np.zeros_like(img_np, dtype=np.float32)
        fg[:, :] = CFR_SILHOUETTE_FILL_255
        out = np.where(mask_3d, fg, neutral)
    else:
        raise ValueError(f"Unsupported image variant: {image_variant}")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "ann_ids": [item["ann_id"] for item in batch],
        "identities": [item["identity"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "encounters": [item["encounter"] for item in batch],
        "mask_area_frac": [item["mask_area_frac"] for item in batch],
    }


def load_feature_model(
    spec: ModelSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> LoadedFeatureModel:
    if spec.kind == "animal_reid":
        ns = argparse.Namespace(
            model=spec.model_name,
            model_tag=spec.model_tag,
            checkpoint_path=spec.checkpoint_path,
            resize_size=args.resize_size,
        )
        model, resize_size, model_tag, checkpoint_report = load_animal_reid_model(ns, device)
        return LoadedFeatureModel(
            model=model,
            resize_size=int(resize_size),
            model_tag=model_tag,
            checkpoint_report=checkpoint_report,
        )

    raise ValueError(f"Unsupported model kind: {spec.kind}")


def cache_tag(model_spec: ModelSpec, args: argparse.Namespace) -> str:
    """Config fingerprint embedded in cached embedding filenames.

    Encodes every knob that changes the produced embeddings but is NOT already
    captured by (dataset, variant, model) so that a different ``--crop_mode``
    or ``--resize_size`` never silently reuses a stale ``.npz``.
    """
    rs = args.resize_size if getattr(args, "resize_size", None) else "default"
    return "_".join([f"crop-{args.crop_mode}", f"rs-{rs}"])


def embedding_path(
    output_root: Path, dataset_key: str, model_key: str, variant: str, tag: str
) -> Path:
    return (
        output_root
        / "embeddings"
        / model_key
        / f"{dataset_key}_{variant}_{model_key}_{tag}.npz"
    )


def extract_variant_embeddings(
    dataset_spec: DatasetSpec,
    model_spec: ModelSpec,
    loaded: LoadedFeatureModel,
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    tag = cache_tag(model_spec, args)
    out_path = embedding_path(args.output_root, dataset_spec.key, model_spec.key, variant, tag)
    summary_path = (
        args.output_root
        / "extract_summaries"
        / model_spec.key
        / f"{dataset_spec.key}_{variant}_{model_spec.key}_{tag}_extract_summary.json"
    )
    if out_path.exists() and not args.force_extract:
        return {"output": str(out_path), "skipped_existing": True}

    transform = transforms.Compose(
        [
            transforms.Resize((loaded.resize_size, loaded.resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = CfrImageDataset(
        dataset_spec,
        transform=transform,
        image_variant=variant,
        crop_mode=args.crop_mode,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
        collate_fn=collate_batch,
    )

    all_embeddings: list[np.ndarray] = []
    ann_ids: list[str] = []
    identities: list[str] = []
    image_paths: list[str] = []
    encounters: list[str] = []
    mask_area_frac: list[float] = []
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"Extract {dataset_spec.key}/{model_spec.key}/{variant}",
        ):
            images = batch["images"].to(device, non_blocking=True)
            feats = loaded.model(images)
            feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
            all_embeddings.append(feats.detach().cpu().numpy().astype(np.float32))
            ann_ids.extend(batch["ann_ids"])
            identities.extend(batch["identities"])
            image_paths.extend(batch["image_paths"])
            encounters.extend(batch["encounters"])
            mask_area_frac.extend(float(x) for x in batch["mask_area_frac"])

    embeddings = (
        np.concatenate(all_embeddings, axis=0)
        if all_embeddings
        else np.empty((0, 0), dtype=np.float32)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        ann_ids=np.asarray(ann_ids),
        identities=np.asarray(identities),
        image_paths=np.asarray(image_paths),
        encounters=np.asarray(encounters),
        mask_area_frac=np.asarray(mask_area_frac, dtype=np.float32),
        dataset=np.asarray(dataset_spec.key),
        model=np.asarray(model_spec.key),
        model_label=np.asarray(model_spec.label),
        model_tag=np.asarray(loaded.model_tag),
        checkpoint_path=np.asarray(str(model_spec.checkpoint_path) if model_spec.checkpoint_path else ""),
        variant=np.asarray(variant),
        metadata_path=np.asarray(str(dataset_spec.metadata_path)),
        image_root=np.asarray(str(dataset_spec.image_root)),
        mask_pkl=np.asarray(str(dataset_spec.mask_pkl)),
    )

    summary = {
        "dataset": asdict(dataset_spec),
        "model": asdict(model_spec),
        "variant": variant,
        "output": str(out_path),
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "records_before_filter": int(dataset.records_before_filter),
        "records_after_mask_identity_filter": int(len(dataset)),
        "skipped_unknown_identity": int(dataset.skipped_unknown_identity),
        "skipped_missing_masks": int(dataset.skipped_missing_masks),
        "mean_mask_area_frac": float(np.mean(mask_area_frac)) if mask_area_frac else None,
        "resize_size": int(loaded.resize_size),
        "device": str(device),
        "model_tag": loaded.model_tag,
        "checkpoint_report": loaded.checkpoint_report,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, default=json_default)
    return summary


def load_embedding_npz(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    return {key: np.asarray(npz[key]) for key in npz.files}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def random_topk_probability(gallery_size: int, positives: int, k: int) -> float:
    if gallery_size <= 0 or positives <= 0:
        return 0.0
    k_eff = min(k, gallery_size)
    negatives = gallery_size - positives
    if negatives < k_eff:
        return 1.0
    prob_no_positive = 1.0
    for t in range(k_eff):
        prob_no_positive *= (negatives - t) / (gallery_size - t)
    return float(1.0 - prob_no_positive)


def topk_search_numpy(features: np.ndarray, max_k: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = features.shape[0]
    k_eff = min(max_k, max(0, n - 1))
    if k_eff <= 0:
        return (
            np.empty((n, 0), dtype=np.int64),
            np.empty((n, 0), dtype=np.float32),
        )
    all_indices = np.empty((n, k_eff), dtype=np.int64)
    all_scores = np.empty((n, k_eff), dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc="Top-k retrieval"):
        end = min(n, start + batch_size)
        sims = features[start:end] @ features.T
        row_idx = np.arange(end - start)
        sims[row_idx, np.arange(start, end)] = -np.inf
        part = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        sorted_idx = np.take_along_axis(part, order, axis=1)
        sorted_scores = np.take_along_axis(part_scores, order, axis=1)
        all_indices[start:end] = sorted_idx
        all_scores[start:end] = sorted_scores
    return all_indices, all_scores


def topk_search_faiss(features: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    n, dim = features.shape
    k_eff = min(max_k + 1, n)
    index = faiss.IndexFlatIP(dim)
    index.add(features.astype(np.float32))
    scores, indices = index.search(features.astype(np.float32), k_eff)
    out_indices: list[list[int]] = []
    out_scores: list[list[float]] = []
    final_k = min(max_k, max(0, n - 1))
    for i in range(n):
        row_idx: list[int] = []
        row_scores: list[float] = []
        for idx, score in zip(indices[i].tolist(), scores[i].tolist()):
            if idx == i or idx < 0:
                continue
            row_idx.append(int(idx))
            row_scores.append(float(score))
            if len(row_idx) == final_k:
                break
        out_indices.append(row_idx)
        out_scores.append(row_scores)
    return np.asarray(out_indices, dtype=np.int64), np.asarray(out_scores, dtype=np.float32)


def topk_search(
    features: np.ndarray,
    max_k: int,
    batch_size: int,
    use_faiss: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    if use_faiss:
        result = topk_search_faiss(features, max_k)
        if result is not None:
            indices, scores = result
            return indices, scores, "faiss"
    indices, scores = topk_search_numpy(features, max_k, batch_size)
    return indices, scores, "numpy_chunked"


def validate_variant_alignment(tables: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_variant = next(iter(tables))
    base = tables[base_variant]
    ann_ids = np.asarray(base["ann_ids"]).astype(str)
    identities = np.asarray(base["identities"]).astype(str)
    image_paths = np.asarray(base["image_paths"]).astype(str)
    for variant, table in tables.items():
        if not np.array_equal(ann_ids, np.asarray(table["ann_ids"]).astype(str)):
            raise ValueError(f"Annotation order mismatch for variant {variant}")
        if not np.array_equal(identities, np.asarray(table["identities"]).astype(str)):
            raise ValueError(f"Identity order mismatch for variant {variant}")
    return ann_ids, identities, image_paths


def factorize_labels(values: Iterable[Any], drop_empty: bool = False) -> np.ndarray:
    """Integer-code labels; ``drop_empty`` maps empty/unknown values to -1.

    A code of -1 never matches any other code, so empty group keys (e.g. a
    missing encounter id) are treated as "no group" for exclusion purposes.
    """
    values = list(values)
    codes = np.full(len(values), -1, dtype=np.int64)
    mapping: dict[str, int] = {}
    nxt = 0
    for i, v in enumerate(values):
        s = str(v).strip()
        if drop_empty and s.lower() in MISSING_TEXT_VALUES:
            continue
        if s not in mapping:
            mapping[s] = nxt
            nxt += 1
        codes[i] = mapping[s]
    return codes


def chunk_exclusion_mask(
    start: int, end: int, n: int, group_codes_list: list[np.ndarray]
) -> np.ndarray:
    """Boolean (end-start, n) mask: True where row i and col j share a valid
    group code in any of the provided code arrays.  Self is NOT handled here."""
    mask = np.zeros((end - start, n), dtype=bool)
    for codes in group_codes_list:
        col = codes[start:end][:, None]
        mask |= (col >= 0) & (col == codes[None, :])
    return mask


def compute_gallery_stats(
    n: int,
    id_codes: np.ndarray,
    group_codes_list: list[np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-query effective gallery size and positive count after self- and
    group-exclusion (same-image / same-encounter)."""
    gallery_size = np.empty(n, dtype=np.int64)
    positives = np.empty(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        rows = np.arange(end - start)
        excl = chunk_exclusion_mask(start, end, n, group_codes_list)
        excl[rows, np.arange(start, end)] = True  # exclude self
        same_id = id_codes[start:end][:, None] == id_codes[None, :]
        gallery_size[start:end] = n - excl.sum(axis=1)
        positives[start:end] = (same_id & ~excl).sum(axis=1)
    return gallery_size, positives


def topk_search_numpy_excl(
    features: np.ndarray,
    max_k: int,
    batch_size: int,
    group_codes_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k retrieval excluding self and any same-group gallery items.

    Excluded slots receive a ``-inf`` score; callers must drop non-finite
    scores before using the corresponding indices."""
    n = features.shape[0]
    k_eff = min(max_k, max(0, n - 1))
    if k_eff <= 0:
        return (
            np.empty((n, 0), dtype=np.int64),
            np.empty((n, 0), dtype=np.float32),
        )
    all_indices = np.empty((n, k_eff), dtype=np.int64)
    all_scores = np.full((n, k_eff), -np.inf, dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc="Top-k retrieval (excl)"):
        end = min(n, start + batch_size)
        rows = np.arange(end - start)
        sims = features[start:end] @ features.T
        sims[rows, np.arange(start, end)] = -np.inf
        if group_codes_list:
            sims[chunk_exclusion_mask(start, end, n, group_codes_list)] = -np.inf
        part = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        all_indices[start:end] = np.take_along_axis(part, order, axis=1)
        all_scores[start:end] = np.take_along_axis(part_scores, order, axis=1)
    return all_indices, all_scores


def evaluate_cfr(
    dataset_spec: DatasetSpec,
    model_spec: ModelSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tag = cache_tag(model_spec, args)
    tables = {
        variant: load_embedding_npz(
            embedding_path(args.output_root, dataset_spec.key, model_spec.key, variant, tag)
        )
        for variant in args.variants
    }
    ann_ids, identities, image_paths = validate_variant_alignment(tables)
    n = len(ann_ids)
    base_table = tables[next(iter(tables))]
    encounters = np.asarray(base_table.get("encounters", np.full(n, ""))).astype(str)
    if encounters.shape[0] != n:
        encounters = np.full(n, "")

    exclude_same_image = bool(getattr(args, "exclude_same_image", False))
    exclude_same_encounter = bool(getattr(args, "exclude_same_encounter", False))

    id_codes = factorize_labels(identities)
    group_codes_list: list[np.ndarray] = []
    if exclude_same_image:
        group_codes_list.append(factorize_labels(image_paths, drop_empty=True))
    if exclude_same_encounter:
        enc_codes = factorize_labels(encounters, drop_empty=True)
        if not np.any(enc_codes >= 0):
            raise SystemExit(
                "--exclude_same_encounter requested but dataset "
                f"'{dataset_spec.key}' has no usable encounter/date/track metadata."
            )
        group_codes_list.append(enc_codes)

    id_counts = Counter(identities.tolist())
    gallery_size_arr, positives_arr = compute_gallery_stats(
        n, id_codes, group_codes_list, args.retrieval_batch_size
    )
    valid_query = (positives_arr > 0) & (gallery_size_arr > 0)
    valid_indices = np.flatnonzero(valid_query)
    max_k = max(args.ks)

    random_by_k = {
        k: float(
            np.mean(
                [
                    random_topk_probability(
                        int(gallery_size_arr[i]), int(positives_arr[i]), k
                    )
                    for i in valid_indices
                ]
            )
            if len(valid_indices)
            else 0.0
        )
        for k in args.ks
    }
    per_query: dict[int, dict[str, Any]] = {}
    for i in valid_indices.tolist():
        gallery_size = int(gallery_size_arr[i])
        positives = int(positives_arr[i])
        per_query[i] = {
            "dataset": dataset_spec.key,
            "model": model_spec.key,
            "query_index": i,
            "query_ann_id": ann_ids[i],
            "identity": identities[i],
            "image_path": image_paths[i],
            "encounter": encounters[i],
            "gallery_size": gallery_size,
            "positive_count": positives,
            **{f"random_p_at_{k}": random_topk_probability(gallery_size, positives, k) for k in args.ks},
        }

    use_exclusion = bool(group_codes_list)
    variant_metrics: dict[str, Any] = {}
    retrieval_backend: dict[str, str] = {}
    for variant in args.variants:
        features = l2_normalize(np.asarray(tables[variant]["embeddings"], dtype=np.float32))
        if use_exclusion:
            top_indices, top_scores = topk_search_numpy_excl(
                features,
                max_k=max_k,
                batch_size=args.retrieval_batch_size,
                group_codes_list=group_codes_list,
            )
            backend = "numpy_chunked_excl"
        else:
            top_indices, top_scores, backend = topk_search(
                features,
                max_k=max_k,
                batch_size=args.retrieval_batch_size,
                use_faiss=not args.no_faiss,
            )
        retrieval_backend[variant] = backend
        hits_by_k = {k: [] for k in args.ks}
        for i in valid_indices.tolist():
            raw_idxs = top_indices[i].tolist()
            raw_scores = top_scores[i].tolist()
            # Drop excluded padding (non-finite scores) so neighbours reflect
            # the true effective gallery ranking under the validity rule.
            idxs = [j for j, s in zip(raw_idxs, raw_scores) if math.isfinite(s)]
            scores = [s for s in raw_scores if math.isfinite(s)]
            top_identities = [identities[j] for j in idxs]
            positive_flags = [identity == identities[i] for identity in top_identities]
            for k in args.ks:
                hits_by_k[k].append(bool(any(positive_flags[: min(k, len(positive_flags))])))
            prefix = f"{variant}_"
            row = per_query[i]
            row[prefix + "top_ann_ids"] = json.dumps([ann_ids[j] for j in idxs], default=json_default)
            row[prefix + "top_identities"] = json.dumps([str(x) for x in top_identities])
            row[prefix + "top_scores"] = json.dumps([float(x) for x in scores])
            row[prefix + "top1_ann_id"] = ann_ids[idxs[0]] if idxs else ""
            row[prefix + "top1_identity"] = str(top_identities[0]) if top_identities else ""
            row[prefix + "top1_score"] = float(scores[0]) if scores else None
            for k in args.ks:
                row[prefix + f"hit_at_{k}"] = bool(any(positive_flags[: min(k, len(positive_flags))]))

        variant_metrics[variant] = {
            "valid_queries": int(len(valid_indices)),
            "num_embeddings": int(n),
            "r_at_k": {
                str(k): float(np.mean(hits_by_k[k])) if hits_by_k[k] else 0.0
                for k in args.ks
            },
        }

    cfr_by_k: dict[str, Any] = {}
    for k in args.ks:
        random_rate = random_by_k[k]
        r = {
            variant: float(variant_metrics[variant]["r_at_k"][str(k)])
            for variant in args.variants
        }
        signals = {
            variant: max(0.0, value - random_rate)
            for variant, value in r.items()
        }
        silhouette_identity = {
            clean_dataset_key(value)
            for value in getattr(args, "silhouette_identity_datasets", [])
        }
        dataset_is_silhouette_identity = (
            clean_dataset_key(dataset_spec.key) in silhouette_identity
            or clean_dataset_key(dataset_spec.label) in silhouette_identity
        )
        effective_shortcut_views = [
            view
            for view in args.shortcut_views
            if not (view == "silhouette" and dataset_is_silhouette_identity)
        ]
        shortcut_candidates = {
            variant: signals[variant]
            for variant in effective_shortcut_views
            if variant in signals
        }
        shortcut_view = (
            max(shortcut_candidates, key=shortcut_candidates.get)
            if shortcut_candidates
            else None
        )
        shortcut_signal = shortcut_candidates[shortcut_view] if shortcut_view else 0.0
        fg_signal = signals.get("foreground", 0.0)
        full_signal = signals.get("full", 0.0)
        headroom = max(0.0, 1.0 - random_rate)
        cfr = min(full_signal, fg_signal) / headroom if headroom > 0 else None
        ssi = min(1.0, shortcut_signal / full_signal) if full_signal > 0 else None
        cfr_by_k[str(k)] = {
            "random": random_rate,
            "retrieval": r,
            "signals_above_chance": signals,
            "shortcut_views": list(args.shortcut_views),
            "effective_shortcut_views": effective_shortcut_views,
            "shortcut_view": shortcut_view,
            "shortcut_signal": shortcut_signal,
            "foreground_signal": fg_signal,
            "full_signal": full_signal,
            "headroom": headroom,
            "ssi": ssi,
            "cfr": cfr,
            "defined": cfr is not None,
        }

    result = {
        "dataset": asdict(dataset_spec),
        "model": asdict(model_spec),
        "variants": list(args.variants),
        "ks": list(args.ks),
        "counts": {
            "num_embeddings": int(n),
            "identities": int(len(id_counts)),
            "valid_queries": int(len(valid_indices)),
            "queries_without_positive": int(n - len(valid_indices)),
        },
        "validity": {
            "exclude_same_image": exclude_same_image,
            "exclude_same_encounter": exclude_same_encounter,
            "mean_gallery_size": float(np.mean(gallery_size_arr[valid_indices]))
            if len(valid_indices)
            else 0.0,
            "mean_positive_count": float(np.mean(positives_arr[valid_indices]))
            if len(valid_indices)
            else 0.0,
        },
        "random_by_k": {str(k): random_by_k[k] for k in args.ks},
        "variant_metrics": variant_metrics,
        "cfr_by_k": cfr_by_k,
        "retrieval_backend": retrieval_backend,
    }

    result_dir = args.output_root / "results" / model_spec.key
    result_dir.mkdir(parents=True, exist_ok=True)
    result_json = result_dir / f"{dataset_spec.key}_{model_spec.key}_cfr.json"
    with result_json.open("w") as handle:
        json.dump(result, handle, indent=2, default=json_default)

    per_query_path = result_dir / f"{dataset_spec.key}_{model_spec.key}_cfr_per_query.csv"
    write_dict_rows(list(per_query.values()), per_query_path)
    return result


def write_dict_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result_for_k(result: dict[str, Any], k: int) -> dict[str, Any]:
    cfr = result["cfr_by_k"][str(k)]
    retrieval = cfr["retrieval"]
    signals = cfr["signals_above_chance"]
    dataset = result["dataset"]
    model = result["model"]
    row = {
        "dataset": dataset["key"],
        "dataset_label": dataset["label"],
        "model": model["key"],
        "model_label": model["label"],
        "k": k,
        "num_embeddings": result["counts"]["num_embeddings"],
        "identities": result["counts"]["identities"],
        "valid_queries": result["counts"]["valid_queries"],
        "random": cfr["random"],
        "full": retrieval.get("full"),
        "foreground": retrieval.get("foreground"),
        "background": retrieval.get("background"),
        "silhouette": retrieval.get("silhouette"),
        "s_full": signals.get("full"),
        "s_foreground": signals.get("foreground"),
        "s_background": signals.get("background"),
        "s_silhouette": signals.get("silhouette"),
        "shortcut_view": cfr["shortcut_view"],
        "shortcut_signal": cfr["shortcut_signal"],
        "headroom": cfr.get("headroom"),
        "ssi": cfr.get("ssi"),
        "cfr": cfr["cfr"],
        "cfr_defined": cfr["defined"],
    }
    return row


def format_score(value: Any) -> str:
    if value is None:
        return "--"
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "--"
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def write_latex_table(rows: list[dict[str, Any]], path: Path, caption: str, label: str) -> None:
    order = []
    seen = set()
    for row in rows:
        key = row["dataset"]
        if key not in seen:
            seen.add(key)
            order.append(key)
    by_dataset = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in order}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Rand.} & \textbf{Full} & \textbf{Fg} & \textbf{Bg} & \textbf{Sil} & \textbf{SSI} & \textbf{CFR} \\",
        r"\midrule",
    ]
    for d_idx, dataset in enumerate(order):
        ds_rows = by_dataset[dataset]
        for r_idx, row in enumerate(ds_rows):
            ds_label = row["dataset_label"] if r_idx == 0 else ""
            lines.append(
                f"{ds_label} & {row['model_label']} & "
                f"{format_score(row['random'])} & {format_score(row['full'])} & "
                f"{format_score(row['foreground'])} & {format_score(row['background'])} & "
                f"{format_score(row['silhouette'])} & {format_score(row.get('ssi'))} & "
                f"{format_score(row['cfr'])} \\\\"
            )
        if d_idx != len(order) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def select_datasets(args: argparse.Namespace, specs: dict[str, DatasetSpec]) -> list[DatasetSpec]:
    if args.datasets:
        keys = args.datasets
    elif args.suite == "headline":
        keys = list(HEADLINE_DATASETS)
    elif args.suite == "all":
        keys = [key for key, spec in specs.items() if spec.annotations_accepted > 0]
    else:
        raise SystemExit("--datasets is required when --suite custom")
    missing = [key for key in keys if key not in specs]
    if missing:
        raise SystemExit(f"Datasets missing from central index: {missing}")
    return [specs[key] for key in keys]


def select_models(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> list[ModelSpec]:
    if args.models:
        keys = args.models
    else:
        keys = ["miewid", "megadescriptor"]
    missing = [key for key in keys if key not in specs]
    if missing:
        raise SystemExit(f"Unknown models: {missing}")
    return [specs[key] for key in keys]


def ensure_inputs_exist(datasets: Iterable[DatasetSpec]) -> None:
    problems = []
    for spec in datasets:
        if not spec.metadata_path.exists():
            problems.append(f"{spec.key}: missing metadata {spec.metadata_path}")
        if not spec.image_root.exists():
            problems.append(f"{spec.key}: missing image root {spec.image_root}")
        if not spec.mask_pkl.exists():
            problems.append(f"{spec.key}: missing mask file {spec.mask_pkl}")
    if problems:
        raise SystemExit("Input validation failed:\n" + "\n".join(problems))


def write_toy_embedding(
    output_root: Path,
    dataset_key: str,
    model_key: str,
    variant: str,
    embeddings: np.ndarray,
    tag: str,
    encounters: list[str] | None = None,
) -> None:
    out_path = embedding_path(output_root, dataset_key, model_key, variant, tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        ann_ids=np.asarray(["a0", "a1", "b0", "b1"]),
        identities=np.asarray(["A", "A", "B", "B"]),
        image_paths=np.asarray(["a0.jpg", "a1.jpg", "b0.jpg", "b1.jpg"]),
        encounters=np.asarray(encounters if encounters is not None else ["", "", "", ""]),
        mask_area_frac=np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        dataset=np.asarray(dataset_key),
        model=np.asarray(model_key),
        variant=np.asarray(variant),
    )


def run_self_test() -> None:
    assert math.isclose(random_topk_probability(3, 1, 1), 1.0 / 3.0)
    assert math.isclose(random_topk_probability(3, 1, 2), 2.0 / 3.0)
    assert math.isclose(random_topk_probability(3, 1, 3), 1.0)
    assert math.isclose(random_topk_probability(4, 2, 2), 5.0 / 6.0)
    assert get_encounter({"datetime": "NA", "date": "2020-01-01"}) == "2020-01-01"
    assert get_encounter({"encounter_id": "e7", "datetime": "2020-01-01 12:00:00"}) == "e7"
    assert get_encounter({"date": "2020-01-01", "time": "12:34:56"}) == "2020-01-01 12:34:56"

    with tempfile.TemporaryDirectory(prefix="cfr_self_test_") as tmp:
        output_root = Path(tmp)
        dataset = DatasetSpec(
            key="toy",
            label="Toy",
            group="self_test",
            metadata_path=output_root / "toy.json",
            image_root=output_root,
            mask_dir=output_root,
            mask_pkl=output_root / "annotation_masks.pkl",
            input_type="json",
            records_total=4,
            annotations_accepted=4,
            notes="",
        )
        model = ModelSpec("toy_model", "ToyModel", "toy")
        args = argparse.Namespace(
            output_root=output_root,
            variants=list(VARIANTS),
            ks=[1, 2, 3],
            shortcut_views=["background", "silhouette"],
            retrieval_batch_size=2,
            no_faiss=True,
            crop_mode="bbox",
            resize_size=None,
            exclude_same_image=False,
            exclude_same_encounter=False,
        )
        tag = cache_tag(model, args)

        foreground = np.asarray(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [-1.0, 0.0],
                [-0.95, -0.05],
            ],
            dtype=np.float32,
        )
        shortcut = np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        write_toy_embedding(output_root, "toy", "toy_model", "full", foreground, tag)
        write_toy_embedding(output_root, "toy", "toy_model", "foreground", foreground, tag)
        write_toy_embedding(output_root, "toy", "toy_model", "background", shortcut, tag)
        write_toy_embedding(output_root, "toy", "toy_model", "silhouette", shortcut, tag)

        result = evaluate_cfr(dataset, model, args)
        cfr1 = result["cfr_by_k"]["1"]
        assert result["counts"]["valid_queries"] == 4
        assert math.isclose(cfr1["random"], 1.0 / 3.0, rel_tol=1e-7)
        assert math.isclose(cfr1["retrieval"]["foreground"], 1.0, rel_tol=1e-7)
        assert math.isclose(cfr1["retrieval"]["full"], 1.0, rel_tol=1e-7)
        assert math.isclose(cfr1["retrieval"]["background"], 0.0, abs_tol=1e-7)
        assert math.isclose(cfr1["retrieval"]["silhouette"], 0.0, abs_tol=1e-7)
        assert math.isclose(cfr1["foreground_signal"], 2.0 / 3.0, rel_tol=1e-7)
        assert math.isclose(cfr1["full_signal"], 2.0 / 3.0, rel_tol=1e-7)
        assert math.isclose(cfr1["shortcut_signal"], 0.0, abs_tol=1e-7)
        assert math.isclose(cfr1["ssi"], 0.0, abs_tol=1e-7)
        assert math.isclose(cfr1["cfr"], 1.0, rel_tol=1e-7)

        # Same-image exclusion: with each identity's two annotations sharing an
        # image, every query loses its only positive -> no valid queries.
        excl_args = argparse.Namespace(**vars(args))
        excl_args.exclude_same_image = True
        for variant in VARIANTS:
            write_toy_embedding(
                output_root,
                "toy",
                "toy_model",
                variant,
                foreground if variant in ("full", "foreground") else shortcut,
                tag,
                encounters=["e0", "e0", "e1", "e1"],
            )
        # Override image_paths so a0/a1 share an image and b0/b1 share an image.
        for variant in VARIANTS:
            path = embedding_path(output_root, "toy", "toy_model", variant, tag)
            data = dict(load_embedding_npz(path))
            data["image_paths"] = np.asarray(["img_a.jpg", "img_a.jpg", "img_b.jpg", "img_b.jpg"])
            np.savez_compressed(path, **data)
        excl_result = evaluate_cfr(dataset, model, excl_args)
        assert excl_result["counts"]["valid_queries"] == 0
        assert excl_result["validity"]["exclude_same_image"] is True

    print("CFR self-test passed.", flush=True)


def run(args: argparse.Namespace) -> None:
    if args.self_test:
        run_self_test()
        return

    args.output_root = Path(args.output_root)
    dataset_specs = dataset_specs_from_index(Path(args.central_index))
    model_specs = get_model_specs()
    datasets = select_datasets(args, dataset_specs)
    models = select_models(args, model_specs)
    ensure_inputs_exist(datasets)

    if args.dry_run:
        payload = {
            "suite": args.suite,
            "datasets": [asdict(spec) for spec in datasets],
            "models": [asdict(spec) for spec in models],
            "variants": list(args.variants),
            "ks": list(args.ks),
            "shortcut_views": list(args.shortcut_views),
            "exclude_same_image": bool(args.exclude_same_image),
            "exclude_same_encounter": bool(args.exclude_same_encounter),
            "output_root": str(args.output_root),
        }
        print(json.dumps(payload, indent=2, default=json_default))
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    all_results: list[dict[str, Any]] = []
    for model_spec in models:
        print(f"Loading model {model_spec.key} on {device}", flush=True)
        loaded = load_feature_model(model_spec, args, device)
        for dataset_spec in datasets:
            print(f"=== CFR {dataset_spec.key} / {model_spec.key} ===", flush=True)
            if args.mode in {"extract", "extract_eval"}:
                for variant in args.variants:
                    extract_variant_embeddings(dataset_spec, model_spec, loaded, variant, args, device)
            if args.mode in {"eval", "extract_eval"}:
                result = evaluate_cfr(dataset_spec, model_spec, args)
                all_results.append(result)
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.mode in {"eval", "extract_eval"}:
        summary_rows = []
        for result in all_results:
            for k in args.ks:
                summary_rows.append(flatten_result_for_k(result, k))
        summary_csv = args.output_root / f"cfr_summary_{args.suite}.csv"
        write_dict_rows(summary_rows, summary_csv)
        for k in args.ks:
            rows_k = [row for row in summary_rows if row["k"] == k]
            write_dict_rows(rows_k, args.output_root / f"cfr_summary_{args.suite}_k{k}.csv")
            write_latex_table(
                rows_k,
                args.output_root / f"cfr_table_{args.suite}_k{k}.tex",
                caption=(
                    f"Counterfactual Foreground Retrieval components at rank {k}. "
                    "SSI is shortcut recoverability relative to full-image signal; "
                    "CFR is foreground-supported retrieval."
                ),
                label=f"tab:cfr-{args.suite}-k{k}",
            )
        print(f"Wrote summaries under {args.output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central_index", type=Path, default=CENTRAL_INDEX)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--suite", choices=["headline", "all", "custom"], default="headline")
    parser.add_argument("--datasets", nargs="*", help="Dataset keys from the central index.")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["miewid", "megadescriptor"],
    )
    parser.add_argument("--mode", choices=["extract", "eval", "extract_eval"], default="extract_eval")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--shortcut_views", nargs="+", default=["background", "silhouette"])
    parser.add_argument(
        "--silhouette_identity_datasets",
        nargs="*",
        default=sorted(DEFAULT_SILHOUETTE_IDENTITY_DATASETS),
        help=(
            "Datasets for which silhouette is treated as valid identity evidence "
            "rather than shortcut evidence."
        ),
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--crop_mode", choices=["bbox", "full"], default="bbox")
    parser.add_argument(
        "--exclude_same_image",
        action="store_true",
        help="Exclude gallery annotations sharing the query's image path.",
    )
    parser.add_argument(
        "--exclude_same_encounter",
        action="store_true",
        help="Exclude gallery annotations sharing the query's encounter/date/track key.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--retrieval_batch_size", type=int, default=1024)
    parser.add_argument("--device", help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--resize_size", type=int, help="Override model resize size.")
    parser.add_argument("--limit", type=int, help="Debug limit on mask-valid records.")
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--no_faiss", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--self_test", action="store_true", help="Run a synthetic CFR sanity test and exit.")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
