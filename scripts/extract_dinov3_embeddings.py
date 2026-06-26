#!/usr/bin/env python3
"""Extract DINOv3 zero-shot embeddings for cross-view animal ReID JSONs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cross_view_utils import (
    annotation_records,
    get_annotation_key,
    get_image_key,
    image_lookup,
    load_json,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN_255 = np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
SILHOUETTE_COLOR_255 = np.asarray([128.0, 128.0, 128.0], dtype=np.float32)


def _load_custom_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_dinov3_backbone(
    semantic_matching_root: str,
    config_yaml: str,
    device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Load the same DINOv3 backbone used by the training scripts."""
    sm_root = Path(semantic_matching_root)
    if str(sm_root) not in sys.path:
        sys.path.insert(0, str(sm_root))

    _load_custom_module(
        "src.features.dinov3_loader",
        sm_root / "src" / "features" / "dinov3_loader.py",
    )
    dino_mod = _load_custom_module(
        "src.features.dino_extractor",
        sm_root / "src" / "features" / "dino_features_ankit_updated.py",
    )
    from src.config.config import MainConfig

    config = MainConfig.from_yaml(config_yaml)
    if hasattr(config.dino, "device"):
        config.dino.device = str(device)
    try:
        extractor = dino_mod.DINOv3Extractor(config.dino)
        model = extractor.model
        embed_dim = int(extractor.get_feature_dim())
    except ModuleNotFoundError as exc:
        if exc.name != "torchmetrics":
            raise
        print(
            "[WARN] torchmetrics is missing, so torch.hub hubconf import failed. "
            "Falling back to direct DINOv3 backbone import from the local torch hub cache."
        )
        model = load_dinov3_backbone_direct(
            model_name=str(config.dino.model_name),
            weights_file=Path(config.dino.weights_file_path),
            device=device,
        )
        embed_dim = {
            "dinov3_vits16": 384,
            "dinov3_vits16plus": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
            "dinov3_vitl16plus": 1024,
            "dinov3_vith16": 1280,
            "dinov3_vith16plus": 1280,
        }.get(str(config.dino.model_name), 1024)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    resize_size = int(config.dino.resize_size)

    with torch.no_grad():
        dummy = torch.randn(1, 3, resize_size, resize_size, device=device)
        out = model.forward_features(dummy)
        probed_dim = out["x_norm_clstoken"].shape[-1]
        if probed_dim != embed_dim:
            raise RuntimeError(
                f"DINO dimension mismatch: extractor={embed_dim}, probed={probed_dim}"
            )

    return model, embed_dim, resize_size


def load_dinov3_backbone_direct(
    model_name: str,
    weights_file: Path,
    device: torch.device,
) -> nn.Module:
    """Load DINOv3 without importing hubconf optional eval dependencies."""
    hub_root = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if not hub_root.exists():
        raise FileNotFoundError(
            f"Local DINOv3 hub cache not found at {hub_root}. "
            "Run the existing Semantic-Matching DINO loader once or install torchmetrics."
        )
    if str(hub_root) not in sys.path:
        sys.path.insert(0, str(hub_root))

    from dinov3.hub import backbones
    from src.features.dinov3_loader import parse_dinov3_weights_file

    fn = getattr(backbones, model_name)
    weight_arg: str | None = None

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    matches = sorted(checkpoint_dir.glob(f"{model_name}_pretrain_*.pth"))
    if matches:
        weight_arg = str(matches[0])
    else:
        weight_urls = parse_dinov3_weights_file(weights_file)
        if model_name in weight_urls:
            weight_arg = weight_urls[model_name]
        else:
            candidates = [key for key in weight_urls if key.startswith(model_name)]
            if not candidates:
                raise KeyError(f"No DINOv3 weight URL found for {model_name}")
            weight_arg = weight_urls[candidates[0]]

    model = fn(pretrained=True, weights=weight_arg)
    model.to(device)
    model.eval()
    return model


class DinoFeatureWrapper(nn.Module):
    """Return CLS or mean patch-token embeddings from a DINO-style ViT."""

    def __init__(self, model: nn.Module, pooling: str = "cls"):
        super().__init__()
        self.model = model
        self.pooling = pooling

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "forward_features"):
            out = self.model(x)
            if isinstance(out, dict):
                out = out.get("x_norm_clstoken", out.get("cls_token"))
            return out

        out = self.model.forward_features(x)
        if self.pooling == "cls":
            return out["x_norm_clstoken"]
        if self.pooling == "patch_mean":
            patches = out.get("x_norm_patchtokens")
            if patches is None:
                patches = out.get("patch_tokens")
            if patches is None:
                raise KeyError(
                    "Model forward_features did not return x_norm_patchtokens "
                    "or patch_tokens for --pooling patch_mean."
                )
            return patches.mean(dim=1)
        raise ValueError(f"Unsupported pooling: {self.pooling}")


def _image_path_from_record(
    record: dict[str, Any],
    images_by_key: dict[str, dict[str, Any]],
    image_root: Path,
) -> Path:
    image = images_by_key.get(record["image_key"])
    if image is None:
        raise KeyError(f"Could not find image metadata for key {record['image_key']}")

    rel = image.get("image_path") or image.get("file_name") or image.get("path")
    if rel in (None, ""):
        raise KeyError(f"Image metadata has no image_path/file_name/path: {image}")
    rel_path = Path(str(rel))
    if rel_path.is_absolute():
        return rel_path
    return image_root / rel_path


def _bbox_from_ann(ann: dict[str, Any]) -> list[float] | None:
    bbox = ann.get("bbox")
    if bbox is None:
        # Support converted WildlifeDatasets-style bbox columns.
        if all(k in ann for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
            bbox = [ann["bbox_x"], ann["bbox_y"], ann["bbox_w"], ann["bbox_h"]]
    if bbox is None:
        return None
    return [float(v) for v in bbox]


def _crop_box(img: Image.Image, ann: dict[str, Any], crop_mode: str) -> tuple[int, int, int, int] | None:
    if crop_mode == "full":
        return (0, 0, img.width, img.height)
    bbox = _bbox_from_ann(ann)
    if bbox is None:
        return None
    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img.width, int(round(x + w)))
    y2 = min(img.height, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _crop_image(img: Image.Image, ann: dict[str, Any], crop_mode: str) -> Image.Image:
    box = _crop_box(img, ann, crop_mode)
    if box is None:
        return img
    return img.crop(box)


def _candidate_mask_keys(record: dict[str, Any]) -> list[Any]:
    ann = record["ann"]
    candidates: list[Any] = [record["ann_key"]]
    for key in ("id", "uuid", "annot_uuid"):
        value = ann.get(key)
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


def _load_mask_pkl(path: Path | None) -> dict[Any, Any]:
    if path is None:
        return {}
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Mask PKL must contain a dict, got {type(obj)} from {path}")
    return obj


def _lookup_mask(record: dict[str, Any], masks: dict[Any, Any]) -> np.ndarray | None:
    for key in _candidate_mask_keys(record):
        if key in masks:
            value = masks[key]
            if value is None:
                return None
            return np.asarray(value)
    return None


def _mask_for_crop(
    mask: np.ndarray,
    img: Image.Image,
    ann: dict[str, Any],
    crop: Image.Image,
    crop_mode: str,
) -> np.ndarray:
    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim > 2:
        mask_bool = np.squeeze(mask_bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask_bool.shape}")

    # Some stored masks are full-image masks; most existing annotation_masks.pkl
    # files store bbox-cropped masks. Support both without changing pixels.
    if mask_bool.shape == (img.height, img.width):
        box = _crop_box(img, ann, crop_mode)
        if box is not None:
            x1, y1, x2, y2 = box
            mask_bool = mask_bool[y1:y2, x1:x2]

    if mask_bool.shape != (crop.height, crop.width):
        mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
        resample = getattr(Image, "Resampling", Image).NEAREST
        mask_img = mask_img.resize((crop.width, crop.height), resample)
        mask_bool = np.asarray(mask_img) > 127
    return mask_bool


def _apply_image_variant(crop: Image.Image, mask_bool: np.ndarray | None, image_variant: str) -> Image.Image:
    if image_variant == "unmasked":
        return crop
    if image_variant == "masked":
        image_variant = "foreground"
    if mask_bool is None:
        raise ValueError(f"Image variant '{image_variant}' requires a foreground mask")

    img_np = np.asarray(crop).astype(np.float32)
    bg = np.zeros_like(img_np, dtype=np.float32)
    bg[:, :] = IMAGENET_MEAN_255
    mask_3d = mask_bool[:, :, None]

    if image_variant == "foreground":
        out = np.where(mask_3d, img_np, bg)
    elif image_variant == "background":
        out = np.where(mask_3d, bg, img_np)
    elif image_variant == "silhouette":
        fg = np.zeros_like(img_np, dtype=np.float32)
        fg[:, :] = SILHOUETTE_COLOR_255
        out = np.where(mask_3d, fg, bg)
    else:
        raise ValueError(f"Unsupported image variant: {image_variant}")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class CrossViewJsonDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        image_root: Path,
        transform,
        crop_mode: str = "bbox",
        image_variant: str = "unmasked",
        mask_pkl: Path | None = None,
        mask_required: bool = False,
        missing_mask_policy: str = "error",
        limit: int | None = None,
    ):
        self.json_path = json_path
        self.image_root = image_root
        self.transform = transform
        self.crop_mode = crop_mode
        self.image_variant = "foreground" if image_variant == "masked" else image_variant
        self.mask_required = mask_required or self.image_variant != "unmasked"
        self.missing_mask_policy = missing_mask_policy
        self.masks = _load_mask_pkl(mask_pkl)
        self.mask_pkl = mask_pkl
        self.data = load_json(json_path)
        self.images_by_key = image_lookup(self.data)
        records = annotation_records(self.data)
        self.records_before_filter = len(records)
        self.skipped_missing_masks = 0
        self.mask_available_records = 0
        if self.mask_required:
            kept = []
            for record in records:
                if _lookup_mask(record, self.masks) is None:
                    self.skipped_missing_masks += 1
                    if self.missing_mask_policy == "error":
                        raise KeyError(
                            f"Missing mask for annotation {record['ann_key']} in {mask_pkl}. "
                            "Use --missing_mask_policy skip for paired mask-valid evaluation."
                        )
                    if self.missing_mask_policy == "skip":
                        continue
                    raise ValueError(f"Unsupported missing mask policy: {self.missing_mask_policy}")
                self.mask_available_records += 1
                kept.append(record)
            records = kept
        if limit is not None:
            records = records[:limit]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        ann = record["ann"]
        path = _image_path_from_record(record, self.images_by_key, self.image_root)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image {path}") from exc
        crop = _crop_image(image, ann, self.crop_mode)
        mask_bool = None
        mask_area_frac = np.nan
        if self.mask_required or self.masks:
            mask = _lookup_mask(record, self.masks)
            if mask is not None:
                mask_bool = _mask_for_crop(mask, image, ann, crop, self.crop_mode)
                mask_area_frac = float(mask_bool.mean())
        image_variant = _apply_image_variant(crop, mask_bool, self.image_variant)
        tensor = self.transform(image_variant)
        return {
            "image": tensor,
            "ann_id": record["ann_key"],
            "identity": record["identity"],
            "viewpoint": record["viewpoint"],
            "image_path": str(path),
            "mask_available": mask_bool is not None,
            "mask_area_frac": mask_area_frac,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "ann_ids": [item["ann_id"] for item in batch],
        "identities": [item["identity"] for item in batch],
        "viewpoints": [item["viewpoint"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "mask_available": [item["mask_available"] for item in batch],
        "mask_area_frac": [item["mask_area_frac"] for item in batch],
    }


def extract_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_model, embed_dim, resize_size = load_dinov3_backbone(
        args.sm_root, args.sm_config, device
    )
    model = DinoFeatureWrapper(raw_model, pooling=args.pooling).to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.resize_size or resize_size, args.resize_size or resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = CrossViewJsonDataset(
        args.json,
        args.image_root,
        transform,
        crop_mode=args.crop_mode,
        image_variant=args.image_variant,
        mask_pkl=args.mask_pkl,
        mask_required=args.mask_required,
        missing_mask_policy=args.missing_mask_policy,
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
    viewpoints: list[str] = []
    image_paths: list[str] = []
    mask_available: list[bool] = []
    mask_area_frac: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {args.output.name}"):
            images = batch["images"].to(device, non_blocking=True)
            feats = model(images)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))
            ann_ids.extend(batch["ann_ids"])
            identities.extend(batch["identities"])
            viewpoints.extend(batch["viewpoints"])
            image_paths.extend(batch["image_paths"])
            mask_available.extend(batch["mask_available"])
            mask_area_frac.extend(batch["mask_area_frac"])

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, embed_dim), dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        ann_ids=np.asarray(ann_ids),
        identities=np.asarray(identities),
        viewpoints=np.asarray(viewpoints),
        image_paths=np.asarray(image_paths),
        mask_available=np.asarray(mask_available, dtype=bool),
        mask_area_frac=np.asarray(mask_area_frac, dtype=np.float32),
        image_variant=np.asarray(args.image_variant),
        mask_pkl=np.asarray(str(args.mask_pkl) if args.mask_pkl else ""),
    )

    summary = {
        "json": str(args.json),
        "image_root": str(args.image_root),
        "output": str(args.output),
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "pooling": args.pooling,
        "crop_mode": args.crop_mode,
        "image_variant": args.image_variant,
        "mask_pkl": str(args.mask_pkl) if args.mask_pkl else None,
        "mask_required": bool(args.mask_required or args.image_variant not in ("unmasked",)),
        "missing_mask_policy": args.missing_mask_policy,
        "records_before_mask_filter": int(dataset.records_before_filter),
        "records_after_mask_filter": int(len(dataset)),
        "skipped_missing_masks": int(dataset.skipped_missing_masks),
        "mask_available_records": int(dataset.mask_available_records),
        "mean_mask_area_frac": (
            float(np.nanmean(mask_area_frac)) if len(mask_area_frac) else None
        ),
        "resize_size": int(args.resize_size or resize_size),
        "backbone_embed_dim": embed_dim,
        "device": str(device),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path, help="COCO-like metadata JSON.")
    parser.add_argument("--image_root", required=True, type=Path, help="Root for relative image paths.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz embedding cache.")
    parser.add_argument("--summary_json", type=Path, help="Optional extraction summary JSON.")
    parser.add_argument("--sm_root", default=os.environ.get("SEMANTIC_MATCHING_ROOT", ""))
    parser.add_argument("--sm_config", default=os.environ.get("DINO_CONFIG", ""))
    parser.add_argument("--pooling", choices=["cls", "patch_mean"], default="cls")
    parser.add_argument("--crop_mode", choices=["bbox", "full"], default="bbox")
    parser.add_argument(
        "--image_variant",
        choices=["unmasked", "masked", "foreground", "background", "silhouette"],
        default="unmasked",
        help=(
            "Image control to extract. masked is an alias for foreground. "
            "foreground/background/silhouette require --mask_pkl."
        ),
    )
    parser.add_argument("--mask_pkl", type=Path, help="annotation_masks.pkl keyed by annotation ID.")
    parser.add_argument(
        "--mask_required",
        action="store_true",
        help=(
            "Require a valid mask even for --image_variant unmasked. Use this to "
            "extract a paired unmasked subset directly comparable to masked controls."
        ),
    )
    parser.add_argument(
        "--missing_mask_policy",
        choices=["error", "skip"],
        default="error",
        help="How to handle records without masks when masks are required.",
    )
    parser.add_argument("--resize_size", type=int, help="Override DINO config resize size.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--limit", type=int, help="Debug limit on number of annotations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_embeddings(args)


if __name__ == "__main__":
    main()
