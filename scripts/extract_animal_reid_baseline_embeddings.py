#!/usr/bin/env python3
"""Extract MiewID or MegaDescriptor embeddings in the unified evaluator format."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from extract_dinov3_embeddings import (
    CrossViewJsonDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    collate_batch,
)


DEFAULT_MODEL_TAGS = {
    "miewid": "conservationxlabs/miewid-msv3",
    "megadescriptor": "hf-hub:BVRA/MegaDescriptor-L-384",
}


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
    report = {
        "path": str(checkpoint_path),
        "key_transform": best_name,
        "num_keys": len(best_state_dict),
        "missing_count": len(best_missing),
        "unexpected_count": len(best_unexpected),
        "missing_keys_sample": best_missing[:20],
        "unexpected_keys_sample": best_unexpected[:20],
    }
    print(json.dumps({"checkpoint_load": report}, indent=2))
    return report


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, int, str, dict[str, Any] | None]:
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
    return FeatureModel(model).to(device).eval(), resize_size, model_tag, checkpoint_report


def extract_embeddings(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, resize_size, model_tag, checkpoint_report = load_model(args, device)

    transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
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

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {args.output.name}"):
            images = batch["images"].to(device, non_blocking=True)
            feats = model(images)
            all_embeddings.append(feats.cpu().numpy().astype(np.float32))
            ann_ids.extend(batch["ann_ids"])
            identities.extend(batch["identities"])
            viewpoints.extend(batch["viewpoints"])
            image_paths.extend(batch["image_paths"])

    embeddings = (
        np.concatenate(all_embeddings, axis=0)
        if all_embeddings
        else np.empty((0, 0), dtype=np.float32)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        ann_ids=np.asarray(ann_ids),
        identities=np.asarray(identities),
        viewpoints=np.asarray(viewpoints),
        image_paths=np.asarray(image_paths),
        model=np.asarray(args.model),
        model_tag=np.asarray(model_tag),
        checkpoint_path=np.asarray(str(args.checkpoint_path) if args.checkpoint_path else ""),
        image_variant=np.asarray(args.image_variant),
        mask_pkl=np.asarray(str(args.mask_pkl) if args.mask_pkl else ""),
    )

    summary = {
        "json": str(args.json),
        "image_root": str(args.image_root),
        "output": str(args.output),
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "model": args.model,
        "model_tag": model_tag,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "checkpoint_load": checkpoint_report,
        "crop_mode": args.crop_mode,
        "image_variant": args.image_variant,
        "mask_pkl": str(args.mask_pkl) if args.mask_pkl else None,
        "mask_required": bool(args.mask_required or args.image_variant not in ("unmasked",)),
        "missing_mask_policy": args.missing_mask_policy,
        "records_before_mask_filter": int(dataset.records_before_filter),
        "records_after_mask_filter": int(len(dataset)),
        "skipped_missing_masks": int(dataset.skipped_missing_masks),
        "mask_available_records": int(dataset.mask_available_records),
        "resize_size": int(resize_size),
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
    parser.add_argument("--model", choices=["miewid", "megadescriptor"], required=True)
    parser.add_argument("--model_tag", help="Override Hugging Face/timm model identifier.")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        help="Optional local checkpoint to load after constructing the model architecture.",
    )
    parser.add_argument("--json", required=True, type=Path, help="COCO-like metadata JSON.")
    parser.add_argument("--image_root", required=True, type=Path, help="Root for relative image paths.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz embedding cache.")
    parser.add_argument("--summary_json", type=Path, help="Optional extraction summary JSON.")
    parser.add_argument("--crop_mode", choices=["bbox", "full"], default="bbox")
    parser.add_argument(
        "--image_variant",
        choices=["unmasked", "masked", "foreground", "background", "silhouette"],
        default="unmasked",
        help="Image control to extract. masked is an alias for foreground.",
    )
    parser.add_argument("--mask_pkl", type=Path, help="annotation_masks.pkl keyed by annotation ID.")
    parser.add_argument(
        "--mask_required",
        action="store_true",
        help="Require a valid mask even for --image_variant unmasked.",
    )
    parser.add_argument(
        "--missing_mask_policy",
        choices=["error", "skip"],
        default="error",
        help="How to handle records without masks when masks are required.",
    )
    parser.add_argument("--resize_size", type=int, help="Override model resize size.")
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
