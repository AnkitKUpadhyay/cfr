#!/usr/bin/env python3
"""Evaluate CFR/SSI sensitivity to mask errors and query-image corruption.

The runner deliberately reuses ``run_cfr_evaluation.py`` for dataset parsing,
accepted-mask loading, annotation crops, CFR view rendering, model loading, and
the final CFR/SSI definitions.  It changes only the mask or query crop specified
by an explicit perturbation condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import run_cfr_evaluation as base


MASK_PERTURBATIONS = {
    "mask_erode",
    "mask_dilate",
    "mask_interior_false_negative",
    "mask_exterior_false_positive",
}
IMAGE_PERTURBATIONS = {
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
    "foreground_cutout",
    "background_cutout",
}
PERTURBATIONS = ["clean", *sorted(MASK_PERTURBATIONS), *sorted(IMAGE_PERTURBATIONS)]

SEVERITY_VALUES = {
    "mask_erode": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "mask_dilate": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "mask_interior_false_negative": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "mask_exterior_false_positive": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "gaussian_noise": {"low": 0.02, "medium": 0.05, "high": 0.10},
    "gaussian_blur": {"low": 1.0, "medium": 2.0, "high": 4.0},
    "jpeg_compression": {"low": 80.0, "medium": 50.0, "high": 20.0},
    "foreground_cutout": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "background_cutout": {"low": 0.05, "medium": 0.10, "high": 0.20},
}


@dataclass(frozen=True)
class Perturbation:
    name: str
    family: str
    level: str
    value: float
    seed: int
    scope: str

    @property
    def tag(self) -> str:
        value = f"{self.value:g}".replace(".", "p")
        return f"{self.name}__{self.level}__v-{value}__seed-{self.seed}__scope-{self.scope}"


def build_perturbation(args: argparse.Namespace) -> Perturbation:
    if args.perturbation == "clean":
        return Perturbation("clean", "clean", "clean", 0.0, 0, "both")
    family = "mask" if args.perturbation in MASK_PERTURBATIONS else "image"
    value = (
        float(args.severity_value)
        if args.severity_value is not None
        else SEVERITY_VALUES[args.perturbation][args.severity]
    )
    scope = args.scope
    if scope == "auto":
        # Mask errors model an imperfect released annotation layer and therefore
        # affect query and gallery views. Image degradation models a field query
        # against the same clean reference gallery used in Table S3.
        scope = "both" if family == "mask" else "query"
    return Perturbation(args.perturbation, family, args.severity, value, args.seed, scope)


def stable_rng(condition: Perturbation, dataset: str, ann_id: str) -> np.random.Generator:
    payload = f"{condition.tag}|{dataset}|{ann_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def choose_smallest(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(max(int(count), 0), int(scores.size))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == scores.size:
        return np.arange(scores.size, dtype=np.int64)
    return np.argpartition(scores, count - 1)[:count]


def compact_subset(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Select an exact-size compact region from a Boolean candidate map."""
    coords = np.argwhere(candidates)
    count = min(max(int(count), 0), len(coords))
    selected = np.zeros(candidates.shape, dtype=bool)
    if count == 0:
        return selected
    center = coords[int(rng.integers(0, len(coords)))]
    distances = np.sum((coords.astype(np.int64) - center.astype(np.int64)) ** 2, axis=1)
    local = choose_smallest(distances, count)
    chosen = coords[local]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def perturb_mask(
    mask: np.ndarray,
    condition: Perturbation,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    original = np.asarray(mask, dtype=bool)
    area = int(original.sum())
    if condition.family != "mask" or area == 0:
        return original.copy(), {
            "mask_original_area": float(area),
            "mask_perturbed_area": float(area),
            "mask_changed_pixels": 0.0,
            "mask_changed_fraction": 0.0,
        }

    target = max(1, int(round(area * condition.value)))
    perturbed = original.copy()
    if condition.name == "mask_erode":
        target = min(target, max(0, area - 1))
        inside = np.flatnonzero(original)
        distances = distance_transform_edt(original).ravel()[inside]
        chosen = inside[choose_smallest(distances, target)]
        perturbed.ravel()[chosen] = False
    elif condition.name == "mask_dilate":
        outside = np.flatnonzero(~original)
        target = min(target, len(outside))
        distances = distance_transform_edt(~original).ravel()[outside]
        chosen = outside[choose_smallest(distances, target)]
        perturbed.ravel()[chosen] = True
    elif condition.name == "mask_interior_false_negative":
        target = min(target, max(0, area - 1))
        removed = compact_subset(original, target, rng)
        perturbed[removed] = False
    elif condition.name == "mask_exterior_false_positive":
        outside = ~original
        target = min(target, int(outside.sum()))
        distance_from_animal = distance_transform_edt(outside)
        minimum_gap = max(2.0, 0.02 * math.sqrt(area))
        separated = outside & (distance_from_animal >= minimum_gap)
        candidates = separated if int(separated.sum()) >= target else outside
        added = compact_subset(candidates, target, rng)
        perturbed[added] = True
    else:
        raise ValueError(f"Unsupported mask perturbation: {condition.name}")

    changed = int(np.count_nonzero(perturbed != original))
    return perturbed, {
        "mask_original_area": float(area),
        "mask_perturbed_area": float(perturbed.sum()),
        "mask_changed_pixels": float(changed),
        "mask_changed_fraction": float(changed / area) if area else 0.0,
    }


def corrupt_image(
    crop: Image.Image,
    mask: np.ndarray,
    condition: Perturbation,
    rng: np.random.Generator,
) -> tuple[Image.Image, dict[str, float]]:
    if condition.family != "image":
        return crop, {"image_changed_pixels": 0.0, "image_changed_fraction": 0.0}

    if condition.name == "gaussian_blur":
        return crop.filter(ImageFilter.GaussianBlur(radius=condition.value)), {
            "image_changed_pixels": float("nan"),
            "image_changed_fraction": float("nan"),
        }
    if condition.name == "jpeg_compression":
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=int(round(condition.value)), subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB"), {
                "image_changed_pixels": float("nan"),
                "image_changed_fraction": float("nan"),
            }

    image = np.asarray(crop).astype(np.float32)
    if condition.name == "gaussian_noise":
        noise = rng.normal(0.0, condition.value * 255.0, size=image.shape)
        output = np.clip(image + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(output), {
            "image_changed_pixels": float("nan"),
            "image_changed_fraction": float("nan"),
        }

    mask_area = int(mask.sum())
    target = max(1, int(round(mask_area * condition.value))) if mask_area else 0
    if condition.name == "foreground_cutout":
        candidates = mask
    elif condition.name == "background_cutout":
        candidates = ~mask
    else:
        raise ValueError(f"Unsupported image perturbation: {condition.name}")
    occluded = compact_subset(candidates, target, rng)
    output = image.copy()
    output[occluded] = base.IMAGENET_MEAN_255
    changed = int(occluded.sum())
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), {
        "image_changed_pixels": float(changed),
        "image_changed_fraction": float(changed / mask_area) if mask_area else 0.0,
    }


class RobustnessImageDataset(Dataset):
    def __init__(
        self,
        spec: base.DatasetSpec,
        transform: transforms.Compose,
        variants: list[str],
        condition: Perturbation,
        crop_mode: str = "bbox",
        limit: int | None = None,
    ) -> None:
        self.spec = spec
        self.transform = transform
        self.variants = variants
        self.condition = condition
        self.crop_mode = crop_mode
        self.masks = base.load_mask_pkl(spec.mask_pkl)
        records = base.load_records(spec)
        self.records_before_filter = len(records)
        self.skipped_unknown_identity = 0
        self.skipped_missing_masks = 0
        kept: list[base.CfrRecord] = []
        for record in records:
            if not base.is_valid_identity(record.identity):
                self.skipped_unknown_identity += 1
                continue
            if base.lookup_mask(record, self.masks) is None:
                self.skipped_missing_masks += 1
                continue
            kept.append(record)
        self.records = kept[:limit] if limit is not None else kept

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        try:
            image = Image.open(record.image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image {record.image_path}") from exc
        crop = base.crop_image(image, record.bbox_xywh, self.crop_mode)
        stored_mask = base.lookup_mask(record, self.masks)
        if stored_mask is None:
            raise KeyError(f"Missing accepted mask for {record.ann_id}")
        mask = base.mask_for_crop(
            stored_mask, image, record.bbox_xywh, crop, self.crop_mode
        )
        rng = stable_rng(self.condition, self.spec.key, record.ann_id)
        view_mask, mask_stats = perturb_mask(mask, self.condition, rng)
        corrupted_crop, image_stats = corrupt_image(crop, mask, self.condition, rng)
        rendered = {
            variant: self.transform(
                base.apply_image_variant(corrupted_crop, view_mask, variant)
            )
            for variant in self.variants
        }
        return {
            "images": rendered,
            "ann_id": record.ann_id,
            "identity": record.identity,
            "image_path": str(record.image_path),
            "encounter": record.encounter,
            "mask_area_frac": float(mask.mean()) if mask.size else 0.0,
            **mask_stats,
            **image_stats,
        }


STAT_KEYS = (
    "mask_area_frac",
    "mask_original_area",
    "mask_perturbed_area",
    "mask_changed_pixels",
    "mask_changed_fraction",
    "image_changed_pixels",
    "image_changed_fraction",
)


def collate_robustness_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    variants = list(batch[0]["images"])
    return {
        "images": {
            variant: torch.stack([item["images"][variant] for item in batch])
            for variant in variants
        },
        "ann_ids": [item["ann_id"] for item in batch],
        "identities": [item["identity"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "encounters": [item["encounter"] for item in batch],
        **{key: [float(item[key]) for item in batch] for key in STAT_KEYS},
    }


def condition_root(output_root: Path, condition: Perturbation) -> Path:
    return output_root / "conditions" / condition.tag


def condition_embedding_path(
    output_root: Path,
    condition: Perturbation,
    dataset_key: str,
    model_key: str,
    variant: str,
    base_tag: str,
) -> Path:
    return (
        condition_root(output_root, condition)
        / "embeddings"
        / model_key
        / f"{dataset_key}_{variant}_{model_key}_{base_tag}_{condition.tag}.npz"
    )


def variants_requiring_extraction(condition: Perturbation, variants: list[str]) -> list[str]:
    if condition.family == "clean":
        return []
    if condition.family == "mask":
        return [variant for variant in variants if variant != "full"]
    return list(variants)


def finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def extract_condition_embeddings(
    dataset_spec: base.DatasetSpec,
    model_spec: base.ModelSpec,
    loaded: base.LoadedFeatureModel,
    condition: Perturbation,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    base_tag = base.cache_tag(model_spec, args)
    requested = variants_requiring_extraction(condition, list(args.variants))
    pending = [
        variant
        for variant in requested
        if args.force_extract
        or not condition_embedding_path(
            args.output_root, condition, dataset_spec.key, model_spec.key, variant, base_tag
        ).exists()
    ]
    if not pending:
        return {"skipped_existing": True, "variants": requested}

    transform = transforms.Compose(
        [
            transforms.Resize((loaded.resize_size, loaded.resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(base.IMAGENET_MEAN, base.IMAGENET_STD),
        ]
    )
    dataset = RobustnessImageDataset(
        dataset_spec,
        transform,
        pending,
        condition,
        crop_mode=args.crop_mode,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and device.type == "cuda",
        collate_fn=collate_robustness_batch,
    )

    embeddings: dict[str, list[np.ndarray]] = {variant: [] for variant in pending}
    metadata = {key: [] for key in ("ann_ids", "identities", "image_paths", "encounters")}
    stats = {key: [] for key in STAT_KEYS}
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"Robustness {condition.tag}/{dataset_spec.key}/{model_spec.key}",
        ):
            for variant in pending:
                images = batch["images"][variant].to(device, non_blocking=True)
                features = loaded.model(images)
                features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
                embeddings[variant].append(
                    features.detach().cpu().numpy().astype(np.float32)
                )
            for key in metadata:
                metadata[key].extend(batch[key])
            for key in stats:
                stats[key].extend(float(value) for value in batch[key])

    for variant in pending:
        array = (
            np.concatenate(embeddings[variant], axis=0)
            if embeddings[variant]
            else np.empty((0, 0), dtype=np.float32)
        )
        out_path = condition_embedding_path(
            args.output_root, condition, dataset_spec.key, model_spec.key, variant, base_tag
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            embeddings=array,
            ann_ids=np.asarray(metadata["ann_ids"]),
            identities=np.asarray(metadata["identities"]),
            image_paths=np.asarray(metadata["image_paths"]),
            encounters=np.asarray(metadata["encounters"]),
            mask_area_frac=np.asarray(stats["mask_area_frac"], dtype=np.float32),
            dataset=np.asarray(dataset_spec.key),
            model=np.asarray(model_spec.key),
            model_label=np.asarray(model_spec.label),
            model_tag=np.asarray(loaded.model_tag),
            variant=np.asarray(variant),
            perturbation=np.asarray(condition.name),
            perturbation_level=np.asarray(condition.level),
            perturbation_value=np.asarray(condition.value),
            perturbation_seed=np.asarray(condition.seed),
            perturbation_scope=np.asarray(condition.scope),
        )

    summary = {
        "dataset": asdict(dataset_spec),
        "model": asdict(model_spec),
        "condition": asdict(condition),
        "condition_tag": condition.tag,
        "variants": pending,
        "num_embeddings": len(metadata["ann_ids"]),
        "records_before_filter": dataset.records_before_filter,
        "records_after_mask_identity_filter": len(dataset),
        "skipped_unknown_identity": dataset.skipped_unknown_identity,
        "skipped_missing_masks": dataset.skipped_missing_masks,
        "mean_mask_changed_fraction": finite_mean(stats["mask_changed_fraction"]),
        "mean_image_changed_fraction": finite_mean(stats["image_changed_fraction"]),
        "resize_size": loaded.resize_size,
        "device": str(device),
        "model_tag": loaded.model_tag,
    }
    summary_path = (
        condition_root(args.output_root, condition)
        / "extract_summaries"
        / model_spec.key
        / f"{dataset_spec.key}_{model_spec.key}_{condition.tag}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=base.json_default))
    return summary


def slice_table(table: dict[str, np.ndarray], limit: int | None) -> dict[str, np.ndarray]:
    if limit is None:
        return table
    output: dict[str, np.ndarray] = {}
    n = len(np.asarray(table["ann_ids"]))
    for key, value in table.items():
        array = np.asarray(value)
        output[key] = array[:limit] if array.ndim and array.shape[0] == n else array
    return output


def load_clean_table(
    args: argparse.Namespace,
    dataset_spec: base.DatasetSpec,
    model_spec: base.ModelSpec,
    variant: str,
) -> dict[str, np.ndarray]:
    path = base.embedding_path(
        args.clean_embedding_root,
        dataset_spec.key,
        model_spec.key,
        variant,
        base.cache_tag(model_spec, args),
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing Table S3 clean embedding: {path}")
    return slice_table(base.load_embedding_npz(path), args.limit)


def load_condition_table(
    args: argparse.Namespace,
    condition: Perturbation,
    dataset_spec: base.DatasetSpec,
    model_spec: base.ModelSpec,
    variant: str,
) -> dict[str, np.ndarray]:
    path = condition_embedding_path(
        args.output_root,
        condition,
        dataset_spec.key,
        model_spec.key,
        variant,
        base.cache_tag(model_spec, args),
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing condition embedding: {path}")
    return base.load_embedding_npz(path)


def verify_clean_embedding_parity(
    args: argparse.Namespace,
    condition: Perturbation,
    dataset_spec: base.DatasetSpec,
    model_spec: base.ModelSpec,
) -> None:
    if condition.family == "clean" or condition.value != 0:
        raise ValueError(
            "--verify_clean_embedding_parity requires a non-clean condition "
            "with --severity_value 0."
        )
    for variant in variants_requiring_extraction(condition, list(args.variants)):
        candidate = load_condition_table(
            args, condition, dataset_spec, model_spec, variant
        )
        reference = load_clean_table(args, dataset_spec, model_spec, variant)
        if not np.array_equal(candidate["ann_ids"], reference["ann_ids"]):
            raise AssertionError(f"Annotation mismatch for zero-strength {variant}")
        candidate_features = np.asarray(candidate["embeddings"], dtype=np.float32)
        reference_features = np.asarray(reference["embeddings"], dtype=np.float32)
        max_abs = float(np.max(np.abs(candidate_features - reference_features)))
        cosine = np.sum(candidate_features * reference_features, axis=1) / (
            np.linalg.norm(candidate_features, axis=1)
            * np.linalg.norm(reference_features, axis=1)
        )
        min_cosine = float(np.min(cosine))
        if max_abs >= 1e-4 or min_cosine <= 0.99999:
            raise AssertionError(
                f"Zero-strength embedding parity failed for {variant}: "
                f"max_abs={max_abs}, min_cosine={min_cosine}"
            )
        print(
            f"Zero-strength embedding parity passed for {variant}: "
            f"max_abs={max_abs:.3g}, min_cosine={min_cosine:.8f}",
            flush=True,
        )


def query_gallery_tables(
    args: argparse.Namespace,
    condition: Perturbation,
    dataset_spec: base.DatasetSpec,
    model_spec: base.ModelSpec,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    query: dict[str, dict[str, np.ndarray]] = {}
    gallery: dict[str, dict[str, np.ndarray]] = {}
    for variant in args.variants:
        uses_clean = condition.family == "clean" or (
            condition.family == "mask" and variant == "full"
        )
        query[variant] = (
            load_clean_table(args, dataset_spec, model_spec, variant)
            if uses_clean
            else load_condition_table(args, condition, dataset_spec, model_spec, variant)
        )
        gallery[variant] = (
            query[variant]
            if condition.scope == "both" and not uses_clean
            else load_clean_table(args, dataset_spec, model_spec, variant)
        )
    return query, gallery


def validate_query_gallery_alignment(
    query: dict[str, dict[str, np.ndarray]],
    gallery: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_ann, q_ids, q_paths = base.validate_variant_alignment(query)
    g_ann, g_ids, g_paths = base.validate_variant_alignment(gallery)
    if not np.array_equal(q_ann, g_ann):
        raise ValueError("Query/gallery annotation order mismatch")
    if not np.array_equal(q_ids, g_ids):
        raise ValueError("Query/gallery identity order mismatch")
    return q_ann, q_ids, q_paths, g_paths


def topk_cross_numpy(
    queries: np.ndarray,
    gallery: np.ndarray,
    max_k: int,
    batch_size: int,
    group_codes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n = queries.shape[0]
    k_eff = min(max_k, max(0, gallery.shape[0] - 1))
    indices = np.empty((n, k_eff), dtype=np.int64)
    scores = np.full((n, k_eff), -np.inf, dtype=np.float32)
    for start in tqdm(range(0, n, batch_size), desc="Cross-view retrieval"):
        end = min(n, start + batch_size)
        sims = queries[start:end] @ gallery.T
        rows = np.arange(end - start)
        sims[rows, np.arange(start, end)] = -np.inf
        if group_codes:
            sims[base.chunk_exclusion_mask(start, end, n, group_codes)] = -np.inf
        part = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        indices[start:end] = np.take_along_axis(part, order, axis=1)
        scores[start:end] = np.take_along_axis(part_scores, order, axis=1)
    return indices, scores


def topk_cross_faiss(
    queries: np.ndarray, gallery: np.ndarray, max_k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    n, dim = gallery.shape
    final_k = min(max_k, max(0, n - 1))
    index = faiss.IndexFlatIP(dim)
    index.add(gallery.astype(np.float32))
    raw_scores, raw_indices = index.search(
        queries.astype(np.float32), min(n, final_k + 1)
    )
    indices = np.empty((len(queries), final_k), dtype=np.int64)
    scores = np.empty((len(queries), final_k), dtype=np.float32)
    for i in range(len(queries)):
        kept = [
            (idx, score)
            for idx, score in zip(raw_indices[i].tolist(), raw_scores[i].tolist())
            if idx >= 0 and idx != i
        ][:final_k]
        indices[i] = [idx for idx, _ in kept]
        scores[i] = [score for _, score in kept]
    return indices, scores


def topk_cross(
    queries: np.ndarray,
    gallery: np.ndarray,
    max_k: int,
    batch_size: int,
    group_codes: list[np.ndarray],
    use_faiss: bool,
    require_faiss: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    if use_faiss and not group_codes:
        result = topk_cross_faiss(queries, gallery, max_k)
        if result is not None:
            return *result, "faiss_cross"
        if require_faiss:
            raise RuntimeError(
                "FAISS is required for Table S3 parity, but it is unavailable. "
                "Install faiss-cpu/faiss-gpu or omit --require_faiss only for debugging."
            )
    return *topk_cross_numpy(queries, gallery, max_k, batch_size, group_codes), "numpy_cross"


def evaluate_condition(
    dataset_spec: base.DatasetSpec,
    model_spec: base.ModelSpec,
    condition: Perturbation,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_tables, gallery_tables = query_gallery_tables(
        args, condition, dataset_spec, model_spec
    )
    ann_ids, identities, image_paths, gallery_paths = validate_query_gallery_alignment(
        query_tables, gallery_tables
    )
    n = len(ann_ids)
    base_table = query_tables[next(iter(query_tables))]
    encounters = np.asarray(base_table.get("encounters", np.full(n, ""))).astype(str)
    if encounters.shape[0] != n:
        encounters = np.full(n, "")

    group_codes: list[np.ndarray] = []
    if args.exclude_same_image:
        group_codes.append(base.factorize_labels(gallery_paths, drop_empty=True))
    if args.exclude_same_encounter:
        encoded = base.factorize_labels(encounters, drop_empty=True)
        if not np.any(encoded >= 0):
            raise SystemExit(
                f"Dataset '{dataset_spec.key}' has no usable encounter metadata."
            )
        group_codes.append(encoded)

    id_codes = base.factorize_labels(identities)
    gallery_size, positives = base.compute_gallery_stats(
        n, id_codes, group_codes, args.retrieval_batch_size
    )
    valid = np.flatnonzero((positives > 0) & (gallery_size > 0))
    random_by_k = {
        k: float(
            np.mean(
                [
                    base.random_topk_probability(
                        int(gallery_size[i]), int(positives[i]), k
                    )
                    for i in valid
                ]
            )
            if len(valid)
            else 0.0
        )
        for k in args.ks
    }

    per_query: dict[int, dict[str, Any]] = {}
    if args.write_per_query:
        for i in valid.tolist():
            per_query[i] = {
                "dataset": dataset_spec.key,
                "model": model_spec.key,
                "condition": condition.tag,
                "query_index": i,
                "query_ann_id": ann_ids[i],
                "identity": identities[i],
                "image_path": image_paths[i],
                "gallery_size": int(gallery_size[i]),
                "positive_count": int(positives[i]),
            }

    variant_metrics: dict[str, Any] = {}
    retrieval_backend: dict[str, str] = {}
    for variant in args.variants:
        q_features = base.l2_normalize(query_tables[variant]["embeddings"])
        g_features = base.l2_normalize(gallery_tables[variant]["embeddings"])
        top_indices, top_scores, backend = topk_cross(
            q_features,
            g_features,
            max(args.ks),
            args.retrieval_batch_size,
            group_codes,
            not args.no_faiss,
            args.require_faiss,
        )
        retrieval_backend[variant] = backend
        hits = {k: [] for k in args.ks}
        for i in valid.tolist():
            pairs = [
                (j, score)
                for j, score in zip(top_indices[i].tolist(), top_scores[i].tolist())
                if math.isfinite(score)
            ]
            flags = [identities[j] == identities[i] for j, _ in pairs]
            for k in args.ks:
                hits[k].append(any(flags[: min(k, len(flags))]))
            if args.write_per_query:
                row = per_query[i]
                row[f"{variant}_top_ann_ids"] = json.dumps(
                    [ann_ids[j] for j, _ in pairs]
                )
                for k in args.ks:
                    row[f"{variant}_hit_at_{k}"] = bool(
                        any(flags[: min(k, len(flags))])
                    )
        variant_metrics[variant] = {
            "valid_queries": len(valid),
            "num_embeddings": n,
            "r_at_k": {
                str(k): float(np.mean(hits[k])) if hits[k] else 0.0
                for k in args.ks
            },
        }

    silhouette_identity = {
        base.clean_dataset_key(value) for value in args.silhouette_identity_datasets
    }
    silhouette_is_identity = (
        base.clean_dataset_key(dataset_spec.key) in silhouette_identity
        or base.clean_dataset_key(dataset_spec.label) in silhouette_identity
    )
    cfr_by_k: dict[str, Any] = {}
    for k in args.ks:
        random_rate = random_by_k[k]
        retrieval = {
            variant: float(variant_metrics[variant]["r_at_k"][str(k)])
            for variant in args.variants
        }
        signals = {
            variant: max(0.0, value - random_rate)
            for variant, value in retrieval.items()
        }
        effective_shortcuts = [
            view
            for view in args.shortcut_views
            if not (view == "silhouette" and silhouette_is_identity)
        ]
        candidates = {
            view: signals[view] for view in effective_shortcuts if view in signals
        }
        shortcut_view = max(candidates, key=candidates.get) if candidates else None
        shortcut_signal = candidates[shortcut_view] if shortcut_view else 0.0
        full_signal = signals.get("full", 0.0)
        foreground_signal = signals.get("foreground", 0.0)
        headroom = max(0.0, 1.0 - random_rate)
        cfr = (
            min(full_signal, foreground_signal) / headroom if headroom > 0 else None
        )
        ssi = min(1.0, shortcut_signal / full_signal) if full_signal > 0 else None
        cfr_by_k[str(k)] = {
            "random": random_rate,
            "retrieval": retrieval,
            "signals_above_chance": signals,
            "shortcut_views": list(args.shortcut_views),
            "effective_shortcut_views": effective_shortcuts,
            "shortcut_view": shortcut_view,
            "shortcut_signal": shortcut_signal,
            "foreground_signal": foreground_signal,
            "full_signal": full_signal,
            "headroom": headroom,
            "ssi": ssi,
            "cfr": cfr,
            "defined": cfr is not None,
        }

    result = {
        "dataset": asdict(dataset_spec),
        "model": asdict(model_spec),
        "condition": asdict(condition),
        "condition_tag": condition.tag,
        "query_scope": condition.scope,
        "gallery_source": "perturbed" if condition.scope == "both" else "clean_table_s3",
        "variants": list(args.variants),
        "ks": list(args.ks),
        "counts": {
            "num_embeddings": n,
            "identities": len(Counter(identities.tolist())),
            "valid_queries": len(valid),
            "queries_without_positive": n - len(valid),
        },
        "validity": {
            "exclude_same_image": args.exclude_same_image,
            "exclude_same_encounter": args.exclude_same_encounter,
            "crop_mode": args.crop_mode,
            "same_annotation_order_as_table_s3": True,
        },
        "random_by_k": {str(k): value for k, value in random_by_k.items()},
        "variant_metrics": variant_metrics,
        "cfr_by_k": cfr_by_k,
        "retrieval_backend": retrieval_backend,
    }
    result_dir = condition_root(args.output_root, condition) / "results" / model_spec.key
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{dataset_spec.key}_{model_spec.key}_robustness.json"
    result_path.write_text(json.dumps(result, indent=2, default=base.json_default))
    if args.write_per_query:
        base.write_dict_rows(
            list(per_query.values()),
            result_dir / f"{dataset_spec.key}_{model_spec.key}_robustness_per_query.csv",
        )
    return result


def run_self_test() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:65, 25:80] = True
    for name in sorted(MASK_PERTURBATIONS):
        condition = Perturbation(name, "mask", "medium", 0.10, 7, "both")
        first, stats = perturb_mask(mask, condition, stable_rng(condition, "d", "a"))
        second, _ = perturb_mask(mask, condition, stable_rng(condition, "d", "a"))
        assert np.array_equal(first, second), name
        assert abs(stats["mask_changed_fraction"] - 0.10) <= 1.0 / mask.sum(), stats
        if name in {"mask_erode", "mask_interior_false_negative"}:
            assert np.all(~first | mask), name
        else:
            assert np.all(~mask | first), name

    crop = Image.fromarray(np.full((80, 100, 3), 127, dtype=np.uint8))
    for name in sorted(IMAGE_PERTURBATIONS):
        value = SEVERITY_VALUES[name]["medium"]
        condition = Perturbation(name, "image", "medium", value, 11, "query")
        first, stats = corrupt_image(crop, mask, condition, stable_rng(condition, "d", "a"))
        second, _ = corrupt_image(crop, mask, condition, stable_rng(condition, "d", "a"))
        assert np.array_equal(first, second), name
        assert first.size == crop.size
        if name.endswith("cutout"):
            assert abs(stats["image_changed_fraction"] - 0.10) <= 1.0 / mask.sum()

    clean = Perturbation("clean", "clean", "clean", 0.0, 0, "both")
    unchanged, stats = perturb_mask(mask, clean, stable_rng(clean, "d", "a"))
    assert np.array_equal(mask, unchanged)
    assert stats["mask_changed_pixels"] == 0

    rng = np.random.default_rng(5)
    features = rng.normal(size=(20, 12)).astype(np.float32)
    features = base.l2_normalize(features)
    old_idx, old_scores = base.topk_search_numpy(features, 5, 7)
    new_idx, new_scores = topk_cross_numpy(features, features, 5, 7, [])
    assert np.array_equal(old_idx, new_idx)
    assert np.allclose(old_scores, new_scores)
    print("CFR robustness self-test passed.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central_index", type=Path, default=base.CENTRAL_INDEX)
    parser.add_argument("--clean_embedding_root", type=Path, required=False)
    parser.add_argument("--output_root", type=Path, required=False)
    parser.add_argument("--suite", choices=["headline", "all", "custom"], default="all")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--models", nargs="*", choices=["miewid", "megadescriptor"])
    parser.add_argument("--mode", choices=["extract", "eval", "extract_eval"], default="extract_eval")
    parser.add_argument("--variants", nargs="+", choices=list(base.VARIANTS), default=list(base.VARIANTS))
    parser.add_argument("--shortcut_views", nargs="+", default=["background", "silhouette"])
    parser.add_argument(
        "--silhouette_identity_datasets",
        nargs="*",
        default=sorted(base.DEFAULT_SILHOUETTE_IDENTITY_DATASETS),
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1])
    parser.add_argument("--crop_mode", choices=["bbox", "full"], default="bbox")
    parser.add_argument("--perturbation", choices=PERTURBATIONS, default="clean")
    parser.add_argument("--severity", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--severity_value", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scope", choices=["auto", "query", "both"], default="auto")
    parser.add_argument("--exclude_same_image", action="store_true")
    parser.add_argument("--exclude_same_encounter", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--retrieval_batch_size", type=int, default=1024)
    parser.add_argument("--device")
    parser.add_argument("--resize_size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force_extract", action="store_true")
    parser.add_argument("--no_faiss", action="store_true")
    parser.add_argument(
        "--require_faiss",
        action="store_true",
        help="Fail instead of changing tie behavior through the NumPy fallback.",
    )
    parser.add_argument("--write_per_query", action="store_true")
    parser.add_argument("--verify_clean_embedding_parity", action="store_true")
    parser.add_argument("--remove_condition_embeddings_after_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.self_test:
        run_self_test()
        return
    if args.output_root is None or args.clean_embedding_root is None:
        raise SystemExit("--output_root and --clean_embedding_root are required")
    args.output_root = Path(args.output_root)
    args.clean_embedding_root = Path(args.clean_embedding_root)
    condition = build_perturbation(args)
    dataset_specs = base.dataset_specs_from_index(Path(args.central_index))
    model_specs = base.get_model_specs()
    datasets = base.select_datasets(args, dataset_specs)
    models = base.select_models(args, model_specs)
    base.ensure_inputs_exist(datasets)

    extraction_variants = variants_requiring_extraction(condition, list(args.variants))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "condition": asdict(condition),
                    "condition_tag": condition.tag,
                    "datasets": [spec.key for spec in datasets],
                    "models": [spec.key for spec in models],
                    "variants": list(args.variants),
                    "extraction_variants": extraction_variants,
                    "clean_embedding_root": str(args.clean_embedding_root),
                    "output_root": str(args.output_root),
                },
                indent=2,
            )
        )
        return

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    for model_spec in models:
        loaded = None
        if args.mode in {"extract", "extract_eval"} and extraction_variants:
            loaded = base.load_feature_model(model_spec, args, device)
        for dataset_spec in datasets:
            print(
                f"=== Robustness {condition.tag} / {dataset_spec.key} / {model_spec.key} ===",
                flush=True,
            )
            if loaded is not None:
                extract_condition_embeddings(
                    dataset_spec, model_spec, loaded, condition, args, device
                )
                if args.verify_clean_embedding_parity:
                    verify_clean_embedding_parity(
                        args, condition, dataset_spec, model_spec
                    )
            if args.mode in {"eval", "extract_eval"}:
                evaluate_condition(dataset_spec, model_spec, condition, args)
                if args.remove_condition_embeddings_after_eval:
                    for variant in extraction_variants:
                        path = condition_embedding_path(
                            args.output_root,
                            condition,
                            dataset_spec.key,
                            model_spec.key,
                            variant,
                            base.cache_tag(model_spec, args),
                        )
                        path.unlink(missing_ok=True)
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
