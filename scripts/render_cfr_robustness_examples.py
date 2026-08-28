#!/usr/bin/env python3
"""Render real examples of every CFR robustness perturbation."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_dilation, binary_erosion

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cfr_evaluation as base
from run_cfr_robustness import (
    Perturbation,
    SEVERITY_VALUES,
    corrupt_image,
    perturb_mask,
    stable_rng,
)


EXAMPLES = (
    ("leopard", "Leopard", "d795a489-8ee2-463f-9332-494dbfc53a87"),
    ("GiraffeZebraID", "Zebra", "2222"),
    ("AerialCattle2017", "Aerial cattle", "14ca504998c07583"),
    ("seaturtle_head", "Sea turtle head", "seaturtle_head_annot_026705"),
    ("HumpbackWhaleID", "Humpback whale", "d4ef1eb25"),
)

MASK_COLUMNS = (
    ("clean", "Clean mask"),
    ("mask_erode", "Erosion\n10%"),
    ("mask_dilate", "Dilation\n10%"),
    ("mask_interior_false_negative", "Interior FN\n10%"),
    ("mask_exterior_false_positive", "Exterior FP\n10%"),
)

IMAGE_COLUMNS = (
    ("clean", "Clean query"),
    ("gaussian_noise", "Gaussian noise\n$\\sigma=0.05$"),
    ("gaussian_blur", "Gaussian blur\n$r=2$"),
    ("jpeg_compression", "JPEG\n$q=50$"),
    ("foreground_cutout", "Foreground CutOut\n10%"),
    ("background_cutout", "Background CutOut\n10%"),
)

MODEL_INPUT_ROWS = (
    ("clean", "clean", "Clean"),
    ("mask_erode", "mask", "Mask erosion (10%)"),
    ("mask_dilate", "mask", "Mask dilation (10%)"),
    ("mask_interior_false_negative", "mask", "Interior false negative (10%)"),
    ("mask_exterior_false_positive", "mask", "Exterior false positive (10%)"),
    ("gaussian_noise", "image", "Gaussian noise ($\\sigma=0.05$)"),
    ("gaussian_blur", "image", "Gaussian blur ($r=2$)"),
    ("jpeg_compression", "image", "JPEG compression ($q=50$)"),
    ("foreground_cutout", "image", "Foreground CutOut (10%)"),
    ("background_cutout", "image", "Background CutOut (10%)"),
)

MODEL_INPUT_COLUMNS = (
    ("full", "Full"),
    ("foreground", "Foreground"),
    ("background", "Background"),
    ("silhouette", "Silhouette"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central_index", type=Path, required=True)
    parser.add_argument(
        "--rle_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Masks_data",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_input_dir", type=Path)
    parser.add_argument("--candidate_pool", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def iter_json_array(path: Path, array_name: str):
    """Stream objects from a named top-level JSON array."""
    decoder = json.JSONDecoder()
    marker = f'"{array_name}"'
    buffer = ""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                raise ValueError(f"Could not find annotations array in {path}")
            buffer += chunk
            marker_index = buffer.find(marker)
            if marker_index < 0:
                buffer = buffer[-len(marker) :]
                continue
            array_index = buffer.find("[", marker_index + len(marker))
            if array_index < 0:
                continue
            buffer = buffer[array_index + 1 :]
            break

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(1 << 20)
                if not chunk:
                    raise ValueError(f"Truncated annotations array in {path}")
                buffer += chunk
                continue
            buffer = buffer[end:]
            yield row


def iter_rle_entries(path: Path):
    yield from iter_json_array(path, "annotations")


def load_candidate_entries(
    path: Path,
    preferred_id: str | None,
    candidate_pool: int,
) -> list[dict[str, Any]]:
    if preferred_id is not None:
        for row in iter_rle_entries(path):
            if (
                str(row.get("annotation_id")) == preferred_id
                and row.get("present")
                and row.get("segmentation") is not None
            ):
                return [row]
        raise KeyError(f"Missing preferred annotation {preferred_id} in {path}")

    summaries: list[tuple[str, float]] = []
    for row in iter_rle_entries(path):
        if not row.get("present") or row.get("segmentation") is None:
            continue
        height, width = row["segmentation"]["size"]
        summaries.append((str(row["annotation_id"]), row["area"] / (height * width)))
    median_fraction = float(np.median([fraction for _, fraction in summaries]))
    selected_ids = {
        ann_id
        for ann_id, _ in sorted(
            summaries,
            key=lambda item: abs(item[1] - median_fraction),
        )[:candidate_pool]
    }
    return [
        row
        for row in iter_rle_entries(path)
        if str(row.get("annotation_id")) in selected_ids
        and row.get("present")
        and row.get("segmentation") is not None
    ]


def csv_records_for_ids(
    spec: base.DatasetSpec,
    annotation_ids: set[str],
) -> dict[str, base.CfrRecord]:
    records: dict[str, base.CfrRecord] = {}
    with spec.metadata_path.open("r", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            ann_id = str(
                row.get("annotation_id")
                or row.get("annot_uuid")
                or row.get("image_id")
                or row.get("id")
                or idx
            )
            if ann_id not in annotation_ids:
                continue
            relpath = row.get("path") or row.get("image_path") or row.get("file_name")
            if not relpath:
                continue
            dataset = str(row.get("dataset") or spec.key)
            image_id = str(row.get("image_id") or row.get("id") or idx)
            records[ann_id] = base.CfrRecord(
                ann_id=ann_id,
                image_key=image_id,
                image_path=base.resolve_image_path(relpath, spec.image_root, dataset),
                identity=base.get_identity(row),
                bbox_xywh=base.bbox_xywh(row),
                raw=dict(row),
                source_index=idx,
                dataset=dataset,
                encounter=base.get_encounter(row),
            )
    return records


def json_records_for_ids(
    spec: base.DatasetSpec,
    annotation_ids: set[str],
) -> dict[str, base.CfrRecord]:
    selected: dict[str, tuple[int, dict[str, Any], str]] = {}
    for idx, annotation in enumerate(
        iter_json_array(spec.metadata_path, "annotations")
    ):
        ann_id = base.get_ann_id(annotation, idx)
        if ann_id in annotation_ids:
            selected[ann_id] = (
                idx,
                annotation,
                base.get_image_key_from_ann(annotation),
            )
    image_keys = {image_key for _, _, image_key in selected.values()}
    images: dict[str, dict[str, Any]] = {}
    for image in iter_json_array(spec.metadata_path, "images"):
        for key in (
            "id",
            "uuid",
            "image_uuid",
            "image_id",
            "image_path",
            "file_name",
            "path",
        ):
            value = image.get(key)
            if value not in (None, "") and str(value) in image_keys:
                images[str(value)] = image

    records: dict[str, base.CfrRecord] = {}
    for ann_id, (idx, annotation, image_key) in selected.items():
        image = images.get(image_key)
        if image is None:
            continue
        relpath = base.image_relpath(image)
        if not relpath:
            continue
        records[ann_id] = base.CfrRecord(
            ann_id=ann_id,
            image_key=image_key,
            image_path=base.resolve_image_path(relpath, spec.image_root),
            identity=base.get_identity(annotation),
            bbox_xywh=base.bbox_xywh(annotation),
            raw=dict(annotation),
            source_index=idx,
            dataset=spec.key,
            encounter=base.get_encounter(annotation) or base.get_encounter(image),
        )
    return records


def records_for_ids(
    spec: base.DatasetSpec,
    annotation_ids: set[str],
) -> dict[str, base.CfrRecord]:
    if spec.input_type == "csv":
        return csv_records_for_ids(spec, annotation_ids)
    return json_records_for_ids(spec, annotation_ids)


def image_quality(record: base.CfrRecord, area_fraction: float) -> float | None:
    if not base.is_valid_identity(record.identity) or not record.image_path.exists():
        return None
    try:
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
            crop = base.crop_image(image, record.bbox_xywh, "bbox")
    except Exception:
        return None
    if min(crop.size) < 40 or max(crop.size) / max(1, min(crop.size)) > 3.5:
        return None
    thumb = crop.copy()
    thumb.thumbnail((320, 320), getattr(Image, "Resampling", Image).BILINEAR)
    values = np.asarray(thumb, dtype=np.float32)
    gray = values.mean(axis=2)
    contrast = float(gray.std())
    sharpness = 0.5 * (
        float(np.abs(np.diff(gray, axis=0)).mean())
        + float(np.abs(np.diff(gray, axis=1)).mean())
    )
    exposure = max(0.0, 1.0 - abs(float(gray.mean()) - 127.5) / 127.5)
    resolution = min(1.0, math.sqrt(crop.width * crop.height) / 500.0)
    area_balance = max(0.0, 1.0 - abs(area_fraction - 0.5) / 0.5)
    return (
        1.5 * min(2.0, contrast / 55.0)
        + 1.5 * min(2.0, sharpness / 22.0)
        + exposure
        + 0.5 * resolution
        + area_balance
    )


def select_annotation(
    spec: base.DatasetSpec,
    entries: list[dict[str, Any]],
    preferred_id: str | None,
    candidate_pool: int,
) -> tuple[base.CfrRecord, dict[str, Any], float | None]:
    by_id = {str(row["annotation_id"]): row for row in entries}
    if preferred_id is not None:
        entry = by_id.get(preferred_id)
        if entry is None:
            raise KeyError(f"Missing preferred annotation {preferred_id} for {spec.key}")
        records = records_for_ids(spec, {preferred_id})
        if preferred_id not in records:
            raise KeyError(f"Missing metadata record {preferred_id} for {spec.key}")
        return records[preferred_id], entry, None

    fractions = np.asarray(
        [
            row["area"]
            / (row["segmentation"]["size"][0] * row["segmentation"]["size"][1])
            for row in entries
        ],
        dtype=np.float64,
    )
    median_fraction = float(np.median(fractions))
    ranked = sorted(
        entries,
        key=lambda row: abs(
            row["area"]
            / (row["segmentation"]["size"][0] * row["segmentation"]["size"][1])
            - median_fraction
        ),
    )[:candidate_pool]
    candidate_ids = {str(row["annotation_id"]) for row in ranked}
    records = records_for_ids(spec, candidate_ids)

    best: tuple[float, base.CfrRecord, dict[str, Any]] | None = None
    for entry in ranked:
        ann_id = str(entry["annotation_id"])
        record = records.get(ann_id)
        if record is None:
            continue
        height, width = entry["segmentation"]["size"]
        fraction = float(entry["area"] / (height * width))
        quality = image_quality(record, fraction)
        if quality is None:
            continue
        if best is None or quality > best[0]:
            best = (quality, record, entry)
    if best is None:
        raise RuntimeError(f"Could not select a valid example for {spec.key}")
    return best[1], best[2], best[0]


def decode_crop(
    record: base.CfrRecord,
    entry: dict[str, Any],
) -> tuple[Image.Image, np.ndarray]:
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
        crop = base.crop_image(image, record.bbox_xywh, "bbox")
        decoded = mask_utils.decode(entry["segmentation"]).astype(np.uint8)
        mask = base.mask_for_crop(decoded, image, record.bbox_xywh, crop, "bbox")
    return crop, mask


def condition(name: str, family: str) -> Perturbation:
    if name == "clean":
        return Perturbation("clean", "clean", "clean", 0.0, 0, "both")
    return Perturbation(
        name,
        family,
        "medium",
        SEVERITY_VALUES[name]["medium"],
        0,
        "both" if family == "mask" else "query",
    )


def boundary(mask: np.ndarray) -> np.ndarray:
    width = max(1, int(round(min(mask.shape) / 180.0)))
    outer = binary_dilation(mask, iterations=width)
    inner = binary_erosion(mask, iterations=width, border_value=0)
    return outer ^ inner


def blend(array: np.ndarray, region: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if not np.any(region):
        return
    array[region] = (1.0 - alpha) * array[region] + alpha * np.asarray(color)


def mask_overlay(
    crop: Image.Image,
    clean_mask: np.ndarray,
    perturbed_mask: np.ndarray,
) -> Image.Image:
    output = np.asarray(crop, dtype=np.float32).copy()
    removed = clean_mask & ~perturbed_mask
    added = ~clean_mask & perturbed_mask
    blend(output, boundary(clean_mask), (255, 255, 255), 0.88)
    blend(output, removed, (220, 45, 55), 0.78)
    blend(output, added, (255, 190, 20), 0.82)
    blend(output, boundary(perturbed_mask), (0, 210, 225), 0.95)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def image_with_mask_boundary(crop: Image.Image, mask: np.ndarray) -> Image.Image:
    output = np.asarray(crop, dtype=np.float32).copy()
    blend(output, boundary(mask), (0, 210, 225), 0.95)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def render_mask_examples(
    samples: list[dict[str, Any]],
) -> list[list[Image.Image]]:
    rendered: list[list[Image.Image]] = []
    for sample in samples:
        row = []
        for name, _ in MASK_COLUMNS:
            if name == "clean":
                perturbed = sample["mask"]
            else:
                current = condition(name, "mask")
                rng = stable_rng(current, sample["dataset"], sample["ann_id"])
                perturbed, _ = perturb_mask(sample["mask"], current, rng)
            row.append(mask_overlay(sample["crop"], sample["mask"], perturbed))
        rendered.append(row)
    return rendered


def render_image_examples(
    samples: list[dict[str, Any]],
) -> list[list[Image.Image]]:
    rendered: list[list[Image.Image]] = []
    for sample in samples:
        row = []
        for name, _ in IMAGE_COLUMNS:
            if name == "clean":
                corrupted = sample["crop"]
            else:
                current = condition(name, "image")
                rng = stable_rng(current, sample["dataset"], sample["ann_id"])
                corrupted, _ = corrupt_image(
                    sample["crop"], sample["mask"], current, rng
                )
            row.append(image_with_mask_boundary(corrupted, sample["mask"]))
        rendered.append(row)
    return rendered


def exact_model_views(sample: dict[str, Any]) -> list[list[Image.Image]]:
    """Render CFR query content before model-specific normalization."""
    rows: list[list[Image.Image]] = []
    resize = getattr(Image, "Resampling", Image).BILINEAR
    for name, family, _ in MODEL_INPUT_ROWS:
        current = condition(name, family)
        rng = stable_rng(current, sample["dataset"], sample["ann_id"])
        view_mask, _ = perturb_mask(sample["mask"], current, rng)
        corrupted_crop, _ = corrupt_image(
            sample["crop"], sample["mask"], current, rng
        )
        rows.append(
            [
                base.apply_image_variant(corrupted_crop, view_mask, variant).resize(
                    (384, 384), resize
                )
                for variant, _ in MODEL_INPUT_COLUMNS
            ]
        )
    return rows


def write_model_input_figures(
    samples: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "cfr_robustness_model_inputs_all_datasets.pdf"
    with PdfPages(combined_path, metadata={"Creator": "Matplotlib"}) as pdf:
        for sample in samples:
            views = exact_model_views(sample)
            figure, axes = plt.subplots(
                len(MODEL_INPUT_ROWS),
                len(MODEL_INPUT_COLUMNS),
                figsize=(11.2, 24.5),
                squeeze=False,
                facecolor="white",
            )
            figure.suptitle(
                f"{sample['label']}: CFR model inputs under medium perturbations",
                fontsize=17,
                fontweight="bold",
                y=0.992,
            )
            for row_index, (_, family, row_label) in enumerate(MODEL_INPUT_ROWS):
                for column_index, (_, column_label) in enumerate(MODEL_INPUT_COLUMNS):
                    axis = axes[row_index, column_index]
                    axis.imshow(views[row_index][column_index])
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    if row_index == 0:
                        axis.set_title(column_label, fontsize=13, fontweight="bold", pad=6)
                    if column_index == 0:
                        axis.set_ylabel(
                            row_label,
                            fontsize=11,
                            fontweight="bold" if family == "clean" else "normal",
                            rotation=0,
                            ha="right",
                            va="center",
                            labelpad=14,
                        )
                    if row_index in {1, 5}:
                        axis.spines["top"].set_visible(True)
                        axis.spines["top"].set_color("0.25")
                        axis.spines["top"].set_linewidth(1.2)
            figure.text(
                0.56,
                0.012,
                "Exact CFR RGB views are shown without overlays. Panels are visualized at "
                "384 x 384 before normalization; MiewID uses the same content at 440 x 440.",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
            figure.subplots_adjust(
                left=0.205,
                right=0.99,
                top=0.968,
                bottom=0.035,
                wspace=0.025,
                hspace=0.08,
            )
            slug = sample["dataset"].lower().replace("_", "-")
            figure.savefig(
                output_dir / f"cfr_robustness_model_inputs_{slug}.png",
                dpi=max(160, min(dpi, 200)),
                bbox_inches="tight",
                facecolor="white",
                metadata={"Software": "Matplotlib"},
            )
            pdf.savefig(figure, bbox_inches="tight", facecolor="white")
            plt.close(figure)
    print(f"Wrote {combined_path}", flush=True)


def draw_grid(
    subfigure: Any,
    images: list[list[Image.Image]],
    samples: list[dict[str, Any]],
    columns: tuple[tuple[str, str], ...],
    title: str,
) -> None:
    axes = subfigure.subplots(len(samples), len(columns), squeeze=False)
    subfigure.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    for row_index, sample in enumerate(samples):
        for column_index, (_, column_title) in enumerate(columns):
            axis = axes[row_index, column_index]
            axis.imshow(images[row_index][column_index])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                axis.set_title(column_title, fontsize=11, pad=5)
            if column_index == 0:
                axis.set_ylabel(
                    sample["label"],
                    fontsize=11,
                    fontweight="bold",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                )
    subfigure.subplots_adjust(
        left=0.105,
        right=0.995,
        top=0.88,
        bottom=0.04,
        wspace=0.025,
        hspace=0.08,
    )


def main() -> None:
    args = parse_args()
    specs = base.dataset_specs_from_index(args.central_index)
    samples: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for dataset, label, preferred_id in EXAMPLES:
        spec = specs[dataset]
        rle_path = args.rle_root / dataset / "annotation_masks_rle.json.gz"
        entries = load_candidate_entries(
            rle_path, preferred_id, args.candidate_pool
        )
        record, entry, score = select_annotation(
            spec, entries, preferred_id, args.candidate_pool
        )
        crop, mask = decode_crop(record, entry)
        samples.append(
            {
                "dataset": dataset,
                "label": label,
                "ann_id": str(record.ann_id),
                "crop": crop,
                "mask": mask,
            }
        )
        manifest.append(
            {
                "dataset": dataset,
                "label": label,
                "annotation_id": str(record.ann_id),
                "identity": str(record.identity),
                "crop_width": crop.width,
                "crop_height": crop.height,
                "mask_area_fraction": float(mask.mean()),
                "selection_score": score,
            }
        )
        print(
            f"Selected {dataset}: ann={record.ann_id}, crop={crop.width}x{crop.height}, "
            f"mask={mask.mean():.3f}",
            flush=True,
        )
        del entries
        gc.collect()

    mask_images = render_mask_examples(samples)
    corruption_images = render_image_examples(samples)

    figure = plt.figure(figsize=(19.5, 15.5), facecolor="white")
    subfigures = figure.subfigures(2, 1, height_ratios=(1.0, 1.08), hspace=0.055)
    draw_grid(
        subfigures[0],
        mask_images,
        samples,
        MASK_COLUMNS,
        "A. Mask errors: image unchanged; query and gallery masks modified",
    )
    draw_grid(
        subfigures[1],
        corruption_images,
        samples,
        IMAGE_COLUMNS,
        "B. Query corruptions: pixels modified; mask and clean gallery unchanged",
    )

    mask_legend = [
        Line2D([0], [0], color=(0.0, 0.82, 0.88), lw=3, label="Resulting mask boundary"),
        Line2D([0], [0], color="0.55", lw=3, label="Clean boundary (white in panels)"),
        Patch(facecolor=(0.86, 0.18, 0.22), label="Pixels removed from mask"),
        Patch(facecolor=(1.0, 0.75, 0.08), label="Pixels added to mask"),
    ]
    subfigures[0].legend(
        handles=mask_legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.55, -0.002),
    )
    subfigures[1].text(
        0.55,
        0.006,
        "Cyan contours show the accepted mask for orientation and are not model inputs.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Matplotlib"},
    )
    figure.savefig(
        args.output.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "Matplotlib"},
    )
    plt.close(figure)
    args.output.with_name(f"{args.output.stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    model_input_dir = (
        args.model_input_dir
        if args.model_input_dir is not None
        else args.output.parent / "cfr_robustness_model_inputs"
    )
    write_model_input_figures(samples, model_input_dir, args.dpi)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
