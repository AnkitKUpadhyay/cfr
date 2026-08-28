#!/usr/bin/env python3
"""Render paper panels of the exact CFR inputs under robustness perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import render_cfr_robustness_examples as examples


PAPER_EXAMPLES = (
    ("GiraffeZebraID", "Zebra", "2222"),
    ("seaturtle_head", "Sea turtle head", "seaturtle_head_annot_026705"),
)

MASK_COLUMNS = (
    ("clean", "clean", "Clean"),
    ("mask_erode", "mask", "Erosion"),
    ("mask_dilate", "mask", "Dilation"),
    ("mask_interior_false_negative", "mask", "Interior\nfalse negative"),
    ("mask_exterior_false_positive", "mask", "Exterior\nfalse positive"),
)

IMAGE_COLUMNS = (
    ("clean", "clean", "Clean"),
    ("gaussian_noise", "image", "Gaussian\nnoise"),
    ("gaussian_blur", "image", "Gaussian\nblur"),
    ("jpeg_compression", "image", "JPEG"),
    ("foreground_cutout", "image", "Foreground\nCutOut"),
    ("background_cutout", "image", "Background\nCutOut"),
)

MASK_VIEWS = (
    ("foreground", "Foreground"),
    ("background", "Background"),
    ("silhouette", "Silhouette"),
)

IMAGE_VIEWS = (
    ("full", "Full"),
    ("foreground", "Foreground"),
    ("background", "Background"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central_index", type=Path, required=True)
    parser.add_argument("--rle_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=420)
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = examples.base.dataset_specs_from_index(args.central_index)
    samples: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for dataset, label, annotation_id in PAPER_EXAMPLES:
        spec = specs[dataset]
        rle_path = args.rle_root / dataset / "annotation_masks_rle.json.gz"
        entries = examples.load_candidate_entries(rle_path, annotation_id, 1)
        record, entry, _ = examples.select_annotation(spec, entries, annotation_id, 1)
        crop, mask = examples.decode_crop(record, entry)
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
            }
        )
    return samples, manifest


def exact_views(
    sample: dict[str, Any],
    columns: tuple[tuple[str, str, str], ...],
    views: tuple[tuple[str, str], ...],
) -> list[list[Image.Image]]:
    """Return the RGB content passed to the model before normalization."""
    resize = getattr(Image, "Resampling", Image).BILINEAR
    rendered: list[list[Image.Image]] = []
    for view, _ in views:
        row: list[Image.Image] = []
        for name, family, _ in columns:
            current = examples.condition(name, family)
            rng = examples.stable_rng(current, sample["dataset"], sample["ann_id"])
            view_mask, _ = examples.perturb_mask(sample["mask"], current, rng)
            corrupted_crop, _ = examples.corrupt_image(
                sample["crop"], sample["mask"], current, rng
            )
            row.append(
                examples.base.apply_image_variant(
                    corrupted_crop, view_mask, view
                ).resize((384, 384), resize)
            )
        rendered.append(row)
    return rendered


def style_axis(
    axis: Any,
    image: Image.Image,
    column_title: str | None,
    row_title: str | None,
) -> None:
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    if column_title:
        axis.set_title(column_title, fontsize=7.4, pad=3.0, linespacing=0.95)
    if row_title:
        axis.set_ylabel(
            row_title,
            fontsize=7.1,
            rotation=0,
            ha="right",
            va="center",
            labelpad=6,
            linespacing=0.95,
        )


def save_panel(
    samples: list[dict[str, Any]],
    columns: tuple[tuple[str, str, str], ...],
    views: tuple[tuple[str, str], ...],
    output: Path,
    dpi: int,
) -> None:
    rows = len(samples) * len(views)
    cols = len(columns)
    figure_width = 7.15
    figure_height = figure_width * rows / cols * 0.91
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(figure_width, figure_height),
        squeeze=False,
        facecolor="white",
    )

    row_index = 0
    for sample_index, sample in enumerate(samples):
        images = exact_views(sample, columns, views)
        for view_index, (_, view_label) in enumerate(views):
            for column_index, (_, _, column_label) in enumerate(columns):
                style_axis(
                    axes[row_index, column_index],
                    images[view_index][column_index],
                    column_label if row_index == 0 else None,
                    f"{sample['label']}\n{view_label}" if column_index == 0 else None,
                )
            row_index += 1
        if sample_index < len(samples) - 1:
            for axis in axes[row_index, :]:
                axis.spines["top"].set_visible(True)
                axis.spines["top"].set_color("0.65")
                axis.spines["top"].set_linewidth(0.65)

    figure.subplots_adjust(
        left=0.14,
        right=0.998,
        top=0.945,
        bottom=0.005,
        wspace=0.025,
        hspace=0.045,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        facecolor="white",
        metadata={"Creator": "Matplotlib"},
    )
    figure.savefig(
        output.with_suffix(".png"),
        dpi=dpi,
        facecolor="white",
        metadata={"Software": "Matplotlib"},
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    samples, manifest = load_samples(args)
    mask_output = args.output_dir / "cfr_mask_perturbation_model_inputs.pdf"
    image_output = args.output_dir / "cfr_image_degradation_model_inputs.pdf"
    save_panel(samples, MASK_COLUMNS, MASK_VIEWS, mask_output, args.dpi)
    save_panel(samples, IMAGE_COLUMNS, IMAGE_VIEWS, image_output, args.dpi)
    (args.output_dir / "cfr_robustness_model_input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote {mask_output}")
    print(f"Wrote {image_output}")


if __name__ == "__main__":
    main()
