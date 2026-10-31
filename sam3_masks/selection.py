"""Deterministic, threshold-free selection among SAM3's returned candidates.

The backend owns candidate generation and its confidence filtering. This module
adds no confidence or coverage cutoff and performs no morphological cleanup.
Boxes use continuous input coordinates and half-open containment around
(W/2,H/2). The mask itself need not contain this center point.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import numpy as np


@dataclass
class SelectionResult:
    """Selected original candidate index, audit records, and a binary mask copy."""

    selected_index: int | None
    status: str
    candidates: list[dict[str, Any]]
    selected_mask: np.ndarray | None = None


def _finite_number(value: Any) -> float | None:
    """Represent nonfinite numeric metadata as JSON null, never NaN/Infinity."""
    number = float(value)
    return number if np.isfinite(number) else None


def select_candidate(
    output: Mapping[str, Any], image_size: tuple[int, int],
) -> SelectionResult:
    """Choose the largest central mask, then confidence, then original index.

    Parameters
    ----------
    output:
        ``masks`` is a binary 0/1 array of shape (N,H,W) or (N,1,H,W),
        ``boxes`` is a numeric array of shape (N,4) in XYXY crop coordinates,
        and ``scores`` is a numeric array of shape (N,). Optional
        ``prompt_texts`` contains one string per candidate. Arrays must already
        reside in CPU memory (e.g. NumPy arrays).
    image_size:
        Positive integer (width, height) of the annotation-level input crop.

    Returns
    -------
    SelectionResult
        ``selected_mask`` is a copied uint8 0/1 mask without cleanup or cropping.
        ``mask_coverage`` uses H*W. ``candidates`` preserves
        the original order and includes reasons for individually invalid
        candidates. Empty, nonbinary, and nonfinite masks; nonfinite scores;
        nonfinite/invalid boxes; and boxes empty after clipping are invalid.

    Raises
    ------
    ValueError
        Required fields are missing, arrays have inconsistent lengths/shapes or
        nonnumeric dtypes, or image_size is invalid. Such errors represent an
        adapter contract error, rather than an annotation rejection.
    """
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
               or v <= 0 for v in image_size)
    ):
        raise ValueError("image_size must be positive integer (width, height)")
    width, height = (int(v) for v in image_size)
    if not isinstance(output, Mapping):
        raise ValueError("output must provide masks, boxes, and scores")
    missing = [key for key in ("masks", "boxes", "scores") if key not in output]
    if missing:
        raise ValueError(f"Missing required candidate fields: {', '.join(missing)}")

    try:
        masks = np.asarray(output["masks"])
        boxes = np.asarray(output["boxes"], dtype=np.float64)
        scores = np.asarray(output["scores"], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Candidate arrays must be numeric and rectangular") from exc
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(
            f"masks must have shape (N,{height},{width}) or "
            f"(N,1,{height},{width}); got {masks.shape}"
        )
    if masks.dtype.kind not in "buif":
        raise ValueError("masks must use a real numeric or boolean dtype")
    count = masks.shape[0]
    if boxes.shape != (count, 4):
        raise ValueError(f"boxes must have shape ({count},4); got {boxes.shape}")
    if scores.shape != (count,):
        raise ValueError(f"scores must have shape ({count},); got {scores.shape}")
    prompts = output.get("prompt_texts")
    if prompts is None:
        prompts = [None] * count
    elif isinstance(prompts, (str, bytes)):
        raise ValueError("prompt_texts must contain one string per candidate")
    else:
        try:
            prompts = list(prompts)
        except TypeError as exc:
            raise ValueError("prompt_texts must contain one string per candidate") from exc
        if len(prompts) != count or any(not isinstance(p, str) for p in prompts):
            raise ValueError("prompt_texts must contain one string per candidate")

    if count == 0:
        return SelectionResult(None, "no_candidate", [])

    center_x, center_y = width / 2.0, height / 2.0
    image_area = width * height
    candidates: list[dict[str, Any]] = []
    eligible: list[int] = []
    for index, (mask, box, score, prompt) in enumerate(zip(masks, boxes, scores, prompts)):
        finite_mask = bool(np.isfinite(mask).all())
        binary_mask = finite_mask and bool(((mask == 0) | (mask == 1)).all())
        pixels = int(np.count_nonzero(mask)) if binary_mask else None
        record: dict[str, Any] = {
            "index": index,
            "prompt_text": prompt,
            "raw_box": [_finite_number(v) for v in box],
            "clipped_box": None,
            "valid": False,
            "invalid_reason": None,
            "contains_center": False,
            "eligible": False,
            "foreground_pixels": pixels,
            "mask_coverage": pixels / image_area if pixels is not None else None,
            "score": _finite_number(score),
        }
        reason = None
        if not finite_mask:
            reason = "non_finite_mask"
        elif not binary_mask:
            reason = "non_binary_mask"
        elif pixels == 0:
            reason = "empty_mask"
        if not np.isfinite(box).all():
            reason = reason or "non_finite_box"
        else:
            clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
            record["clipped_box"] = clipped.tolist()
            x1, y1, x2, y2 = clipped
            record["contains_center"] = bool(x1 <= center_x < x2 and y1 <= center_y < y2)
            if box[2] <= box[0] or box[3] <= box[1]:
                reason = reason or "invalid_box"
            elif x2 <= x1 or y2 <= y1:
                reason = reason or "empty_box_after_clipping"
        if not np.isfinite(score):
            reason = reason or "non_finite_score"
        record["invalid_reason"] = reason
        record["valid"] = reason is None
        record["eligible"] = bool(record["valid"] and record["contains_center"])
        if record["eligible"]:
            eligible.append(index)
        candidates.append(record)

    if not any(record["valid"] for record in candidates):
        return SelectionResult(None, "no_valid_candidate", candidates)
    if not eligible:
        return SelectionResult(None, "no_central_candidate", candidates)
    selected = min(
        eligible,
        key=lambda i: (-candidates[i]["foreground_pixels"], -scores[i], i),
    )
    return SelectionResult(
        selected, "accepted", candidates,
        np.array(masks[selected], dtype=np.uint8, copy=True),
    )
