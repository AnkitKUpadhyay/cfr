# CFR Artifact Manifest

This document describes the non-code artifacts expected by the CFR scripts.
The image datasets themselves should be obtained from their original sources.

## Dataset Index

The evaluator reads a TSV index with one row per dataset. A template is
provided at:

```text
configs/cfr_dataset_index.template.tsv
```

Important columns:

- `dataset`: dataset key used by the evaluator.
- `group`: dataset grouping used only for organization.
- `raw_image_root`: root directory containing source images.
- `central_metadata_input`: JSON or CSV metadata file used by the evaluator.
- `central_mask_dir`: directory containing mask files for the dataset.
- `records_total`: eligible annotation records.
- `annotations_accepted`: annotation records with accepted masks.
- `manifest_prompts`: text prompts used for SAM extraction.
- `notes`: dataset-specific caveats.

## Included Per-Dataset Mask Directory

This repository includes the release-facing mask files under:

```text
Masks_data/
  <dataset>/
    annotation_masks_rle.json.gz
    annotation_best_any_iou_masks_rle.json.gz
```

Both files are gzipped JSON files containing COCO-style compressed RLE masks
from `pycocotools.mask.encode`. Their mask coordinate frame is the same as the
internal extraction output: annotation-box coordinates for tight boxes, and
image coordinates for full-image or crop-level annotations.

`annotation_masks_rle.json.gz` is the file consumed by CFR evaluation. It
stores accepted masks only; annotations without an accepted mask have
`present=false` and `segmentation=null`.

`annotation_best_any_iou_masks_rle.json.gz` stores the best available candidate
for each annotation, including candidates that did not pass the acceptance
rule. It is included to support mask audit and failure inspection.

Each JSON file has this structure:

```json
{
  "format": "cfr-mask-rle-v1",
  "rle_encoding": "COCO compressed RLE from pycocotools.mask.encode",
  "dataset": "<dataset>",
  "annotations": [
    {
      "annotation_id": "<annotation id>",
      "present": true,
      "segmentation": {"size": [height, width], "counts": "<compressed RLE>"},
      "area": 123,
      "bbox": [x, y, width, height]
    }
  ]
}
```

Use `pycocotools.mask.decode(record["segmentation"])` to recover a binary mask
for any record with `present=true`.

## Internal Audit Mask Directory

If users regenerate masks from SAM outputs, the richer internal audit directory
has this structure:

```text
<central_mask_dir>/
  annotation_masks.pkl
  annotation_best_any_iou_masks.pkl
  annotation_match_metadata.jsonl
  image_candidate_metadata.jsonl
  extraction_summary.json
  candidate_masks/
    *.npz
```

`annotation_masks.pkl` maps annotation identifiers to accepted binary
foreground masks. Records without accepted masks are stored as missing or
`None` and are excluded from CFR evaluation. The release script
`scripts/convert_mask_pickles_to_coco_rle.py` converts this pickle to
`annotation_masks_rle.json.gz`.

`annotation_best_any_iou_masks.pkl` stores the best available candidate even
when it does not pass the acceptance rule. The release script converts this
pickle to `annotation_best_any_iou_masks_rle.json.gz`. The JSONL files store
candidate scores, candidate boxes, IoU values, prompt metadata, and links to
candidate mask arrays. These files are for audit and quality control.

## Acceptance Policy

The paper extraction used:

- `MIN_SAM_SCORE=0.0`
- `ACCEPT_IOU=0.5` for tight source boxes
- `CANDIDATE_MASK_MODE=box_crop`

For cropped or full-image annotations, the highest-scoring candidate is used.
StripeSpotter is handled with flank-region overlap alignment because its
benchmark annotations are flank regions rather than whole-animal boxes.

## Result Directory

CFR evaluation writes:

```text
outputs/cfr/
  embeddings/
    <model>/
      *.npz
  results/
    <model>/
      <dataset>_<model>_cfr.json
      <dataset>_<model>_cfr_per_query.csv
  cfr_summary_reference.csv
  cfr_summary_reference_k1.csv
  cfr_results_compact_k1.csv
```

The JSON files contain retrieval rates, above-random signals, shortcut view
choice, SSI/CFR fields, and validity metadata. Per-query CSVs preserve the
top-ranked annotations and hit flags, which makes table values auditable
without rerunning embedding extraction.
