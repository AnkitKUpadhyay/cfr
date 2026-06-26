# CFR Mask Data

This directory contains the released SAM-derived masks used by the CFR
evaluation.

```text
Masks_data/
  <dataset>/
    annotation_masks_rle.json.gz
    annotation_best_any_iou_masks_rle.json.gz
```

- `annotation_masks_rle.json.gz`: accepted masks used by evaluation. Records
  without an accepted mask have `present=false` and `segmentation=null`.
- `annotation_best_any_iou_masks_rle.json.gz`: best available candidate mask
  for each annotation, including candidates that did not pass the acceptance
  rule.

Masks are stored as COCO-style compressed RLE from `pycocotools.mask.encode`.
Use `pycocotools.mask.decode(record["segmentation"])` to recover a binary mask
for records with `present=true`.

Raw images and source metadata are not included in this directory.
