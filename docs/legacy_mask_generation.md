# Legacy SAM3 mask generation

The original extractor remains at [`scripts/extract_wildlifedatasets_sam_masks.py`](../scripts/extract_wildlifedatasets_sam_masks.py). Its selection behavior is preserved. The [updated extractor](mask_generation.md) is separate; its options and selection rule must not be mixed with the legacy procedure.

For the frozen 34-dataset resource, [download the historical masks](mask_releases.md). Regeneration additionally requires the original prepared images and metadata, compatible SAM3 weights/software, and the recorded corrections; it is not guaranteed to recreate every historical archive byte for byte.

## Selection rule

- With a retained source annotation box, rank candidates by box IoU, then confidence; accept at IoU ≥ 0.5.
- For full-image or already-cropped annotations, use the highest-confidence returned candidate.
- `min_sam_score=0.0` adds no positive confidence cutoff after SAM's own processing.
- The historical StripeSpotter archive uses the corrected flank-alignment masks recorded in the dataset index. Rerunning the generic box-IoU job is not a reproduction of that correction.

## Run a job

Install a SAM3-capable environment as described in [mask generation](mask_generation.md), then copy the legacy manifest and edit the paths:

```bash
cp configs/sam_mask_jobs.template.jsonl configs/sam_mask_jobs.jsonl
python scripts/run_wildlifedatasets_sam_mask_job.py \
  --manifest configs/sam_mask_jobs.jsonl --index 0 --dry_run
python scripts/run_wildlifedatasets_sam_mask_job.py \
  --manifest configs/sam_mask_jobs.jsonl --index 0 --device cuda:0 \
  --min_sam_score 0.0 --accept_iou 0.5 \
  --candidate_mask_mode box_crop --save_candidate_masks
```

The index is zero-based. Manifest paths in this legacy launcher are used as supplied; use absolute paths or paths relative to the repository root. Each output directory should be separate from downloaded masks and other runs. These underscore-style flags belong to the legacy launcher; the updated CLI uses hyphenated flags.

The legacy extractor writes accepted and best-candidate pickle files plus audit metadata. Convert your trusted local outputs to COCO RLE using:

```bash
python scripts/convert_mask_pickles_to_coco_rle.py \
  --index configs/cfr_dataset_index.tsv \
  --output_root outputs/legacy_rle --convert
```

Use `annotation_masks_rle.json.gz` for evaluation. The best-candidate archive can contain rejected candidates and is only for inspection.
