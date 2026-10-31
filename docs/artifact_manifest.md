# CFR evaluation inputs and outputs

## Dataset index

Copy [the TSV template](../configs/cfr_dataset_index.template.tsv) to a local file and edit its paths. Its counts and prompts describe the historical mask collection; use the appropriate inventory when evaluating updated masks.

| Field | Meaning |
|---|---|
| `dataset` | Evaluation key; historical keys are preserved in the template. |
| `group` | Organizational label retained from the input inventory. |
| `raw_image_root` | Root for paths in the annotation metadata. |
| `central_metadata_input` | CSV or COCO/GZCD-style JSON containing annotation IDs, image references, identities, and any annotation boxes. |
| `central_mask_dir` | Folder containing `annotation_masks_rle.json.gz` (or a trusted local legacy pickle). |
| `annotation_masks_rle` | Optional explicit mask-file path; set this for the updated HF `masks.json.gz` filename. |
| `annotations_accepted` | Accepted-mask count. `--suite all` includes only rows with a positive value; a missing value is treated as zero. |
| `records_total` | Number of annotation records in the selected inventory. |

Other template columns preserve the original extraction inventory and prompts. `python scripts/run_cfr_evaluation.py --help` lists model, retrieval, view, and exclusion options.

## Matching and coordinates

Masks are gzipped JSON with an `annotations` list. Each record has a string `annotation_id`, `present`, and a COCO compressed RLE `segmentation`; rejected records have null segmentation. RLE dimensions are `[height, width]`. Join masks to metadata by annotation ID, not row number. The auxiliary historical best-candidate file is not an evaluation input.

CSV metadata must supply image paths (for example `path`) and identities (for example `identity`), with IDs matching the mask records. JSON uses `images` and `annotations` arrays with consistent references. A source annotation box is `[x,y,width,height]`; the evaluator rounds endpoints, clips to image bounds, and applies its recorded crop behavior. A mask's foreground `bbox` is not a replacement for that source annotation box.

The historical evaluator handles image-frame masks by cropping and uses nearest-neighbor resizing for dimension mismatches. Invalid or empty historical crop boxes fall back to the full image. This is compatibility behavior, not proof of correct alignment. Check prepared crops, resizing, EXIF orientation, and annotation IDs against the selected mask version. StripeSpotter's updated full-zebra mask must be aligned with the stored flank region when evaluating flanks. Unknown identities and missing masks are excluded; historically empty accepted masks are retained.

See the [HF cards and version guide](mask_releases.md) for the distinct historical and updated coordinate metadata. The historical archives do not contain the original identity/image/crop catalogue; source metadata is required.

## Reference protocol

The default reference gallery contains accepted-mask annotations with known identities, excluding the query itself. A valid query must have at least one other same-identity annotation in its gallery. Same-image and same-encounter exclusions are optional and are off in the reference protocol. Full, foreground, background-with-animal-shaped-hole, and silhouette views use the same accepted annotation set. Preserve the model, masks, metadata, gallery protocol, and view settings when comparing runs.

For recall `R_T`, random recall `R_rand`, and `S_T = max(0, R_T - R_rand)`:

```text
CFR = min(S_full, S_foreground) / (1 - R_rand)
SSI = min(1, S_shortcut / S_full)
```

`S_shortcut` is the maximum above-chance signal among enabled shortcut views (background and silhouette by default), after any silhouette-policy exclusion. The code handles zero signal and degenerate denominators explicitly. The historical silhouette policy treats shape as identity evidence for configured datasets and changes which shortcut signal enters SSI; it does not change the CFR formula. Use `--silhouette_identity_datasets` to select that policy explicitly when changing the protocol. The background view retains an animal-shaped hole and can therefore contain silhouette information.

## Outputs

Evaluation writes local embeddings and per-dataset/per-query JSON/CSV metrics beneath `--output_root`. To aggregate numerical metrics:

```bash
python scripts/summarize_cfr_results.py \
  --results_root outputs/cfr --central_index configs/cfr_dataset_index.tsv \
  --suite all --models miewid megadescriptor --ks 1 5 10 \
  --output_name cfr_summary_reference
```

Generated outputs are ignored by Git. The repository contains computation and experiment configuration, without committed result tables or presentation-generation scripts.
