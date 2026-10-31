# Mask releases

Both mask collections are hosted in [imageomics/cfr-wildlife-masks](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks). The downloader pins exact commits in [the release registry](../configs/mask_releases.json); it never silently selects a changing `main` branch.

| Release | Use | Accepted mask records | Pinned revision |
|---|---|---:|---|
| `paper` | Original 34-dataset CFR evaluation resource and original extractor | 457,238 | [`7655704d…`](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks/tree/7655704d058af7fa4ed5e3b901cff1d521d7ede7) |
| `standardized` | Updated extraction procedure, including PolarBearVidID; 34 datasets | 462,428 | [`3baa6a58…`](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks/tree/3baa6a58da04cc18bf5e21f77cf0d366efe1e5f7) |

These snapshots are currently available through [archive PR #2](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks/discussions/2) and [PolarBearVidID PR #1](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks/discussions/1). Permanent release tags are pending. The pinned commits are downloadable now and do not depend on either PR being merged. The two collections have different dataset coverage and annotation populations; updated masks do not reproduce the original inputs.

## Download

From the repository root:

```bash
python scripts/download_masks.py --release paper --list
python scripts/download_masks.py --release standardized --datasets NyalaData PolarBearVidID --dry-run

pip install huggingface_hub
python scripts/download_masks.py --release paper --datasets NyalaData SeaTurtleID2022
python scripts/download_masks.py --release standardized --datasets NyalaData PolarBearVidID
```

`--list` and `--dry-run` need only Python and work offline. Use `--all` instead of `--datasets` to download a complete collection. `--output-dir` defaults to `data/masks`; the downloader keeps versions in separate `paper/` and `standardized/` subdirectories. The original folder aliases, such as `seaturtle_head`, are also accepted as input.

Only selected dataset files and their release documentation/checksums are downloaded. Each payload is verified against that revision's `SHA256SUMS`. Already verified files are reused; an unexpected revision or modified local file causes an error instead of being overwritten.

## Files and evaluation

| Release | Dataset files under its download directory |
|---|---|
| `paper` | `Masks_data/<archive_key>/annotation_masks_rle.json.gz` |
| `standardized` | `data/<Dataset>/masks.json.gz`, `metadata.parquet`, `summary.json` |

The paper download also includes the original dataset-index template and root `summary.json`, which maps official names to historical folder keys. Add `--include-candidates` to download its auxiliary `annotation_best_any_iou_masks_rle.json.gz` files; these audit candidates are not evaluation masks. Two empty accepted masks are retained unchanged in the historical collection.

Set the evaluator's `annotation_masks_rle` column to the downloaded accepted-mask file, or use `central_mask_dir` for the paper directory. **Matching source metadata is still required:** original images, identities, annotation IDs and crop geometry must agree with the chosen masks. The paper archive does not supply those source mappings. The updated `metadata.parquet` documents its own input geometry; it is not a replacement mapping for the historical masks.

Use the original extractor instructions for paper-pipeline regeneration and the standardized extractor instructions for new extraction. See [mask generation](mask_generation.md). Exact historical results require the matching prepared inputs and evaluation protocol as well as the frozen masks.

**Six datasets with source-provided masks or foregrounds:** BirdIndividualID, CzechLynx, FriesianCattle2015, NDD20, SMALST, and SeaTurtleID2022. The historical archive retains our original pipeline-generated masks for these datasets; the standardized collection excludes them. For new work on those datasets, use the provider's masks, segmentation annotations or foreground images.
