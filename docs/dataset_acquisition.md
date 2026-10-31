# Source images and annotations

Get source images and annotations from the original providers. [WildlifeDatasets](https://wildlifedatasets.github.io/wildlife-datasets/) supports acquisition and catalogues for many collections; some require Kaggle access or a separate provider download. Source links are listed in the [HF dataset card](https://huggingface.co/datasets/imageomics/cfr-wildlife-masks).

For supported catalogue classes:

```bash
python scripts/download_wildlifedatasets_catalogues.py \
  --root data/WildlifeDatasets --stats_root outputs/catalogues \
  --datasets CzechLynx ReunionTurtles AmvrakikosTurtles
```

Class availability and provider access depend on the installed toolkit version. Keep credentials in the provider's normal authentication mechanism, outside this repository.

## Match a mask version

The historical masks use the original prepared evaluation IDs and crop geometry. A fresh WildlifeDatasets catalogue may not reproduce those identifiers or preprocessing. The updated HF collection supplies per-annotation source references and geometry in `metadata.parquet`, but identity labels still come from the source annotations. Verify the join and coordinate frame before evaluation; do not join by row order.

For the current collection, datasets that already supply masks or foregrounds should use those assets. The historical archive retains original pipeline-generated masks for six such datasets to preserve the original resource; its card lists them explicitly.

Create a local dataset index using [the template](../configs/cfr_dataset_index.template.tsv). Set `raw_image_root`, `central_metadata_input`, and `central_mask_dir`, or provide an explicit `annotation_masks_rle` column. Keep only the rows you intend to evaluate. Local populated indices and generated data are ignored by Git.

See [mask downloads](mask_releases.md) for version-specific paths and [evaluation inputs](artifact_manifest.md) for metadata fields and geometry requirements. No source images, identity catalogues, model weights, or generated evaluation results are bundled with the code.
