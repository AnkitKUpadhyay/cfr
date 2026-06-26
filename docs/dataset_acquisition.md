# Dataset Acquisition Notes

The code repository is designed to be path-configurable, not to redistribute
source images. Reviewers can use the scripts directly after obtaining the
source datasets and editing `configs/cfr_dataset_index.tsv`.

## What Can Be Downloaded Automatically

`scripts/download_wildlifedatasets_catalogues.py` uses the
`wildlife-datasets` Python package to download/export catalogues for datasets
that package exposes. This covers several of the WildlifeDatasets-backed
collections used in the paper, such as CzechLynx, ReunionTurtles,
AmvrakikosTurtles, ZakynthosTurtles, Melops, and BrownBearHeads when available
through the installed package version or the local wrapper.

Example:

```bash
python scripts/download_wildlifedatasets_catalogues.py \
  --root data/WildlifeDatasets \
  --stats_root outputs/catalogues \
  --datasets CzechLynx ReunionTurtles AmvrakikosTurtles Melops BrownBearHeads
```

After download, update `raw_image_root` and `central_metadata_input` in
`configs/cfr_dataset_index.tsv`.

## What Requires Separate Source Access

Not every dataset in the 34-dataset CFR suite can be fetched by one anonymous
script. Some collections require:

- original dataset portals,
- Kaggle credentials,
- LILA/Wild Me downloads,
- locally converted COCO/GZCD metadata,
- or source-specific access conditions.

For those datasets, the release provides code, path templates, prompt metadata,
and expected metadata/mask formats. The raw images should be obtained from the
original source and cited according to the paper.

## Direct Use With the CFR Scripts

WildlifeDatasets download is not, by itself, the whole CFR input.  It provides
the source images and a catalogue.  The CFR evaluator then needs a row in
`configs/cfr_dataset_index.tsv` that points to:

1. source images,
2. the metadata file listed in the dataset index,
3. `annotation_masks.pkl` in each `central_mask_dir`,

with metadata identifiers that match the keys in `annotation_masks.pkl`.
The evaluator can read CSV or JSON metadata.  For CSV files it expects an image
path column such as `path`, `image_path`, or `file_name`, and an identity column
such as `identity`, `name`, `individual_id`, or `individual`.

Once those paths and identifiers are aligned, the evaluator can be run directly:

```bash
python scripts/run_cfr_evaluation.py \
  --central_index configs/cfr_dataset_index.tsv \
  --suite all \
  --models miewid megadescriptor \
  --mode extract_eval \
  --output_root outputs/cfr \
  --ks 1 5 10 \
  --device cuda:0
```

If the separate mask artifact is provided, reviewers do not need to rerun SAM
to reproduce the CFR retrieval tables. They only need the source images and the
accepted-mask artifact.

In short: for WildlifeDatasets-backed datasets, the intended workflow is
download/export catalogue, point the dataset-index row to that image root and
metadata file, add the released `annotation_masks.pkl`, then run CFR.  Without
the released mask artifact, reviewers must first run the SAM extraction scripts
to create the accepted masks.
