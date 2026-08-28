# Counterfactual Foreground Retrieval Code Release

This repository contains the code needed to reproduce the Counterfactual
Foreground Retrieval (CFR) evaluation layer from the WACV submission. It is
intended for anonymous review. Raw source images are not redistributed here;
users should obtain them from the original dataset sources and follow the
original licenses.

## What This Repository Covers

The paper promises the following release components:

- CFR evaluation code: `scripts/run_cfr_evaluation.py`
- Mask-generation code: `scripts/extract_wildlifedatasets_sam_masks.py`
- Mask-artifact conversion code: `scripts/convert_mask_pickles_to_coco_rle.py`
- Released RLE masks for the evaluated datasets: `Masks_data/<dataset>/`
- Dataset manifests and prompt metadata: `configs/*.template.*`
- Evaluation configs and Slurm examples: `slurm/*.sh`
- Table-generation scripts: `scripts/summarize_cfr_results.py` and
  `scripts/recompute_paper_cfr_scores.py`
- Paper result tables: `results/*.csv` and `results/*.tex`
- Revision robustness code: `scripts/run_cfr_robustness.py`
- Foreground-mixed MegaDescriptor ablation code:
  `experiments/megadescriptor_mix_ablation/`

The code computes the final paper definitions of CFR and SSI directly. The
released foreground masks are included in this repository as COCO-style
compressed RLE JSON files. Their directory layout is described in
`docs/artifact_manifest.md`.
The released CFR table uses the public MiewID and MegaDescriptor models.

## Included Mask Resource

This repository includes accepted and best-candidate SAM masks for the 34
datasets used in the CFR evaluation:

```text
Masks_data/
  <dataset>/
    annotation_masks_rle.json.gz
    annotation_best_any_iou_masks_rle.json.gz
```

`annotation_masks_rle.json.gz` is the evaluation file: annotations without an
accepted mask have `present=false` and `segmentation=null`.
`annotation_best_any_iou_masks_rle.json.gz` stores the best available candidate
for each annotation, including candidates that did not pass the acceptance
rule. Raw source images and source metadata are not redistributed here; obtain
those from the original dataset sources.

## Main Pipeline

### 1. Prepare Dataset Paths

Copy the template index and edit the path columns:

```bash
cp configs/cfr_dataset_index.template.tsv configs/cfr_dataset_index.tsv
```

The evaluator expects the TSV columns in the template, including:

- `dataset`
- `raw_image_root`
- `central_metadata_input`
- `central_mask_dir`
- optionally `annotation_masks_rle`
- `records_total`
- `annotations_accepted`

The `central_mask_dir` for each dataset must contain either
`annotation_masks_rle.json.gz` or `annotation_masks.pkl`. For the included mask
resource, set `central_mask_dir` to `Masks_data/<dataset>`. If the TSV contains
an `annotation_masks_rle` column, that file is used directly.

If you have local pickle masks and want to create the anonymous RLE artifact,
use:

```bash
python scripts/convert_mask_pickles_to_coco_rle.py \
  --index configs/cfr_dataset_index.tsv \
  --output_root outputs/cfr_sam_mask_rle_release \
  --prepare_release \
  --convert
```

Some datasets can be obtained through the `wildlife-datasets` package, while
others require original-source downloads, Kaggle access, or author-provided
metadata. See `docs/dataset_acquisition.md`.

### 2. Optional: Extract SAM Masks

This step is only needed if you want to regenerate masks. To reproduce the CFR
retrieval tables, use the released accepted-mask artifact and skip to Step 3.

SAM extraction uses text prompts from `configs/sam_mask_jobs.template.jsonl`.
Edit the image, metadata, and output paths, then run one job in an environment
that provides SAM3 inference:

```bash
python scripts/run_wildlifedatasets_sam_mask_job.py \
  --manifest configs/sam_mask_jobs.template.jsonl \
  --index 0 \
  --device cuda:0 \
  --min_sam_score 0.0 \
  --accept_iou 0.5 \
  --candidate_mask_mode box_crop \
  --save_candidate_masks
```

For cluster runs, adapt `slurm/run_wildlifedatasets_sam_masks_slurm.sh`.

### 3. Run CFR Evaluation

For the public reference models used in the main result table:

```bash
python scripts/run_cfr_evaluation.py \
  --central_index configs/cfr_dataset_index.tsv \
  --suite all \
  --models miewid megadescriptor \
  --mode extract_eval \
  --output_root outputs/cfr \
  --ks 1 5 10 \
  --shortcut_views background silhouette \
  --batch_size 16 \
  --num_workers 4 \
  --retrieval_batch_size 1024 \
  --device cuda:0
```

The gallery is all accepted-mask annotations except the query annotation
itself. Same-image and same-encounter exclusions are optional flags and were
not used for the main table.

### 4. Collect Results

```bash
python scripts/summarize_cfr_results.py \
  --results_root outputs/cfr \
  --central_index configs/cfr_dataset_index.tsv \
  --suite all \
  --models miewid megadescriptor \
  --ks 1 5 10 \
  --output_name cfr_summary_reference
```

### 5. Optional: Recompute Paper SSI and CFR Columns

`run_cfr_evaluation.py` already writes the final paper definitions. If you
already have a saved summary CSV and want to recompute only the final columns
without rerunning retrieval, use `scripts/recompute_paper_cfr_scores.py`.
The final paper reports:

- `SSI = min(1, S_shortcut / S_full)`
- `CFR = min(A_full, A_fg)`

where `S_T = max(0, R_T - R_rand)`,
`A_full = S_full / (1 - R_rand)`, and
`A_fg = S_fg / (1 - R_rand)`.

Run:

```bash
python scripts/recompute_paper_cfr_scores.py \
  --input_csv outputs/cfr/cfr_summary_reference_k1.csv \
  --output_csv outputs/cfr/cfr_results_compact_k1.csv \
  --output_tex outputs/cfr/cfr_results_compact_k1.tex \
  --k 1
```

By default, silhouette is not counted as shortcut evidence for whale-style
datasets where contour can be valid identity evidence. Override this policy
with `--silhouette_identity_datasets`.

### 6. Optional: Robustness Analysis

The post-review robustness runner tests mask-quality sensitivity and degraded
field queries while preserving the Table S3 annotation crops, accepted record
set, neutral fill, gallery rule, public models, and final CFR/SSI formulas. See
`ROBUSTNESS_EXPERIMENTS.md` for the experimental design and
`configs/cfr_robustness_conditions.tsv` for the three severity levels.

Mask perturbations are applied to both query and gallery masks. Image
corruptions are applied to queries while the original clean embeddings form the
gallery. Run one condition with:

```bash
python scripts/run_cfr_robustness.py \
  --central_index configs/cfr_dataset_index.tsv \
  --clean_embedding_root outputs/cfr \
  --output_root outputs/cfr_robustness \
  --suite all \
  --models miewid megadescriptor \
  --perturbation gaussian_blur \
  --severity medium \
  --require_faiss
```

After the conditions finish, collect dataset-level and macro results with
`scripts/collect_cfr_robustness.py`. The collector writes paired dataset-level
CSVs and compact LaTeX summary tables for low, medium, and high severity. The
tables report changes in Full, CFR, and SSI together with rank correlations
against the clean evidence profiles. Stochastic perturbations use seeds 0, 1,
and 2; the additional seed conditions and aggregation script are documented in
`ROBUSTNESS_EXPERIMENTS.md`.

### 7. Optional: Foreground-Mixed Training Ablation

The revision also evaluates a matched MegaDescriptor family to test whether
foreground-only inputs are merely unfamiliar at inference time. The experiment
compares a full-image baseline with checkpoints trained using foreground-only
inputs for 25%, 50%, or 75% of selected samples. When a selected sample has no
accepted mask, these runs use its full image.

Evaluation, collection, and evidence-quadrant plotting code is under
`experiments/megadescriptor_mix_ablation/`. Checkpoint paths are supplied
through a private local TSV and are never copied into generated artifacts. See
the experiment README for the registry format and commands.

## Reproducing the Included CSVs

The included files under `results/` are the final CSV and LaTeX tables used for
paper drafting:

- `cfr_results_compact_k1.csv`: compact table columns for sharing.
- `cfr_results_detailed_k1.csv`: detailed table with final intermediate signals.
- `table_all_miewid_megad_ssi_cfr_formula_k1.tex`: LaTeX table.

Generated robustness and foreground-mixing result files are intentionally not
versioned. The paper reports those results, while the released scripts recreate
the corresponding CSVs, tables, and figures from local evaluation outputs.

## Dependencies

Install the Python packages in `requirements.txt`. Reproducing the CFR tables
requires the source images, the released accepted masks, and the MiewID and
MegaDescriptor dependencies. Regenerating masks is optional and requires a
SAM3-capable environment.

MiewID is loaded through Hugging Face with `trust_remote_code=True`.
MegaDescriptor is loaded through `timm`.
