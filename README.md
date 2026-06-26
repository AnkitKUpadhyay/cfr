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
- Dataset manifests and prompt metadata: `configs/*.template.*`
- Evaluation configs and Slurm examples: `slurm/*.sh`
- Table-generation scripts: `scripts/summarize_cfr_results.py` and
  `scripts/recompute_paper_cfr_scores.py`
- Paper result tables: `results/*.csv` and `results/*.tex`

The code computes the final paper definitions of CFR and SSI directly. The
large mask artifact is separate from this code repository. Its expected
directory layout is described in `docs/artifact_manifest.md`.
The released CFR table uses the public MiewID and MegaDescriptor models.

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
- `records_total`
- `annotations_accepted`

The `central_mask_dir` for each dataset must contain an
`annotation_masks.pkl` file.

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

## Reproducing the Included CSVs

The included files under `results/` are the final CSV and LaTeX tables used for
paper drafting:

- `cfr_results_compact_k1.csv`: compact table columns for sharing.
- `cfr_results_detailed_k1.csv`: detailed table with final intermediate signals.
- `table_all_miewid_megad_ssi_cfr_formula_k1.tex`: LaTeX table.

## Dependencies

Install the Python packages in `requirements.txt`. Reproducing the CFR tables
requires the source images, the released accepted masks, and the MiewID and
MegaDescriptor dependencies. Regenerating masks is optional and requires a
SAM3-capable environment.

MiewID is loaded through Hugging Face with `trust_remote_code=True`.
MegaDescriptor is loaded through `timm`.

