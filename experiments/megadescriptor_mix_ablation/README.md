# MegaDescriptor Foreground-Mixing Ablation

This directory contains the isolated evaluator and analysis utilities for the
revision experiment. It does not modify the main CFR evaluator or reuse its
output directory.

## Compared Checkpoints

The matched baseline, `megad_full22`, is MegaD-C trained with full images only
on the same 22-dataset training collection. It uses 0% foreground-only inputs.
Three checkpoints use the same architecture and training collection while
sampling foreground-only inputs with probability 0.25, 0.50, or 0.75. All
remaining samples use the full image. If a selected foreground sample has no
accepted mask, these runs fall back to its full image.

The checkpoint files are not included in this repository. Copy the two registry
templates and fill in local checkpoint paths:

```bash
cp experiments/megadescriptor_mix_ablation/configs/baseline_registry.template.tsv \
  /private/path/full_image_baseline.tsv
cp experiments/megadescriptor_mix_ablation/configs/model_registry.template.tsv \
  /private/path/foreground_mix_models.tsv
```

Do not commit populated registries. Evaluation JSON records checkpoint file
names and model keys, but never absolute checkpoint paths.

## Evaluation Protocol

All four checkpoints use the same settings:

- all 34 mask-accepted datasets
- the same accepted annotations and annotation crops
- full, foreground, background, and silhouette inputs
- the same all-vs-all gallery and query exclusion
- top-1 retrieval, SSI, and CFR using the final paper formulas
- silhouette excluded from SSI for whale-style datasets where contour is an
  intended identity cue

The collector reports separately the 22 datasets represented in the training
collection and the following 12 unseen datasets:

`SeaTurtle head`, `StripeSpotter`, `HyenaID2022`, `NyalaData`, `Reunion
turtles`, `Amvrakikos turtles`, `Melops`, `CzechLynx`, `BrownBearHeads`,
`Whale shark`, `Leopard`, and `Saimaa seal`.

## Run the Sweep

First validate each local registry:

```bash
python experiments/megadescriptor_mix_ablation/preflight_checkpoints.py \
  --model_registry /private/path/foreground_mix_models.tsv
```

The foreground-mixed sweep contains `34 x 3 = 102` tasks:

```bash
cd /path/to/cfr-code
mkdir -p /path/to/ablation/logs

sbatch \
  --account=<ACCOUNT> \
  --partition=<GPU_PARTITION> \
  --array=0-101%20 \
  --output=/path/to/ablation/logs/megad_mix_%A_%a.out \
  --error=/path/to/ablation/logs/megad_mix_%A_%a.err \
  --export=ALL,CFR_CODE_ROOT=$PWD,CFR_CENTRAL_INDEX=/path/to/dataset_index.tsv,CFR_ABLATION_MODEL_REGISTRY=/private/path/foreground_mix_models.tsv,CFR_OUTPUT_ROOT=/path/to/ablation \
  experiments/megadescriptor_mix_ablation/slurm/run_mix_ablation_array.sh
```

Evaluate the full-image baseline with the same wrapper, its one-row registry,
and `--array=0-33%20`. Store both arrays under the same output root. Distinct
model keys prevent output collisions. Set `CFR_DRY_RUN=1` to inspect task
mapping without loading a checkpoint or writing results.

## Collect and Plot

```bash
python experiments/megadescriptor_mix_ablation/collect_mix_ablation.py \
  --central_index /path/to/dataset_index.tsv \
  --baseline_registry /private/path/full_image_baseline.tsv \
  --model_registry /private/path/foreground_mix_models.tsv \
  --ablation_root /path/to/ablation \
  --output_dir outputs/megadescriptor_mix_ablation

python experiments/megadescriptor_mix_ablation/summarize_mix_ablation.py \
  --detailed_csv outputs/megadescriptor_mix_ablation/megadescriptor_mix_ablation_detailed_k1.csv \
  --output_dir outputs/megadescriptor_mix_ablation
```

The collector expects `34 x 4 = 136` complete results unless
`--allow_missing` is used for a progress check. The summarizer applies the
Figure 6 descriptive boundaries, writes quadrant transitions and counts, and
creates macro and dataset-movement figures. Generated CSVs, LaTeX tables, and
figures belong under `outputs/` and are intentionally not versioned.
