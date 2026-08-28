#!/usr/bin/env bash
# One task per (dataset, retrained checkpoint). For 34 datasets and three
# foreground-mixed models, submit --array=0-101. Command-line sbatch options
# provide account and partition.

#SBATCH --job-name=megad_mix
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${CFR_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CENTRAL_INDEX="${CFR_CENTRAL_INDEX:?Set CFR_CENTRAL_INDEX to the populated dataset index.}"
MODEL_REGISTRY="${CFR_ABLATION_MODEL_REGISTRY:?Set CFR_ABLATION_MODEL_REGISTRY to the private TSV.}"
OUTPUT_ROOT="${CFR_OUTPUT_ROOT:?Set CFR_OUTPUT_ROOT to a new ablation output directory.}"

mkdir -p "${OUTPUT_ROOT}/logs"
if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-cfr}"
fi
cd "${REPO_DIR}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

mapfile -t DATASETS < <(
  python - "${CENTRAL_INDEX}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if int(float(row.get("annotations_accepted") or 0)) > 0:
            print(row["dataset"])
PY
)

mapfile -t MODELS < <(
  python - "${MODEL_REGISTRY}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if (row.get("key") or "").strip():
            print(row["key"].strip())
PY
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_MODELS="${#MODELS[@]}"
DATASET_INDEX=$((TASK_ID / NUM_MODELS))
MODEL_INDEX=$((TASK_ID % NUM_MODELS))

if (( DATASET_INDEX >= ${#DATASETS[@]} )); then
  echo "Task ${TASK_ID} is outside ${#DATASETS[@]} datasets x ${NUM_MODELS} models." >&2
  exit 2
fi

DATASET="${DATASETS[$DATASET_INDEX]}"
MODEL="${MODELS[$MODEL_INDEX]}"

echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Protocol: accepted-mask annotations, bbox crops, all-vs-all gallery, k=1."

EXTRA_ARGS=()
if [[ "${CFR_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry_run)
fi

python experiments/megadescriptor_mix_ablation/run_mix_ablation_evaluation.py \
  --central_index "${CENTRAL_INDEX}" \
  --model_registry "${MODEL_REGISTRY}" \
  --suite custom \
  --datasets "${DATASET}" \
  --models "${MODEL}" \
  --mode extract_eval \
  --output_root "${OUTPUT_ROOT}" \
  --ks 1 \
  --crop_mode bbox \
  --shortcut_views background silhouette \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" \
  --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE:-1024}" \
  --device "${DEVICE:-cuda:0}" \
  --no_root_summaries \
  "${EXTRA_ARGS[@]}"
