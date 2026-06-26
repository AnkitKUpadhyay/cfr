#!/usr/bin/env bash
# Example Slurm array wrapper for CFR extraction plus evaluation.
#
# Submit after editing ACCOUNT/PARTITION and array size. For the paper run,
# array size was number_of_datasets x number_of_models.

#SBATCH --job-name=cfr_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cfr_eval_%A_%a.out
#SBATCH --error=logs/cfr_eval_%A_%a.err

set -euo pipefail

REPO_DIR="${CFR_CODE_ROOT:-$(pwd)}"
OUTPUT_ROOT="${CFR_OUTPUT_ROOT:-${REPO_DIR}/outputs/cfr}"
CENTRAL_INDEX="${CFR_CENTRAL_INDEX:-${REPO_DIR}/configs/cfr_dataset_index.template.tsv}"
LOG_DIR="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-cfr}"
fi

cd "${REPO_DIR}"

export DEVICE="${DEVICE:-cuda:0}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
export RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-1024}"
export CFR_SUITE="${CFR_SUITE:-all}"
export CFR_MODELS="${CFR_MODELS:-miewid megadescriptor}"
export CFR_KS="${CFR_KS:-1 5 10}"
export CFR_EXTRA_ARGS="${CFR_EXTRA_ARGS:-}"

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

read -r -a MODELS <<< "${CFR_MODELS//,/ }"
read -r -a KS_ARGS <<< "${CFR_KS}"
read -r -a EXTRA_ARGS <<< "${CFR_EXTRA_ARGS}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_MODELS="${#MODELS[@]}"
DATASET_INDEX=$(( TASK_ID / NUM_MODELS ))
MODEL_INDEX=$(( TASK_ID % NUM_MODELS ))

if (( DATASET_INDEX >= ${#DATASETS[@]} )); then
  echo "Task ${TASK_ID} is out of range." >&2
  exit 2
fi

DATASET="${DATASETS[$DATASET_INDEX]}"
MODEL="${MODELS[$MODEL_INDEX]}"

python scripts/run_cfr_evaluation.py \
  --central_index "${CENTRAL_INDEX}" \
  --suite custom \
  --datasets "${DATASET}" \
  --models "${MODEL}" \
  --mode extract_eval \
  --output_root "${OUTPUT_ROOT}" \
  --ks "${KS_ARGS[@]}" \
  --shortcut_views background silhouette \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
