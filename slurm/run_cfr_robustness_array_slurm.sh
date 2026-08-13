#!/usr/bin/env bash
#SBATCH --job-name=cfr_rob
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${CFR_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CENTRAL_INDEX="${CFR_CENTRAL_INDEX:?Set CFR_CENTRAL_INDEX to the populated dataset index.}"
CLEAN_ROOT="${CFR_CLEAN_ROOT:?Set CFR_CLEAN_ROOT to the immutable Table S3 embedding root.}"
OUTPUT_ROOT="${CFR_OUTPUT_ROOT:-${REPO_DIR}/outputs/cfr_robustness}"
CONDITIONS_FILE="${CFR_CONDITIONS_FILE:-${REPO_DIR}/configs/cfr_robustness_conditions.tsv}"
LEVEL="${CFR_LEVEL:-medium}"

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

mapfile -t CONDITIONS < <(
  python - "${CONDITIONS_FILE}" "${LEVEL}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row["severity"] == sys.argv[2]:
            print("\t".join([row["perturbation"], row["severity"], row["seed"], row["scope"]]))
PY
)

MODELS=(miewid megadescriptor)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASKS_PER_CONDITION=$(( ${#DATASETS[@]} * ${#MODELS[@]} ))
CONDITION_INDEX=$((TASK_ID / TASKS_PER_CONDITION))
WITHIN_CONDITION=$((TASK_ID % TASKS_PER_CONDITION))
DATASET_INDEX=$((WITHIN_CONDITION / ${#MODELS[@]}))
MODEL_INDEX=$((WITHIN_CONDITION % ${#MODELS[@]}))

if (( CONDITION_INDEX >= ${#CONDITIONS[@]} )); then
  echo "Task ${TASK_ID} is outside ${#CONDITIONS[@]} conditions x ${TASKS_PER_CONDITION} tasks." >&2
  exit 2
fi

IFS=$'\t' read -r PERTURBATION SEVERITY SEED SCOPE <<< "${CONDITIONS[$CONDITION_INDEX]}"
DATASET="${DATASETS[$DATASET_INDEX]}"
MODEL="${MODELS[$MODEL_INDEX]}"

echo "Condition: ${PERTURBATION}/${SEVERITY}/seed=${SEED}/scope=${SCOPE}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Clean embeddings: ${CLEAN_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Protocol: Table S3 crop/mask/gallery/model settings; image corruptions use clean galleries."

python scripts/run_cfr_robustness.py \
  --central_index "${CENTRAL_INDEX}" \
  --clean_embedding_root "${CLEAN_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --suite custom \
  --datasets "${DATASET}" \
  --models "${MODEL}" \
  --mode extract_eval \
  --perturbation "${PERTURBATION}" \
  --severity "${SEVERITY}" \
  --seed "${SEED}" \
  --scope "${SCOPE}" \
  --ks 1 \
  --crop_mode bbox \
  --shortcut_views background silhouette \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}" \
  --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE:-1024}" \
  --device "${DEVICE:-cuda:0}" \
  --require_faiss \
  --remove_condition_embeddings_after_eval
