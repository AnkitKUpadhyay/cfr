#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${CFR_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CENTRAL_INDEX="${CFR_CENTRAL_INDEX:?Set CFR_CENTRAL_INDEX to the populated dataset index.}"
CLEAN_ROOT="${CFR_CLEAN_ROOT:?Set CFR_CLEAN_ROOT to the immutable Table S3 embedding root.}"
OUTPUT_ROOT="${CFR_OUTPUT_ROOT:-${REPO_DIR}/outputs/cfr_robustness}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT.}"
SLURM_PARTITION="${SLURM_PARTITION:?Set SLURM_PARTITION.}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
if (( $# > 0 )); then
  LEVELS=("$@")
else
  LEVELS=(medium)
fi

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_DIR}"

for level in "${LEVELS[@]}"; do
  condition_count=$(awk -F '\t' -v level="${level}" 'NR > 1 && $2 == level {count++} END {print count+0}' configs/cfr_robustness_conditions.tsv)
  dataset_count=$(awk -F '\t' 'NR > 1 && ($11+0) > 0 {count++} END {print count+0}' "${CENTRAL_INDEX}")
  task_count=$((condition_count * dataset_count * 2))
  last_task=$((task_count - 1))
  echo "Submitting ${level}: ${condition_count} conditions x ${dataset_count} datasets x 2 models = ${task_count} tasks"
  sbatch \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --output="${OUTPUT_ROOT}/logs/cfr_rob_%A_%a.out" \
    --error="${OUTPUT_ROOT}/logs/cfr_rob_%A_%a.err" \
    --array="0-${last_task}%${MAX_CONCURRENT}" \
    --export="ALL,CFR_LEVEL=${level},CFR_CODE_ROOT=${REPO_DIR},CFR_CENTRAL_INDEX=${CENTRAL_INDEX},CFR_CLEAN_ROOT=${CLEAN_ROOT},CFR_OUTPUT_ROOT=${OUTPUT_ROOT}" \
    slurm/run_cfr_robustness_array_slurm.sh
done
