#!/usr/bin/env bash
# Example Slurm array wrapper for SAM mask extraction.

#SBATCH --job-name=cfr_masks
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cfr_masks_%A_%a.out
#SBATCH --error=logs/cfr_masks_%A_%a.err

set -euo pipefail

REPO_DIR="${CFR_CODE_ROOT:-$(pwd)}"
MANIFEST="${MANIFEST:-${REPO_DIR}/configs/sam_mask_jobs.template.jsonl}"

if [[ -n "${SAM_ENV_ACTIVATE:-}" ]]; then
  source "${SAM_ENV_ACTIVATE}"
fi

cd "${REPO_DIR}"

python scripts/run_wildlifedatasets_sam_mask_job.py \
  --manifest "${MANIFEST}" \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --min_sam_score "${MIN_SAM_SCORE:-0.0}" \
  --accept_iou "${ACCEPT_IOU:-0.5}" \
  --candidate_mask_mode "${CANDIDATE_MASK_MODE:-box_crop}" \
  --save_candidate_masks
