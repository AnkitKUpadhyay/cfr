#!/usr/bin/env bash
#SBATCH --job-name=cfr_rob_smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${CFR_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CENTRAL_INDEX="${CFR_CENTRAL_INDEX:?Set CFR_CENTRAL_INDEX.}"
CLEAN_ROOT="${CFR_CLEAN_ROOT:?Set CFR_CLEAN_ROOT.}"
OUTPUT_ROOT="${CFR_OUTPUT_ROOT:-${REPO_DIR}/outputs/cfr_robustness_smoke}"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-cfr}"
fi
cd "${REPO_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

python - <<'PY'
import faiss
import torch
print("FAISS:", faiss.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY

python scripts/run_cfr_robustness.py \
  --central_index "${CENTRAL_INDEX}" \
  --clean_embedding_root "${CLEAN_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --suite custom \
  --datasets "${CFR_SMOKE_DATASET:-reunionturtles}" \
  --models "${CFR_SMOKE_MODEL:-miewid}" \
  --mode extract \
  --perturbation gaussian_noise \
  --severity medium \
  --severity_value 0 \
  --limit "${CFR_SMOKE_LIMIT:-16}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --device "${DEVICE:-cuda:0}" \
  --verify_clean_embedding_parity

if [[ "${CFR_RUN_CONDITION_SMOKE:-0}" == "1" ]]; then
  python scripts/run_cfr_robustness.py \
    --central_index "${CENTRAL_INDEX}" \
    --clean_embedding_root "${CLEAN_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --suite custom \
    --datasets "${CFR_SMOKE_DATASET:-reunionturtles}" \
    --models "${CFR_SMOKE_MODEL:-miewid}" \
    --mode extract_eval \
    --perturbation mask_exterior_false_positive \
    --severity medium \
    --ks 1 \
    --batch_size "${BATCH_SIZE:-8}" \
    --num_workers "${NUM_WORKERS:-2}" \
    --retrieval_batch_size 1024 \
    --device "${DEVICE:-cuda:0}" \
    --require_faiss \
    --remove_condition_embeddings_after_eval
fi
