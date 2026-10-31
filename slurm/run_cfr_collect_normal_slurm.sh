#!/usr/bin/env bash
# Example Slurm wrapper for collecting CFR result JSONs into CSV summaries.

#SBATCH --job-name=cfr_collect
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_CPU_OR_GPU_PARTITION
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/cfr_collect_%j.out
#SBATCH --error=logs/cfr_collect_%j.err

set -euo pipefail

REPO_DIR="${CFR_CODE_ROOT:-$(pwd)}"
OUTPUT_ROOT="${CFR_OUTPUT_ROOT:-${REPO_DIR}/outputs/cfr}"
CENTRAL_INDEX="${CFR_CENTRAL_INDEX:-${REPO_DIR}/configs/cfr_dataset_index.template.tsv}"

cd "${REPO_DIR}"

python scripts/summarize_cfr_results.py \
  --results_root "${OUTPUT_ROOT}" \
  --central_index "${CENTRAL_INDEX}" \
  --suite all \
  --models miewid megadescriptor \
  --ks 1 5 10 \
  --output_name cfr_summary_reference

python scripts/recompute_paper_cfr_scores.py \
  --input_csv "${OUTPUT_ROOT}/cfr_summary_reference_k1.csv" \
  --output_csv "${OUTPUT_ROOT}/cfr_results_compact_k1.csv" \
  --k 1
