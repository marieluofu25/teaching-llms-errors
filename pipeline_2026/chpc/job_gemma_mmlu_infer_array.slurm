#!/bin/bash
#SBATCH --job-name=gemma-mmlu-inf
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-gemma-inf-%A_%a.out
#SBATCH --error=slurm-gemma-inf-%A_%a.err
## Optional: #SBATCH --array=0-7
## CHPC: set partition/account per cluster policy

set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT to your clone path}"
: "${SLURM_ARRAY_TASK_ID:=0}"
# Must match Slurm array width (e.g. #SBATCH --array=0-7 → N_SHARDS=8).
: "${N_SHARDS:=1}"

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh" ]] && source "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh"

export PYTHONPATH="${REPO_ROOT}/pipeline_2026${PYTHONPATH:+:$PYTHONPATH}"

# Full pickle rows (~7851); split for this task
TOTAL_ROWS="${TOTAL_MMLU_ROWS:-7851}"
N="${N_SHARDS}"
TASK="${SLURM_ARRAY_TASK_ID}"
CHUNK=$(( (TOTAL_ROWS + N - 1) / N ))
ROW_START=$(( TASK * CHUNK ))
ROW_END=$(( ROW_START + CHUNK ))
if (( ROW_END > TOTAL_ROWS )); then ROW_END="${TOTAL_ROWS}"; fi

OUT_DIR="${GEMMA_PRED_OUT_DIR:-${REPO_ROOT}/pipeline_2026/stage2/s01_residual/results/gemma_shards}"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/predictions_${SLURM_ARRAY_JOB_ID}_${TASK}.csv"

python3 -m stage2.s01_residual.code.run_gemma_mmlu_inference \
  --model "${GEMMA_MODEL:-google/gemma-2-9b}" \
  --output-csv "${OUT_CSV}" \
  --micro-batch-size "${GEMMA_MICRO_BATCH:-2}" \
  --dtype "${GEMMA_DTYPE:-bfloat16}" \
  --device cuda \
  --row-start "${ROW_START}" \
  --row-end "${ROW_END}"

echo "Shard ${TASK}: rows ${ROW_START}-${ROW_END} → ${OUT_CSV}"
