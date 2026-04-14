#!/bin/bash
#SBATCH --job-name=tll-mmlu-gemma-pipe
# One job: Gemma MMLU inference (optional sharded, sequential on this GPU) → merge (if needed) → mmlu-gemma-full.
# Edit #SBATCH account/partition/qos for CHPC (e.g. granite-gpu-guest).
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-mmlu-gemma-pipe-%j.out
#SBATCH --error=slurm-mmlu-gemma-pipe-%j.err

set -euo pipefail
# Slurm often does not forward the submit shell's env; SLURM_SUBMIT_DIR is the cwd when sbatch ran.
: "${REPO_ROOT:=${SLURM_SUBMIT_DIR:-}}"
: "${REPO_ROOT:?Set REPO_ROOT or run sbatch from the repository root (cd teaching-llms-errors first)}"

# Non-array Slurm jobs do not set SLURM_ARRAY_*; `env.local.sh` or other snippets
# that reference them under `set -u` would otherwise abort (e.g. unbound SLURM_ARRAY_JOB_ID).
: "${SLURM_JOB_ID:=0}"
export SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "${REPO_ROOT}"

# GH200 / Grace nodes are aarch64; a venv built on Granite (x86_64 login) cannot run here
# ("cannot execute binary file: Exec format error"). Drop that venv from PATH if sbatch exported it.
_drop_path_entry() {
  local drop="$1" p cleaned="" first=1
  local -a parts=()
  IFS=':' read -r -a parts <<< "${PATH:-}"
  for p in "${parts[@]}"; do
    [[ -z "$p" || "$p" == "$drop" ]] && continue
    if (( first )); then cleaned="$p"; first=0; else cleaned+=":$p"; fi
  done
  PATH="$cleaned"
  export PATH
}
if [[ "$(uname -m)" == aarch64 ]]; then
  _drop_path_entry "${HOME}/venvs/teaching-llms-errors/bin"
  if [[ "${VIRTUAL_ENV:-}" == "${HOME}/venvs/teaching-llms-errors" ]]; then
    unset VIRTUAL_ENV
  fi
fi
unset -f _drop_path_entry

# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh" ]] && source "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh"

# Optional: export TLL_SLURM_MODULES="cuda/12.x python/3.12.4" in env.local.sh (space-separated).
if [[ -n "${TLL_SLURM_MODULES:-}" ]] && command -v module &>/dev/null; then
  read -r -a _tl_mods <<< "${TLL_SLURM_MODULES}"
  module load "${_tl_mods[@]}"
  unset _tl_mods
fi
hash -r

export PYTHONPATH="${REPO_ROOT}/pipeline_2026${PYTHONPATH:+:$PYTHONPATH}"

MERGED="${GEMMA_PREDICTIONS_CSV:-${REPO_ROOT}/pipeline_2026/stage2/s01_residual/results/mmlu_gemma_predictions_merged.csv}"
OUT_DIR="${GEMMA_PRED_OUT_DIR:-${REPO_ROOT}/pipeline_2026/stage2/s01_residual/results/gemma_shards}"
JOB_TAG="${SLURM_JOB_ID:-local}"
TOTAL_ROWS="${TOTAL_MMLU_ROWS:-7851}"
N_SHARDS="${N_SHARDS:-1}"

# --- Phase A: inference (skip if SKIP_INFER=1 and merged file exists) ---
if [[ "${SKIP_INFER:-0}" == "1" ]]; then
  if [[ ! -f "${MERGED}" ]]; then
    echo "SKIP_INFER=1 but missing ${MERGED}" >&2
    exit 1
  fi
  echo "==> Skip inference (SKIP_INFER=1), using ${MERGED}"
else
  echo "==> Phase A: Gemma MMLU inference (N_SHARDS=${N_SHARDS})"
  mkdir -p "${OUT_DIR}"
  if [[ "${N_SHARDS}" -eq 1 ]]; then
    python3 -m stage2.s01_residual.code.run_gemma_mmlu_inference \
      --model "${GEMMA_MODEL:-google/gemma-2-9b}" \
      --output-csv "${MERGED}" \
      --micro-batch-size "${GEMMA_MICRO_BATCH:-2}" \
      --dtype "${GEMMA_DTYPE:-bfloat16}" \
      --device cuda
  else
    for ((i = 0; i < N_SHARDS; i++)); do
      CHUNK=$(( (TOTAL_ROWS + N_SHARDS - 1) / N_SHARDS ))
      ROW_START=$(( i * CHUNK ))
      ROW_END=$(( ROW_START + CHUNK ))
      if (( ROW_END > TOTAL_ROWS )); then ROW_END="${TOTAL_ROWS}"; fi
      OUT_CSV="${OUT_DIR}/predictions_${JOB_TAG}_${i}.csv"
      echo "    shard ${i}: rows ${ROW_START}-${ROW_END} → ${OUT_CSV}"
      python3 -m stage2.s01_residual.code.run_gemma_mmlu_inference \
        --model "${GEMMA_MODEL:-google/gemma-2-9b}" \
        --output-csv "${OUT_CSV}" \
        --micro-batch-size "${GEMMA_MICRO_BATCH:-2}" \
        --dtype "${GEMMA_DTYPE:-bfloat16}" \
        --device cuda \
        --row-start "${ROW_START}" \
        --row-end "${ROW_END}"
    done
    SHARD_FILES=()
    for ((i = 0; i < N_SHARDS; i++)); do
      SHARD_FILES+=("${OUT_DIR}/predictions_${JOB_TAG}_${i}.csv")
    done
    echo "==> Merge ${#SHARD_FILES[@]} shards → ${MERGED}"
    python3 -m stage2.s01_residual.code.merge_mmlu_prediction_shards \
      --shards "${SHARD_FILES[@]}" \
      --output "${MERGED}"
  fi
fi

if [[ ! -f "${MERGED}" ]]; then
  echo "Missing predictions CSV: ${MERGED}" >&2
  exit 1
fi

# --- Phase B: export + SAE + diff + eval + report ---
export GEMMA_PREDICTIONS_CSV="${MERGED}"
export EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}"
export EXPORT_FP16="${EXPORT_FP16:-1}"
export EXPORT_MODEL="${EXPORT_MODEL:-google/gemma-2-9b}"
export EXPORT_LAYER_INDEX="${EXPORT_LAYER_INDEX:-}"
export SAE_RELEASE="${SAE_RELEASE:-gemma-scope-9b-pt-res-canonical}"
export SAE_ID="${SAE_ID:-layer_20/width_16k/canonical}"
export RELEASE_THRESHOLDS_YAML="${RELEASE_THRESHOLDS_YAML:-${REPO_ROOT}/pipeline_2026/config/release_thresholds_gemma_full.yaml}"
export EXPORT_META_RELATIVE_TO="${EXPORT_META_RELATIVE_TO:-${REPO_ROOT}}"

echo "==> Phase B: mmlu-gemma-full"
bash "${REPO_ROOT}/pipeline_2026/scripts/run_pipeline_2026.sh" mmlu-gemma-full
