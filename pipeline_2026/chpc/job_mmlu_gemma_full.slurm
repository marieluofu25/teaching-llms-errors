#!/bin/bash
#SBATCH --job-name=tll-mmlu-gemma
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-mmlu-gemma-%j.out
#SBATCH --error=slurm-mmlu-gemma-%j.err

set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT to your clone path}"

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh" ]] && source "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh"

export PYTHONPATH="${REPO_ROOT}/pipeline_2026${PYTHONPATH:+:$PYTHONPATH}"

export EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}"
export EXPORT_FP16="${EXPORT_FP16:-1}"

# Gemma-2-9B + Gemma Scope (see docs/sae_checkpoints.md)
export EXPORT_MODEL="${EXPORT_MODEL:-google/gemma-2-9b}"
export EXPORT_LAYER_INDEX="${EXPORT_LAYER_INDEX:-}"
export SAE_RELEASE="${SAE_RELEASE:-gemma-scope-9b-pt-res-canonical}"
export SAE_ID="${SAE_ID:-layer_20/width_16k/canonical}"

export GEMMA_PREDICTIONS_CSV="${GEMMA_PREDICTIONS_CSV:-${REPO_ROOT}/pipeline_2026/stage2/s01_residual/results/mmlu_gemma_predictions_merged.csv}"
export RELEASE_THRESHOLDS_YAML="${RELEASE_THRESHOLDS_YAML:-${REPO_ROOT}/pipeline_2026/config/release_thresholds_gemma_full.yaml}"
export EXPORT_META_RELATIVE_TO="${EXPORT_META_RELATIVE_TO:-${REPO_ROOT}}"

bash "${REPO_ROOT}/pipeline_2026/scripts/run_pipeline_2026.sh" mmlu-gemma-full
