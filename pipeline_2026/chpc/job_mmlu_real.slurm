#!/bin/bash
#SBATCH --job-name=tll-mmlu-real
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
## CHPC: set partition and account to valid values for your project, e.g.:
## #SBATCH --partition=notchpeak-gpu
## #SBATCH --account=your_account

set -euo pipefail

# Absolute path to teaching-llms-errors repo root (edit or export before sbatch).
: "${REPO_ROOT:?Set REPO_ROOT to the clone path, e.g. export REPO_ROOT=\$HOME/teaching-llms-errors}"

cd "${REPO_ROOT}"

# Optional: source secrets and model/SAE overrides (not in git).
# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh" ]] && source "${REPO_ROOT}/pipeline_2026/chpc/env.local.sh"

export PYTHONPATH="${REPO_ROOT}/pipeline_2026${PYTHONPATH:+:$PYTHONPATH}"

# GPU pipeline (override in env.local.sh or sbatch --export)
export EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}"
export EXPORT_FP16="${EXPORT_FP16:-1}"

# Example Gemma-oriented overrides (comment out to keep repo defaults: gpt2 + gpt2-small-res-jb):
# export EXPORT_MODEL="google/gemma-2-2b"
# export EXPORT_LAYER_INDEX=12
# export SAE_RELEASE="gemma-scope-2b-pt-res-canonical"
# export SAE_ID="layer_12/width_16k/canonical"

bash "${REPO_ROOT}/pipeline_2026/scripts/run_pipeline_2026.sh" mmlu-real
