# CHPC (University of Utah) — running `pipeline_2026`

This folder holds **example Slurm** and **environment templates**. It does not contain secrets; copy [`env_chpc.example`](env_chpc.example) locally and fill in tokens, or use the repo-root [`.env.example`](../../.env.example) as the canonical list of variable names.

## Prerequisites

1. Clone the repo on CHPC and `cd` to the repository root (`teaching-llms-errors`).
2. Python 3.10+ with CUDA-enabled PyTorch (match your cluster CUDA module).
3. Install dependencies from `pipeline_2026/requirements.txt` inside a venv or conda env.

Example (adjust module names to your cluster documentation):

```bash
module load cuda/12.x   # or the version CHPC documents for your GPU node
module load python/3.11
python3 -m venv ~/venvs/teaching-llms-errors
source ~/venvs/teaching-llms-errors/bin/activate
pip install -U pip
pip install -r pipeline_2026/requirements.txt
```

## Hugging Face and SAELens

- Export and SAE encode download model weights and SAE checkpoints from the Hugging Face Hub.
- Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated models and reliable downloads (`huggingface-cli login` also works).
- First run may be slow while caches populate under `~/.cache/huggingface`.

## Submitting a job

1. Edit `job_mmlu_real.slurm`: set `#SBATCH` account/partition/GPU lines to match [CHPC Slurm](https://www.chpc.utah.edu/documentation/software/slurm.php) guidance.
2. Set `REPO_ROOT` in the script to your clone path (or export it before `sbatch`).
3. Optionally `source` a file derived from `env_chpc.example` to set `EXPORT_MODEL`, `SAE_RELEASE`, `SAE_ID`, etc.

```bash
export REPO_ROOT=/path/to/teaching-llms-errors
sbatch pipeline_2026/chpc/job_mmlu_real.slurm
```

## Environment variables (orchestrator)

[`scripts/run_pipeline_2026.sh`](../scripts/run_pipeline_2026.sh) mode `mmlu-real` respects:

| Variable | Role |
|----------|------|
| `EXPORT_DEVICE` | Passed through to export/encode (e.g. `cuda`) |
| `EXPORT_MODEL` | HF model id for `export_activations` |
| `EXPORT_LAYER_INDEX` | Must align with chosen SAE hook (see `docs/sae_checkpoints.md`) |
| `EXPORT_BATCH_SIZE`, `SAE_BATCH_SIZE` | Throughput tuning |
| `EXPORT_FP16` | `1` to enable `--fp16` on CUDA export |
| `SAE_RELEASE`, `SAE_ID` | SAELens pretrained SAE |
| `EXPORT_MAX_ROWS` | Truncate rows for quick tests |
| `SAE_ENCODE` | `0` to skip SAE encode (diff on raw hidden `.npz`) |

Interactive debugging on a GPU node: same exports, then:

```bash
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/pipeline_2026${PYTHONPATH:+:$PYTHONPATH}"
bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu-real
```
