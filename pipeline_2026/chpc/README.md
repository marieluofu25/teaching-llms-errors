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

## Unified Gemma-2-9B pipeline (one `sbatch`)

[`job_mmlu_gemma_pipeline.slurm`](job_mmlu_gemma_pipeline.slurm) runs **in one allocation**:

1. **Inference** — default `N_SHARDS=1` (full MMLU in one pass). If `N_SHARDS>1`, shards run **sequentially** on the same GPU (same row math as the old array script), then shards are merged.
2. **`mmlu-gemma-full`** — runs **Stage 1 paper baseline** (`run_paper_baseline`) then **all of Stage 2** in order: unified residuals → HF export → SAE encode → diff → eval → release summary → five-tab HTML report (through `s06_report`), as long as the job does not hit OOM, missing `EXPORT_LAYER_INDEX`, or HF auth errors.

Edit the script’s `#SBATCH` lines (account, partition, QoS, `--time` — 72h is a starting point for infer + full stack). Set `REPO_ROOT` inside the script or export before `sbatch`:

```bash
export REPO_ROOT=/path/to/teaching-llms-errors
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PART --qos=YOUR_QOS \
  pipeline_2026/chpc/job_mmlu_gemma_pipeline.slurm
```

The script uses `set -u`; it sets defaults for `SLURM_ARRAY_JOB_ID` / `SLURM_ARRAY_TASK_ID` when you submit a **non-array** job so `env.local.sh` (or copied snippets from array jobs) does not hit “unbound variable”.

Optional environment (also via `env.local.sh` derived from [`env_chpc.example`](env_chpc.example)):

| Variable | Role |
|----------|------|
| `SKIP_INFER=1` | Skip Phase A; require existing `GEMMA_PREDICTIONS_CSV` (merged path). |
| `N_SHARDS` | `1` = one inference run; `>1` = sequential shards then merge (walltime sums; use for memory/time splits, not multi-node parallelism). |
| `TOTAL_MMLU_ROWS` | Default `7851`; must match pickle size if you change data. |
| `GEMMA_PREDICTIONS_CSV` | Merged predictions output path (default under `stage2/s01_residual/results/`). |
| `EXPORT_LAYER_INDEX` | **Required** to match `SAE_ID` / hook (see [`docs/sae_checkpoints.md`](../docs/sae_checkpoints.md)). |
| `EXPORT_MODEL`, `SAE_RELEASE`, `SAE_ID` | Gemma-2-9B + Gemma Scope defaults are set in the Slurm script; override as needed. |

Row-count context: [`docs/mmlu_canonical_frame.md`](../docs/mmlu_canonical_frame.md).

**Parallel inference across nodes** is not in this single job; for that, run shard jobs yourself, merge with `merge_mmlu_prediction_shards`, then `SKIP_INFER=1` and this script (or only `bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu-gemma-full`).

## Legacy GPT-2 / `mmlu-real` Slurm

There is **no** CHPC template for the old GPT-2 + JB SAE stack. To run that mode interactively on a GPU node:

```bash
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/pipeline_2026${PYTHONPATH:+:$PYTHONPATH}"
bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu-real
```

## Environment variables (orchestrator reference)

[`scripts/run_pipeline_2026.sh`](../scripts/run_pipeline_2026.sh) modes **`mmlu-real`** and **`mmlu-gemma-full`** respect the exports documented in [`env_chpc.example`](env_chpc.example): `EXPORT_DEVICE`, `EXPORT_MODEL`, `EXPORT_LAYER_INDEX`, `EXPORT_BATCH_SIZE`, `SAE_BATCH_SIZE`, `EXPORT_FP16`, `SAE_RELEASE`, `SAE_ID`, `EXPORT_MAX_ROWS`, `SAE_ENCODE`, etc.
