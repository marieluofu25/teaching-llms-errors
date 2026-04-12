# Stage 2 — s01_residual — Residual control & instance table

**Goal (per project plan):** Build a per-question table with correctness, a **difficulty proxy** (`expected_error` from surface features), **residual** `is_error − expected_error`, and **groups** (e.g. High / Low residual) so later steps focus on errors not explained by generic difficulty alone.

**Code:** `stage2/s01_residual/code/run_residual_control.py`, `difficulty_control.py`, `stage1_profile.py` · shared: `lib/io_schema.py`

**Inputs:** MMLU pickle or MathCAMPs JSON under legacy `pipeline/stage1_do_errors_exist/datasets/` (paths resolved via `lib/repo_paths.py`).

**Outputs (in `results/`):**

- Residual CSV (+ `.meta.json`): per row `questions`, `is_error`, `expected_error`, `residual_error`, `group`, …
- **`<stem>_profile.json`** — aggregates: global error rate, group counts, residual quantiles, **by_subject / by_subcat** (MMLU) or **by_standard / by_model** (MathCAMPs).
- **`<stem>_error_by_subject.png`** (or `_error_by_standard.png` for MathCAMPs) — bar chart (skip with `--no-profile-plot`).

**Audit:** Open `results/audit.html`. CSV preview **hides wide embedding columns** and truncates long text.

**CLI:** `python -m stage2.s01_residual.code.run_residual_control …` · `--skip-profile`, `--no-profile-plot`.

**Unified Gemma / full pickle (CHPC):**

- `python -m stage2.s01_residual.code.run_gemma_mmlu_inference --output-csv …` — HF MMLU MC inference (default `google/gemma-2-9b`); `--row-start` / `--row-end` for shards; `--resume` for single-writer resume. Gold `data_y` may be **1–4** (choice index) or **A–D**; the runner maps 1→A, …, 4→D when scoring `ai_correct`. After any fix to gold mapping or prompts, **re-run inference**, merge shards, **`run_mmlu_full_residuals`**, then **s04/s05** (old predictions CSVs are not comparable).
- `python -m stage2.s01_residual.code.merge_mmlu_prediction_shards --shards … --output …` — merge array outputs.
- `python -m stage2.s01_residual.code.run_mmlu_full_residuals --predictions-csv … --output-csv …` — topic-aware residuals + profile next to output CSV.

Orchestrator: `bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu-gemma-full` (requires merged predictions path). On CHPC, prefer **`pipeline_2026/chpc/job_mmlu_gemma_pipeline.slurm`** (infer + pipeline in one `sbatch`); see `chpc/README.md`.
