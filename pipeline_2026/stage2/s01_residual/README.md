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
