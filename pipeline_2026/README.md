# pipeline_2026 — staged failure-discovery track

This folder is the **2026 additive implementation** of the residual-controlled + SAE-based discovery line from [`../project_plan.md`](../project_plan.md) (Improving LLM Failure Pattern Discovery with Residual-Controlled and SAE-diffing).

## Two top-level stages

| Stage | Role |
|-------|------|
| **stage1/** | **Paper baseline** — aligns with arXiv:2512.21422 Stage 1 style analysis (`run_paper_baseline`, `stage1/code/notebook_paper_figures.py` + notebook; reads legacy `pipeline/` data only). |
| **stage2/** | **Improvement track** — numbered segments `s01_residual` … `s06_report` (residual table → HF export → SAE → diff → metrics → HTML). |

## Alignment with `project_plan.md` (inside Stage 2)

| Proposal theme | Where it lives |
|----------------|----------------|
| **Difficulty control** (expected error, residual, groups) | **s01_residual** — `run_residual_control`, `difficulty_control` |
| **Internal representations** (HF surrogate) | **s02_export** — `export_activations` |
| **SAE latent space** | **s03_sae_encode** — `encode_sae_latents` (SAELens) |
| **Coordinate diffing** (Welch tests) | **s04_diff** — `run_sae_diffing` |
| **Set-level metrics** | **s05_eval** — `evaluate_pattern_sets`, `plot_residual_summary` |
| **Reporting & gates** | **s06_report** — `release_summary`, `build_html_report`, `generate_stage_audit_html` |

The frozen paper **datasets** still live under legacy [`../pipeline/`](../pipeline/). `pipeline_2026` **does not modify** that code; it reads paths via `lib.repo_paths.teaching_llms_errors_repo_root()`.

## Layout

```
pipeline_2026/
  lib/
  stage1/           # paper baseline runner + results/
  stage2/
    s01_residual/   # code/ + results/
    s02_export/
    s03_sae_encode/
    s04_diff/
    s05_eval/
    s06_report/
    README.md       # segment index
  config/
  docs/
  scripts/          # run_pipeline_2026.sh, smoke_test.sh, sae_encode_smoke.sh
  requirements.txt
```

**Run modules** with `PYTHONPATH=<this dir>`:

`python -m stage1.code.run_paper_baseline`, `python -m stage2.s01_residual.code.run_residual_control`, … (see `scripts/run_pipeline_2026.sh`).

## Run

```bash
pip install -r pipeline_2026/requirements.txt
bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu-real   # full chain
bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu-gemma-full   # Gemma predictions → full residual + SAE track
bash pipeline_2026/scripts/run_pipeline_2026.sh smoke       # tiny fixture
```

Open **`stage1/results/audit.html`** for the paper baseline, then **`stage2/<segment>/results/audit.html`** for each segment (nav links connect them). Full five-tab report: **`stage2/s06_report/results/mmlu_report.html`**.

## Entry point from repo root

[`../scripts/run_additive_track.sh`](../scripts/run_additive_track.sh) forwards to `run_pipeline_2026.sh`.
