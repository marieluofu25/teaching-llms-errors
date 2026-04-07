# Stage 2 — Additive / improvement track (numbered segments)

This stage holds the **residual-controlled + SAE + reporting** pipeline from the project plan. Each segment has its own `code/`, `results/`, and `audit.html`.

| Segment | Role |
|---------|------|
| **s01_residual** | Residual CSV, difficulty proxy, High/Low groups, profile JSON/PNG |
| **s02_export** | Hugging Face surrogate hidden states (`.npz`) |
| **s03_sae_encode** | SAELens → SAE latents |
| **s04_diff** | Welch / coordinate diffing → latents CSV |
| **s05_eval** | Coverage/concentration metrics, leaderboard, bar chart |
| **s06_report** | `release_readiness.json`, five-tab `mmlu_report.html`, audit generator |

**Run all:** `bash scripts/run_pipeline_2026.sh mmlu-real` (from `pipeline_2026/`).

**Developer reference (every segment file/function I/O):** [STAGE2_DEVELOPER_MAP.md](STAGE2_DEVELOPER_MAP.md).

**CHPC (Slurm):** [chpc/README.md](../chpc/README.md).

**Audits:** open `stage2/<segment>/results/audit.html`. Navigation links to **Stage 1 — Paper baseline** and sibling segments.
