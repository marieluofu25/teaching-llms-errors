# Stage 2 — s06_report — Release summary & HTML reporting

**Goal:** Executive-style JSON (error rates, per-subject table, threshold gates) and the **five-tab HTML** report tying together residual data, activations, and metrics. Main human-readable deliverable for posters / committee review.

**Code:** `stage2/s06_report/code/release_summary.py`, `build_html_report.py`, `generate_stage_audit_html.py` · shared: `lib/`

**Config:** `config/feature_meanings.yaml`, `config/release_thresholds.yaml`

**Outputs (in `results/`):** `release_readiness.json`, **`mmlu_report.html`** (full report), plus `audit.html`.

**Audit:** Open `results/audit.html` for links to Stage 1 paper baseline and other Stage 2 segments; open **`mmlu_report.html`** for the full narrative.
