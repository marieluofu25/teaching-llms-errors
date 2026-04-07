# Stage 2 — s05_eval — Set-level metrics & plots

**Goal (per project plan):** full set-level metrics (**predictive utility, coverage, concentration, redundancy, stability**) from pattern memberships, plus residual-group proxies and optional Stage-2 judge merge.

**Code:** `stage2/s05_eval/code/evaluate_pattern_sets.py`, `aggregate_stability.py`, `plot_residual_summary.py` · shared: `lib/`

**Outputs (in `results/`):**
- `*_metrics.json` (includes `set_level_metrics` + residual proxies)
- `leaderboard.csv` (columns for all five metrics)
- `*_residual_bar.png` (proxy visualization)
- optional stability aggregate JSON from `aggregate_stability.py`

**Audit:** `results/audit.html`
