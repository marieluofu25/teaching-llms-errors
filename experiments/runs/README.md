# Per-run archives

For each experiment tag, create a folder and copy:

- `requirements-freeze.txt` (optional)
- `metrics.json` / `leaderboard` row
- Paths to baseline Stage 2 JSON + judge outputs
- Any `artifacts/` files you want preserved

Example:

```bash
TAG=$(date +%Y%m%d_%H%M)
mkdir -p experiments/runs/$TAG
cp pipeline_2026/stage2/s05_eval/results/*_metrics.json experiments/runs/$TAG/ 2>/dev/null || true
cp pipeline_2026/stage2/s05_eval/results/leaderboard.csv experiments/runs/$TAG/ 2>/dev/null || true
cp pipeline_2026/stage2/s06_report/results/release_readiness.json experiments/runs/$TAG/ 2>/dev/null || true
```
