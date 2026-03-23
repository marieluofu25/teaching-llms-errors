#!/usr/bin/env bash
# End-to-end smoke (tiny fixture). Writes to s04_diff + s05_eval results; refreshes audit HTML.
set -euo pipefail
P26="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P26"
export PYTHONPATH="${P26}${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

FIX="${P26}/config/fixtures/tiny_residual.csv"
S4="${P26}/stage2/s04_diff/results"
S5="${P26}/stage2/s05_eval/results"
mkdir -p "$S4" "$S5"

"${PY}" -m stage2.s05_eval.code.evaluate_pattern_sets \
  --residual-csv "${FIX}" \
  --label smoke_fixture \
  --output-json "${S5}/smoke_metrics.json" \
  --output-table "${S5}/leaderboard.csv"

"${PY}" -m stage2.s04_diff.code.generate_dummy_activations \
  --residual-csv "${FIX}" \
  --latent-dim 16 \
  --output "${S4}/smoke_activations.npz"

"${PY}" -m stage2.s04_diff.code.run_sae_diffing \
  --activations "${S4}/smoke_activations.npz" \
  --residual-csv "${FIX}" \
  --min-group-size 2 \
  --p-threshold 0.2 \
  --output "${S4}/smoke_latents.csv"

"${PY}" -m stage2.s05_eval.code.plot_residual_summary \
  --metrics-json "${S5}/smoke_metrics.json" \
  --output "${S5}/smoke_residual_bar.png"

AUDIT="stage2.s06_report.code.generate_stage_audit_html"
for seg in s04_diff s05_eval; do
  "${PY}" -m "${AUDIT}" \
    --stage "${seg}" \
    --title "${seg} (smoke)" \
    --readme "${P26}/stage2/${seg}/README.md" \
    --results-dir "${P26}/stage2/${seg}/results"
done

echo "Smoke test OK. See ${S4}/audit.html and ${S5}/audit.html"
