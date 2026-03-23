#!/usr/bin/env bash
# Example: run frozen baseline Stage 2 from pipeline/ (needs LLM API key).
# Usage from repo root:
#   export MISTRAL_API_KEY=...   # or OPENAI_API_KEY if you point config back to OpenAI
#   ./scripts/run_baseline_pipeline_example.sh
#
# Edit MODEL_TO_ANALYZE / DATASET below if needed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/pipeline"
PY="${PYTHON:-python3}"

DATASET="${BASELINE_DATASET:-mathcamps}"
METHOD="${BASELINE_METHOD:-direct}"
MODEL_TO_ANALYZE="${BASELINE_MODEL:-claude-3-haiku-20240307}"
PATH_DIR="${BASELINE_OUT_DIR:-generated_failure_patterns}"

echo "Running describe-landscapes (dataset=${DATASET}, method=${METHOD})…"
"${PY}" stage2_can_we_generate_them/describe-landscapes.py \
  --dataset "${DATASET}" \
  --method "${METHOD}" \
  --model_to_analyze "${MODEL_TO_ANALYZE}" \
  --num_gold_specified \
  --path_dir "${PATH_DIR}"

echo ""
echo "Find the JSON under ${PATH_DIR}/ and run judge manually, e.g.:"
echo "  cd pipeline && ${PY} stage2_can_we_generate_them/judge.py --seed 1 --dataset mathcamps --path '<path-to-json>'"
