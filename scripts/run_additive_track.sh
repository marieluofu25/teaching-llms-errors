#!/usr/bin/env bash
# Forwarder: additive / interpretability track now lives under pipeline_2026/.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pipeline_2026/scripts/run_pipeline_2026.sh" "$@"
