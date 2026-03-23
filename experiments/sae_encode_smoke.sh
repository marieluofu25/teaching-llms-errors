#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")/.." && pwd)/pipeline_2026/scripts/sae_encode_smoke.sh" "$@"
