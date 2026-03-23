#!/usr/bin/env bash
# Minimal encode_sae_latents check: random (4, 768) "hidden" batch → SAELens → (4, d_sae).
# Requires: pip install -r requirements.txt (includes sae-lens). First run downloads the SAE from Hugging Face.
set -euo pipefail
P26="$(cd "$(dirname "$0")/.." && pwd)"
cd "$P26"
export PYTHONPATH="${P26}${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

mkdir -p stage2/s02_export/results stage2/s03_sae_encode/results

"${PY}" -c "
import json
from pathlib import Path
import numpy as np
root = Path('${P26}')
hidden = root / 'stage2/s02_export/results/sae_smoke_hidden.npz'
meta_p = root / 'stage2/s02_export/results/sae_smoke_hidden.meta.json'
np.savez(hidden, activations=np.random.randn(4, 768).astype(np.float32))
meta = {
    'n_rows': 4,
    'representation_type': 'hidden_pooled',
    'd_in': 768,
    'model': 'synthetic-smoke',
    'layer_index': 11,
    'residual_csv': 'config/fixtures/tiny_residual.csv',
}
meta_p.write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('Wrote', hidden, 'and', meta_p)
"

"${PY}" -m stage2.s03_sae_encode.code.encode_sae_latents \
  --input-npz stage2/s02_export/results/sae_smoke_hidden.npz \
  --output-npz stage2/s03_sae_encode/results/sae_smoke_out.npz \
  --batch-size 2

"${PY}" -c "import numpy as np; d=np.load('stage2/s03_sae_encode/results/sae_smoke_out.npz'); print('encode_sae_latents smoke OK, out shape', d['activations'].shape)"
