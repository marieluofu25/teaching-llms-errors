# Reproducibility log

This log records **frozen baseline inputs** and how to reproduce **additive** experiments without changing `pipeline/`.

## Recorded checksums (baseline data)

| Artifact | SHA256 | Size (bytes) | Date recorded |
|----------|--------|--------------|---------------|
| `pipeline/stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl` | `41c7a89bd0762490365adbaaaba0c123f3d57042c92d2d9af4cf9d8663c44e3e` | 132842225 | 2026-03-20 |

To refresh checksums locally:

```bash
cd teaching-llms-errors
shasum -a 256 pipeline/stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl
```

## Baseline code freeze

- **Frozen directory:** `pipeline/` (no edits for this project track).
- **Manifest:** `config/manifest.yaml` lists expected paths and baseline shell commands.

## Additive outputs (version per run)

After running residual / SAE / evaluation scripts, append a row:

| Run ID | Command | Output paths | Git commit | Notes |
|--------|---------|--------------|------------|-------|
| (fill) | | | | |

## Environment

- Python 3.9+ recommended (matches historical conda env in D5).
- Install: `pip install -r requirements.txt` from repo root.
- LLM keys: only required for baseline Stage 2 generation/judge, not for residual-only analysis.
