# experiments/ (legacy location)

**Canonical additive code and configs** have moved to **[`../pipeline_2026/`](../pipeline_2026/)**:

- Python: `pipeline_2026/stage1/` + `pipeline_2026/stage2/s*_*/code/` + `pipeline_2026/lib/`
- YAML / fixtures: `pipeline_2026/config/`
- Staged runs + `results/audit.html`: `pipeline_2026/stage1/results/` + `pipeline_2026/stage2/s*_*/results/`

Run:

```bash
bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu
```

Some paths in older notes may still say `experiments/`; treat `pipeline_2026/config/` as the replacement.
