# Reproducibility checklist

- [ ] `experiments/manifest.yaml` matches your local dataset paths.
- [ ] `mmlu_df_gpt35.pkl` SHA256 matches `docs/reproducibility_log.md` (or update the log).
- [ ] Record Python version: `python --version`.
- [ ] Record dependency freeze: `pip freeze > experiments/runs/<tag>/requirements-freeze.txt`.
- [ ] Run `cd pipeline_2026 && PYTHONPATH=. python -m stage2.s01_residual.code.run_residual_control ...` and archive output under `stage2/s01_residual/results/`.
- [ ] If using SAE: save `activations.npz` schema + `docs/experiment_protocol.md` preprocessing steps.
- [ ] Run `python -m stage2.s05_eval.code.evaluate_pattern_sets ...` and copy `stage2/s05_eval/results/*.json` into the run folder.
- [ ] For baseline Stage 2: store path to generated JSON + optional judge output JSON in the run log table.
