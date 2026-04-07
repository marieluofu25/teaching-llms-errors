# Stage 2 developer map — inputs, process, outputs

Quick reference for each script and important function. Paths are relative to `pipeline_2026/` unless noted.

**Shared validation:** [`lib/io_schema.py`](../lib/io_schema.py) — `validate_mmlu_frame` (MMLU columns), `validate_residual_output` (`expected_error`, `residual_error`, `group`), `validate_activations_shape` (CSV row count vs `.npz` rows), `assert_columns` (generic).

---

## s01_residual — Residual control table

### `s01_residual/code/run_residual_control.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s01_residual.code.run_residual_control` |
| **Inputs** | `--dataset mmlu \| mathcamps`; MMLU: `--mmlu-pickle` (default: repo `pipeline/stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl`); MathCAMPs: `--mathcamps-json`, optional `--model-filter`; `--fit-mode full \| train_only`; `--output` `.csv` or `.parquet`; optional `--skip-profile`, `--no-profile-plot` |
| **Process** | Loads dataset, builds `is_error`, runs `compute_residuals_and_group`, validates columns, writes CSV/parquet + `*.meta.json`, then optional profile JSON/PNG via `write_stage1_artifacts` |
| **Outputs** | Residual table with `expected_error`, `residual_error`, `group` (`High Residual` / `Low Residual` / `neutral`), plus `<stem>_profile.json` and optional bar chart PNG |

**`load_mmlu_dataframe(pickle_path)`** — *Input:* path to pickle. *Process:* `read_pickle`, derive `subject`/`subcat`/`cat` from `metadata`, set `ai_correct`, validate. *Output:* `DataFrame`.

**`_flatten_mathcamps(obj, model_filter)`** — *Input:* nested JSON dict, optional model name. *Process:* DFS collect leaves with `question`+`correct`. *Output:* `list[dict]` rows.

**`main()`** — wires CLI to loaders, `add_is_error_*`, `compute_residuals_and_group`, I/O.

### `s01_residual/code/difficulty_control.py`

**`extract_difficulty_features(text)`** — *Input:* question string. *Process:* length, digit count, simple keyword counts. *Output:* `list` of numeric features for logistic regression.

**`compute_residuals_and_group(df, text_col, error_col, fit_mode=..., high_q=0.75, low_q=0.25)`** — *Input:* frame with text + binary error column. *Process:* fit logistic regression (`full` on all rows or `train_only` with holdout), `expected_error` = P(error), `residual_error` = observed − expected, quantile buckets → `group`. *Output:* augmented `DataFrame`.

**`add_is_error_from_ai_correct(df)`** — MMLU: `is_error = ~ai_correct`.

**`add_is_error_from_correct(df)`** — MathCAMPs: `is_error = ~correct`.

### `s01_residual/code/stage1_profile.py`

**`build_stage1_profile(df, dataset)`** — *Input:* residual frame, dataset name. *Process:* aggregates error rate, group counts, residual quantiles, per-dimension breakdown tables (`subject`/`subcat`/`cat` for MMLU; `standard`/`model` for MathCAMPs). *Output:* JSON-serializable `dict`.

**`write_subject_error_plot(df, out_png, key=..., top_n=...)`** — *Input:* frame with `is_error` and grouping key. *Process:* matplotlib horizontal bar chart of error rates. *Output:* PNG if matplotlib available and data nonempty; else no-op.

**`write_stage1_artifacts(df, dataset, output_csv, write_plot=True)`** — *Input:* residual CSV path (for stem/parent). *Process:* writes `{stem}_profile.json`; for MMLU `{stem}_error_by_subject.png`, for MathCAMPs `{stem}_error_by_standard.png` when plot succeeds. *Output:* `(profile_path, png_path_or_none)`.

---

## s02_export — Hugging Face hidden states

### `s02_export/code/export_activations.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s02_export.code.export_activations` |
| **Inputs** | `--residual-csv` (must have `questions` or `question`); `--model` HF id (default `gpt2`); `--layer-index` (default `-1` last); `--pooling last \| mean`; `--batch-size`, `--max-length`, `--max-rows`, `--device`, `--fp16`, `--output` `.npz` |
| **Process** | Tokenize batches, forward `AutoModelForCausalLM` with `output_hidden_states=True`, take one layer, pool to one vector per row, stack to `(N, d_in)` |
| **Outputs** | `output.npz` key `activations`; sidecar `output.meta.json` (`representation_type: hidden_pooled`, `model`, `layer_index`, `residual_csv`, etc.) |

**`_text_column(df)`** — resolves `questions` vs `question`.

**`_pool_hidden(hidden, attention_mask, pooling)`** — last real token or masked mean.

**`export_activations(texts, model_name, layer_index, ...)`** — core forward loop; returns `float32` `(N, hidden_dim)`.

**`main()`** — reads CSV, calls `export_activations`, validates row count, writes `.npz` + meta.

---

## s03_sae_encode — SAELens latents

### `s03_sae_encode/code/encode_sae_latents.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s03_sae_encode.code.encode_sae_latents` |
| **Inputs** | `--input-npz` (`activations` `(N, d_in)`); optional `--input-meta`; `--output-npz`; `--sae-release`, `--sae-id` (SAELens registry); `--device`, `--batch-size`; `--skip-d-in-check` (discouraged) |
| **Process** | `SAE.from_pretrained`, encode batches with `sae.encode`, enforce `d_in` match vs checkpoint unless skipped |
| **Outputs** | `output.npz` key `activations` shape `(N, d_sae)`; `output.meta.json` with `representation_type: sae_latent`, `d_sae`, release/id, provenance pointers |

**`_device_from_arg(device)`** — cuda / mps / cpu auto-pick.

**`encode_batches(sae, acts, batch_size, device)`** — batched `encode` to numpy.

**`main()`** — load npz, load SAE, encode, write npz + meta.

---

## s04_diff — Group-wise coordinate tests

### `s04_diff/code/run_sae_diffing.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s04_diff.code.run_sae_diffing` |
| **Inputs** | `--activations` `.npz` with `activations` `(N, D)`; `--residual-csv` with `group`; optional `--group-col`, `--p-threshold`, `--min-group-size`, `--fdr`, `--output` CSV, `--output-diagnostics` |
| **Process** | Aligns row counts, calls `discover_failure_patterns` |
| **Outputs** | Ranked CSV: `latent_id`, `V_prime`, `p_value`, `p_adjusted`; `*.diagnostics.json` |

**`main()`** — I/O + validation only.

### `s04_diff/code/sae_diffing.py`

**`discover_failure_patterns(latent_acts, groups, p_value_threshold, min_group_size, use_fdr, ...)`** — *Input:* `(N, D)` array, per-row group labels. *Process:* subset `High Residual` vs `Low Residual`, Welch `ttest_ind` per column, optional Benjamini–Hochberg; keep effects where High > Low and p passes threshold. *Output:* `(list[dict], diagnostics_dict)`.

**`benjamini_hochberg(p_values)`** — FDR-adjusted p-values.

**`extract_sae_activations(model, sae, dataloader)`** — *Note:* legacy-style helper (full forward + SAE); not used by the current `run_sae_diffing` npz path.

### `s04_diff/code/generate_dummy_activations.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s04_diff.code.generate_dummy_activations` |
| **Inputs** | `--residual-csv`, `--latent-dim`, `--seed`, `--output` `.npz` |
| **Process** | Gaussian noise with small bias on first dims by residual group (demo only) |
| **Outputs** | `.npz` `activations` + `*.meta.json` (`representation_type: synthetic_demo`) |

---

## s05_eval — Set-level metrics

### `s05_eval/code/evaluate_pattern_sets.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s05_eval.code.evaluate_pattern_sets` |
| **Inputs** | `--residual-csv` (`is_error`, `group`); optional `--judge-json`; `--label`; `--output-json`; optional `--output-table` leaderboard CSV |
| **Process** | `residual_proxy_metrics`; optional judge summary merge |
| **Outputs** | JSON with `residual_metrics` (coverage/concentration of errors in High/Low groups); optional appended row to leaderboard |

**`residual_proxy_metrics(df)`** — *Input:* validated residual frame. *Process:* counts errors overlapping High/Low residual buckets. *Output:* nested metric dict.

**`load_judge_summary(path)`** — *Input:* legacy judge JSON. *Output:* slimmed summary dict for metrics bundle.

**`main()`** — CLI I/O.

### `s05_eval/code/plot_residual_summary.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s05_eval.code.plot_residual_summary` |
| **Inputs** | `--metrics-json` from `evaluate_pattern_sets`; `--output` PNG |
| **Process** | Four-bar chart: High/Low coverage and concentration |
| **Outputs** | PNG figure |

---

## s06_report — HTML and release readiness

### `s06_report/code/release_summary.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s06_report.code.release_summary` |
| **Inputs** | `--residual-csv`; optional `--metrics-json`, `--activations-meta` (export/sae sidecar); `--thresholds-yaml` (default `config/release_thresholds.yaml`); `--output` JSON |
| **Process** | Overall/per-subject error stats, optional merge of eval metrics and representation meta, threshold gates from YAML |
| **Outputs** | `release_readiness.json` (`behavioral_model`, `representation_model`, `gates.verdict`, etc.) |

**`_load_thresholds(path)`** — reads `gates` block from YAML.

**`_behavioral_model_info(df)`** — infers logged LLM from `ai_model` column if present.

**`_per_subject_stats(df)`** — error rate table by `subject`.

**`_apply_gates(overall_er, per_subject, gates)`** — returns `(verdict, violations)`.

**`build_release_summary(df, metrics_json, activations_meta, thresholds)`** — assembles the release dict.

**`main()`** — CLI I/O.

### `s06_report/code/build_html_report.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s06_report.code.build_html_report` |
| **Inputs** | `--residual-csv`; `--meanings-yaml` (`feature_meanings.yaml` latent labels); `--output` HTML; optional `--activations` `.npz`, `--latents-csv`, `--release-json`, `--tab4-top-questions` |
| **Process** | Builds five-tab static HTML: tables, residual views, optional latent tertile analysis, executive block from release JSON |
| **Outputs** | Single self-contained `mmlu_report.html` (or path given) |

*Many internal `_format_*` / rendering helpers* — HTML fragments; same pattern: *Input:* data frames / dicts; *Process:* escape and template; *Output:* HTML strings.

### `s06_report/code/generate_stage_audit_html.py`

| | |
|--|--|
| **CLI** | `python -m stage2.s06_report.code.generate_stage_audit_html` |
| **Inputs** | `--results-dir`, `--stage`, `--title`, `--readme`; optional `--csv-hide-columns`, `--max-cell-chars` |
| **Process** | Lists artifacts under `results/`, embeds previews for CSV/JSON, navigation links across Stage 1 + Stage 2 segments |
| **Outputs** | `results/audit.html` |

**`_nav_html(results_dir, root)`** — cross-links between segment audits.

**`_stage1_compare_block(...)`** — optional Stage 1 comparison snippet when paths match.

*Other `_truncate_cell`, `_coerce_bool`, etc.* — small utilities for audit rendering.

---

## Orchestration

- **End-to-end:** `bash scripts/run_pipeline_2026.sh mmlu-real` from `pipeline_2026/` (see script header for env vars).
- **CHPC example:** [`chpc/README.md`](../chpc/README.md).
