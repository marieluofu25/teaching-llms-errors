# Additive experiment protocol (no `pipeline/` changes)

## Goal

Evaluate **residual-controlled** grouping and optional **SAE-diffing** on top of frozen baseline data, with **set-level metrics** comparable across runs.

**Layout:** **Stage 1** (`pipeline_2026/stage1/`) = paper-baseline figures/summary (`run_paper_baseline`). **Stage 2** (`pipeline_2026/stage2/s01_…s06_*/`) = additive track below.

## 0. Paper baseline (optional, Stage 1)

Run `python -m stage1.code.run_paper_baseline --output-dir stage1/results` (or rely on `run_pipeline_2026.sh`, which calls it first). Figures are implemented in `pipeline_2026/stage1/code/notebook_paper_figures.py`; it reads legacy `pipeline/` (`Dataset`, `utils`, pickles) with `cwd=pipeline/`.

## 1. Residual control (difficulty proxy)

1. Build instance table with text column and correctness.
2. Define `is_error = 1` if incorrect else `0`.
3. Fit logistic regression on surface features (length, digit count, logical keywords) → `expected_error`.
4. Residual: `residual_error = is_error - expected_error`.
5. Groups: top 25% residual → `High Residual`, bottom 25% → `Low Residual`, else `neutral`.

**Outputs:** `pipeline_2026/stage2/s01_residual/results/*_residuals.csv` (+ `*.meta.json` in the same folder).

## 2. Activations `.npz` (required for statistical “latent” views)

Row \(i\) of `activations` must correspond to row \(i\) of the residual CSV.

**Option A — Real vectors from an open Hugging Face model (recommended in-repo):**  
`stage2/s02_export/code/export_activations.py` runs a local causal LM, takes hidden states at `--layer-index` (default `-1` = last layer), pools each sequence (`--pooling last|mean`), and writes `np.savez(..., activations=(N, d_in))` plus a sidecar `*.meta.json` with `representation_type: hidden_pooled`. This is **real neural activity** from the chosen checkpoint. It is **not** the same as internal states of a closed API model used in the original MMLU pickle; it is a **surrogate** representation on the same question text and order.

**Option A′ — SAE latent space (for “true” SAE analysis):**  
After Option A, run `stage2/s03_sae_encode/code/encode_sae_latents.py` (SAELens pretrained SAE). This writes a new `.npz` with shape `(N, d_sae)` where **`d_sae` is the SAE dictionary size** (often \(\gg d_\text{in}\); e.g. 24576 for default GPT-2 small release in SAELens 6.x), and `*.meta.json` with `representation_type: sae_latent`. **Layer/hook alignment is mandatory:** see `docs/sae_checkpoints.md`.

**Option B — Your own pipeline:** any code that produces `(N, D)` aligned with the CSV (e.g. custom SAE training).

**Option C — Demo only:** `stage2/s04_diff/code/generate_dummy_activations.py` (random) for smoke tests (`representation_type: synthetic_demo`).

## 3. Statistical diffing on activation coordinates (optional)

1. Use row-aligned `activations` from §2 (raw hidden **or** SAE latents).
2. Run `stage2/s04_diff/code/run_sae_diffing.py` with `--min-group-size`, optional `--fdr` (Welch \(t\)-tests across `group` column).
3. **Outputs:** ranked CSV of column indices (`latent_id`) with effect sizes and p-values. **Interpretation depends on the `.npz`:** for `hidden_pooled`, `latent_id` indexes **hidden dimensions** (e.g. 0…767 for `gpt2`); for `sae_latent`, it indexes **SAE feature indices** (0…`d_sae`-1).

Framing for sparse-feature comparisons across groups or corpora: Jiang et al., *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*, [arXiv:2512.10092](https://arxiv.org/abs/2512.10092).

## 4. Set-level metrics (proxy definitions)

Without instance-level pattern membership from generated hypotheses, we report **residual-group proxies**:

- **Coverage (High Residual):** `|{i : is_error(i)=1 ∧ group(i)=High}| / |{i : is_error(i)=1}|`
- **Concentration (High Residual):** `|{i : is_error(i)=1 ∧ group(i)=High}| / |{i : group(i)=High}|`

Optional judge metrics from Stage 2: if `*_judge-seed=*.json` exists, parser records `f1`, `avg_rating_hyps` from that file.

## 5. Baseline comparison

- **Frozen baseline:** Stage 2 JSON + judge outputs produced by original `pipeline/` scripts.
- **Additive track:** residual CSV + latents CSV (s04_diff) + `pipeline_2026/stage2/s05_eval/results/*_metrics.json` (and leaderboard).

Document all paths in `docs/reproducibility_log.md` run table.

## 6. Five-tab HTML deliverable (`stage2/s06_report/code/build_html_report.py`)

One self-contained HTML file with CSS/JS tabs, built from the residual CSV (required) and optional row-aligned SAE activations.

| Tab | Content |
|-----|--------|
| 1 | Correct vs incorrect subsets (`is_error`); columns include question text and `subject` / `subcat` / `cat` when present. |
| 2 | `questions` \| `expected_error`. |
| 3 | `questions` \| `expected_error` \| `actual_error` (derived from `is_error`) plus `residual_error` / `group` when present. |
| 4 | `latent_k` \| **MMLU subject (highest mean activation)** on that dimension (from CSV `subject` column) \| optional interpretation from `config/feature_meanings.yaml`; optional ranked `latent_id` from `run_sae_diffing` when `--latents-csv` is passed. |
| 5 | Per selected latent: **low / mid / high** labels for the **mean** activation on correct rows vs on incorrect rows. |

**Tab 5 definition (fixed for reproducibility):** For each latent \(j\), compute tertile cutoffs on `activations[:, j]` over **all** instances (33rd and 67th percentiles). Let \(\bar{a}_c\) be the mean activation on rows with `is_error=0` and \(\bar{a}_w\) on rows with `is_error=1`. Map each mean to `low` / `mid` / `high` by comparing to those global cutoffs. The same rule is repeated in the HTML footer.

**CLI:** `--residual-csv` (required), `--activations` (optional `.npz` with key `activations`), `--latents-csv` (optional), `--meanings-yaml` (default `config/feature_meanings.yaml`), `--output`, `--max-rows`, `--top-k-tab5`.

**Shell:** `bash pipeline_2026/scripts/run_pipeline_2026.sh mmlu-report`; full path: `… mmlu-real`. Staged outputs under `pipeline_2026/stage1/results/` and `pipeline_2026/stage2/s*/results/` + `audit.html`. See script header for env vars.

## 7. Release-style summary (`stage2/s06_report/code/release_summary.py`)

1. **Inputs:** residual CSV; optional `evaluate_pattern_sets` JSON (merges `high_residual` coverage); optional `export_activations` sidecar `*.meta.json` (representation model id).
2. **Outputs:** e.g. `stage2/s06_report/results/release_readiness.json` with overall **error rate / accuracy**, **per-MMLU-subject** rates, and a **pass/fail verdict** against `config/release_thresholds.yaml`.
3. **HTML:** `build_html_report --release-json …` embeds an **Executive summary** block (read with **Read this first** + glossary). Neuronpedia-style third-party APIs are **not** required; external links may appear in the glossary only.
