# MMLU row counts: canonical frames in this repo

Two different **subset sizes** appear in artifacts; they are consistent once you know which frame each stage uses.

## Full pickle frame (Stage 2 default)

- **Source:** `pipeline/stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl` loaded **without** the train/val split inside `Dataset.py`.
- **Typical size:** **7 851** rows (all examples in the pickle used by `run_residual_control` for MMLU).
- **Use for:** `pipeline_2026` residual CSV, export, SAE, diff, eval—the **full** additive track on disk.

## Train-only frame (~70%) from the legacy notebook path

- **Source:** `MMLUDataset.pre_process_mmlu` applies `train_test_split(..., test_size=0.3, random_state=42)` and keeps the **training** split for `self.data`.
- **Typical size:** **5 495** rows ≈ **70% × 7 851**.
- **Use for:** Stage 1 paper-aligned figures (`notebook_paper_figures.py` / `paper_mmlu_summary.json`), matching `dataset-model-analysis.ipynb` style analysis on the **train** slice.

## Practical rule

- **“Full MMLU” / unified Gemma pipeline:** target **7 851** (full pickle) unless you explicitly re-filter.
- **Paper baseline figures:** expect summaries keyed off **5 495** unless the notebook code path is changed to use the full frame.

When publishing, **state which frame** (full vs train) any error rate refers to.
