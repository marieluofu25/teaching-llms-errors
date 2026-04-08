# Stage 1 — Paper baseline (arXiv:2512.21422 alignment)

**Purpose:** Reproduce the **existence-of-failure-patterns** view from the teaching paper using the **legacy** code under [`../../pipeline/stage1_do_errors_exist`](../../pipeline/stage1_do_errors_exist) (not the additive residual/SAE track).

**Runner (pipeline_2026):** `python -m stage1.code.run_paper_baseline --output-dir stage1/results`

- Runs [`stage1/code/notebook_paper_figures.py`](code/notebook_paper_figures.py) (this repo). It **only reads** legacy [`Dataset.py`](../../pipeline/stage1_do_errors_exist/Dataset.py), [`utils.py`](../../pipeline/utils.py), and datasets under [`pipeline/stage1_do_errors_exist/datasets/`](../../pipeline/stage1_do_errors_exist/datasets/) (subprocess `cwd` = `pipeline/`).
- Writes `paper_baseline_manifest.json` under `results/` (always).
- Regenerates **`stage1/results/audit.html`** so previews match whatever files are in `results/` (including all `paper_nb_*.png`). To run figures alone: `python stage1/code/notebook_paper_figures.py --output-dir /abs/path/to/stage1/results` (the script switches to `pipeline/` internally); or use `run_paper_baseline`.
- When data is present, writes **`paper_mmlu_summary.json`** (includes `figures_written` / `figures_skipped`) and **11 PNGs** aligned with [`dataset-model-analysis.ipynb`](../../pipeline/stage1_do_errors_exist/dataset-model-analysis.ipynb) (one file per notebook figure).

## Notebook cell → PNG filename

| Notebook cell | Output file | Description |
|---------------|-------------|-------------|
| 3 | `paper_nb_cell03_mmlu_math_reg.png` | `plot_metrics`, MMLU math-reg by subject |
| 5 | `paper_nb_cell05_mmlu_health_reg.png` | `plot_metrics`, MMLU health-reg by subject |
| 7 | `paper_nb_cell07_mmlu_subcat_computer_science.png` | `plot_metrics`, MMLU filtered to subcat *computer science* |
| 9 | `paper_nb_cell09_mmlu_all_subjects.png` | `plot_metrics`, full MMLU by subject |
| 11 | `paper_nb_cell11_mmlu_by_subcat.png` | `plot_metrics`, by subcat (uses same full MMLU frame as cell 9) |
| 13 | `paper_nb_cell13_mmlu_subcats_psych_physics_math.png` | `plot_metrics`, subjects within subcats psychology / physics / math |
| 16 | `paper_nb_cell16_mmlu_math_health_vertical.png` | Math + Health stacked + colorbar |
| 18 | `paper_nb_cell18_mmlu_math_health_horizontal.png` | Math + Health 1×2 + colorbar |
| 19 | `paper_nb_cell19_mmlu_subcats_grid.png` | All MMLU subcats except math & health, 3-column grid |
| 24 | `paper_nb_cell24_mathcamps_gpt_claude_sonnet.png` | MathCAMPS: GPT-4o, GPT-3.5-turbo, Claude Sonnet |
| 26 | `paper_nb_cell26_mathcamps_claude_family.png` | MathCAMPS: Sonnet, Opus, Haiku |

**Data requirements:** `stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl` for MMLU figures; `stage1_do_errors_exist/datasets/mathcamps/model-responses/v1/openai.json` and `anthropic.json` for MathCAMPS figures. Missing inputs are skipped and recorded under `figures_skipped` in `paper_mmlu_summary.json`.

**Manual exploration:** [`dataset-model-analysis.ipynb`](../../pipeline/stage1_do_errors_exist/dataset-model-analysis.ipynb) remains the full interactive counterpart.

**Next:** Stage 2 segments under `pipeline_2026/stage2/s01_residual/` … `s06_report/` (residual control, HF export, SAE, diff, metrics, HTML).
