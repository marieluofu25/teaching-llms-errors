# teaching-llms-errors

## The AI-Integration Teaching Pipeline
The `pipeline/` folder contains code for running experiments for Stages 1&2 found in the [paper](https://arxiv.org/abs/2512.21422): 

### LLM Provider Configuration (Centralized)
All model/provider settings are now centralized in `config/`:
- `config/llm_settings.json` for default provider, base URL, and model names
- `config/llm.py` for shared client/config loading used by pipeline scripts

Default provider is set to Mistral. Export your key before running:

`export MISTRAL_API_KEY="your_mistral_api_key"`

### Analyzing existence of failure patterns worth-teaching using meta-labels
Related code can be found in `stage1_do_errors_exist`
* `dataset-model-analysis.ipynb` contains code for analyzing the existence of sizable groups of failure patterns represented by meta-labels for MMLU and MathCAMPs.

To analyze `mmlu`, you'll need to download the data and predictions from [here](https://drive.google.com/file/d/1vrBo426u8O8UMpnmBZyTH7h4mFZR8RJT/view?usp=sharing) and place it in `stage1_do_errors_exist/datasets`

### Generating Failure Patterns
`describe-landscape.py` is used to generate candidate failure patterns for a model and dataset pair. 

For example, from the `pipeline/` directory run:

`$ python stage2_can_we_generate_them/describe-landscapes.py --'mathcamps' --method "direct" --model_to_analyze "claude-3-haiku-20240307" --num_gold_specified --path_dir "generated_failue_patterns"`

You can then score them with the LLM judge for mathcamps failure patterns by running:
`$ python judge.py --seed 1 --path "generated_failure_patterns/mathcamps/error_landscapes_erthresh=0.5/landscape=claude-3-haiku-20240307_specified=True/direct-response_cot.json"`

and for mmlu failure patterns by running:

`$ python judge-mmlu.py --seed 1 --path "generated_failure_patterns/mathcamps/error_landscapes_erthresh=0.5/landscape=claude-3-haiku-20240307_specified=True/direct-response_cot.json"$"`


* template slurm scripts are provided: `describe-single.sh`, `describe-landscapes.sh`, `judge.sh`. These should be run from the `pipeline/` directory.

## Teaching Generated Failure Patterns to Users
Templates for teaching people can be found in the `user_study`directory

Firebase project templates for our user study measuring the effectiveness of the AI-Integration teaching pipeline can be found in the `user_study` directory. 
`study-S3` contains templates with teaching and `study-S5` contains templates without teaching.

---

## Additive experiment track — **`pipeline_2026/`** (does not modify `pipeline/`)

All Python, configs, **per-stage outputs**, and **per-stage `results/audit.html`** live under **[`pipeline_2026/`](pipeline_2026/README.md)**. This matches the PhD plan in [`../project_plan.md`](../project_plan.md): residual control + SAE latents + set-level metrics + reporting.

| Location | Purpose |
|----------|---------|
| `pipeline_2026/README.md` | Map stages → proposal; how to run |
| `pipeline_2026/stage1/` | Paper baseline: `run_paper_baseline` → `results/` + **`audit.html`** |
| `pipeline_2026/stage2/s01_…s06_*` | Improvement segments: each has `README.md` + `code/` + `results/` + **`audit.html`** |
| `pipeline_2026/lib/` | Shared helpers: `repo_paths.py`, `io_schema.py` |
| `pipeline_2026/config/` | `feature_meanings.yaml`, `release_thresholds.yaml`, `manifest.yaml`, `fixtures/` |
| `pipeline_2026/docs/` | `experiment_protocol.md`, `sae_checkpoints.md`, reproducibility notes |
| `scripts/run_additive_track.sh` | Forwards to `pipeline_2026/scripts/run_pipeline_2026.sh` |

### Quickstart

```bash
pip install -r requirements.txt   # includes pipeline_2026 via -r

# Full staged run (writes stage1/results + stage2/s*/results + audit.html)
./scripts/run_additive_track.sh mmlu-real

# Open audits: pipeline_2026/stage1/results/audit.html + stage2/s*/results/audit.html
# Full five-tab report: pipeline_2026/stage2/s06_report/results/mmlu_report.html
```

Smoke / SAE encode smoke:

```bash
./scripts/run_additive_track.sh smoke
bash pipeline_2026/scripts/sae_encode_smoke.sh
```

Manual CLI (set `PYTHONPATH` to `pipeline_2026`): see `pipeline_2026/config/manifest.yaml` and `pipeline_2026/README.md`.

Baseline paper pipeline (LLM calls; configure `config/llm_settings.json` + key first):

```bash
export MISTRAL_API_KEY="..."
./scripts/run_baseline_pipeline_example.sh
```