# Stage 2 — s02_export — Surrogate neural activations (Hugging Face)

**Goal:** Produce **row-aligned** pooled hidden-state vectors `(N, d_in)` for the same questions as the **s01_residual** CSV, using an **open** causal LM (default `gpt2`). This is a *surrogate* for interpretability: it is **not** the closed API model’s internal state.

**Code:** `stage2/s02_export/code/export_activations.py` · shared: `lib/io_schema.py`, `lib/repo_paths.py`

**Layer index must match** any SAE you use in **s03_sae_encode** (see `docs/sae_checkpoints.md`).

**Outputs (in `results/`):** `*.npz` with array `activations`, plus `*.meta.json` (`representation_type: hidden_pooled`).

**Audit:** `results/audit.html`
