# Stage 2 — s04_diff — Coordinate-wise group comparison (“SAE diffing”)

**Goal:** Rank columns (raw hidden dims **or** SAE feature indices) where **residual groups** differ in mean activation (Welch tests). Interprets **High vs Low residual** (or other `group` labels), not a causal “error neuron” claim.

**Code:** `stage2/s04_diff/code/run_sae_diffing.py`, `sae_diffing.py`, `generate_dummy_activations.py` · shared: `lib/`

**Inputs:** **s01_residual** CSV + **s03_sae_encode** `.npz` (or **s02_export** hidden `.npz` only if SAE encode skipped).

**Outputs (in `results/`):** Ranked `*.csv` (`latent_id`, statistics), optional `*.diagnostics.json`.

**Audit:** `results/audit.html`
