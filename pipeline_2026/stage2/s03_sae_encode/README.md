# Stage 2 — s03_sae_encode — SAE encoding (SAELens)

**Goal (per project plan):** Map pooled hidden vectors through a **pretrained sparse autoencoder** so coordinates live in an **overcomplete SAE dictionary** (`d_sae` ≫ `d_in`), suitable for “SAE diffing” claims.

**Code:** `stage2/s03_sae_encode/code/encode_sae_latents.py` · shared: `lib/`

**Dependency:** `sae-lens` (see `requirements.txt`). Checkpoint must match model + hook used in **s02_export**.

**Outputs (in `results/`):** `*.npz` with `activations` of shape `(N, d_sae)`, plus `*.meta.json` (`representation_type: sae_latent`).

**Audit:** `results/audit.html`
