# SAE checkpoints (SAELens) — alignment with `export_activations`

`stage2/s03_sae_encode/code/encode_sae_latents.py` loads a **public sparse autoencoder** via [SAELens](https://github.com/jbloomAus/SAELens) and maps pooled hidden vectors `(N, d_in)` → SAE feature activations `(N, d_sae)`.

## Why layer index must match the SAE hook

Each pretrained SAE is trained on activations at a **specific hook** (e.g. `blocks.11.hook_resid_pre`). Your Hugging Face export must sample the **same residual position**.

For **GPT-2 small** (`gpt2`), TransformerLens-style `blocks.L.hook_resid_pre` corresponds to the residual **entering** block `L`, i.e. the output of block `L-1`. In HuggingFace `GPT2Model` outputs, `hidden_states[k]` is:

- `k=0`: after token + position embeddings  
- `k=1..n_layer`: after transformer block `k-1`

So **`blocks.L.hook_resid_pre` ↔ use `--layer-index L`** (non-negative) in `export_activations`.

Example defaults used in `scripts/run_additive_track.sh` when `SAE_ENCODE=1` and `EXPORT_MODEL=gpt2`:

| Setting | Typical value |
|--------|----------------|
| `SAE_RELEASE` | `gpt2-small-res-jb` |
| `SAE_ID` | `blocks.11.hook_resid_pre` |
| `EXPORT_LAYER_INDEX` | `11` (auto if unset for `gpt2`) |

Using `--layer-index -1` (last layer) with a **middle-block** SAE is **misaligned** and invalid for interpretation.

## Choosing `d_sae`

`d_sae` is **not** 768. It is the **dictionary size** of the chosen checkpoint (e.g. **24576** for the default `gpt2-small-res-jb` / `blocks.11.hook_resid_pre` SAE in SAELens 6.x). Inspect `*.meta.json` after encoding for the exact `d_sae`.

## Gemma (Gemma Scope / SAELens)

Gemma is a strong choice when you want **public pretrained SAEs** aligned with open **Google Gemma** weights on Hugging Face. The same rules apply as for GPT-2: the vector you export in `export_activations.py` must match the **hook / layer** the SAE was trained on, and `activations.shape[1]` must equal `sae.cfg.d_in` (unless you explicitly bypass the check).

**Gemma 4:** newer Gemma releases may not yet appear in the [SAELens pretrained index](https://decoderesearch.github.io/SAELens/latest/). Before committing to a model, confirm a matching `release` / `sae_id` exists; otherwise use **Gemma 2** or **Gemma 3** entries that are listed for your chosen HF checkpoint.

### Example triple (Gemma 2 2B, residual-stream SAE)

Values below are illustrative; always confirm against the current SAELens docs / `pretrained_saes.yaml` for your installed `sae-lens` version.

| Setting | Example value |
|--------|----------------|
| `EXPORT_MODEL` | `google/gemma-2-2b` (or the exact HF id you use) |
| `SAE_RELEASE` | `gemma-scope-2b-pt-res-canonical` |
| `SAE_ID` | `layer_12/width_16k/canonical` |
| `EXPORT_LAYER_INDEX` | Must match the **residual position** implied by that SAE’s training hook (often a middle layer; **do not** use `-1` blindly). After export, `encode_sae_latents` will error if `d_in` mismatches—use that as a sanity check and adjust `EXPORT_LAYER_INDEX`. |

Gemma 3 and larger checkpoints have additional SAELens releases (e.g. Gemma Scope 2, attention vs residual hooks). Hooks named like `blocks.L.hook_resid_post` need a consistent mapping from **TransformerLens-style block index** to **Hugging Face `hidden_states` index** for your architecture; when in doubt, cross-check the hook metadata on the SAELens model page for your exact `sae_id`.

### CHPC / env

Set `EXPORT_MODEL`, `EXPORT_LAYER_INDEX`, `SAE_RELEASE`, and `SAE_ID` in the shell or in `pipeline_2026/chpc/env_chpc.example` (copy to a local `env.local.sh`). See [`chpc/README.md`](../chpc/README.md).

## Further reading

Jiang et al., *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*, [arXiv:2512.10092](https://arxiv.org/abs/2512.10092) — framing for SAE-based embedding comparisons across groups or corpora.
