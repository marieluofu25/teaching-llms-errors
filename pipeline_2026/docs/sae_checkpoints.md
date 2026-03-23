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

## Further reading

Jiang et al., *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*, [arXiv:2512.10092](https://arxiv.org/abs/2512.10092) — framing for SAE-based embedding comparisons across groups or corpora.
