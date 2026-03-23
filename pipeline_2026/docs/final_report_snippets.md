# Report / poster snippets (fill in numbers from your runs)

## Contribution (1–2 sentences)

We keep the original teaching pipeline’s Stage 1–2 code **unchanged** and add a **residual-controlled** grouping step that down-weights errors predictable from surface difficulty, then optionally **diff SAE latents** between high- and low-residual subsets to surface candidate failure features.

## Methods (short)

- **Residual control:** Logistic regression on length, digit count, and logical keywords predicts \(P(\text{error})\). Residual \(= y_{\text{error}} - \hat{p}\). Top/bottom quartiles define **High** / **Low** residual groups (`train_only` fit mode avoids training on the same labels being explained).
- **SAE diffing:** For each latent, Welch’s \(t\)-test (High vs Low); optional Benjamini–Hochberg FDR; minimum group size guardrail.
- **Metrics (proxy):** **Coverage** = fraction of all errors in High residual; **Concentration** = precision of errors within High residual. Stage 2 **judge JSON** can be merged for hypothesis-quality F1 when available.

## Limitations (honest)

- Surface features only partially capture “difficulty.”
- SAE analysis requires separately exported activations aligned with the residual table.
- Residual metrics are **proxies** until instance-level pattern membership is available for every generated hypothesis.

## What to cite from this repo

- Frozen baseline: `pipeline/` + checksums in `docs/reproducibility_log.md`
- Additive code: `pipeline_2026/stage1/` (paper baseline) + `pipeline_2026/stage2/s*_*/code/` + `pipeline_2026/lib/` + `pipeline_2026/config/manifest.yaml`
