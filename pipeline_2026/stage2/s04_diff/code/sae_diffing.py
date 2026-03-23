"""SAE latent diffing: compare High vs Low residual groups."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ttest_ind


def extract_sae_activations(model, sae, dataloader):
    """
    Passes data through the LLM to get hidden states, then through the SAE
    to get sparse latent activations.
    """
    import torch

    all_latent_acts = []

    with torch.no_grad():
        for batch in dataloader:
            hidden_states = model(
                batch["input_ids"], output_hidden_states=True
            ).hidden_states[-1]
            latent_acts = sae.encode(hidden_states)
            all_latent_acts.append(latent_acts.cpu())

    return torch.cat(all_latent_acts, dim=0)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return FDR-adjusted p-values (Benjamini-Hochberg)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    cumulative_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        idx = order[i]
        val = p[idx] * m / rank
        cumulative_min = min(val, cumulative_min)
        ranked[idx] = cumulative_min
    return np.clip(ranked, 0, 1)


def discover_failure_patterns(
    latent_acts: np.ndarray,
    groups: np.ndarray,
    *,
    p_value_threshold: float = 0.05,
    min_group_size: int = 30,
    use_fdr: bool = False,
    effect_std_floor: float = 1e-8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Rank latents where High Residual activations exceed Low Residual.

    Parameters
    ----------
    latent_acts : (N, D) float array
    groups : (N,) string array-like with values including 'High Residual', 'Low Residual'
    use_fdr : bool
        If True, compare adjusted p-values to p_value_threshold.
    """
    groups = np.asarray(groups, dtype=object)
    high_mask = groups == "High Residual"
    low_mask = groups == "Low Residual"

    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    diagnostics = {
        "n_high_residual": n_high,
        "n_low_residual": n_low,
        "min_group_size": min_group_size,
        "skipped_insufficient_n": n_high < min_group_size or n_low < min_group_size,
    }

    if n_high < min_group_size or n_low < min_group_size:
        return [], diagnostics

    high_res_acts = latent_acts[high_mask]
    low_res_acts = latent_acts[low_mask]

    num_latents = latent_acts.shape[-1]
    p_list: list[float] = []
    effects: list[float] = []

    for i in range(num_latents):
        high_vals = np.asarray(high_res_acts[:, i], dtype=float).ravel()
        low_vals = np.asarray(low_res_acts[:, i], dtype=float).ravel()
        if np.std(low_vals) < effect_std_floor and np.std(high_vals) < effect_std_floor:
            p_list.append(1.0)
            effects.append(0.0)
            continue
        _, p_val = ttest_ind(high_vals, low_vals, equal_var=False)
        p_list.append(float(p_val))
        sd = float(np.std(low_vals))
        effect = (
            (float(np.mean(high_vals)) - float(np.mean(low_vals))) / sd
            if sd > effect_std_floor
            else 0.0
        )
        effects.append(effect)

    p_arr = np.array(p_list)
    if use_fdr:
        p_cmp = benjamini_hochberg(p_arr)
    else:
        p_cmp = p_arr

    discovered: list[dict[str, Any]] = []
    for i in range(num_latents):
        if p_cmp[i] < p_value_threshold and effects[i] > 0:
            discovered.append(
                {
                    "latent_id": i,
                    "V_prime": round(effects[i], 4),
                    "p_value": float(p_arr[i]),
                    "p_adjusted": float(p_cmp[i]) if use_fdr else None,
                }
            )

    discovered.sort(key=lambda x: x["V_prime"], reverse=True)
    diagnostics["n_latents_tested"] = num_latents
    diagnostics["n_passing"] = len(discovered)
    diagnostics["use_fdr"] = use_fdr
    return discovered, diagnostics
