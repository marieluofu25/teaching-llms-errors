#!/usr/bin/env python3
"""CLI: Group-wise statistical comparison on .npz columns (Welch tests).

Works on any row-aligned matrix (N, D): pooled hidden states (D=d_model) or
SAE latents from ``encode_sae_latents`` (D=d_sae). Interpret ``latent_id`` as a
column index in the same .npz you passed in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.io_schema import validate_activations_shape
from stage2.s04_diff.code.sae_diffing import discover_failure_patterns

_MEMBERSHIP_COLS = ("row_id", "pattern_id", "latent_id", "is_member")


def _membership_matrix(
    acts: np.ndarray,
    latent_ids: list[int],
    *,
    quantile: float,
) -> tuple[np.ndarray, dict[int, float]]:
    """Build binary membership matrix from latent activations.

    A row is a member of a pattern if the activation for that pattern's latent
    is above the requested quantile threshold across all rows.
    """
    if not latent_ids:
        return np.zeros((acts.shape[0], 0), dtype=np.int8), {}
    thresholds: dict[int, float] = {}
    columns: list[np.ndarray] = []
    for latent_id in latent_ids:
        vals = np.asarray(acts[:, latent_id], dtype=float)
        thr = float(np.quantile(vals, quantile))
        thresholds[latent_id] = thr
        columns.append((vals >= thr).astype(np.int8))
    return np.stack(columns, axis=1), thresholds


def _build_membership_frame(
    membership: np.ndarray,
    *,
    row_count: int,
    latent_ids: list[int],
) -> pd.DataFrame:
    """Convert matrix to long-form DataFrame for downstream metrics."""
    rows: list[dict[str, int]] = []
    for row_id in range(row_count):
        for j, latent_id in enumerate(latent_ids):
            rows.append(
                {
                    "row_id": row_id,
                    "pattern_id": j,
                    "latent_id": latent_id,
                    "is_member": int(membership[row_id, j]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run Welch diffing and write ranked latents + optional membership artifacts."""
    parser = argparse.ArgumentParser(description="SAE diffing runner")
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="NumPy .npz with array 'activations' shape (N, D), optional 'ids'",
    )
    parser.add_argument(
        "--residual-csv",
        type=Path,
        required=True,
        help="CSV from run_residual_control with 'group' column",
    )
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--min-group-size", type=int, default=30)
    parser.add_argument("--fdr", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="CSV of ranked latents")
    parser.add_argument("--output-diagnostics", type=Path, default=None)
    parser.add_argument(
        "--output-membership",
        type=Path,
        default=None,
        help="Optional long CSV (row_id, pattern_id, latent_id, is_member)",
    )
    parser.add_argument(
        "--output-pattern-catalog",
        type=Path,
        default=None,
        help="Optional JSON catalog of discovered patterns and thresholds",
    )
    parser.add_argument(
        "--membership-quantile",
        type=float,
        default=0.90,
        help="Quantile threshold for membership assignment (0, 1)",
    )
    args = parser.parse_args()
    if not (0.0 < args.membership_quantile < 1.0):
        raise ValueError("--membership-quantile must be in (0, 1)")

    data = np.load(args.activations, allow_pickle=True)
    if "activations" not in data:
        raise KeyError("Expected 'activations' array in npz")
    acts = np.asarray(data["activations"], dtype=float)
    if acts.ndim != 2:
        raise ValueError("activations must be 2D (N, D)")

    df = pd.read_csv(args.residual_csv)
    if args.group_col not in df.columns:
        raise ValueError(f"Missing column {args.group_col}")

    validate_activations_shape(len(df), acts.shape[0])

    groups = df[args.group_col].values
    discovered, diagnostics = discover_failure_patterns(
        acts,
        groups,
        p_value_threshold=args.p_threshold,
        min_group_size=args.min_group_size,
        use_fdr=args.fdr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["latent_id", "V_prime", "p_value", "p_adjusted"]
    out_df = (
        pd.DataFrame(discovered)
        if discovered
        else pd.DataFrame(columns=cols)
    )
    out_df.to_csv(args.output, index=False)

    diag_path = args.output_diagnostics or args.output.with_suffix(".diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    latent_ids = [int(x["latent_id"]) for x in discovered]
    membership, thresholds = _membership_matrix(
        acts, latent_ids, quantile=args.membership_quantile
    )
    membership_path = (
        args.output_membership or args.output.with_suffix(".pattern_membership.csv")
    )
    catalog_path = (
        args.output_pattern_catalog or args.output.with_suffix(".pattern_catalog.json")
    )
    membership_df = _build_membership_frame(
        membership, row_count=acts.shape[0], latent_ids=latent_ids
    )
    if membership_df.empty:
        membership_df = pd.DataFrame(columns=list(_MEMBERSHIP_COLS))
        print(
            "run_sae_diffing: 0 patterns passed thresholds; "
            f"wrote header-only membership CSV → {membership_path}",
            flush=True,
        )
    membership_df.to_csv(membership_path, index=False)
    catalog = {
        "n_rows": int(acts.shape[0]),
        "n_patterns": int(len(latent_ids)),
        "membership_quantile": float(args.membership_quantile),
        "patterns": [
            {
                "pattern_id": i,
                "latent_id": latent_id,
                "threshold": thresholds.get(latent_id),
                "V_prime": discovered[i]["V_prime"],
                "p_value": discovered[i]["p_value"],
                "p_adjusted": discovered[i]["p_adjusted"],
            }
            for i, latent_id in enumerate(latent_ids)
        ],
    }
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(
        "Wrote "
        f"{args.output} ({len(discovered)} latents), {diag_path}, "
        f"{membership_path}, {catalog_path}"
    )


if __name__ == "__main__":
    main()
