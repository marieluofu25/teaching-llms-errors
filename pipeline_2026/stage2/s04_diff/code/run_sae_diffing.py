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


def main() -> None:
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
    args = parser.parse_args()

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
    print(f"Wrote {args.output} ({len(discovered)} latents), {diag_path}")


if __name__ == "__main__":
    main()
