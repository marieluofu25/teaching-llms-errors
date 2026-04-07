#!/usr/bin/env python3
"""Create random activations .npz row-aligned with a residual CSV (smoke tests / demos)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.io_schema import validate_activations_shape


def main() -> None:
    """Create synthetic row-aligned activations for smoke/demo diffing runs."""
    p = argparse.ArgumentParser()
    p.add_argument("--residual-csv", type=Path, required=True)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.residual_csv)
    rng = np.random.default_rng(args.seed)
    n = len(df)
    # Bias a few dimensions so High Residual has slightly higher means (may yield hits)
    acts = rng.standard_normal((n, args.latent_dim))
    high = (df["group"] == "High Residual").values
    low = (df["group"] == "Low Residual").values
    for j in range(min(3, args.latent_dim)):
        acts[high, j] += 0.5
        acts[low, j] -= 0.3

    validate_activations_shape(n, acts.shape[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, activations=acts.astype(np.float32))
    meta = {
        "representation_type": "synthetic_demo",
        "n_rows": n,
        "d_in": int(args.latent_dim),
        "model": "random Gaussian (demo only; not HF or SAE)",
        "residual_csv": str(args.residual_csv.resolve()),
        "note": "Smoke/demo vectors; do not cite as neural or SAE features.",
    }
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} shape={acts.shape} and {meta_path}")


if __name__ == "__main__":
    main()
