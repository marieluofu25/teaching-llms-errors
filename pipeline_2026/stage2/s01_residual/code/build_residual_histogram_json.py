#!/usr/bin/env python3
"""Emit histogram JSON for residual_error (for pipeline-explorer poster).

Includes:
- full_range: min/max over all rows (can make one bin dominate).
- zoom: histogram on [p_lo, p_hi] (default 1st–99th percentile) so the bulk
  shape is visible on the poster without a single giant bar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residual-csv", type=Path, required=True)
    p.add_argument("--column", default="residual_error")
    p.add_argument("--bins-full", type=int, default=40)
    p.add_argument("--bins-zoom", type=int, default=48)
    p.add_argument("--zoom-p-low", type=float, default=1.0, help="Lower percentile for zoom range")
    p.add_argument("--zoom-p-high", type=float, default=99.0, help="Upper percentile for zoom range")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.residual_csv)
    if args.column not in df.columns:
        raise SystemExit(f"Missing column {args.column}")
    r = pd.to_numeric(df[args.column], errors="coerce").dropna()
    arr = np.asarray(r, dtype=float)
    n = int(arr.size)
    if n == 0:
        raise SystemExit("No numeric residuals")

    lo_full, hi_full = float(np.min(arr)), float(np.max(arr))
    if lo_full >= hi_full:
        hi_full = lo_full + 1e-9

    q25, q50, q75 = (
        float(np.quantile(arr, 0.25)),
        float(np.quantile(arr, 0.5)),
        float(np.quantile(arr, 0.75)),
    )

    counts_f, edges_f = np.histogram(arr, bins=args.bins_full, range=(lo_full, hi_full))

    z_lo, z_hi = np.quantile(arr, [args.zoom_p_low / 100.0, args.zoom_p_high / 100.0])
    z_lo, z_hi = float(z_lo), float(z_hi)
    if z_lo >= z_hi:
        z_hi = z_lo + 1e-9

    counts_z, edges_z = np.histogram(arr, bins=args.bins_zoom, range=(z_lo, z_hi))

    out = {
        "column": args.column,
        "n_rows": n,
        "quantiles": {"q25": q25, "q50": q50, "q75": q75},
        "group_split_quantiles": {"high_q": 0.75, "low_q": 0.25},
        "full_range": {"min": lo_full, "max": hi_full, "n_bins": args.bins_full},
        "full": {
            "min": lo_full,
            "max": hi_full,
            "n_bins": args.bins_full,
            "edges": [float(x) for x in edges_f.tolist()],
            "counts": [int(x) for x in counts_f.tolist()],
        },
        "zoom": {
            "percentile_window": [args.zoom_p_low, args.zoom_p_high],
            "min": z_lo,
            "max": z_hi,
            "n_bins": args.bins_zoom,
            "edges": [float(x) for x in edges_z.tolist()],
            "counts": [int(x) for x in counts_z.tolist()],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(
        f"Wrote {args.output} (n={n}, zoom [{z_lo:.4f}, {z_hi:.4f}] p{args.zoom_p_low:.0f}–p{args.zoom_p_high:.0f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
