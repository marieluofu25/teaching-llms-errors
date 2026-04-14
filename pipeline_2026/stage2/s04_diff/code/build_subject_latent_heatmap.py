#!/usr/bin/env python3
"""Build subject × top-K latent heatmap of mean(High) − mean(Low) SAE activation.

Rows align with ``mmlu_full_residuals.csv`` order; columns are the first K
``latent_id`` entries from ``pattern_catalog.json``. Subjects are ranked by how
many High+Low residual rows they contain (most informative first), then truncated.

Outputs a small JSON for the pipeline-explorer poster / UI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="NPZ with 'activations' (N, D)",
    )
    p.add_argument(
        "--residual-csv",
        type=Path,
        required=True,
        help="Residual CSV with 'subject', 'group'",
    )
    p.add_argument(
        "--pattern-catalog",
        type=Path,
        required=True,
        help="pattern_catalog.json from run_sae_diffing",
    )
    p.add_argument("--top-latents", type=int, default=12, help="Number of latent columns")
    p.add_argument(
        "--max-subjects",
        type=int,
        default=24,
        help="Max subject rows (ranked by High+Low count)",
    )
    p.add_argument(
        "--min-per-side",
        type=int,
        default=5,
        help="Minimum rows per cohort in a cell; else null",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    if "activations" not in data:
        raise SystemExit("NPZ must contain 'activations'")
    acts = np.asarray(data["activations"], dtype=np.float64)
    if acts.ndim != 2:
        raise SystemExit("activations must be 2D")

    df = pd.read_csv(args.residual_csv)
    if len(df) != acts.shape[0]:
        raise SystemExit(
            f"Row count mismatch: CSV {len(df)} vs activations {acts.shape[0]}"
        )
    if "subject" not in df.columns or "group" not in df.columns:
        raise SystemExit("CSV needs 'subject' and 'group'")

    with open(args.pattern_catalog, encoding="utf-8") as f:
        catalog = json.load(f)
    patterns = catalog.get("patterns") or []
    if not patterns:
        raise SystemExit("pattern_catalog has no patterns")

    top_k = min(args.top_latents, len(patterns))
    latent_ids = [int(patterns[i]["latent_id"]) for i in range(top_k)]
    d_sae = acts.shape[1]
    for lid in latent_ids:
        if lid < 0 or lid >= d_sae:
            raise SystemExit(f"latent_id {lid} out of range for D={d_sae}")

    sub_acts = acts[:, latent_ids]
    groups = df["group"].astype(str).values
    subjects = df["subject"].astype(str).values

    high = groups == "High Residual"
    low = groups == "Low Residual"

    # Rank subjects by number of high+low rows (stability)
    subj_counts: dict[str, int] = {}
    for s in np.unique(subjects):
        m = subjects == s
        subj_counts[s] = int((m & high).sum() + (m & low).sum())
    ranked_subjects = sorted(subj_counts.keys(), key=lambda x: -subj_counts[x])[
        : args.max_subjects
    ]

    values: list[list[float | None]] = []
    n_high_grid: list[list[int]] = []
    n_low_grid: list[list[int]] = []
    for subj in ranked_subjects:
        sm = subjects == subj
        hi_idx = np.flatnonzero(sm & high)
        lo_idx = np.flatnonzero(sm & low)
        nh, nl = int(len(hi_idx)), int(len(lo_idx))
        if nh < args.min_per_side or nl < args.min_per_side:
            values.append([None] * top_k)
            n_high_grid.append([nh] * top_k)
            n_low_grid.append([nl] * top_k)
            continue
        row: list[float | None] = []
        row_nh: list[int] = []
        row_nl: list[int] = []
        for j in range(top_k):
            m_h = float(np.mean(sub_acts[hi_idx, j]))
            m_l = float(np.mean(sub_acts[lo_idx, j]))
            row.append(m_h - m_l)
            row_nh.append(nh)
            row_nl.append(nl)
        values.append(row)
        n_high_grid.append(row_nh)
        n_low_grid.append(row_nl)

    out = {
        "unit": "mean_sae_activation_high_residual_minus_low_residual",
        "min_per_side": args.min_per_side,
        "n_rows_total": int(acts.shape[0]),
        "d_sae": int(d_sae),
        "subjects": ranked_subjects,
        "latent_ids": latent_ids,
        "values": values,
        "n_high": n_high_grid,
        "n_low": n_low_grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output} ({len(ranked_subjects)}×{top_k} cells)", flush=True)


if __name__ == "__main__":
    main()
