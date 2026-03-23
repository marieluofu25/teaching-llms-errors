#!/usr/bin/env python3
"""Set-level metrics: residual-group proxies + optional judge JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.io_schema import assert_columns


def residual_proxy_metrics(df: pd.DataFrame) -> dict:
    """Coverage / concentration for High Residual vs all errors."""
    assert_columns(df, ["is_error", "group"], name="residual df")
    err = df["is_error"].astype(int) == 1
    high = df["group"] == "High Residual"
    low = df["group"] == "Low Residual"

    n_errors = int(err.sum())
    n_high = int(high.sum())
    errors_and_high = int((err & high).sum())

    coverage_high = errors_and_high / n_errors if n_errors else 0.0
    concentration_high = errors_and_high / n_high if n_high else 0.0

    errors_and_low = int((err & low).sum())
    n_low = int(low.sum())
    coverage_low = errors_and_low / n_errors if n_errors else 0.0
    concentration_low = errors_and_low / n_low if n_low else 0.0

    return {
        "n_instances": len(df),
        "n_errors": n_errors,
        "error_rate": float(df["is_error"].mean()),
        "high_residual": {
            "n": n_high,
            "coverage_of_errors": coverage_high,
            "concentration_in_group": concentration_high,
        },
        "low_residual": {
            "n": n_low,
            "coverage_of_errors": coverage_low,
            "concentration_in_group": concentration_low,
        },
    }


def load_judge_summary(path: Path) -> dict:
    """Extract summary stats from pipeline judge output JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Structure: { setting_key: { f1, avg_rating_hyps, ... } }
    out: dict = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = {
                "f1": v.get("f1"),
                "avg_rating_hyps": v.get("avg_rating_hyps"),
                "avg_rating_stds": v.get("avg_rating_stds"),
            }
    return {"judge_blocks": out, "source": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual / judge metrics")
    parser.add_argument("--residual-csv", type=Path, required=True)
    parser.add_argument("--judge-json", type=Path, default=None)
    parser.add_argument("--label", default="run", help="Label for this row in summary table")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-table",
        type=Path,
        default=None,
        help="Append one CSV row to this leaderboard file",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.residual_csv)
    metrics = {
        "label": args.label,
        "residual_metrics": residual_proxy_metrics(df),
    }
    if args.judge_json is not None:
        metrics["stage2_judge"] = load_judge_summary(args.judge_json)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.output_table:
        row = {
            "label": args.label,
            "n_instances": metrics["residual_metrics"]["n_instances"],
            "n_errors": metrics["residual_metrics"]["n_errors"],
            "high_coverage": metrics["residual_metrics"]["high_residual"][
                "coverage_of_errors"
            ],
            "high_concentration": metrics["residual_metrics"]["high_residual"][
                "concentration_in_group"
            ],
        }
        args.output_table.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame([row])
        if args.output_table.exists():
            old = pd.read_csv(args.output_table)
            old = old.dropna(how="all")
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df.to_csv(args.output_table, index=False)

    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
