#!/usr/bin/env python3
"""Build ``mmlu_full_residuals.csv`` from unified-model predictions (Phase 1 residuals).

Expects columns: questions, data_y, ai_preds, ai_correct, subject (optional), row_id, ai_model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.io_schema import validate_mmlu_frame, validate_residual_output
from stage2.s01_residual.code.difficulty_control import (
    add_is_error_from_ai_correct,
    compute_residuals_and_group,
)
from stage2.s01_residual.code.stage1_profile import write_stage1_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description="Residual table from Gemma/unified predictions CSV")
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument(
        "--fit-mode",
        choices=("full", "train_only"),
        default="train_only",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--subject-col",
        default="subject",
        help="Topic column for difficulty LR; '' disables",
    )
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--no-profile-plot", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.predictions_csv)
    if "ai_correct" not in df.columns:
        raise ValueError("predictions CSV must include ai_correct")
    validate_mmlu_frame(df)
    df = add_is_error_from_ai_correct(df)
    subj = args.subject_col.strip() or None

    out_df = compute_residuals_and_group(
        df,
        "questions",
        "is_error",
        fit_mode=args.fit_mode,
        random_state=args.random_state,
        subject_col=subj,
    )
    validate_residual_output(out_df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    meta = {
        "source_predictions": str(args.predictions_csv.resolve()),
        "n_rows": len(out_df),
        "fit_mode": args.fit_mode,
        "group_counts": out_df["group"].value_counts().to_dict(),
        "mean_is_error": float(out_df["is_error"].mean()),
    }
    meta_path = args.output_csv.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output_csv} and {meta_path}")

    if not args.skip_profile:
        prof_path, plot_path = write_stage1_artifacts(
            out_df,
            dataset="mmlu",
            output_csv=args.output_csv,
            write_plot=not args.no_profile_plot,
        )
        print(f"Wrote {prof_path}")
        if plot_path:
            print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
