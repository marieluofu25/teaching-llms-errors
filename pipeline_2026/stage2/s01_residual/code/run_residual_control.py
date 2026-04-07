#!/usr/bin/env python3
"""CLI: build residual-control table from MMLU pickle or MathCAMPs JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.repo_paths import teaching_llms_errors_repo_root
from lib.io_schema import validate_mmlu_frame, validate_residual_output
from stage2.s01_residual.code.difficulty_control import (
    add_is_error_from_ai_correct,
    add_is_error_from_correct,
    compute_residuals_and_group,
)
from stage2.s01_residual.code.stage1_profile import write_stage1_artifacts
DATA_ROOT = teaching_llms_errors_repo_root()


def _flatten_mathcamps(obj, model_filter: str | None = None) -> list[dict]:
    """Flatten nested MathCAMPs JSON into row records for one model (optional)."""
    rows: list[dict] = []

    def visit(node):
        if not isinstance(node, dict):
            return
        if "question" in node and "correct" in node:
            m = node.get("model")
            if model_filter is None or m == model_filter:
                rows.append(
                    {
                        "problem": node.get("problem"),
                        "standard": node.get("standard"),
                        "question": node.get("question"),
                        "correct": bool(node.get("correct")),
                        "model": m,
                    }
                )
        for v in node.values():
            if isinstance(v, dict):
                visit(v)
            elif isinstance(v, list):
                for item in v:
                    visit(item)

    if isinstance(obj, dict):
        visit(obj)
    return rows


def load_mmlu_dataframe(pickle_path: Path) -> pd.DataFrame:
    """Load MMLU pickle and derive canonical columns for Stage 2 processing."""
    df = pd.read_pickle(pickle_path)
    df = df.copy()
    df["subject"] = df["metadata"].apply(lambda x: x[0])
    df["subcat"] = df["metadata"].apply(lambda x: x[1])
    df["cat"] = df["metadata"].apply(lambda x: x[2])
    df["ai_correct"] = df.apply(lambda row: row["data_y"] == row["ai_preds"], axis=1)
    validate_mmlu_frame(df)
    return df


def main() -> None:
    """CLI entrypoint for residual-control preprocessing and artifact writing."""
    parser = argparse.ArgumentParser(description="Residual-control preprocessing")
    parser.add_argument(
        "--dataset",
        choices=("mmlu", "mathcamps"),
        required=True,
    )
    parser.add_argument(
        "--mmlu-pickle",
        type=Path,
        default=DATA_ROOT
        / "pipeline/stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl",
        help="Path to mmlu_df_*.pkl",
    )
    parser.add_argument(
        "--model-id",
        default="gpt35",
        help="Suffix for default pickle name when using shortcuts (gpt35)",
    )
    parser.add_argument(
        "--mathcamps-json",
        type=Path,
        default=DATA_ROOT
        / "pipeline/stage1_do_errors_exist/datasets/mathcamps/model-responses/v1/openai.json",
    )
    parser.add_argument(
        "--model-filter",
        default=None,
        help="For mathcamps: e.g. gpt-3.5-turbo-0125",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .csv or .parquet",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("full", "train_only"),
        default="train_only",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-profile-plot",
        action="store_true",
        help="Skip matplotlib PNG (still writes *_profile.json)",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip *_profile.json and overview plot entirely",
    )
    args = parser.parse_args()

    if args.dataset == "mmlu":
        path = args.mmlu_pickle
        if not path.exists():
            raise FileNotFoundError(path)
        df = load_mmlu_dataframe(path)
        df = add_is_error_from_ai_correct(df)
        text_col = "questions"
        error_col = "is_error"
    else:
        path = args.mathcamps_json
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = _flatten_mathcamps(raw, model_filter=args.model_filter)
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No mathcamps rows after flatten (check --model-filter).")
        df = add_is_error_from_correct(df)
        text_col = "question"
        error_col = "is_error"

    out_df = compute_residuals_and_group(
        df,
        text_col,
        error_col,
        fit_mode=args.fit_mode,
        random_state=args.random_state,
    )
    validate_residual_output(out_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        try:
            out_df.to_parquet(args.output, index=False)
        except ImportError as e:
            raise SystemExit(
                "Parquet output requires pyarrow or fastparquet. "
                "Use .csv or: pip install pyarrow"
            ) from e
    else:
        out_df.to_csv(args.output, index=False)

    meta = {
        "dataset": args.dataset,
        "n_rows": len(out_df),
        "fit_mode": args.fit_mode,
        "text_col": text_col,
        "error_col": error_col,
        "output": str(args.output.resolve()),
        "group_counts": out_df["group"].value_counts().to_dict(),
        "mean_is_error": float(out_df["is_error"].mean()),
    }
    if args.dataset == "mathcamps":
        meta["model_filter"] = args.model_filter
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output} and {meta_path}")

    if not args.skip_profile:
        prof_path, plot_path = write_stage1_artifacts(
            out_df,
            dataset=args.dataset,
            output_csv=args.output,
            write_plot=not args.no_profile_plot,
        )
        print(f"Wrote {prof_path}")
        if plot_path is not None:
            print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
