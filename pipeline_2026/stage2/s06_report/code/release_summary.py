#!/usr/bin/env python3
"""
Executive / release-style summary: behavioral model id, error rates overall and per MMLU subject,
optional merge of evaluate_pattern_sets JSON, optional activations export meta, and threshold gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from lib.repo_paths import pipeline_2026_root

ROOT = pipeline_2026_root()


def _load_thresholds(path: Path) -> dict:
    """Load threshold gate settings from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("gates") or {}


def _behavioral_model_info(df: pd.DataFrame) -> dict:
    """Infer behavioral model metadata from residual CSV columns."""
    if "ai_model" in df.columns:
        vals = df["ai_model"].dropna().astype(str).unique().tolist()
        mode = df["ai_model"].dropna().astype(str).mode()
        mode_s = str(mode.iloc[0]) if len(mode) else None
        return {
            "source": "residual_csv_column_ai_model",
            "unique_values": sorted(vals)[:20],
            "mode": mode_s,
        }
    return {
        "source": "unknown",
        "note": "No ai_model column; use MMLU pickle / paper provenance for the logged LLM.",
    }


def _per_subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-subject error rate table when subject is available."""
    if "subject" not in df.columns:
        return pd.DataFrame()
    g = df.groupby("subject", dropna=False)["is_error"].agg(["count", "mean"])
    g = g.rename(columns={"count": "n", "mean": "error_rate"})
    g["accuracy"] = 1.0 - g["error_rate"]
    g = g.reset_index()
    return g.sort_values("error_rate", ascending=False)


def _apply_gates(
    overall_er: float,
    per_subject: pd.DataFrame,
    gates: dict,
) -> tuple[str, list[dict]]:
    """Apply threshold gates and produce pass/fail verdict with violations."""
    violations: list[dict] = []
    max_g = float(gates.get("max_error_rate_global", 1.0))
    max_w = float(gates.get("max_error_rate_worst_subject", 1.0))
    min_n = int(gates.get("min_rows_per_subject", 1))

    if overall_er > max_g:
        violations.append(
            {
                "gate": "max_error_rate_global",
                "detail": f"overall error_rate {overall_er:.4f} > {max_g}",
            }
        )

    eligible = per_subject[per_subject["n"] >= min_n] if not per_subject.empty else per_subject
    if not eligible.empty:
        worst = float(eligible["error_rate"].max())
        if worst > max_w:
            worst_row = eligible.loc[eligible["error_rate"].idxmax()]
            violations.append(
                {
                    "gate": "max_error_rate_worst_subject",
                    "detail": (
                        f"worst subject among n>={min_n} is {worst_row['subject']!r} "
                        f"with error_rate {worst:.4f} > {max_w}"
                    ),
                }
            )

    verdict = "fail" if violations else "pass"
    return verdict, violations


def build_release_summary(
    df: pd.DataFrame,
    *,
    metrics_json: dict | None,
    activations_meta: dict | None,
    thresholds: dict,
) -> dict:
    """Build release summary dictionary for JSON export."""
    df = df.copy()
    if "is_error" not in df.columns:
        raise ValueError("DataFrame must contain is_error")
    is_err = df["is_error"].astype(int)
    n = len(df)
    overall_er = float(is_err.mean())
    per = _per_subject_stats(df)
    per_records = per.to_dict(orient="records") if not per.empty else []

    verdict, violations = _apply_gates(overall_er, per, thresholds)

    per_asc = per.sort_values("error_rate", ascending=True) if not per.empty else per
    best_records = per_asc.head(10).to_dict(orient="records") if not per_asc.empty else []

    out: dict = {
        "behavioral_model": _behavioral_model_info(df),
        "representation_model": activations_meta,
        "overall": {
            "n_instances": n,
            "n_errors": int(is_err.sum()),
            "error_rate": overall_er,
            "accuracy": 1.0 - overall_er,
        },
        "per_subject": per_records,
        "worst_subjects": per_records[: min(10, len(per_records))],
        "best_subjects": best_records,
        "gates": {
            "thresholds_used": thresholds,
            "verdict": verdict,
            "violations": violations,
        },
    }
    if metrics_json and "residual_metrics" in metrics_json:
        out["residual_metrics_from_eval"] = metrics_json["residual_metrics"]
    if metrics_json and "set_level_metrics" in metrics_json:
        out["final_set_level_metrics"] = metrics_json["set_level_metrics"]
    return out


def main() -> None:
    """CLI entrypoint for release summary generation."""
    p = argparse.ArgumentParser(description="Build release_readiness.json from residual CSV")
    p.add_argument("--residual-csv", type=Path, required=True)
    p.add_argument("--metrics-json", type=Path, default=None, help="Output from evaluate_pattern_sets")
    p.add_argument(
        "--activations-meta",
        type=Path,
        default=None,
        help="Sidecar JSON next to export_activations .npz",
    )
    p.add_argument(
        "--thresholds-yaml",
        type=Path,
        default=ROOT / "config/release_thresholds.yaml",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "stage2/s06_report/results/release_readiness.json",
    )
    args = p.parse_args()

    df = pd.read_csv(args.residual_csv)
    if "is_error" not in df.columns and "ai_correct" in df.columns:
        df["is_error"] = (~df["ai_correct"].astype(bool)).astype(int)
    if "is_error" not in df.columns:
        raise ValueError("Need is_error or ai_correct in CSV")

    metrics = None
    if args.metrics_json is not None and args.metrics_json.is_file():
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    act_meta = None
    if args.activations_meta is not None and args.activations_meta.is_file():
        with open(args.activations_meta, "r", encoding="utf-8") as f:
            act_meta = json.load(f)

    thresholds = _load_thresholds(args.thresholds_yaml)
    summary = build_release_summary(
        df,
        metrics_json=metrics,
        activations_meta=act_meta,
        thresholds=thresholds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output} verdict={summary['gates']['verdict']}")


if __name__ == "__main__":
    main()
