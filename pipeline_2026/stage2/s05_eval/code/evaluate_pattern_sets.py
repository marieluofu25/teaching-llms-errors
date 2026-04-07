#!/usr/bin/env python3
"""Set-level metrics: final five metrics + residual proxies + optional judge JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from lib.io_schema import assert_columns


def residual_proxy_metrics(df: pd.DataFrame) -> dict:
    """Compute residual-group proxy metrics for backward compatibility."""
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


def _load_membership_matrix(path: Path) -> tuple[np.ndarray, list[int]]:
    """Load long-form pattern membership CSV into a dense 2D matrix.

    Returns:
        tuple[np.ndarray, list[int]]: Matrix ``(N, P)`` with 0/1 values and
        ordered pattern ids.
    """
    mdf = pd.read_csv(path)
    assert_columns(
        mdf, ["row_id", "pattern_id", "is_member"], name="pattern membership"
    )
    if mdf.empty:
        return np.zeros((0, 0), dtype=np.int8), []
    pattern_ids = sorted(int(x) for x in mdf["pattern_id"].dropna().unique().tolist())
    n_rows = int(mdf["row_id"].max()) + 1
    out = np.zeros((n_rows, len(pattern_ids)), dtype=np.int8)
    pidx = {p: i for i, p in enumerate(pattern_ids)}
    for row in mdf.itertuples(index=False):
        out[int(row.row_id), pidx[int(row.pattern_id)]] = int(row.is_member)
    return out, pattern_ids


def _coverage_concentration(y: np.ndarray, x: np.ndarray) -> dict:
    """Coverage and concentration using explicit pattern memberships."""
    n_errors = int(y.sum())
    covered = x.sum(axis=1) > 0 if x.size else np.zeros_like(y, dtype=bool)
    covered_errors = int(np.logical_and(covered, y == 1).sum())
    n_covered = int(covered.sum())
    coverage = covered_errors / n_errors if n_errors else 0.0
    concentration = covered_errors / n_covered if n_covered else 0.0
    return {
        "n_errors": n_errors,
        "n_covered_instances": n_covered,
        "n_covered_errors": covered_errors,
        "coverage": coverage,
        "concentration": concentration,
    }


def _predictive_utility(y: np.ndarray, x: np.ndarray, *, split_seed: int) -> dict:
    """Evaluate predictive utility of pattern membership features."""
    if x.size == 0 or x.shape[1] == 0:
        return {
            "split_seed": split_seed,
            "n_features": int(x.shape[1] if x.ndim == 2 else 0),
            "auc": None,
            "f1": None,
            "note": "No discovered patterns; predictive utility undefined.",
        }
    if len(np.unique(y)) < 2:
        return {
            "split_seed": split_seed,
            "n_features": int(x.shape[1]),
            "auc": None,
            "f1": None,
            "note": "Single-class labels; predictive utility undefined.",
        }
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=split_seed,
        stratify=y,
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "split_seed": split_seed,
        "n_features": int(x.shape[1]),
        "auc": float(roc_auc_score(y_test, prob)),
        "f1": float(f1_score(y_test, pred)),
    }


def _redundancy(x: np.ndarray) -> dict:
    """Estimate redundancy using average pairwise Jaccard overlap."""
    if x.size == 0 or x.shape[1] < 2:
        return {
            "n_patterns": int(x.shape[1] if x.ndim == 2 else 0),
            "avg_pairwise_jaccard": 0.0,
            "max_pairwise_jaccard": 0.0,
            "note": "Need at least two patterns for pairwise redundancy.",
        }
    pairs: list[float] = []
    for i in range(x.shape[1]):
        ai = x[:, i].astype(bool)
        for j in range(i + 1, x.shape[1]):
            bj = x[:, j].astype(bool)
            union = np.logical_or(ai, bj).sum()
            inter = np.logical_and(ai, bj).sum()
            pairs.append(float(inter / union) if union else 0.0)
    return {
        "n_patterns": int(x.shape[1]),
        "avg_pairwise_jaccard": float(np.mean(pairs)) if pairs else 0.0,
        "max_pairwise_jaccard": float(np.max(pairs)) if pairs else 0.0,
    }


def _load_catalog_latents(path: Path, *, top_k: int) -> list[int]:
    """Load ordered latent ids from pattern catalog JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patterns = data.get("patterns") or []
    out = [int(p["latent_id"]) for p in patterns if "latent_id" in p]
    return out[:top_k] if top_k > 0 else out


def _stability(
    current_catalog: Path | None,
    other_catalogs: list[Path],
    *,
    top_k: int,
) -> dict:
    """Compute pattern-set stability as overlap@k across catalogs."""
    if current_catalog is None or not current_catalog.is_file():
        return {"overlap_at_k_mean": None, "pairs_evaluated": 0, "note": "No current catalog."}
    base = set(_load_catalog_latents(current_catalog, top_k=top_k))
    if not base:
        return {"overlap_at_k_mean": 0.0, "pairs_evaluated": 0, "note": "No base patterns."}
    vals: list[float] = []
    for path in other_catalogs:
        if not path.is_file():
            continue
        comp = set(_load_catalog_latents(path, top_k=top_k))
        denom = max(len(base), 1)
        vals.append(len(base & comp) / denom)
    return {
        "top_k": top_k,
        "pairs_evaluated": len(vals),
        "overlap_at_k_mean": float(np.mean(vals)) if vals else None,
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
    """CLI entrypoint for evaluation metrics and optional leaderboard update."""
    parser = argparse.ArgumentParser(description="Evaluate final metrics + residual proxies")
    parser.add_argument("--residual-csv", type=Path, required=True)
    parser.add_argument(
        "--pattern-membership",
        type=Path,
        default=None,
        help="Long-form membership CSV from run_sae_diffing",
    )
    parser.add_argument(
        "--pattern-catalog",
        type=Path,
        default=None,
        help="Pattern catalog JSON from run_sae_diffing",
    )
    parser.add_argument(
        "--stability-catalogs",
        type=str,
        default="",
        help="Comma-separated additional pattern catalog JSON paths for stability.",
    )
    parser.add_argument("--stability-top-k", type=int, default=25)
    parser.add_argument("--split-seed", type=int, default=42)
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
    y = df["is_error"].astype(int).values if "is_error" in df.columns else np.array([], dtype=int)
    if y.size == 0:
        raise ValueError("residual CSV must contain is_error for final metrics")
    metrics = {
        "label": args.label,
        "residual_metrics": residual_proxy_metrics(df),
    }
    if args.pattern_membership is not None and args.pattern_membership.is_file():
        x, pattern_ids = _load_membership_matrix(args.pattern_membership)
        if x.shape[0] != len(df):
            raise ValueError(
                f"pattern membership rows {x.shape[0]} != residual rows {len(df)}"
            )
        metrics["set_level_metrics"] = {
            "coverage_concentration": _coverage_concentration(y, x),
            "predictive_utility": _predictive_utility(y, x, split_seed=args.split_seed),
            "redundancy": _redundancy(x),
            "stability": _stability(
                args.pattern_catalog,
                [Path(p) for p in args.stability_catalogs.split(",") if p.strip()],
                top_k=args.stability_top_k,
            ),
            "n_patterns": int(len(pattern_ids)),
            "pattern_membership_source": str(args.pattern_membership),
            "pattern_catalog_source": str(args.pattern_catalog) if args.pattern_catalog else None,
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
        sl = metrics.get("set_level_metrics") or {}
        cc = sl.get("coverage_concentration") or {}
        pu = sl.get("predictive_utility") or {}
        rd = sl.get("redundancy") or {}
        st = sl.get("stability") or {}
        row.update(
            {
                "coverage": cc.get("coverage"),
                "concentration": cc.get("concentration"),
                "predictive_utility_auc": pu.get("auc"),
                "predictive_utility_f1": pu.get("f1"),
                "redundancy_avg_jaccard": rd.get("avg_pairwise_jaccard"),
                "stability_overlap_at_k": st.get("overlap_at_k_mean"),
            }
        )
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
