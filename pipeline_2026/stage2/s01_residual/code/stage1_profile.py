"""Stage 1 summary stats + optional figures (notebook-style overview, lightweight)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _group_error_table(
    df: pd.DataFrame,
    key: str,
    *,
    min_n: int = 5,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    if key not in df.columns:
        return []
    sub = df[[key, "is_error"]].dropna(subset=[key])
    if sub.empty:
        return []
    g = (
        sub.groupby(key, dropna=False)
        .agg(n=("is_error", "size"), errors=("is_error", "sum"))
        .reset_index()
    )
    g["error_rate"] = (g["errors"] / g["n"].replace(0, np.nan)).astype(float)
    g = g[g["n"] >= min_n].sort_values("n", ascending=False).head(top_k)
    rows: list[dict[str, Any]] = []
    for _, r in g.iterrows():
        rows.append(
            {
                str(key): str(r[key]) if r[key] is not None else "",
                "n": int(r["n"]),
                "errors": int(r["errors"]),
                "error_rate": round(float(r["error_rate"]), 4) if r["n"] > 0 else None,
            }
        )
    return rows


def build_stage1_profile(df: pd.DataFrame, *, dataset: str) -> dict[str, Any]:
    """
    Aggregate view similar in spirit to ``dataset-model-analysis.ipynb``:
    global error rate, residual groups, and (for MMLU) subject/subcat breakdowns.
    """
    profile: dict[str, Any] = {
        "dataset": dataset,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }
    if "is_error" in df.columns:
        profile["error_rate"] = round(float(df["is_error"].mean()), 6)
        profile["n_errors"] = int(df["is_error"].sum())
    if "group" in df.columns:
        vc = df["group"].value_counts()
        profile["group_counts"] = {str(k): int(v) for k, v in vc.items()}
    if "expected_error" in df.columns:
        profile["expected_error_mean"] = round(float(df["expected_error"].mean()), 6)
    if "residual_error" in df.columns:
        profile["residual_error_quantiles"] = {
            "q25": round(float(df["residual_error"].quantile(0.25)), 6),
            "q50": round(float(df["residual_error"].quantile(0.50)), 6),
            "q75": round(float(df["residual_error"].quantile(0.75)), 6),
        }

    if dataset == "mmlu":
        profile["by_subject"] = _group_error_table(df, "subject")
        profile["by_subcat"] = _group_error_table(df, "subcat")
        profile["by_cat"] = _group_error_table(df, "cat")
    elif dataset == "mathcamps":
        profile["by_standard"] = _group_error_table(df, "standard")
        profile["by_model"] = _group_error_table(df, "model")

    return _json_safe(profile)


def write_subject_error_plot(
    df: pd.DataFrame,
    out_png: Path,
    *,
    key: str = "subject",
    top_n: int = 25,
    title: str = "Error rate by subject (top n by volume)",
) -> bool:
    """Bar chart of error_rate for the top ``top_n`` groups by count. Returns True if written."""
    if key not in df.columns or "is_error" not in df.columns:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    sub = df[[key, "is_error"]].dropna(subset=[key])
    if sub.empty:
        return False
    g = (
        sub.groupby(key, dropna=False)
        .agg(n=("is_error", "size"), errors=("is_error", "sum"))
        .reset_index()
    )
    g["error_rate"] = g["errors"] / g["n"].replace(0, np.nan)
    g = g.sort_values("n", ascending=False).head(top_n)
    if g.empty:
        return False

    labels = [str(x)[:40] + ("…" if len(str(x)) > 40 else "") for x in g[key]]
    y = g["error_rate"].values

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(labels))))
    ax.barh(range(len(labels)), y, color="#2c7fb8")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Error rate (is_error mean)")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_stage1_artifacts(
    df: pd.DataFrame,
    *,
    dataset: str,
    output_csv: Path,
    write_plot: bool = True,
) -> tuple[Path, Path | None]:
    """
    Write ``<stem>_profile.json`` and optional ``<stem>_error_by_subject.png``
    next to the residual CSV.
    """
    stem = output_csv.stem
    parent = output_csv.parent
    profile_path = parent / f"{stem}_profile.json"
    plot_path = parent / f"{stem}_error_by_subject.png"

    prof = build_stage1_profile(df, dataset=dataset)
    profile_path.write_text(json.dumps(prof, indent=2), encoding="utf-8")

    png_written: Path | None = None
    if write_plot and dataset == "mmlu":
        if write_subject_error_plot(df, plot_path, key="subject"):
            png_written = plot_path
    elif write_plot and dataset == "mathcamps":
        alt = parent / f"{stem}_error_by_standard.png"
        if write_subject_error_plot(
            df,
            alt,
            key="standard",
            title="Error rate by standard (top by volume)",
        ):
            png_written = alt

    return profile_path, png_written
