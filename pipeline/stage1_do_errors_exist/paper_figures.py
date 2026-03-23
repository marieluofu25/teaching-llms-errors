#!/usr/bin/env python3
"""
Optional legacy helper: quick MMLU summary + one bar chart.

For **notebook-aligned figures (11 PNGs)**, use ``pipeline_2026``:
``python -m stage1.code.run_paper_baseline`` (runs ``stage1/code/notebook_paper_figures.py``).

Run this file with **CWD = ``pipeline/``** (same as ``dataset-model-analysis.ipynb``) so
``from Dataset import MMLUDataset`` and pickle paths work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        from Dataset import MMLUDataset  # noqa: PLC0415 — legacy layout
    except ImportError as e:
        print(f"paper_figures: skip Dataset import ({e})", file=sys.stderr)
        return

    pkl = Path("stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl")  # CWD = pipeline/
    if not pkl.is_file():
        print(f"paper_figures: missing {pkl.resolve()} — skip plots", file=sys.stderr)
        summary = {"error": "missing_pickle", "path": str(pkl.resolve())}
        (out / "paper_mmlu_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return

    ds = MMLUDataset("mmlu", model_id="chatgpt")
    df = ds.data
    err_rate = float((~df["ai_correct"]).mean())
    by_sub = (
        df.groupby("subject", dropna=False)["ai_correct"]
        .apply(lambda s: float((~s).mean()))
        .sort_values(ascending=False)
        .head(30)
    )
    summary = {
        "n_rows": int(len(df)),
        "error_rate": round(err_rate, 6),
        "top_subjects_by_error_rate": [
            {"subject": str(k), "error_rate": round(float(v), 6)}
            for k, v in by_sub.items()
        ],
    }
    (out / "paper_mmlu_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out / 'paper_mmlu_summary.json'}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("paper_figures: matplotlib not installed — skip PNG", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(by_sub.head(20)))))
    sub20 = by_sub.head(20)
    ax.barh(range(len(sub20)), sub20.values, color="#2c7fb8")
    ax.set_yticks(range(len(sub20)))
    ax.set_yticklabels([str(x)[:35] for x in sub20.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Error rate")
    ax.set_title("MMLU — error rate by subject (top 20, paper-style quick view)")
    fig.tight_layout()
    png = out / "paper_mmlu_error_by_subject_top20.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
