#!/usr/bin/env python3
"""
Notebook-aligned paper figures (11 PNGs matching dataset-model-analysis.ipynb).

Lives under ``pipeline_2026``; uses **read-only** legacy code in ``pipeline/``
(``Dataset.py``, ``utils.py``, pickles). On import, switches CWD to
``<repo>/pipeline/`` so relative paths in ``Dataset`` / ``load_mathcamps_data`` work.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def _prepare_legacy_pipeline_cwd() -> Path:
    """Return ``pipeline/`` dir; insert ``stage1_do_errors_exist`` + ``pipeline`` on path; chdir to ``pipeline``."""
    here = Path(__file__).resolve()
    pipeline_2026 = here.parents[2]
    repo = pipeline_2026.parent
    pl = repo / "pipeline"
    if not pl.is_dir():
        raise FileNotFoundError(f"expected legacy pipeline at {pl}")
    st = pl / "stage1_do_errors_exist"
    for p in (str(st), str(pl)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(pl)
    return pl


_prepare_legacy_pipeline_cwd()

# Headless backend before pyplot / utils (utils imports pyplot).
import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from utils import (  # noqa: E402
    calculate_error_metrics,
    df_to_dict,
    load_mathcamps_data,
    plot_metrics,
    plot_metrics_fig,
    plot_metrics_fig_mathcamps,
)

MMLU_PICKLE = Path("stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl")
MC_OPENAI = Path("stage1_do_errors_exist/datasets/mathcamps/model-responses/v1/openai.json")
MC_ANTHROPIC = Path("stage1_do_errors_exist/datasets/mathcamps/model-responses/v1/anthropic.json")


def _cmap_blues():
    """Return a Blues colormap compatible with old/new matplotlib APIs."""
    reg = getattr(matplotlib, "colormaps", None)
    if reg is not None:
        return reg["Blues"]
    return cm.get_cmap("Blues")


def _save_plot_metrics(
    qm: dict,
    out: Path,
    filename: str,
    *,
    written: list[str],
    skipped: list[dict],
) -> None:
    """Render a metric chart via utils and write it to disk safely."""
    try:
        plot_metrics(qm)
        fig = plt.gcf()
        path = out / filename
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        written.append(filename)
    except Exception as e:  # noqa: BLE001
        plt.close("all")
        skipped.append({"figure": filename, "error": str(e)})


def _save_figure(fig, out: Path, filename: str, written: list[str], skipped: list[dict]) -> None:
    """Save an already-built figure and track success/failure."""
    try:
        path = out / filename
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        written.append(filename)
    except Exception as e:  # noqa: BLE001
        skipped.append({"figure": filename, "error": str(e)})
    finally:
        plt.close(fig)


def main() -> None:
    """Generate notebook-aligned paper baseline figures and summary JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[dict] = []

    summary: dict = {
        "figures_written": written,
        "figures_skipped": skipped,
    }

    try:
        from Dataset import MMLUDataset  # noqa: PLC0415
    except ImportError as e:
        summary["error"] = f"Dataset import failed: {e}"
        (out / "paper_mmlu_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"notebook_paper_figures: {summary['error']}", file=sys.stderr)
        return

    mmlu_mpl_names = [
        "paper_nb_cell03_mmlu_math_reg.png",
        "paper_nb_cell05_mmlu_health_reg.png",
        "paper_nb_cell07_mmlu_subcat_computer_science.png",
        "paper_nb_cell09_mmlu_all_subjects.png",
        "paper_nb_cell11_mmlu_by_subcat.png",
        "paper_nb_cell13_mmlu_subcats_psych_physics_math.png",
        "paper_nb_cell16_mmlu_math_health_vertical.png",
        "paper_nb_cell18_mmlu_math_health_horizontal.png",
        "paper_nb_cell19_mmlu_subcats_grid.png",
    ]
    mmlu_ok = MMLU_PICKLE.is_file()
    if not mmlu_ok:
        summary["error"] = "missing_pickle"
        summary["path"] = str(MMLU_PICKLE.resolve())
        for fname in mmlu_mpl_names:
            skipped.append({"figure": fname, "error": "missing_mmlu_pickle"})
    else:
        ds_full = MMLUDataset("mmlu", model_id="chatgpt")
        mmlu_data_all = ds_full.data

        mmlu_math = MMLUDataset("mmlu-math-reg", model_id="chatgpt").data
        df = calculate_error_metrics(mmlu_math, "subject", "ai_correct", mmlu=True)
        qm = df_to_dict(df, "subject")
        _save_plot_metrics(qm, out, "paper_nb_cell03_mmlu_math_reg.png", written=written, skipped=skipped)

        mmlu_health = MMLUDataset("mmlu-health-reg", model_id="chatgpt").data
        df = calculate_error_metrics(mmlu_health, "subject", "ai_correct", mmlu=True)
        qm = df_to_dict(df, "subject")
        _save_plot_metrics(qm, out, "paper_nb_cell05_mmlu_health_reg.png", written=written, skipped=skipped)

        sc = "computer science"
        m7 = MMLUDataset("mmlu", model_id="chatgpt")
        m7.choose_subcat_mmlu(subcat=sc)
        df = calculate_error_metrics(m7.data, "subject", "ai_correct", mmlu=True)
        qm = df_to_dict(df, "subject")
        _save_plot_metrics(
            qm, out, "paper_nb_cell07_mmlu_subcat_computer_science.png", written=written, skipped=skipped
        )

        mmlu_data = mmlu_data_all
        df = calculate_error_metrics(mmlu_data, "subject", "ai_correct", mmlu=True)
        qm = df_to_dict(df, "subject")
        _save_plot_metrics(qm, out, "paper_nb_cell09_mmlu_all_subjects.png", written=written, skipped=skipped)

        df = calculate_error_metrics(mmlu_data, "subcat", "ai_correct", mmlu=True)
        qm = df_to_dict(df, "subcat")
        _save_plot_metrics(qm, out, "paper_nb_cell11_mmlu_by_subcat.png", written=written, skipped=skipped)

        subcats_f = ["psyhology", "physics", "math"]
        m13 = mmlu_data_all[mmlu_data_all["subcat"].isin(subcats_f)]
        df = calculate_error_metrics(m13, "subject", "ai_correct", mmlu=True)
        qm = df_to_dict(df, "subject")
        _save_plot_metrics(
            qm, out, "paper_nb_cell13_mmlu_subcats_psych_physics_math.png", written=written, skipped=skipped
        )

        try:
            mmlu_math_data = MMLUDataset("mmlu-math-reg", model_id="chatgpt").data
            mmlu_health_data = MMLUDataset("mmlu-health-reg", model_id="chatgpt").data
            fig, axes = plt.subplots(2, 1, figsize=(5, 10), constrained_layout=False)
            df1 = calculate_error_metrics(mmlu_math_data, "subject", "ai_correct", mmlu=True)
            df2 = calculate_error_metrics(mmlu_health_data, "subject", "ai_correct", mmlu=True)
            qm1 = df_to_dict(df1, "subject")
            qm2 = df_to_dict(df2, "subject")
            fig_data = [qm1, qm2]
            plot_names = ["MMLU-Math", "MMLU-Health"]
            all_coverages = np.concatenate(
                [np.array([item["coverage"] for item in data.values()]) for data in fig_data]
            )
            norm = mcolors.Normalize(vmin=min(all_coverages), vmax=max(all_coverages))
            cmap = _cmap_blues()
            all_wrong_vs_correct = np.concatenate(
                [np.array([item["wrongVScorrect"] for item in data.values()]) for data in fig_data]
            )
            xlim = (0, max(all_wrong_vs_correct) * 1.1)
            for ax, data, title in zip(axes, fig_data, plot_names):
                plot_metrics_fig(data, ax, title, norm, cmap, xlim)
            fig.text(0.04, 0.5, "Subjects", ha="center", va="center", rotation="vertical", fontsize=14)
            plt.subplots_adjust(left=0.1, right=0.9, hspace=0.3)
            fig.canvas.draw()
            bbox0 = axes[0].get_position()
            bbox1 = axes[1].get_position()
            bottom = bbox1.y0
            top = bbox0.y1
            left = bbox0.x1 + 0.02
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([left, bottom, 0.02, top - bottom])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
            cbar.set_label("Coverage", fontsize=12, labelpad=20)
            _save_figure(fig, out, "paper_nb_cell16_mmlu_math_health_vertical.png", written, skipped)
        except Exception as e:  # noqa: BLE001
            plt.close("all")
            skipped.append({"figure": "paper_nb_cell16_mmlu_math_health_vertical.png", "error": str(e)})

        try:
            mmlu_math_data = MMLUDataset("mmlu-math-reg", model_id="chatgpt").data
            mmlu_health_data = MMLUDataset("mmlu-health-reg", model_id="chatgpt").data
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
            df1 = calculate_error_metrics(mmlu_math_data, "subject", "ai_correct", mmlu=True)
            df2 = calculate_error_metrics(mmlu_health_data, "subject", "ai_correct", mmlu=True)
            qm1 = df_to_dict(df1, "subject")
            qm2 = df_to_dict(df2, "subject")
            fig_data = [qm1, qm2]
            plot_names = ["MMLU-Math", "MMLU-Health"]
            all_coverages = np.concatenate(
                [np.array([item["coverage"] for item in data.values()]) for data in fig_data]
            )
            norm = mcolors.Normalize(vmin=min(all_coverages), vmax=max(all_coverages))
            cmap = _cmap_blues()
            all_wrong_vs_correct = np.concatenate(
                [np.array([item["wrongVScorrect"] for item in data.values()]) for data in fig_data]
            )
            xlim = (0, max(all_wrong_vs_correct) * 1.1)
            axes[0].set_ylabel("MMLU Subjects", fontsize=14)
            plt.subplots_adjust(left=0.05)
            for ax, data, title in zip(axes, fig_data, plot_names):
                plot_metrics_fig(data, ax, title, norm, cmap, xlim)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
            cbar.set_label("Coverage", fontsize=12, labelpad=20)
            _save_figure(fig, out, "paper_nb_cell18_mmlu_math_health_horizontal.png", written, skipped)
        except Exception as e:  # noqa: BLE001
            plt.close("all")
            skipped.append({"figure": "paper_nb_cell18_mmlu_math_health_horizontal.png", "error": str(e)})

        try:
            m0 = MMLUDataset("mmlu", model_id="chatgpt")
            subcats = list(m0.data.subcat.unique())
            subcats = [s for s in subcats if s not in ["math", "health"]]
            qms = []
            for subcat in subcats:
                md = MMLUDataset("mmlu", model_id="chatgpt")
                md.choose_subcat_mmlu(subcat=subcat)
                d = md.data
                df_m = calculate_error_metrics(d, "subject", "ai_correct", mmlu=True)
                qms.append(df_to_dict(df_m, "subject"))
            plot_names = [f"MMLU-{s}" for s in subcats]
            num_plots = len(qms)
            ncols = 3
            nrows = math.ceil(num_plots / ncols)
            fig_width = 15
            fig_height = 4 * nrows
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=False)
            axes = np.ravel(axes)
            all_coverages = np.concatenate(
                [np.array([item["coverage"] for item in data.values()]) for data in qms]
            )
            norm = mcolors.Normalize(vmin=np.min(all_coverages), vmax=np.max(all_coverages))
            cmap = _cmap_blues()
            all_wrong_vs_correct = np.concatenate(
                [np.array([item["wrongVScorrect"] for item in data.values()]) for data in qms]
            )
            xlim = (0, np.max(all_wrong_vs_correct) * 1.1)
            for ax, data, title in zip(axes, qms, plot_names):
                plot_metrics_fig(data, ax, title, norm, cmap, xlim)
            for ax in axes[len(qms) :]:
                ax.axis("off")
            fig.text(0, 0.5, "Subjects", ha="center", va="center", rotation="vertical", fontsize=14)
            plt.subplots_adjust(left=0.1, right=0.9, hspace=0.4, wspace=1.8)
            fig.canvas.draw()
            bboxes = [ax.get_position() for ax in axes if ax.get_visible()]
            bottom = min(b.y0 for b in bboxes)
            top = max(b.y1 for b in bboxes)
            left = max(b.x1 for b in bboxes) + 0.02
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar_ax = fig.add_axes([left, bottom, 0.02, top - bottom])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
            cbar.set_label("Coverage", fontsize=12, labelpad=20)
            _save_figure(fig, out, "paper_nb_cell19_mmlu_subcats_grid.png", written, skipped)
        except Exception as e:  # noqa: BLE001
            plt.close("all")
            skipped.append({"figure": "paper_nb_cell19_mmlu_subcats_grid.png", "error": str(e)})

        err_rate = float((~mmlu_data_all["ai_correct"]).mean())
        by_sub = (
            mmlu_data_all.groupby("subject", dropna=False)["ai_correct"]
            .apply(lambda s: float((~s).mean()))
            .sort_values(ascending=False)
            .head(30)
        )
        summary.update(
            {
                "n_rows": int(len(mmlu_data_all)),
                "error_rate": round(err_rate, 6),
                "top_subjects_by_error_rate": [
                    {"subject": str(k), "error_rate": round(float(v), 6)} for k, v in by_sub.items()
                ],
            }
        )

    mc_ok = MC_OPENAI.is_file() and MC_ANTHROPIC.is_file()
    mc_figures = [
        "paper_nb_cell24_mathcamps_gpt_claude_sonnet.png",
        "paper_nb_cell26_mathcamps_claude_family.png",
    ]
    if not mc_ok:
        summary["mathcamps_error"] = "missing_json"
        summary["mathcamps_paths"] = [str(MC_OPENAI.resolve()), str(MC_ANTHROPIC.resolve())]
        for fname in mc_figures:
            skipped.append({"figure": fname, "error": "missing_mathcamps_json"})
    else:
        try:
            mathcamps_gpt4 = load_mathcamps_data(str(MC_OPENAI), "gpt-4o-2024-05-13")
            mathcamps_claude_sonnet = load_mathcamps_data(str(MC_ANTHROPIC), "claude-3-sonnet-20240229")
            mathcamps_gpt35_turbo = load_mathcamps_data(str(MC_OPENAI), "gpt-3.5-turbo-0125")
            mathcamps_claude_opus = load_mathcamps_data(str(MC_ANTHROPIC), "claude-3-opus-20240229")
            mathcamps_claude_haiku = load_mathcamps_data(str(MC_ANTHROPIC), "claude-3-haiku-20240307")
        except Exception as e:  # noqa: BLE001
            summary["mathcamps_load_error"] = str(e)
            for fname in mc_figures:
                skipped.append({"figure": fname, "error": f"load_mathcamps: {e}"})

        if summary.get("mathcamps_load_error") is None:
            try:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                df1 = calculate_error_metrics(mathcamps_gpt4, "standard", "correct")
                df2 = calculate_error_metrics(mathcamps_gpt35_turbo, "standard", "correct")
                df3 = calculate_error_metrics(mathcamps_claude_sonnet, "standard", "correct")
                qm1 = df_to_dict(df1, "standard")
                qm2 = df_to_dict(df2, "standard")
                qm3 = df_to_dict(df3, "standard")
                for qm in (qm1, qm2, qm3):
                    qm.pop("8.EE.C.8", None)
                fig_data = [qm1, qm2, qm3]
                plot_names = ["GPT-4o", "GPT-3.5-turbo", "Claude-3-Sonnet"]
                all_coverages = np.concatenate(
                    [np.array([item["coverage"] for item in data.values()]) for data in fig_data]
                )
                norm = mcolors.Normalize(vmin=min(all_coverages), vmax=max(all_coverages))
                cmap = _cmap_blues()
                all_wrong_vs_correct = np.concatenate(
                    [np.array([item["wrongVScorrect"] for item in data.values()]) for data in fig_data]
                )
                xlim = (0, max(all_wrong_vs_correct) * 1.1)
                axes[0].set_ylabel("MathCAMPS Standard", fontsize=14)
                plt.subplots_adjust(left=0.05)
                for ax, data, title in zip(axes, fig_data, plot_names):
                    plot_metrics_fig_mathcamps(data, ax, title, norm, cmap, xlim)
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
                cbar.set_label("Coverage", fontsize=12, labelpad=20)
                _save_figure(fig, out, "paper_nb_cell24_mathcamps_gpt_claude_sonnet.png", written, skipped)
            except Exception as e:  # noqa: BLE001
                plt.close("all")
                skipped.append({"figure": "paper_nb_cell24_mathcamps_gpt_claude_sonnet.png", "error": str(e)})

            try:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                df1 = calculate_error_metrics(mathcamps_claude_sonnet, "standard", "correct")
                df2 = calculate_error_metrics(mathcamps_claude_opus, "standard", "correct")
                df3 = calculate_error_metrics(mathcamps_claude_haiku, "standard", "correct")
                qm1 = df_to_dict(df1, "standard")
                qm2 = df_to_dict(df2, "standard")
                qm3 = df_to_dict(df3, "standard")
                for qm in (qm1, qm2, qm3):
                    qm.pop("8.EE.C.8", None)
                fig_data = [qm1, qm2, qm3]
                plot_names = ["Sonnet", "Opus", "Haiku"]
                all_coverages = np.concatenate(
                    [np.array([item["coverage"] for item in data.values()]) for data in fig_data]
                )
                norm = mcolors.Normalize(vmin=min(all_coverages), vmax=max(all_coverages))
                cmap = _cmap_blues()
                all_wrong_vs_correct = np.concatenate(
                    [np.array([item["wrongVScorrect"] for item in data.values()]) for data in fig_data]
                )
                xlim = (0, max(all_wrong_vs_correct) * 1.1)
                axes[0].set_ylabel("MathCAMPS Standard", fontsize=14)
                plt.subplots_adjust(left=0.05)
                for ax, data, title in zip(axes, fig_data, plot_names):
                    plot_metrics_fig_mathcamps(data, ax, title, norm, cmap, xlim)
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
                cbar.set_label("Coverage", fontsize=12, labelpad=20)
                _save_figure(fig, out, "paper_nb_cell26_mathcamps_claude_family.png", written, skipped)
            except Exception as e:  # noqa: BLE001
                plt.close("all")
                skipped.append({"figure": "paper_nb_cell26_mathcamps_claude_family.png", "error": str(e)})

    (out / "paper_mmlu_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'paper_mmlu_summary.json'}")
    for w in written:
        print(f"Wrote {out / w}")
    if skipped:
        print(
            f"notebook_paper_figures: {len(skipped)} figure(s) skipped — see paper_mmlu_summary.json",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
