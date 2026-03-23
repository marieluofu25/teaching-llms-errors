#!/usr/bin/env python3
"""
Five-tab HTML report: MMLU residual pipeline + optional SAE activations.

Tab 5 bucketing (documented in HTML footer):
  For each latent j, tertile boundaries q33, q66 are computed on activations[:, j]
  across ALL rows. Let mean_c = mean activation on correct rows (is_error==0),
  mean_w = mean on incorrect rows (is_error==1). Each mean is mapped to
  low / mid / high by comparing to q33 and q66.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lib.io_schema import validate_activations_shape
from lib.repo_paths import pipeline_2026_root

ROOT = pipeline_2026_root()


def _text_col(df: pd.DataFrame) -> str:
    if "questions" in df.columns:
        return "questions"
    if "question" in df.columns:
        return "question"
    raise ValueError("DataFrame needs column 'questions' (MMLU) or 'question' (MathCAMPs).")


def _load_meanings(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    lm = raw.get("latent_meanings") or {}
    return {int(k): str(v) for k, v in lm.items()}


def _format_read_first_html() -> str:
    return """
  <div class="read-first">
    <h2>Read this first</h2>
    <ol>
      <li><strong>Executive summary</strong> (when <code>release_readiness.json</code> is linked): model names, overall error %, per-subject table, and a threshold-based verdict.</li>
      <li><strong>Pipeline overview</strong> (blue box): what each of the five tabs means, in order.</li>
      <li><strong>Tabs 1&ndash;5</strong>: raw split, expected error, residuals, internal dimensions, correct vs failure buckets.</li>
      <li><strong>Glossary</strong> (end of page): definitions and limits of this report.</li>
    </ol>
  </div>
"""


def _format_executive_summary_html(release: dict) -> str:
    """Render release_readiness.json (from src/release_summary.py) as HTML."""
    beh = release.get("behavioral_model") or {}
    rep = release.get("representation_model")
    ov = release.get("overall") or {}
    gates = release.get("gates") or {}
    verdict = gates.get("verdict", "—")
    violations = gates.get("violations") or []
    thresholds = gates.get("thresholds_used") or {}

    vclass = "verdict-pass" if verdict == "pass" else "verdict-fail"

    beh_lines = []
    if beh.get("mode"):
        beh_lines.append(f"<strong>Logged LLM (behavioral):</strong> {html.escape(str(beh['mode']))}")
    elif beh.get("unique_values"):
        beh_lines.append(
            "<strong>Logged LLM (behavioral):</strong> "
            + html.escape(", ".join(str(x) for x in beh["unique_values"][:5]))
        )
    else:
        beh_lines.append(
            "<strong>Logged LLM (behavioral):</strong> "
            + html.escape(str(beh.get("note", "see CSV / pickle provenance")))
        )

    rep_html = ""
    if isinstance(rep, dict) and rep.get("model"):
        rt = rep.get("representation_type", "")
        dim_note = ""
        if rt == "sae_latent":
            dim_note = (
                f" <span class=\"muted\">(<strong>SAE latent space</strong>: "
                f"<code>d_sae={html.escape(str(rep.get('d_sae', '?')))}</code>, "
                f"<code>d_in={html.escape(str(rep.get('d_in', '?')))}</code>; "
                f"{html.escape(str(rep.get('sae_release', '')))}/"
                f"{html.escape(str(rep.get('sae_id', '')))})</span>"
            )
        elif rt == "hidden_pooled":
            dim_note = (
                f" <span class=\"muted\">(<strong>pooled hidden</strong>, not SAE: "
                f"<code>d_in={html.escape(str(rep.get('d_in', rep.get('hidden_dim', '?'))))}</code>)</span>"
            )
        elif rt == "synthetic_demo":
            dim_note = (
                f" <span class=\"muted\">(synthetic demo, <code>d_in={html.escape(str(rep.get('d_in', '?')))}</code>)</span>"
            )
        rep_html = (
            f"<p><strong>Representation model</strong> (activations in <code>.npz</code>): "
            f"{html.escape(str(rep.get('model')))}{dim_note}"
            f"{(' — ' + html.escape(str(rep.get('note', '')))) if rep.get('note') else ''}</p>"
        )
    else:
        rep_html = (
            "<p><strong>Representation model:</strong> not linked (no <code>activations</code> meta). "
            "If you run <code>export_activations</code>, the sidecar <code>*.meta.json</code> is merged here.</p>"
        )

    er = float(ov.get("error_rate", 0.0))
    acc = float(ov.get("accuracy", 1.0 - er))
    n = int(ov.get("n_instances", 0))

    viol_html = ""
    if violations:
        viol_html = "<ul>" + "".join(f"<li>{html.escape(str(v.get('detail', v)))}</li>" for v in violations) + "</ul>"
    else:
        viol_html = "<p><em>No threshold violations.</em></p>"

    thr_bits = []
    if thresholds:
        thr_bits.append(
            f"global ≤ {thresholds.get('max_error_rate_global', '—')}, "
            f"worst subject ≤ {thresholds.get('max_error_rate_worst_subject', '—')} "
            f"(subjects with n ≥ {thresholds.get('min_rows_per_subject', '—')})"
        )
    thr_str = " ".join(thr_bits) if thr_bits else "(see config/release_thresholds.yaml)"

    per = release.get("per_subject") or []
    per_note = ""
    if len(per) > 60:
        per = per[:60]
        per_note = "<p class=\"muted\">Showing first 60 subjects by table size cap.</p>"
    subj_df = pd.DataFrame(per) if per else pd.DataFrame()
    if not subj_df.empty and "error_rate" in subj_df.columns:
        subj_df["error_rate_pct"] = (subj_df["error_rate"] * 100).round(2)
        subj_df["accuracy_pct"] = (subj_df["accuracy"] * 100).round(2)
        cols = [c for c in ["subject", "n", "error_rate_pct", "accuracy_pct"] if c in subj_df.columns]
        subj_df = subj_df[cols]
        subj_df = subj_df.rename(
            columns={
                "error_rate_pct": "error %",
                "accuracy_pct": "accuracy %",
            }
        )
    subj_table = (
        subj_df.to_html(index=False, escape=True, classes="data", border=0)
        if not subj_df.empty
        else "<p><em>No subject column in CSV.</em></p>"
    )

    eval_note = ""
    if release.get("residual_metrics_from_eval"):
        rm = release["residual_metrics_from_eval"]
        hr = rm.get("high_residual", {})
        eval_note = (
            f"<p class=\"muted\"><strong>From evaluate_pattern_sets:</strong> "
            f"coverage of errors in High Residual group = {hr.get('coverage_of_errors', 0):.3f}.</p>"
        )

    return f"""
  <div class="executive-summary {vclass}">
    <h2>Executive summary &amp; release-style gates</h2>
    <p class="muted">Verdict is <strong>policy-based</strong> on YAML thresholds, not a universal product-ship signal.</p>
    <p>{"".join(beh_lines)}</p>
    {rep_html}
    <p>
      <strong>Overall on this CSV:</strong> n = {n}, error rate = <strong>{100 * er:.2f}%</strong>,
      accuracy = <strong>{100 * acc:.2f}%</strong>.
    </p>
    {eval_note}
    <p><strong>Gate verdict:</strong> <span class="verdict-tag">{html.escape(str(verdict).upper())}</span> ({thr_str})</p>
    {viol_html}
    <h3>Performance by MMLU subject</h3>
    {per_note}
    {subj_table}
  </div>
"""


def _representation_meta_html(meta: dict | None) -> str:
    """Short HTML note for pipeline overview when activations sidecar is present."""
    if not meta:
        return ""
    rt = meta.get("representation_type", "")
    if rt == "sae_latent":
        return (
            "<p class=\"muted\"><strong>Loaded representation:</strong> SAE latent vectors "
            f"(<code>d_sae={html.escape(str(meta.get('d_sae', '?')))}</code>, "
            f"release <code>{html.escape(str(meta.get('sae_release', '?')))}</code> / "
            f"<code>{html.escape(str(meta.get('sae_id', '?')))}</code>). "
            "Each Tab 4/5 column index is a sparse feature id, not a raw hidden dimension.</p>"
        )
    if rt == "hidden_pooled":
        return (
            "<p class=\"muted\"><strong>Loaded representation:</strong> pooled Hugging Face "
            f"hidden states (<code>d_in={html.escape(str(meta.get('d_in', meta.get('hidden_dim', '?'))))}</code>). "
            "This is not an SAE feature space unless you run <code>encode_sae_latents</code>.</p>"
        )
    if rt == "synthetic_demo":
        return (
            "<p class=\"muted\"><strong>Loaded representation:</strong> synthetic random demo vectors "
            "(not from a real model or SAE).</p>"
        )
    return ""


def _format_glossary_html() -> str:
    return """
  <details class="glossary">
    <summary><strong>Glossary &amp; limits</strong></summary>
    <dl>
      <dt>Behavioral model</dt>
      <dd>The LLM whose answers are logged in the MMLU residual CSV (<code>ai_model</code> when present). This is what &ldquo;error rate&rdquo; describes.</dd>
      <dt>Representation model</dt>
      <dd>Optional: a different model used only to compute vectors in <code>activations.npz</code> (e.g. local Hugging Face). It does not change which answers were right or wrong in the CSV.</dd>
      <dt>Expected error / residual</dt>
      <dd><code>expected_error</code> is a difficulty proxy; <code>residual_error = is_error − expected_error</code> highlights mistakes that are hard to explain by surface difficulty alone.</dd>
      <dt>Release verdict</dt>
      <dd>pass/fail against <code>config/release_thresholds.yaml</code>. Tune thresholds for your course or product; this is not a substitute for production monitoring or safety review.</dd>
      <dt>SAE latent vs pooled hidden</dt>
      <dd>If <code>*.meta.json</code> has <code>representation_type: sae_latent</code>, each Tab 4/5 coordinate is a <strong>sparse dictionary feature</strong> (<code>d_sae</code> columns). If it is <code>hidden_pooled</code>, coordinates are raw Hugging Face hidden dimensions (<code>d_in</code>, often 768 for <code>gpt2</code>)—do not call that &ldquo;SAE diffing&rdquo; in a thesis without an encode step. See <code>docs/sae_checkpoints.md</code> and Jiang et al., <a href="https://arxiv.org/abs/2512.10092">arXiv:2512.10092</a>.</dd>
      <dt>Further reading (feature dashboards)</dt>
      <dd>External resources such as <a href="https://www.neuronpedia.org/">Neuronpedia</a> map public-model features to human-readable labels; your run may or may not use the same SAE release.</dd>
    </dl>
  </details>
"""


def _df_to_html_table(df: pd.DataFrame, max_rows: int | None) -> str:
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    if df.empty:
        return "<p><em>(no rows)</em></p>"
    return df.to_html(index=False, escape=True, classes="data", border=0)


def _truncate_q(text: str, max_chars: int) -> str:
    t = " ".join(str(text).split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _tab4_question_trace_section(
    df: pd.DataFrame,
    acts: np.ndarray,
    feat_ids: list[int],
    text_col: str,
    *,
    k: int,
    max_chars: int,
    max_latents: int,
) -> str:
    """
    For each latent index j, show which CSV rows (questions) have the highest/lowest
    activations[:, j]. Clarifies that latent_j is one coordinate shared by all rows.
    """
    validate_activations_shape(len(df), acts.shape[0])
    eligible = [j for j in feat_ids if 0 <= j < acts.shape[1]]
    parts: list[str] = [
        "<h3>Which questions have the strongest / weakest signal on each dimension?</h3>",
        "<p class=\"muted\"><strong>Important:</strong> <code>latent_j</code> does <em>not</em> come from a single question. "
        "It is the <em>j</em>-th coordinate of the activation vector. <strong>Every row</strong> (every question) has one "
        "scalar for that coordinate. <strong>CSV row indices</strong> below are 0-based and match the row order of your "
        "residual CSV and the <code>.npz</code> file.</p>",
    ]
    shown = 0
    for j in feat_ids:
        if shown >= max_latents:
            break
        if j < 0 or j >= acts.shape[1]:
            continue
        v = acts[:, j].astype(float)
        order = np.argsort(v)
        bot_i = order[:k]
        top_i = order[-k:][::-1]

        def _rows(idxs: np.ndarray) -> list[dict[str, object]]:
            out: list[dict[str, object]] = []
            for ii in idxs:
                i = int(ii)
                q = _truncate_q(df.iloc[i][text_col], max_chars)
                sub = df.iloc[i]["subject"] if "subject" in df.columns else "—"
                out.append(
                    {
                        "csv_row_index": i,
                        "subject": sub,
                        "activation": float(v[i]),
                        "question (truncated)": q,
                    }
                )
            return out

        top_df = pd.DataFrame(_rows(top_i))
        bot_df = pd.DataFrame(_rows(bot_i))
        parts.append(f"<h4><code>latent_{j}</code> &mdash; top {k} vs bottom {k} rows by this coordinate</h4>")
        parts.append("<p><strong>Highest</strong> values on this dimension:</p>")
        parts.append(top_df.to_html(index=False, escape=True, classes="data", border=0))
        parts.append("<p><strong>Lowest</strong> values:</p>")
        parts.append(bot_df.to_html(index=False, escape=True, classes="data", border=0))
        shown += 1
    if shown == 0:
        return ""
    if shown < len(eligible):
        parts.append(
            f"<p class=\"muted\">Traces shown for the first {shown} of {len(eligible)} listed dimensions "
            f"(cap <code>--tab4-max-latents-for-trace</code>). Increase the cap or shorten the latents list.</p>"
        )
    return "\n".join(parts)


def _strongest_mmlu_subject_for_dim(
    df: pd.DataFrame,
    acts: np.ndarray,
    j: int,
    *,
    subject_col: str = "subject",
) -> str | None:
    """
    MMLU taxonomy: each row has a subject label. For dimension j, return the subject
    whose rows have the highest mean activation — data-driven 'grouping' from the task.
    """
    if subject_col not in df.columns:
        return None
    if len(df) != acts.shape[0]:
        return None
    sub = df[subject_col].fillna("(missing subject)").astype(str)
    v = acts[:, j].astype(float)
    g = pd.DataFrame({"sub": sub, "v": v}).groupby("sub", sort=False)["v"].mean()
    if g.empty:
        return None
    return str(g.idxmax())


def _mean_tertile_label(mean_val: float, q33: float, q66: float) -> str:
    if np.isnan(mean_val):
        return "n/a"
    if mean_val <= q33:
        return "low"
    if mean_val <= q66:
        return "mid"
    return "high"


def _tab5_rows(
    acts: np.ndarray,
    is_error: np.ndarray,
    latent_ids: list[int],
) -> list[dict[str, str]]:
    correct = is_error == 0
    wrong = is_error == 1
    rows_out: list[dict[str, str]] = []
    for j in latent_ids:
        col = acts[:, j].astype(float)
        q33, q66 = np.nanpercentile(col, [100 / 3, 200 / 3])
        if correct.any():
            mean_c = float(np.nanmean(col[correct]))
        else:
            mean_c = float("nan")
        if wrong.any():
            mean_w = float(np.nanmean(col[wrong]))
        else:
            mean_w = float("nan")
        rows_out.append(
            {
                "feature": f"latent_{j}",
                "correct_answers": _mean_tertile_label(mean_c, q33, q66),
                "failures": _mean_tertile_label(mean_w, q33, q66),
            }
        )
    return rows_out


def _tab5_display_rows(
    t5_data: list[dict[str, str]], meanings: dict[int, str]
) -> list[dict[str, str]]:
    """Use human-readable feature labels in Tab 5 when YAML has them (slide-style)."""
    out: list[dict[str, str]] = []
    for row in t5_data:
        lid = row["feature"]
        try:
            j = int(lid.replace("latent_", ""))
        except ValueError:
            j = -1
        label = meanings.get(j)
        feat_display = label if label and label != "TBD" else lid
        out.append(
            {
                "feature": feat_display,
                "correct_answers (low/mid/high)": row["correct_answers"],
                "failures (low/mid/high)": row["failures"],
            }
        )
    return out


def build_html(
    df: pd.DataFrame,
    *,
    max_rows: int | None,
    meanings: dict[int, str],
    activations: np.ndarray | None,
    latent_ids_for_sae: list[int] | None,
    top_k_tab5: int,
    meta_utc: str,
    meta_local: str,
    meta_iso_utc: str,
    tab4_top_questions: int = 0,
    tab4_question_chars: int = 220,
    tab4_max_latents_for_trace: int = 12,
    release_summary: dict | None = None,
    activations_meta: dict | None = None,
) -> str:
    tcol = _text_col(df)
    # --- Tab 1: split correct / incorrect
    ok = df[df["is_error"].astype(int) == 0].copy()
    bad = df[df["is_error"].astype(int) == 1].copy()
    cols1 = [c for c in [tcol, "subject", "subcat", "cat"] if c in ok.columns]
    if not cols1:
        cols1 = [tcol]

    tab1_ok = _df_to_html_table(ok[cols1], max_rows)
    tab1_bad = _df_to_html_table(bad[cols1], max_rows)

    # --- Tab 2: expected error
    t2 = df[[tcol, "expected_error"]].copy()
    tab2_html = _df_to_html_table(t2, max_rows)

    # --- Tab 3: unexpected / actual error
    t3 = df[[tcol, "expected_error"]].copy()
    t3["actual_error"] = df["is_error"].map({0: "no (correct)", 1: "yes (incorrect)"})
    extra = [c for c in ("residual_error", "group") if c in df.columns]
    for c in extra:
        t3[c] = df[c]
    tab3_html = _df_to_html_table(t3, max_rows)

    # --- Tab 4 & 5: SAE
    d = activations.shape[1] if activations is not None else 0

    # If --latents-csv was passed but is empty (e.g. SAE diffing found no latents), do not
    # leave feat_ids empty — fall back to top-k columns or YAML-defined ids.
    if latent_ids_for_sae is not None and len(latent_ids_for_sae) > 0:
        feat_ids = list(latent_ids_for_sae)
    elif activations is not None:
        feat_ids = list(range(min(d, top_k_tab5)))
    else:
        feat_ids = sorted(meanings.keys())

    tab4_rows = []
    for j in feat_ids:
        yaml_note = meanings.get(j, "")
        interp = yaml_note if yaml_note else "—"
        if activations is not None:
            sub_best = _strongest_mmlu_subject_for_dim(df, activations, j)
            mmlu_grp = sub_best if sub_best is not None else "— (no subject column)"
        else:
            mmlu_grp = "— (load activations to link dims → MMLU subject)"
        tab4_rows.append(
            {
                "feature": f"latent_{j}",
                "mmlu_subject (highest mean activation)": mmlu_grp,
                "interpretation (optional YAML)": interp,
            }
        )
    tab4_df = pd.DataFrame(tab4_rows)
    tab4_explainer = (
        "<p class=\"muted\"><strong>Tab 4.</strong> Each <strong>table row</strong> below is one activation "
        "<strong>dimension</strong> (<code>latent_j</code> = coordinate <em>j</em> of the vector). "
        "Every question (every CSV row) has one value per dimension; the dimension is not owned by a single question. "
        "<strong>MMLU subject</strong> is taken from the dataset&rsquo;s <code>subject</code> column: "
        "we group all rows by subject and pick the subject with the <em>highest mean activation</em> "
        "on that dimension (ties: first). This ties features to the official MMLU taxonomy, not to "
        "manual neuron names. Optional YAML adds a second-line interpretation.</p>"
    )
    tab4_html = (
        tab4_explainer
        + (
            tab4_df.to_html(index=False, escape=True, classes="data", border=0)
            if not tab4_df.empty
            else "<p><em>No features listed. Provide --activations or extend feature_meanings.yaml / --latents-csv.</em></p>"
        )
    )
    if activations is not None and tab4_top_questions > 0:
        tab4_html += _tab4_question_trace_section(
            df,
            activations,
            feat_ids,
            tcol,
            k=tab4_top_questions,
            max_chars=tab4_question_chars,
            max_latents=tab4_max_latents_for_trace,
        )

    tab5_methodology = (
        "<p><strong>Tab 5 methodology.</strong> For each latent, global tertile cutoffs "
        "q<sub>33</sub>, q<sub>66</sub> are computed on that latent&rsquo;s activations across "
        "<em>all</em> instances. The mean activation on <strong>correct</strong> rows "
        "(is_error=0) and on <strong>incorrect</strong> rows (is_error=1) each is mapped to "
        "<code>low</code> / <code>mid</code> / <code>high</code> by comparison to those cutoffs.</p>"
    )

    if activations is None:
        stub: list[dict[str, str]] = []
        for j in feat_ids:
            label = meanings.get(j)
            feat_display = label if label and label != "TBD" else f"latent_{j}"
            stub.append(
                {
                    "feature": feat_display,
                    "correct_answers (low/mid/high)": "—",
                    "failures (low/mid/high)": "—",
                }
            )
        if stub:
            tab5_stub = pd.DataFrame(stub).to_html(index=False, escape=True, classes="data", border=0)
            tab5_html = (
                '<div class="callout"><strong>Activations not loaded.</strong> '
                "Pass <code>--activations</code> (row-aligned <code>.npz</code>, key "
                "<code>activations</code>, shape <code>(N, D)</code>) to compute real "
                f"low/mid/high buckets. Below: features from Tab 4 ({len(stub)} row(s)).</div>"
                + tab5_stub
            )
        else:
            tab5_html = (
                '<div class="callout"><strong>No features to show.</strong> Add entries to '
                "<code>config/feature_meanings.yaml</code> and/or pass "
                "<code>--activations</code> (and optionally <code>--latents-csv</code>).</div>"
            )
        tab5_full = tab5_html + tab5_methodology
    else:
        is_err = df["is_error"].astype(int).values
        validate_activations_shape(len(df), activations.shape[0])
        # restrict to valid latent indices
        feat_ids_t5 = [j for j in feat_ids if 0 <= j < activations.shape[1]]
        t5_data = _tab5_rows(activations, is_err, feat_ids_t5)
        t5_display = _tab5_display_rows(t5_data, meanings)
        tab5_df = pd.DataFrame(t5_display)
        tab5_html = (
            tab5_df.to_html(index=False, escape=True, classes="data", border=0)
            if not tab5_df.empty
            else "<p><em>(no latent indices in range)</em></p>"
        )
        tab5_full = tab5_html + tab5_methodology
    if activations is None:
        tab4_note = "<p><em>Optional: pass <code>--activations</code> and <code>--latents-csv</code> to prioritize ranked latents.</em></p>"
    else:
        tab4_note = ""
    tab4_block = tab4_note + tab4_html

    n_total = len(df)
    n_ok = int((df["is_error"].astype(int) == 0).sum())
    n_bad = int((df["is_error"].astype(int) == 1).sum())
    act_loaded = activations is not None
    rep_meta_html = _representation_meta_html(activations_meta) if act_loaded else ""
    act_callout = (
        '<div class="callout"><strong>Activations loaded.</strong> Tabs 4&ndash;5 use your '
        "<code>--activations</code> file (row-aligned with this CSV). Vectors usually come from a "
        "<em>local</em> Hugging Face forward pass on the same question text; they are not OpenAI "
        "internal states. See the <code>*.meta.json</code> next to the <code>.npz</code> for model id."
        f"{rep_meta_html}</div>"
        if act_loaded
        else '<div class="callout"><strong>No activations file.</strong> Tab 4 lists dimensions (YAML / defaults); '
        "Tab 5 may show placeholders. Run <code>export_activations</code> or <code>mmlu-real</code>, then rebuild with "
        "<code>--activations</code>.</div>"
    )
    pipeline_overview_html = f"""
  <div class="pipeline-overview">
    <h2>How to read this report (full pipeline in five tabs)</h2>
    <p>
      This page summarizes a <strong>Stage-2-style failure analysis</strong> on MMLU-style rows: (1) separate where the
      model was right or wrong, (2) estimate how &ldquo;hard&rdquo; each question looks, (3) highlight <em>unexpected</em>
      errors via residuals, (4) relate items to <strong>internal activation coordinates</strong> and MMLU
      <code>subject</code> labels, (5) compare those coordinates on correct vs incorrect answers. Together, the tabs are
      meant to be read <strong>in order</strong> like a short lab notebook.
    </p>
    <ol class="pipeline-steps">
      <li>
        <strong>Tab 1 &mdash; Consume data &amp; split.</strong>
        Input: one row per question from the residual CSV. Output: two tables (correct vs incorrect) with question text
        and MMLU taxonomy columns when present.
      </li>
      <li>
        <strong>Tab 2 &mdash; Expected error.</strong>
        A difficulty proxy: <code>expected_error</code> from surface features (fitted outside this HTML). Shows which
        items the model was <em>expected</em> to miss often.
      </li>
      <li>
        <strong>Tab 3 &mdash; Unexpected error.</strong>
        Joins <code>expected_error</code> with whether the model actually failed. Residual
        <code>is_error &minus; expected_error</code> surfaces surprising mistakes (high residual) or surprisingly easy
        passes.
      </li>
      <li>
        <strong>Tab 4 &mdash; Internal features.</strong>
        Each <code>latent_j</code> is one dimension of the saved activation vector. The
        <strong>MMLU subject</strong> column names which subject has the highest mean activation on that dimension
        (data-driven link to the curriculum). Optional YAML adds a short manual note.
      </li>
      <li>
        <strong>Tab 5 &mdash; Failures vs correct.</strong>
        For each dimension, mean activation on correct rows vs wrong rows is bucketed into low/mid/high (method in the
        footer). This is the direct &ldquo;does this direction differ between outcomes?&rdquo; view.
      </li>
    </ol>
    <p class="muted">
      <strong>Counts in this file:</strong> {n_total} rows total; {n_ok} correct; {n_bad} incorrect.
      Large tables may be truncated by <code>--max-rows</code> when building the report.
    </p>
    {act_callout}
  </div>
"""

    tab_intro_t1 = """
  <div class="tab-intro">
    <h3>Report tab 1 of 5 &mdash; Consume MMLU and split outcomes</h3>
    <p><strong>Purpose:</strong> Show the same set of questions twice: rows where the model was <em>correct</em> vs <em>incorrect</em> (<code>is_error</code>), with MMLU <code>subject</code> / <code>subcat</code> / <code>cat</code> for context.</p>
    <p><strong>Why it matters:</strong> Everything later (expected error, residuals, activations) refers to these rows in the same order.</p>
    <p class="muted"><strong>Next:</strong> Tab 2 attaches a predicted error probability per row (difficulty control).</p>
  </div>"""

    tab_intro_t2 = """
  <div class="tab-intro">
    <h3>Report tab 2 of 5 &mdash; Expected error (difficulty proxy)</h3>
    <p><strong>Purpose:</strong> For each question, show <code>expected_error</code>: how likely a mistake was <em>predicted</em> from cheap surface features (length, digits, keywords, etc.), fitted by <code>run_residual_control</code>.</p>
    <p><strong>Why it matters:</strong> Separates &ldquo;hard-looking&rdquo; items from the raw error label so we can study <em>residuals</em> next.</p>
    <p class="muted"><strong>Next:</strong> Tab 3 combines this with actual correctness to surface unexpected failures.</p>
  </div>"""

    tab_intro_t3 = """
  <div class="tab-intro">
    <h3>Report tab 3 of 5 &mdash; Unexpected errors (residual view)</h3>
    <p><strong>Purpose:</strong> Compare predicted difficulty to what actually happened. <code>actual_error</code> states whether the model got the item wrong; <code>residual_error = is_error &minus; expected_error</code> flags cases that are worse (or better) than the difficulty model expected.</p>
    <p><strong>Why it matters:</strong> This is the <strong>residual-control</strong> step: focus discovery on errors not explained by generic difficulty alone. Columns <code>group</code> / <code>residual_error</code> support subgroup analysis elsewhere (e.g. SAE diffing CSV).</p>
    <p class="muted"><strong>Next:</strong> Tab 4 links rows to internal activation dimensions and MMLU subjects.</p>
  </div>"""

    tab4_repr = ""
    if activations_meta and activations_meta.get("representation_type") == "sae_latent":
        tab4_repr = (
            "<p class=\"muted\"><strong>Representation:</strong> indices are <em>SAE feature</em> coordinates "
            f"(<code>d_sae={html.escape(str(activations_meta.get('d_sae', '?')))}</code>), "
            "from <code>encode_sae_latents</code> + a public SAELens checkpoint—not raw 768-d hidden units.</p>"
        )
    elif activations_meta and activations_meta.get("representation_type") == "hidden_pooled":
        tab4_repr = (
            "<p class=\"muted\"><strong>Representation:</strong> pooled hidden-state dimensions "
            f"(<code>d_in={html.escape(str(activations_meta.get('d_in', activations_meta.get('hidden_dim', '?'))))}</code>); "
            "not an overcomplete SAE dictionary unless you run encoding.</p>"
        )
    tab_intro_t4 = f"""
  <div class="tab-intro">
    <h3>Report tab 4 of 5 &mdash; Internal activations &amp; MMLU subject link</h3>
    <p><strong>Purpose:</strong> <code>latent_j</code> names coordinate <em>j</em> of the activation vector. <strong>All questions share the same coordinate system</strong>: row <em>i</em> has value <code>activations[i, j]</code>. The summary table links each dimension to the MMLU <code>subject</code> with highest <em>mean</em> activation. Optional YAML adds a manual note.</p>
    {tab4_repr}
    <p><strong>Which questions spike on <code>latent_0</code>?</strong> Scroll to <strong>&ldquo;Which questions have the strongest / weakest signal&hellip;&rdquo;</strong> below (shown when the report is built with <code>--tab4-top-questions K</code>): it lists CSV row indices and question text for the top/bottom <em>K</em> rows by that coordinate.</p>
    <p class="muted"><strong>Next:</strong> Tab 5 aggregates correct vs incorrect over the same coordinates.</p>
  </div>"""

    tab_intro_t5 = """
  <div class="tab-intro">
    <h3>Report tab 5 of 5 &mdash; Correct vs failures (low / mid / high)</h3>
    <p><strong>Purpose:</strong> For each selected dimension, compare mean activation on <strong>correct</strong> rows vs <strong>incorrect</strong> rows, each mapped to <code>low</code> / <code>mid</code> / <code>high</code> using global tertiles on that dimension (exact rule below the table). This is still <strong>one number per dimension per outcome group</strong>, not &ldquo;which question&rdquo;&mdash;for per-question values use Tab 4&rsquo;s trace tables or inspect <code>activations[i, j]</code> in Python.</p>
    <p><strong>Why it matters:</strong> Answers &ldquo;does this direction of the representation shift with failure?&rdquo; in a slide-friendly discrete summary.</p>
    <p class="muted"><strong>Set-level metrics</strong> (coverage, concentration, etc.) live in <code>pipeline_2026/stage2/s05_eval/results/*_metrics.json</code> and <code>pipeline_2026/stage2/s05_eval/results/leaderboard.csv</code> from <code>evaluate_pattern_sets</code>, not in this HTML.</p>
  </div>"""

    title = html.escape("MMLU residual + SAE report")
    read_first_block = _format_read_first_html()
    executive_block = (
        _format_executive_summary_html(release_summary) if release_summary else ""
    )
    glossary_block = _format_glossary_html()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="generator" content="stage2.s06_report.code.build_html_report" />
  <meta name="report-generated-at" content="{meta_iso_utc}" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem 1.5rem; line-height: 1.45; }}
    h1 {{ font-size: 1.25rem; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.75rem; border-bottom: 1px solid #ccc; padding-bottom: 0.5rem; }}
    .tabs button {{
      cursor: pointer; padding: 0.4rem 0.75rem; border: 1px solid #bbb; background: #f5f5f5; border-radius: 6px 6px 0 0;
    }}
    .tabs button.active {{ background: #fff; border-bottom-color: #fff; font-weight: 600; }}
    .panel {{ display: none; margin-top: 0.75rem; overflow-x: auto; }}
    .panel.active {{ display: block; }}
    table.data {{ border-collapse: collapse; font-size: 0.85rem; }}
    table.data th, table.data td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; vertical-align: top; }}
    table.data th {{ background: #f0f0f0; }}
    .muted {{ color: #555; font-size: 0.9rem; }}
    .callout {{ background: #fff8e6; border: 1px solid #e6c200; padding: 0.75rem 1rem; margin-bottom: 0.75rem; border-radius: 6px; font-size: 0.9rem; }}
    .meta-generated {{ font-size: 0.85rem; margin-bottom: 1rem; padding: 0.5rem 0.75rem; background: #f6f6f6; border-radius: 6px; border-left: 4px solid #555; }}
    .meta-generated dt {{ font-weight: 600; margin-top: 0.35rem; }}
    .meta-generated dd {{ margin: 0.15rem 0 0 0; padding: 0; }}
    .pipeline-overview {{ background: #f0f7ff; border: 1px solid #b8d4f0; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.25rem; }}
    .pipeline-overview > h2 {{ margin-top: 0; font-size: 1.1rem; }}
    .pipeline-steps {{ margin: 0.75rem 0; padding-left: 1.35rem; }}
    .pipeline-steps li {{ margin: 0.55rem 0; line-height: 1.4; }}
    .tab-intro {{ background: #fafafa; border-left: 4px solid #6a9aca; padding: 0.65rem 0.9rem; margin-bottom: 1rem; font-size: 0.9rem; border-radius: 0 6px 6px 0; }}
    .tab-intro h3 {{ margin: 0 0 0.4rem 0; font-size: 0.98rem; color: #1a3a5c; }}
    .tab-intro p {{ margin: 0.35rem 0 0 0; }}
    .read-first {{ background: #f8fff8; border: 1px solid #c8e6c9; padding: 0.75rem 1rem; margin-bottom: 1rem; border-radius: 8px; }}
    .read-first h2 {{ margin-top: 0; font-size: 1.05rem; }}
    .executive-summary {{ padding: 1rem; margin-bottom: 1.25rem; background: #fafafa; border-radius: 8px; border: 1px solid #ddd; }}
    .verdict-pass {{ border-left: 5px solid #2e7d32; }}
    .verdict-fail {{ border-left: 5px solid #c62828; }}
    .verdict-tag {{ font-weight: 700; letter-spacing: 0.05em; }}
    .glossary {{ margin-top: 1.25rem; padding: 0.75rem 1rem; border: 1px solid #ddd; border-radius: 6px; background: #fcfcfc; }}
    .glossary dt {{ font-weight: 600; margin-top: 0.5rem; }}
    .glossary dd {{ margin: 0.2rem 0 0 1rem; }}
    footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ccc; font-size: 0.8rem; color: #444; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <dl class="meta-generated" title="When this HTML was produced">
    <dt>Report generated (UTC)</dt>
    <dd><time datetime="{meta_iso_utc}">{meta_utc}</time></dd>
    <dt>Local time</dt>
    <dd>{meta_local}</dd>
    <dt>ISO 8601 (UTC)</dt>
    <dd><code>{meta_iso_utc}</code></dd>
  </dl>
  {read_first_block}
  {executive_block}
  {pipeline_overview_html}
  <p class="muted"><strong>Use the tabs</strong> to walk stages 1&ndash;5 in order. Tables may be truncated per <code>--max-rows</code>.</p>

  <div class="tabs" role="tablist">
    <button type="button" class="active" data-tab="t1" role="tab" aria-selected="true" title="Tab 1: split correct vs incorrect">1. Data &amp; split</button>
    <button type="button" data-tab="t2" role="tab" title="Tab 2: expected error">2. Expected error</button>
    <button type="button" data-tab="t3" role="tab" title="Tab 3: residuals / unexpected errors">3. Unexpected error</button>
    <button type="button" data-tab="t4" role="tab" title="Tab 4: activations &amp; subjects">4. Internal features</button>
    <button type="button" data-tab="t5" role="tab" title="Tab 5: correct vs failures">5. vs failures</button>
  </div>

  <div id="t1" class="panel active" role="tabpanel">
    <h2>1. Consume data &amp; separate correct vs incorrect</h2>
    {tab_intro_t1}
    <h3>Correct answers</h3>
    {tab1_ok}
    <h3>Incorrect answers</h3>
    {tab1_bad}
  </div>

  <div id="t2" class="panel" role="tabpanel">
    <h2>2. Expected error per question</h2>
    {tab_intro_t2}
    {tab2_html}
  </div>

  <div id="t3" class="panel" role="tabpanel">
    <h2>3. Unexpected error view</h2>
    {tab_intro_t3}
    <p class="muted"><strong>Reminder:</strong> <code>actual_error</code> = whether the LLM was incorrect; <code>residual_error</code> = <code>is_error &minus; expected_error</code>.</p>
    {tab3_html}
  </div>

  <div id="t4" class="panel" role="tabpanel">
    <h2>4. Look inside the model &mdash; dimensions, MMLU subject link, optional notes</h2>
    {tab_intro_t4}
    {tab4_block}
  </div>

  <div id="t5" class="panel" role="tabpanel">
    <h2>5. Compare failures vs correct (activation buckets)</h2>
    {tab_intro_t5}
    {tab5_full}
  </div>

  {glossary_block}

  <footer>
    <p><strong>Related outputs (not shown in tabs):</strong> run <code>evaluate_pattern_sets</code> for coverage/concentration JSON and leaderboard CSV; aligns with the project plan&rsquo;s set-level metrics.</p>
    <p>Generated by <code>stage2/s06_report/code/build_html_report.py</code> at <strong>{meta_utc}</strong> (<code>{meta_iso_utc}</code>). Additive code only; <code>pipeline/</code> unchanged.</p>
  </footer>

  <script>
    document.querySelectorAll('.tabs button').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        var id = btn.getAttribute('data-tab');
        document.querySelectorAll('.tabs button').forEach(function(b) {{
          b.classList.toggle('active', b === btn);
          b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
        }});
        document.querySelectorAll('.panel').forEach(function(p) {{
          p.classList.toggle('active', p.id === id);
        }});
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Build five-tab HTML report")
    p.add_argument("--residual-csv", type=Path, required=True)
    p.add_argument("--activations", type=Path, default=None, help=".npz with key activations (N, D)")
    p.add_argument(
        "--activations-meta",
        type=Path,
        default=None,
        help="Sidecar JSON (default: same basename as --activations with .meta.json)",
    )
    p.add_argument("--latents-csv", type=Path, default=None, help="Optional ranked latents (latent_id column)")
    p.add_argument(
        "--meanings-yaml",
        type=Path,
        default=ROOT / "config/feature_meanings.yaml",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=500, help="Max rows per table (None = all)")
    p.add_argument(
        "--top-k-tab5",
        type=int,
        default=32,
        help="If no latents-csv, first K latent indices 0..K-1 for tabs 4-5",
    )
    p.add_argument(
        "--tab4-top-questions",
        type=int,
        default=0,
        help="Tab 4: per dimension, list this many CSV rows with highest/lowest activation on that coordinate (0=off)",
    )
    p.add_argument(
        "--tab4-question-chars",
        type=int,
        default=220,
        help="Max characters of question text in Tab 4 trace tables",
    )
    p.add_argument(
        "--tab4-max-latents-for-trace",
        type=int,
        default=12,
        help="Tab 4: only add trace tables for the first N dimensions in the feature list (limits HTML size)",
    )
    p.add_argument(
        "--release-json",
        type=Path,
        default=None,
        help="Optional release_readiness.json from stage2.s06_report.code.release_summary (executive block at top)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.residual_csv)
    if "is_error" not in df.columns:
        if "ai_correct" in df.columns:
            df = df.copy()
            df["is_error"] = (~df["ai_correct"].astype(bool)).astype(int)
        else:
            raise ValueError("residual CSV must contain is_error or ai_correct")
    if "expected_error" not in df.columns:
        raise ValueError("residual CSV must contain expected_error")

    if not args.meanings_yaml.exists():
        print(
            f"Warning: meanings YAML not found at {args.meanings_yaml}; Tab 4 labels will be TBD only.",
            file=sys.stderr,
        )
    meanings = _load_meanings(args.meanings_yaml)

    acts = None
    activations_meta: dict | None = None
    if args.activations is not None:
        act_path = args.activations.resolve()
        if not act_path.is_file():
            raise FileNotFoundError(
                f"Activations file not found: {args.activations}\n"
                "Create row-aligned activations first, e.g.:\n"
                "  ./scripts/run_additive_track.sh mmlu-demo\n"
                "or:\n"
                f"  python3 -m stage2.s04_diff.code.generate_dummy_activations \\\n"
                f"    --residual-csv {args.residual_csv} \\\n"
                f"    --latent-dim 64 \\\n"
                f"    --output pipeline_2026/stage2/s04_diff/results/dummy_activations.npz\n"
                "Then optionally run stage2.s04_diff.code.run_sae_diffing to produce a latents CSV."
            )
        data = np.load(act_path, allow_pickle=True)
        acts = np.asarray(data["activations"], dtype=float)
        meta_path = args.activations_meta or act_path.with_suffix(".meta.json")
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                activations_meta = json.load(f)

    latent_ids: list[int] | None = None
    if args.latents_csv is not None:
        if not args.latents_csv.is_file():
            print(
                f"Warning: --latents-csv not found ({args.latents_csv}); "
                "continuing without it (Tab 4–5 use top-k columns from activations or YAML).",
                file=sys.stderr,
            )
        else:
            ldf = pd.read_csv(args.latents_csv)
            if "latent_id" not in ldf.columns:
                raise ValueError("latents CSV must have latent_id column")
            latent_ids = [int(x) for x in ldf["latent_id"].tolist()]

    max_rows = args.max_rows if args.max_rows > 0 else None

    release_summary: dict | None = None
    if args.release_json is not None:
        rp = args.release_json.resolve()
        if not rp.is_file():
            raise FileNotFoundError(f"Release JSON not found: {args.release_json}")
        with open(rp, "r", encoding="utf-8") as f:
            release_summary = json.load(f)

    utc_now = datetime.now(timezone.utc)
    local_now = datetime.now().astimezone()
    meta_iso_utc = utc_now.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    meta_utc = html.escape(utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"))
    meta_local = html.escape(local_now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)"))
    meta_iso_utc_esc = html.escape(meta_iso_utc)

    html_out = build_html(
        df,
        max_rows=max_rows,
        meanings=meanings,
        activations=acts,
        latent_ids_for_sae=latent_ids,
        top_k_tab5=args.top_k_tab5,
        meta_utc=meta_utc,
        meta_local=meta_local,
        meta_iso_utc=meta_iso_utc_esc,
        tab4_top_questions=args.tab4_top_questions,
        tab4_question_chars=args.tab4_question_chars,
        tab4_max_latents_for_trace=args.tab4_max_latents_for_trace,
        release_summary=release_summary,
        activations_meta=activations_meta,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.output} (generated at {meta_iso_utc})")


if __name__ == "__main__":
    main()
