#!/usr/bin/env python3
"""
Emit a small audit HTML for one pipeline_2026 stage: lists ``results/`` artifacts,
optional CSV/JSON previews, and links to sibling stages.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from lib.repo_paths import pipeline_2026_root

# Ordered Stage 2 segments (additive / improvement track)
STAGE2_SEGMENTS: tuple[str, ...] = (
    "s01_residual",
    "s02_export",
    "s03_sae_encode",
    "s04_diff",
    "s05_eval",
    "s06_report",
)


def _nav_html(results_dir: Path, root: Path) -> str:
    """Links: Stage 1 paper baseline + Stage 2 segments."""
    root = root.resolve()
    rd = results_dir.resolve()
    try:
        rel = rd.relative_to(root)
    except ValueError:
        return ""

    parts = rel.parts
    links: list[str] = []

    if len(parts) >= 2 and parts[0] == "stage1" and parts[1] == "results":
        links.append("<strong>Stage1 — Paper baseline</strong>")
        for seg in STAGE2_SEGMENTS:
            links.append(
                f'<a href="../../stage2/{seg}/results/audit.html">Stage2 — {html.escape(seg)}</a>'
            )
        return " &middot; ".join(links)

    if len(parts) >= 3 and parts[0] == "stage2" and parts[2] == "results":
        seg_here = parts[1]
        links.append(
            '<a href="../../../stage1/results/audit.html">Stage1 — Paper baseline</a>'
        )
        for seg in STAGE2_SEGMENTS:
            if seg == seg_here:
                links.append(f"<strong>{html.escape(seg)}</strong>")
            else:
                links.append(
                    f'<a href="../../{html.escape(seg)}/results/audit.html">{html.escape(seg)}</a>'
                )
        idx = STAGE2_SEGMENTS.index(seg_here) if seg_here in STAGE2_SEGMENTS else -1
        idx_note = (
            f'<span class="muted"> (segment {idx + 1} of {len(STAGE2_SEGMENTS)})</span>'
            if idx >= 0
            else ""
        )
        return " &middot; ".join(links) + idx_note

    return ""


def _truncate_cell(text: str, max_chars: int) -> str:
    t = text if isinstance(text, str) else str(text)
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _coerce_bool(v: str) -> bool | None:
    s = (v or "").strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _stage1_compare_block(results_dir: Path, root: Path) -> str:
    """Render paper-vs-mistral MMLU comparison for stage1/results."""
    try:
        rel = results_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return ""
    if rel.parts[:2] != ("stage1", "results"):
        return ""

    paper_summary = results_dir / "paper_mmlu_summary.json"
    mistral_csv = results_dir / "mistral_mmlu_predictions.csv"
    mistral_summary = results_dir / "mistral_mmlu_summary.json"
    if not paper_summary.is_file() or not mistral_csv.is_file():
        return ""

    try:
        p = json.loads(paper_summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""

    model_label = "mistral"
    if mistral_summary.is_file():
        try:
            ms = json.loads(mistral_summary.read_text(encoding="utf-8"))
            model_label = str(ms.get("model_label") or model_label)
        except (OSError, json.JSONDecodeError):
            pass

    # Compute mistral metrics directly from CSV so report reflects latest run.
    n_rows = 0
    n_wrong = 0
    by_subject: dict[str, list[bool]] = {}
    try:
        with open(mistral_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                subject = str(row.get("subject") or "").strip()
                ok = _coerce_bool(str(row.get("ai_correct") or ""))
                if not subject or ok is None:
                    continue
                n_rows += 1
                if not ok:
                    n_wrong += 1
                by_subject.setdefault(subject, []).append(ok)
    except OSError:
        return ""
    if n_rows == 0:
        return (
            "<h2>MMLU comparison: Paper vs Mistral</h2>"
            "<p class=\"muted\">Mistral CSV exists but has no valid rows with columns "
            "<code>subject</code> and <code>ai_correct</code>.</p>"
        )

    m_err = n_wrong / float(n_rows)
    p_err = float(p.get("error_rate", 0.0))
    m_top = sorted(
        (
            (subject, sum(1 for x in vals if not x) / float(len(vals)))
            for subject, vals in by_subject.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    p_top = [str(x.get("subject", "")) for x in p.get("top_subjects_by_error_rate", [])[:10]]
    overlap = sorted(set(p_top) & set(s for s, _ in m_top))
    overlap_txt = ", ".join(overlap[:8]) if overlap else "none"

    tb = ['<table class="data"><thead><tr>']
    tb.append("<th>model</th><th>n_rows</th><th>error_rate</th><th>delta_vs_paper</th>")
    tb.append("</tr></thead><tbody>")
    tb.append(
        "<tr>"
        "<td>paper_baseline</td>"
        f"<td>{html.escape(str(p.get('n_rows', 'n/a')))}</td>"
        f"<td>{html.escape(str(round(p_err, 6)))}</td>"
        "<td>0.0</td>"
        "</tr>"
    )
    tb.append(
        "<tr>"
        f"<td>{html.escape(model_label)}</td>"
        f"<td>{html.escape(str(n_rows))}</td>"
        f"<td>{html.escape(str(round(m_err, 6)))}</td>"
        f"<td>{html.escape(str(round(m_err - p_err, 6)))}</td>"
        "</tr>"
    )
    tb.append("</tbody></table>")

    # All Stage1 paper figure pairs (paper vs mistral), shown side-by-side.
    pair_map = [
        ("paper_mmlu_error_by_subject_top20.png", "mistral_mmlu_error_by_subject_top20.png", "Top20 quick view"),
        ("paper_nb_cell03_mmlu_math_reg.png", "mistral_nb_cell03_mmlu_math_reg.png", "Cell 03: MMLU math-reg"),
        ("paper_nb_cell05_mmlu_health_reg.png", "mistral_nb_cell05_mmlu_health_reg.png", "Cell 05: MMLU health-reg"),
        (
            "paper_nb_cell07_mmlu_subcat_computer_science.png",
            "mistral_nb_cell07_mmlu_subcat_computer_science.png",
            "Cell 07: MMLU subcat computer science",
        ),
        ("paper_nb_cell09_mmlu_all_subjects.png", "mistral_nb_cell09_mmlu_all_subjects.png", "Cell 09: all subjects"),
        ("paper_nb_cell11_mmlu_by_subcat.png", "mistral_nb_cell11_mmlu_by_subcat.png", "Cell 11: by subcat"),
        (
            "paper_nb_cell13_mmlu_subcats_psych_physics_math.png",
            "mistral_nb_cell13_mmlu_subcats_psych_physics_math.png",
            "Cell 13: psychology/physics/math",
        ),
        (
            "paper_nb_cell16_mmlu_math_health_vertical.png",
            "mistral_nb_cell16_mmlu_math_health_vertical.png",
            "Cell 16: math+health vertical",
        ),
        (
            "paper_nb_cell18_mmlu_math_health_horizontal.png",
            "mistral_nb_cell18_mmlu_math_health_horizontal.png",
            "Cell 18: math+health horizontal",
        ),
        (
            "paper_nb_cell19_mmlu_subcats_grid.png",
            "mistral_nb_cell19_mmlu_subcats_grid.png",
            "Cell 19: subcat grid",
        ),
        (
            "paper_nb_cell24_mathcamps_gpt_claude_sonnet.png",
            "mistral_nb_cell24_mathcamps_gpt_claude_sonnet.png",
            "Cell 24: MathCAMPS GPT/Claude",
        ),
        (
            "paper_nb_cell26_mathcamps_claude_family.png",
            "mistral_nb_cell26_mathcamps_claude_family.png",
            "Cell 26: MathCAMPS Claude family",
        ),
    ]
    pair_rows = []
    for paper_name, mistral_name, label in pair_map:
        p_ok = (results_dir / paper_name).is_file()
        m_ok = (results_dir / mistral_name).is_file()
        if not p_ok:
            continue
        m_cell = (
            f'<img src="{html.escape(mistral_name)}" alt="" style="max-width:100%;height:auto;border:1px solid #ccc;" />'
            if m_ok
            else '<span class="muted">Not available for Mistral in current artifacts.</span>'
        )
        pair_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f'<td><img src="{html.escape(paper_name)}" alt="" style="max-width:100%;height:auto;border:1px solid #ccc;" /></td>'
            f"<td>{m_cell}</td>"
            "</tr>"
        )
    pair_block = ""
    if pair_rows:
        pair_block = (
            "<h3>Paper figures vs Mistral (side-by-side)</h3>"
            '<table class="data"><thead><tr><th>figure</th><th>paper</th><th>mistral</th></tr></thead><tbody>'
            + "".join(pair_rows)
            + "</tbody></table>"
        )

    return (
        "<h2>MMLU comparison: Paper vs Mistral</h2>"
        + "".join(tb)
        + f'<p class="muted">Top-10 subject overlap: {html.escape(overlap_txt)}</p>'
        + pair_block
    )


def _file_block(
    path: Path,
    *,
    max_csv_rows: int,
    max_json_chars: int,
    csv_hide_columns: frozenset[str] | None,
    max_cell_chars: int,
) -> str:
    rel = html.escape(path.name)
    size = path.stat().st_size
    parts = [f"<h4><code>{rel}</code></h4><p class=\"muted\">{size} bytes</p>"]
    suf = path.suffix.lower()
    if suf == ".csv" and max_csv_rows > 0:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if not rows:
                parts.append("<p><em>(empty CSV)</em></p>")
            else:
                header, body = rows[0], rows[1 : max_csv_rows + 1]
                hide = csv_hide_columns or frozenset()
                idx_keep = [i for i, h in enumerate(header) if h.strip() not in hide]
                if not idx_keep:
                    parts.append(
                        "<p class=\"muted\">CSV preview skipped: all columns hidden "
                        "(wide embedding columns). See <code>*_profile.json</code> for aggregates.</p>"
                    )
                else:
                    if hide:
                        parts.append(
                            f"<p class=\"muted\">Hiding {len(header) - len(idx_keep)} wide column(s) "
                            f"from preview: {html.escape(', '.join(sorted(hide)))}</p>"
                        )
                    h2 = [header[i] for i in idx_keep]
                    tb = ['<table class="data"><thead><tr>']
                    tb.append("".join(f"<th>{html.escape(c)}</th>" for c in h2))
                    tb.append("</tr></thead><tbody>")
                    for row in body:
                        cells = []
                        for i in idx_keep:
                            raw = row[i] if i < len(row) else ""
                            cells.append(_truncate_cell(raw, max_cell_chars))
                        tb.append(
                            "<tr>"
                            + "".join(f"<td>{html.escape(c)}</td>" for c in cells)
                            + "</tr>"
                        )
                    tb.append("</tbody></table>")
                    parts.append("".join(tb))
        except OSError as e:
            parts.append(f"<p class=\"err\">Could not read CSV: {html.escape(str(e))}</p>")
    elif suf == ".json" and max_json_chars > 0:
        try:
            raw = path.read_text(encoding="utf-8")
            try:
                obj = json.loads(raw)
                pretty = json.dumps(obj, indent=2)
            except json.JSONDecodeError:
                pretty = raw
            if len(pretty) > max_json_chars:
                pretty = pretty[:max_json_chars] + "\n… (truncated)"
            parts.append(f"<pre>{html.escape(pretty)}</pre>")
        except OSError as e:
            parts.append(f"<p class=\"err\">Could not read JSON: {html.escape(str(e))}</p>")
    elif suf == ".npz":
        parts.append(
            "<p class=\"muted\">NumPy archive (binary). Load in Python: "
            "<code>np.load(path)[&quot;activations&quot;]</code>.</p>"
        )
    elif suf in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        parts.append(
            f"<p><img src=\"{html.escape(path.name)}\" alt=\"\" style=\"max-width:100%;height:auto;border:1px solid #ccc;\" /></p>"
        )
    else:
        parts.append("<p class=\"muted\">No inline preview for this file type.</p>")
    return "\n".join(parts)


def build_audit_html(
    *,
    stage_id: str,
    title: str,
    readme_excerpt: str,
    results_dir: Path,
    max_csv_rows: int,
    max_json_chars: int,
    csv_hide_columns: frozenset[str] | None = None,
    max_cell_chars: int = 200,
) -> str:
    root = pipeline_2026_root()
    if not results_dir.is_dir():
        files: list[Path] = []
    else:
        files = sorted(results_dir.iterdir(), key=lambda p: p.name.lower())

    nav_html = _nav_html(results_dir, root)
    if not nav_html:
        nav_html = html.escape(stage_id)

    report_callout = ""
    try:
        rel = results_dir.resolve().relative_to(root.resolve())
        if (
            len(rel.parts) >= 2
            and rel.parts[0] == "stage2"
            and rel.parts[1] == "s06_report"
        ):
            report = results_dir / "mmlu_report.html"
            if report.is_file():
                report_callout = (
                    '<p class="callout"><a href="mmlu_report.html"><strong>Full five-tab HTML report</strong></a> '
                    "(open this file in a browser).</p>"
                )
    except ValueError:
        pass
    compare_block = _stage1_compare_block(results_dir, root)

    body_files = ""
    if not files:
        body_files = "<p><em>No files in results/ yet. Run this stage&rsquo;s <code>run.sh</code>.</em></p>"
    else:
        chunks = []
        for p in files:
            if p.name == "audit.html":
                continue
            if p.is_file():
                chunks.append(
                    _file_block(
                        p,
                        max_csv_rows=max_csv_rows,
                        max_json_chars=max_json_chars,
                        csv_hide_columns=csv_hide_columns,
                        max_cell_chars=max_cell_chars,
                    )
                )
        body_files = "\n".join(chunks) if chunks else "<p><em>(no artifacts besides audit)</em></p>"

    utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)} — audit</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.2rem 1.5rem; line-height: 1.45; max-width: 960px; }}
    .nav {{ font-size: 0.9rem; margin-bottom: 1rem; padding: 0.5rem; background: #f4f4f4; border-radius: 6px; }}
    .excerpt {{ background: #f9f9f9; padding: 0.75rem 1rem; border-left: 4px solid #336; margin: 1rem 0; }}
    table.data {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 0.5rem 0; }}
    table.data th, table.data td {{ border: 1px solid #ccc; padding: 0.25rem 0.4rem; text-align: left; }}
    pre {{ background: #111; color: #eee; padding: 0.75rem; overflow: auto; font-size: 0.8rem; }}
    .muted {{ color: #555; }}
    .err {{ color: #a00; }}
    .callout {{ background: #e8f4ff; padding: 0.6rem 0.9rem; border-radius: 6px; margin: 0.75rem 0; }}
  </style>
</head>
<body>
  <p class="nav"><strong>pipeline_2026</strong> &mdash; {nav_html}</p>
  <h1>{html.escape(title)}</h1>
  <p class="muted">Audit generated {html.escape(utc)}</p>
  <div class="excerpt">{readme_excerpt}</div>
  {compare_block}
  {report_callout}
  <h2>Artifacts in <code>{html.escape(str(results_dir.relative_to(root)))}</code></h2>
  {body_files}
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Write stage results/audit.html")
    p.add_argument("--stage", required=True, help="e.g. stage1")
    p.add_argument("--title", required=True)
    p.add_argument("--readme", type=Path, required=True, help="Stage README.md (first ~40 lines as excerpt)")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--max-csv-rows", type=int, default=12)
    p.add_argument("--max-json-chars", type=int, default=8000)
    p.add_argument(
        "--csv-hide-columns",
        default="",
        help="Comma-separated CSV column names to omit from HTML table preview (e.g. embeddings)",
    )
    p.add_argument(
        "--max-cell-chars",
        type=int,
        default=200,
        help="Truncate long CSV cells in preview (questions text, etc.)",
    )
    args = p.parse_args()
    hide_set: frozenset[str] | None = None
    if args.csv_hide_columns.strip():
        hide_set = frozenset(
            x.strip() for x in args.csv_hide_columns.split(",") if x.strip()
        )

    readme_lines = args.readme.read_text(encoding="utf-8").splitlines()[:40]
    excerpt = "<br/>\n".join(html.escape(line) for line in readme_lines if line.strip()) or "(see README.md)"
    html_out = build_audit_html(
        stage_id=args.stage,
        title=args.title,
        readme_excerpt=excerpt,
        results_dir=args.results_dir,
        max_csv_rows=args.max_csv_rows,
        max_json_chars=args.max_json_chars,
        csv_hide_columns=hide_set,
        max_cell_chars=args.max_cell_chars,
    )
    out = args.results_dir / "audit.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
