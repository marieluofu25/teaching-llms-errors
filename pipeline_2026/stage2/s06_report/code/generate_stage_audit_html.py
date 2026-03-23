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
