#!/usr/bin/env python3
"""Stage 1 — Paper baseline: run ``notebook_paper_figures.py`` (pipeline_2026) + manifest + audit HTML."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from lib.repo_paths import pipeline_2026_root, teaching_llms_errors_repo_root


def main() -> None:
    p = argparse.ArgumentParser(description="Paper baseline artifacts for Stage 1")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Usually pipeline_2026/stage1/results",
    )
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    repo = teaching_llms_errors_repo_root()
    p26 = pipeline_2026_root()
    fig_script = p26 / "stage1" / "code" / "notebook_paper_figures.py"
    env = {**os.environ, "PYTHONPATH": str(repo / "pipeline")}
    pipeline_cwd = repo / "pipeline"

    ran = False
    err_note: str | None = None
    try:
        if fig_script.is_file():
            r = subprocess.run(
                [sys.executable, str(fig_script), "--output-dir", str(out)],
                cwd=str(pipeline_cwd),
                env=env,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                err_note = (r.stderr or r.stdout or "notebook_paper_figures failed").strip()
                print(err_note, file=sys.stderr)
            else:
                ran = True
                if r.stdout:
                    print(r.stdout, end="")
        else:
            err_note = f"missing {fig_script}"
    except OSError as e:
        err_note = str(e)
        print(err_note, file=sys.stderr)

    manifest: dict = {
        "stage": "paper_baseline",
        "source": "pipeline_2026/stage1/code/notebook_paper_figures.py",
        "legacy_data": "pipeline/stage1_do_errors_exist + pipeline/utils (read-only)",
        "paper_figures_ran": ran,
        "notebook": "dataset-model-analysis.ipynb (manual exploration)",
    }
    if err_note:
        manifest["error"] = err_note
    (out / "paper_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out / 'paper_baseline_manifest.json'}")

    # Refresh audit.html so it lists current artifacts (e.g. 11 notebook-aligned PNGs).
    readme = p26 / "stage1" / "README.md"
    if readme.is_file():
        r_audit = subprocess.run(
            [
                sys.executable,
                "-m",
                "stage2.s06_report.code.generate_stage_audit_html",
                "--stage",
                "paper_baseline",
                "--title",
                "Stage 1 — Paper baseline",
                "--readme",
                str(readme),
                "--results-dir",
                str(out),
            ],
            cwd=str(p26),
            env={**os.environ, "PYTHONPATH": str(p26)},
            capture_output=True,
            text=True,
        )
        if r_audit.returncode != 0:
            note = (r_audit.stderr or r_audit.stdout or "generate_stage_audit_html failed").strip()
            print(note, file=sys.stderr)
            manifest["audit_html_error"] = note
            (out / "paper_baseline_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        elif r_audit.stdout:
            print(r_audit.stdout, end="")
    else:
        print(f"paper_baseline: skip audit.html (missing {readme})", file=sys.stderr)


if __name__ == "__main__":
    main()
