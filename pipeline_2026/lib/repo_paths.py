"""Filesystem roots for pipeline_2026 (staged additive track)."""
from __future__ import annotations

from pathlib import Path


def pipeline_2026_root() -> Path:
    """Directory containing ``lib/``, ``stage1/``, ``stage2/``, ``config/``, etc."""
    return Path(__file__).resolve().parent.parent


def teaching_llms_errors_repo_root() -> Path:
    """Parent of ``pipeline_2026/`` — the ``teaching-llms-errors`` checkout (has legacy ``pipeline/``)."""
    return pipeline_2026_root().parent
