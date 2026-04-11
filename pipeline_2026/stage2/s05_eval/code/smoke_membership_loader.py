#!/usr/bin/env python3
"""Smoke test: header-only / empty membership CSV → (N, 0) matrix.

Run from pipeline_2026 with PYTHONPATH=. :
  python3 -m stage2.s05_eval.code.smoke_membership_loader
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from stage2.s05_eval.code.evaluate_pattern_sets import _load_membership_matrix


def main() -> None:
    d = Path(tempfile.mkdtemp())
    header_only = d / "m.csv"
    pd.DataFrame(
        columns=["row_id", "pattern_id", "latent_id", "is_member"]
    ).to_csv(header_only, index=False)
    x, ids = _load_membership_matrix(header_only, n_rows=3)
    assert x.shape == (3, 0), x.shape
    assert ids == []

    zero = d / "z.csv"
    zero.write_bytes(b"")
    x2, ids2 = _load_membership_matrix(zero, n_rows=5)
    assert x2.shape == (5, 0)
    assert ids2 == []

    print("smoke_membership_loader: ok")


if __name__ == "__main__":
    main()
