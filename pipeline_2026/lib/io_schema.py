"""Lightweight schema checks for additive experiment I/O (no pipeline imports)."""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

REQUIRED_MMLU_COLS = ("questions", "ai_correct")
REQUIRED_RESIDUAL_OUTPUT_COLS = (
    "expected_error",
    "residual_error",
    "group",
)


def assert_columns(df: pd.DataFrame, required: Sequence[str], *, name: str = "dataframe") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")


def validate_mmlu_frame(df: pd.DataFrame) -> None:
    assert_columns(df, REQUIRED_MMLU_COLS, name="MMLU frame")


def validate_residual_output(df: pd.DataFrame) -> None:
    assert_columns(df, REQUIRED_RESIDUAL_OUTPUT_COLS, name="residual output")


def validate_activations_shape(n_rows: int, activations_n_rows: int) -> None:
    if n_rows != activations_n_rows:
        raise ValueError(
            f"Row count mismatch: residual table has {n_rows} rows, "
            f"activations has {activations_n_rows}."
        )
