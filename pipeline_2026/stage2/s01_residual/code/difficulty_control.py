"""Residual-based difficulty control (surface features + logistic regression)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re


def extract_difficulty_features(text: str) -> list:
    """Extract surface-level difficulty features from the input text."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    text = str(text)
    length = len(text.split())
    num_numbers = len(re.findall(r"\d+", text))
    num_logical_ops = len(re.findall(r"\b(if|and|or|not|because)\b", text.lower()))
    return [length, num_numbers, num_logical_ops]


def _build_xy(df: pd.DataFrame, text_col: str, error_col: str):
    """Create model matrix X and binary label vector y from a residual DataFrame."""
    X = np.array([extract_difficulty_features(t) for t in df[text_col]])
    y = df[error_col].values
    return X, y


def compute_residuals_and_group(
    df: pd.DataFrame,
    text_col: str,
    error_col: str,
    *,
    fit_mode: str = "full",
    test_size: float = 0.3,
    random_state: int = 42,
    high_q: float = 0.75,
    low_q: float = 0.25,
) -> pd.DataFrame:
    """
    Train a logistic regression to predict expected error, compute residuals,
    and assign High / Low / neutral residual groups.

    Parameters
    ----------
    df : DataFrame
        Must contain text_col and error_col.
    error_col : str
        Binary 1 = error (incorrect), 0 = correct.
    fit_mode : {"full", "train_only"}
        - full: fit on all rows (legacy; can leak signal into expected_error).
        - train_only: fit on (1-test_size) fraction, apply predict_proba to all rows.
    high_q, low_q : float
        Quantiles for High / Low residual buckets on residual_error.
    """
    if fit_mode not in ("full", "train_only"):
        raise ValueError("fit_mode must be 'full' or 'train_only'")

    out = df.copy()
    X, y = _build_xy(out, text_col, error_col)

    if fit_mode == "full":
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(X, y)
        expected_error_prob = lr.predict_proba(X)[:, 1]
    else:
        idx = np.arange(len(out))
        tr_idx, _ = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=y
        )
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(X[tr_idx], y[tr_idx])
        expected_error_prob = lr.predict_proba(X)[:, 1]

    out["expected_error"] = expected_error_prob
    out["residual_error"] = out[error_col].astype(float).values - out["expected_error"]

    high_threshold = out["residual_error"].quantile(high_q)
    low_threshold = out["residual_error"].quantile(low_q)

    out["group"] = "neutral"
    out.loc[out["residual_error"] >= high_threshold, "group"] = "High Residual"
    out.loc[out["residual_error"] <= low_threshold, "group"] = "Low Residual"

    return out


def add_is_error_from_ai_correct(df: pd.DataFrame) -> pd.DataFrame:
    """MMLU-style: ai_correct True -> is_error 0."""
    out = df.copy()
    if "ai_correct" not in out.columns:
        raise ValueError("Expected column 'ai_correct'")
    out["is_error"] = (~out["ai_correct"].astype(bool)).astype(int)
    return out


def add_is_error_from_correct(df: pd.DataFrame) -> pd.DataFrame:
    """MathCAMPs-style: correct True -> is_error 0."""
    out = df.copy()
    if "correct" not in out.columns:
        raise ValueError("Expected column 'correct'")
    out["is_error"] = (~out["correct"].astype(bool)).astype(int)
    return out
