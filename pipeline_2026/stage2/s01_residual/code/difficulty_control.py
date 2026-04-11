"""Residual-based difficulty control (surface features + logistic regression)."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def extract_difficulty_features(text: str) -> list:
    """Extract surface-level difficulty features from the input text."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    text = str(text)
    length = len(text.split())
    num_numbers = len(re.findall(r"\d+", text))
    num_logical_ops = len(re.findall(r"\b(if|and|or|not|because)\b", text.lower()))
    return [length, num_numbers, num_logical_ops]


def _build_xy(
    df: pd.DataFrame,
    text_col: str,
    error_col: str,
    *,
    subject_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create model matrix X and binary label vector y from a residual DataFrame."""
    base = np.array(
        [extract_difficulty_features(t) for t in df[text_col]], dtype=np.float64
    )
    if subject_col and subject_col in df.columns:
        codes, _ = pd.factorize(df[subject_col].astype(str), sort=True)
        X = np.column_stack([base, codes.astype(np.float64)])
    else:
        X = base
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
    subject_col: str | None = "subject",
) -> pd.DataFrame:
    """
    Train a logistic regression to predict expected error, compute residuals,
    and assign High / Low / neutral residual groups.

    Parameters
    ----------
    df : DataFrame
        Must contain text_col and error_col.
    error_col : str
        Binary 1 = error (incorrect), 0 = correct. If this column is constant,
        logistic regression is skipped and ``expected_error`` is set to the
        empirical error rate for every row (a warning is emitted).
    fit_mode : {"full", "train_only"}
        - full: fit on all rows (legacy; can leak signal into expected_error).
        - train_only: fit on (1-test_size) fraction, apply predict_proba to all rows.
    high_q, low_q : float
        Quantiles for High / Low residual buckets on residual_error.
    subject_col : str | None
        If set and present on ``df``, MMLU **subject** (topic) is included as a categorical
        feature via integer codes (topic-aware difficulty control).
    """
    if fit_mode not in ("full", "train_only"):
        raise ValueError("fit_mode must be 'full' or 'train_only'")

    out = df.copy()
    use_subj = (
        subject_col
        if (subject_col and subject_col in out.columns)
        else None
    )
    X, y = _build_xy(out, text_col, error_col, subject_col=use_subj)
    n = len(out)

    if np.unique(y).size < 2:
        warnings.warn(
            f"{error_col!r} has a single class in the data; skipping LogisticRegression "
            f"and using constant expected_error={float(np.mean(y)):.6g} for all rows.",
            UserWarning,
            stacklevel=2,
        )
        expected_error_prob = np.full(n, float(np.mean(y)), dtype=np.float64)
    elif fit_mode == "full":
        lr = LogisticRegression(class_weight="balanced", max_iter=2000)
        lr.fit(X, y)
        expected_error_prob = lr.predict_proba(X)[:, 1]
    else:
        idx = np.arange(n)
        tr_idx, _ = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=y
        )
        y_train = y[tr_idx]
        if np.unique(y_train).size < 2:
            warnings.warn(
                f"{error_col!r} has a single class in the training split; skipping "
                f"LogisticRegression and using constant expected_error="
                f"{float(np.mean(y_train)):.6g} for all rows.",
                UserWarning,
                stacklevel=2,
            )
            expected_error_prob = np.full(n, float(np.mean(y_train)), dtype=np.float64)
        else:
            lr = LogisticRegression(class_weight="balanced", max_iter=2000)
            lr.fit(X[tr_idx], y_train)
            expected_error_prob = lr.predict_proba(X)[:, 1]

    out["expected_error"] = expected_error_prob
    out["residual_error"] = out[error_col].astype(float).values - out["expected_error"]

    high_threshold = out["residual_error"].quantile(high_q)
    low_threshold = out["residual_error"].quantile(low_q)

    out["group"] = "neutral"
    out.loc[out["residual_error"] >= high_threshold, "group"] = "High Residual"
    out.loc[out["residual_error"] <= low_threshold, "group"] = "Low Residual"

    return out


def _series_to_bool(s: pd.Series) -> pd.Series:
    """Coerce CSV / pickle truthiness to boolean."""

    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int) != 0
    sl = s.astype(str).str.strip().str.lower()
    return sl.isin(("1", "true", "t", "yes", "y"))


def add_is_error_from_ai_correct(df: pd.DataFrame) -> pd.DataFrame:
    """MMLU-style: ai_correct True -> is_error 0."""
    out = df.copy()
    if "ai_correct" not in out.columns:
        raise ValueError("Expected column 'ai_correct'")
    correct = _series_to_bool(out["ai_correct"])
    out["is_error"] = (~correct).astype(int)
    return out


def add_is_error_from_correct(df: pd.DataFrame) -> pd.DataFrame:
    """MathCAMPs-style: correct True -> is_error 0."""
    out = df.copy()
    if "correct" not in out.columns:
        raise ValueError("Expected column 'correct'")
    out["is_error"] = (~out["correct"].astype(bool)).astype(int)
    return out
