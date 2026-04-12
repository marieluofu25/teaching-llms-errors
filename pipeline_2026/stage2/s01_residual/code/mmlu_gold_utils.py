"""Map legacy MMLU ``data_y`` labels to gold letters A–D (no torch / HF imports)."""
from __future__ import annotations

import numbers

import pandas as pd

_NUM_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}


def mmlu_gold_letter(data_y) -> str | None:
    """Return gold letter for ``data_y``: numeric 1–4, float 1.0–4.0, or A–D string."""
    if data_y is None or isinstance(data_y, bool):
        return None
    if isinstance(data_y, numbers.Integral):
        k = int(data_y)
        if 1 <= k <= 4:
            return _NUM_TO_LETTER[k]
        return None
    if isinstance(data_y, numbers.Real):
        fv = float(data_y)
        if pd.isna(fv):
            return None
        if abs(fv - round(fv)) < 1e-9:
            k = int(round(fv))
            if 1 <= k <= 4:
                return _NUM_TO_LETTER[k]
        return None
    s = str(data_y).strip()
    if not s:
        return None
    u = s.upper()
    if len(u) == 1 and u in "ABCD":
        return u
    if len(u) == 1 and u in "1234":
        return _NUM_TO_LETTER[int(u)]
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            k = int(round(v))
            if 1 <= k <= 4:
                return _NUM_TO_LETTER[k]
    except ValueError:
        pass
    return None
