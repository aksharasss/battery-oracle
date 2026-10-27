"""Loader for a single user-uploaded partial discharge (used by the UI).

Accepts a CSV with a voltage column and either a capacity column or
current + time columns (capacity is then integrated).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from oracle.schema import DischargeCurve


def _find(df, *keys):
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None


def load_csv(file: str | io.IOBase, cell_id: str = "uploaded") -> DischargeCurve:
    df = pd.read_csv(file)
    c_v = _find(df, "volt")
    c_q = _find(df, "capacity", "ah")
    c_i = _find(df, "current")
    c_t = _find(df, "time")
    if c_v is None:
        raise ValueError("no voltage column found")
    if c_q is not None:
        q = df[c_q].to_numpy(float)
        q = np.abs(q - q[0])
    elif c_i is not None and c_t is not None:
        i = np.abs(df[c_i].to_numpy(float))
        t = df[c_t].to_numpy(float)
        q = np.concatenate([[0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))]) / 3600
    else:
        raise ValueError("need a capacity column, or current and time columns")
    return DischargeCurve(df[c_v].to_numpy(float), q, cell_id=cell_id, cycle=0, source="upload")
