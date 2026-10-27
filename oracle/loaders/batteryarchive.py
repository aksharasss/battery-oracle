"""Loader for batteryarchive.org time-series CSVs (Sandia LFP cells and others).

Sandia (Preger et al. 2020) cycled A123 LFP 18650s at several temperatures,
depths of discharge and discharge rates (0.5C to 3C), which is exactly the
spread we need for cross-dataset validation. batteryarchive.org serves them
in one standard CSV layout with columns such as:

    Cycle_Index, Test_Time (s), Current (A), Voltage (V),
    Charge_Capacity (Ah), Discharge_Capacity (Ah), Cell_Temperature (C)

NOTE: column names are matched loosely below; confirm on the first file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from oracle.schema import DischargeCurve


def _col(df: pd.DataFrame, *starts: str) -> str | None:
    for c in df.columns:
        lc = c.lower()
        if any(lc.startswith(s) for s in starts):
            return c
    return None


def iter_discharges(
    csv_path: str | Path,
    source: str = "sandia",
    nominal_ah: float = 1.1,
    min_points: int = 50,
) -> Iterator[DischargeCurve]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    c_cyc = _col(df, "cycle_index")
    c_i = _col(df, "current")
    c_v = _col(df, "voltage")
    c_qd = _col(df, "discharge_capacity")
    c_t = _col(df, "cell_temperature", "temperature")
    if None in (c_cyc, c_i, c_v, c_qd):
        raise ValueError(f"{csv_path.name}: missing expected columns, got {list(df.columns)}")

    for cyc, g in df.groupby(c_cyc):
        dis = g[g[c_i] < -0.05 * nominal_ah]
        if len(dis) < min_points:
            continue
        q = dis[c_qd].to_numpy()
        q = q - q[0]
        if q.max() < 0.1 * nominal_ah:
            continue
        i = dis[c_i].to_numpy()
        yield DischargeCurve(
            voltage=dis[c_v].to_numpy(),
            capacity=q,
            current=i,
            temperature=dis[c_t].to_numpy() if c_t else None,
            cell_id=csv_path.stem,
            cycle=int(cyc),
            source=source,
            c_rate=float(np.round(np.median(-i) / nominal_ah, 2)),
        )
