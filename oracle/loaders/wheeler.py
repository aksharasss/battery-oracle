"""Loader for Wheeler et al. (2025), "An ageing study of twenty 18650
lithium-ion Graphite/LFP cells in first and second life use", Sci Data 12, 392.
Data: Recherche Data Gouv, doi:10.57745/OLBXKT (CC BY 4.0).

20 A123 18650 LFP/graphite 1.1 Ah cells (the same cell type as Severson),
aged at 50 degC in 7 groups down to 30-50% SoH. Every 100 cycles each cell
gets a characterisation: C/3 capacity, then a C/25 (0.044 A) pseudo-OCV
discharge and charge, sampled every ~1 s. That C/25 discharge is the
curve used here; at that rate the graphite staging peaks are resolved.

Files (Arbin export, ';'-separated): Cell<id>/agingdata_50degC/
WWH_18650_cell<id>_charac<NNNN>.csv, plus 25 degC initial/final
characterisations in initialcharac_25degC/ and finalcharac_25degC/.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from oracle.schema import DischargeCurve

NOMINAL_AH = 1.1
GROUPS = {   # Table 2 of the paper, all ageing at 50 degC
    1: "WLTP discharge, 70% DoD, C/3 CC-CV charge to 3.65 V (reference)",
    2: "as 1, charge cut-off 3.55 V",
    3: "as 1, CC-only charge",
    4: "2C CC-CV charge to 3.65 V, WLTP discharge",
    5: "2C CC-CV charge to 3.55 V, WLTP discharge",
    6: "2C charge to 3.65 V, 2C CC discharge",
    7: "C/3 charge, WLTP discharge to 2.5 V (100% DoD)",
}
COLS = ["Step_Index", "Test_Time", "Current", "Voltage", "Discharge_Capacity", "Charge_Capacity"]


def slow_discharge(csv_path: str | Path, max_abs_a: float = 0.08) -> pd.DataFrame | None:
    """The C/25 discharge step of one characterisation file: the longest
    step whose median current is negative and below max_abs_a."""
    d = pd.read_csv(csv_path, sep=";", usecols=COLS)
    steps = d.groupby("Step_Index")["Current"].agg(["median", "size"])
    slow = steps[(steps["median"] < -0.01) & (steps["median"] > -max_abs_a)]
    if slow.empty:
        return None
    return d[d["Step_Index"] == slow["size"].idxmax()]


def _cycle_of(name: str) -> tuple[int, str]:
    if m := re.search(r"_charac(\d+)\.csv$", name):
        return int(m.group(1)), "50C"
    if name.endswith("_initialcharac.csv"):
        return 0, "25C"
    if name.endswith("_finalcharac.csv"):
        return -1, "25C"   # cycle filled in from the last 50 degC checkup
    raise ValueError(name)


def iter_discharges(cell_dir: str | Path, include_25c: bool = False,
                    min_points: int = 500) -> Iterator[DischargeCurve]:
    """One DischargeCurve per checkup of one cell folder (Cell6a/ etc.)."""
    cell_dir = Path(cell_dir)
    cell = cell_dir.name.replace("Cell", "")
    files = sorted(cell_dir.rglob("*charac*.csv"))
    last_cycle = max((_cycle_of(f.name)[0] for f in files if "_charac" in f.name), default=0)
    for f in files:
        cycle, temp = _cycle_of(f.name)
        if temp == "25C" and not include_25c:
            continue
        seg = slow_discharge(f)
        if seg is None or len(seg) < min_points:
            continue
        q = seg["Discharge_Capacity"].to_numpy()
        i = seg["Current"].to_numpy()
        yield DischargeCurve(
            voltage=seg["Voltage"].to_numpy(),
            capacity=q - q[0],
            current=i,
            cell_id=f"wheeler_{cell}",
            cycle=last_cycle if cycle == -1 else cycle,
            source="wheeler",
            c_rate=float(np.round(np.median(-i) / NOMINAL_AH, 3)),
            meta={"group": int(cell[0]), "temperature": temp,
                  "condition": GROUPS.get(int(cell[0]), "")},
        )


def iter_npz(npz_path: str | Path, include_25c: bool = False, kind: str = "dis",
             min_points: int = 500) -> Iterator[DischargeCurve]:
    """Same curves as iter_discharges, from a compact per-cell .npz holding
    only the C/25 steps (keys '<file stem>__dis' / '__chg', columns Test_Time,
    Current, Voltage, Discharge_Capacity, Charge_Capacity). Written by
    scripts/extract_wheeler_c25.py so the 4.7 GB of raw CSVs need not be
    copied around. kind="chg" gives the C/25 charge, with capacity counted
    from the start of charge."""
    npz_path = Path(npz_path)
    cell = npz_path.stem.replace("Cell", "")
    z = np.load(npz_path)
    keys = sorted(k for k in z.files if k.endswith(f"__{kind}"))
    last_cycle = max((_cycle_of(k.split("__")[0] + ".csv")[0] for k in keys if "_charac" in k
                      and not k.split("__")[0].endswith(("initialcharac", "finalcharac"))), default=0)
    for k in keys:
        cycle, temp = _cycle_of(k.split("__")[0] + ".csv")
        if temp == "25C" and not include_25c:
            continue
        a = z[k]
        if len(a) < min_points:
            continue
        i = a[:, 1]
        q = a[:, 3] if kind == "dis" else a[:, 4]
        yield DischargeCurve(
            voltage=a[:, 2], capacity=q - q[0], current=i, cell_id=f"wheeler_{cell}",
            cycle=last_cycle if cycle == -1 else cycle, source="wheeler",
            c_rate=float(np.round(np.median(np.abs(i)) / NOMINAL_AH, 3)),
            meta={"group": int(cell[0]), "temperature": temp, "mode": kind,
                  "condition": GROUPS.get(int(cell[0]), "")})
