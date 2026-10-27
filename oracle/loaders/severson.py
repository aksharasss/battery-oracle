"""Loader for the Severson et al. (2019) MIT/Stanford/Toyota LFP dataset.

124 A123 APR18650M1A cells (LFP/graphite, 1.1 Ah), fast-charged with
different protocols and all discharged at 4C to 2.0 V. Data is published
as three MATLAB v7.3 (HDF5) files, one per batch, on data.matr.io.

Structure (from the authors' BuildPkl_Batch scripts):
    batch/cycles[i]   per-cell group of per-cycle references:
                      I (C-rate), V, Qd (Ah), Qc, T, t, Qdlin, Vdlin...
    batch/policy_readable[i], batch/cycle_life[i]

NOTE: written against that published structure; check it on the first
real file. Batch 1 also has known glitches (cells that need stitching with
batch 2, a few bad cycles) that the authors' scripts patch; we only drop
cycles that fail basic checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from oracle.schema import DischargeCurve

NOMINAL_AH = 1.1


def _read_str(f, ref) -> str:
    arr = f[ref][()]
    return arr.tobytes()[::2].decode(errors="ignore")


def iter_discharges(
    mat_path: str | Path,
    cells: list[int] | None = None,
    cycle_step: int = 1,
    min_points: int = 50,
) -> Iterator[DischargeCurve]:
    """Yield one DischargeCurve per (cell, cycle) from a batch .mat file.

    The discharge segment is taken as the samples with current below
    -0.1C; capacity is Qd re-zeroed at the start of that segment.
    """
    mat_path = Path(mat_path)
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        n_cells = batch["cycles"].shape[0]
        for i in cells if cells is not None else range(n_cells):
            policy = _read_str(f, batch["policy_readable"][i, 0])
            cycle_life = float(np.ravel(f[batch["cycle_life"][i, 0]][()])[0])
            cycles = f[batch["cycles"][i, 0]]
            n_cycles = cycles["I"].shape[0]
            cell_id = f"{mat_path.stem}_c{i}"
            for j in range(1, n_cycles, cycle_step):  # cycle 0 is often incomplete
                I = np.hstack(f[cycles["I"][j, 0]][()])
                V = np.hstack(f[cycles["V"][j, 0]][()])
                Qd = np.hstack(f[cycles["Qd"][j, 0]][()])
                T = np.hstack(f[cycles["T"][j, 0]][()])
                dis = I < -0.1
                if dis.sum() < min_points:
                    continue
                q = Qd[dis] - Qd[dis][0]
                if q.max() < 0.5 * NOMINAL_AH or q.max() > 1.3 * NOMINAL_AH:
                    continue  # corrupt cycle
                yield DischargeCurve(
                    voltage=V[dis],
                    capacity=q,
                    current=I[dis] * NOMINAL_AH,
                    temperature=T[dis],
                    cell_id=cell_id,
                    cycle=j,
                    source="severson",
                    c_rate=4.0,
                    meta={"policy": policy, "cycle_life": cycle_life},
                )
