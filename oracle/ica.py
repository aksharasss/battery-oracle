"""Incremental Capacity Analysis: turn a discharge curve into a dQ/dV curve.

Raw dQ/dV from cycler data is mostly noise, because on the flat LFP plateau
dV between samples is close to the voltage resolution. So instead of
differentiating sample by sample we:

1. keep only the voltage window of interest (the partial discharge),
2. make Q a single-valued function of V (sort, merge duplicate voltages),
3. resample Q onto a uniform voltage grid,
4. differentiate with a Savitzky-Golay filter, which smooths and
   differentiates in one step,
5. optionally apply a light Gaussian smooth on top.

On discharge V falls while Q rises, so dQ/dV is negative; we return
-dQ/dV so peaks point up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from oracle.schema import DischargeCurve


@dataclass
class ICAConfig:
    v_min: float = 3.00        # lower edge of the partial-discharge window (V)
    v_max: float = 3.45        # upper edge (V)
    dv: float = 0.002          # resampling step (V)
    savgol_window_v: float = 0.040   # Savitzky-Golay window width, in volts
    savgol_order: int = 3
    gaussian_sigma: float = 1.0      # extra smoothing, in grid points (0 = off)


@dataclass
class ICACurve:
    voltage: np.ndarray        # uniform grid, ascending (V)
    dqdv: np.ndarray           # -dQ/dV (Ah/V), positive peaks
    capacity: np.ndarray       # Q resampled on the same grid (Ah)
    cell_id: str
    cycle: int
    source: str


def _monotonic_q_of_v(v: np.ndarray, q: np.ndarray, decimals: int = 4):
    """Return (v_unique ascending, q) with one Q per voltage.

    Cycler voltage is quantised and noisy, so the same voltage appears many
    times. We round to 0.1 mV and take the mean Q at each voltage.
    """
    v_r = np.round(v, decimals)
    order = np.argsort(v_r, kind="stable")
    v_s, q_s = v_r[order], q[order]
    v_u, idx, counts = np.unique(v_s, return_index=True, return_counts=True)
    q_u = np.add.reduceat(q_s, idx) / counts
    # Q must fall as V rises on a discharge; enforce it so noise can't
    # create negative dQ/dV spikes.
    q_u = np.minimum.accumulate(q_u)
    return v_u, q_u


def compute_ica(curve: DischargeCurve, cfg: ICAConfig | None = None) -> ICACurve:
    cfg = cfg or ICAConfig()
    v, q = curve.voltage, curve.capacity

    mask = (v >= cfg.v_min) & (v <= cfg.v_max)
    if mask.sum() < 10:
        raise ValueError(
            f"{curve.cell_id} cycle {curve.cycle}: fewer than 10 points in "
            f"{cfg.v_min}-{cfg.v_max} V window"
        )
    v_u, q_u = _monotonic_q_of_v(v[mask], q[mask])

    grid = np.arange(cfg.v_min, cfg.v_max + cfg.dv / 2, cfg.dv)
    grid = grid[(grid >= v_u[0]) & (grid <= v_u[-1])]
    q_grid = np.interp(grid, v_u, q_u)

    window = int(round(cfg.savgol_window_v / cfg.dv))
    window = max(window | 1, cfg.savgol_order + 2 | 1)  # odd, > order
    if window >= grid.size:
        raise ValueError("Savitzky-Golay window is wider than the voltage window")
    dqdv = -savgol_filter(q_grid, window, cfg.savgol_order, deriv=1, delta=cfg.dv)
    if cfg.gaussian_sigma > 0:
        dqdv = gaussian_filter1d(dqdv, cfg.gaussian_sigma)
    dqdv = np.clip(dqdv, 0, None)

    return ICACurve(grid, dqdv, q_grid, curve.cell_id, curve.cycle, curve.source)
