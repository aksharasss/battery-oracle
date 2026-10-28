"""Mechanistic digital twin: fit the half-cell model directly to a measured
slow discharge and read LLI / LAM_PE / LAM_NE off the fitted parameters.

This is the classic degradation-mode analysis (Dubarry, Birkl): instead of
learning features -> mechanisms from synthetic curves, which only works
where real curves look like synthetic ones, the model curve itself is
fitted to each measured curve. It needs a slow-rate discharge (C/25 or so)
so that kinetics barely matter.

1. fit_design(fresh): fit the cell design (electrode capacities, lithium
   inventory, overpotential spread, graphite staging widths/offsets) to the
   fresh checkup of a cell or population.
2. fit_mechanisms(aged, design, error=model_error(fresh, design)): with the
   design frozen, fit (lli, lam_pe, lam_ne) to an aged checkup after
   removing the fresh-state model mismatch.

The residual compares Q(V) on a common voltage grid (capacity in Ah) and
dQ/dV (peak shape), so both the total capacity and the staging peaks count.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from oracle import synth
from oracle.ica import ICAConfig, compute_ica
from oracle.schema import DischargeCurve

ICA = ICAConfig(v_min=3.05, v_max=3.42, dv=0.002, savgol_window_v=0.016, gaussian_sigma=0)
GRID = np.arange(2.9, 3.42, 0.005)


def _q_of_v(curve: DischargeCurve) -> np.ndarray:
    v, q = curve.voltage, curve.capacity
    o = np.argsort(v)
    return np.interp(GRID, v[o], q[o], left=np.nan, right=np.nan)


@dataclass
class Signature:
    q: np.ndarray           # Q(V) on GRID (Ah discharged from top of charge)
    dqdv: np.ndarray        # -dQ/dV on ICA grid
    q_total: float


def signature(curve: DischargeCurve) -> Signature:
    ica = compute_ica(curve, ICA)
    grid = np.arange(ICA.v_min, ICA.v_max + 1e-9, ICA.dv)
    return Signature(_q_of_v(curve), np.interp(grid, ica.voltage, ica.dqdv), curve.total_capacity)


def _model(design, c_rate, v_max, v_min, v_shift=0.0, **deg) -> Signature:
    c = synth.generate(**deg, c_rate=c_rate, v_max=v_max, v_min=v_min, n=2000,
                       noise_v=0, design=design)
    if v_shift:
        # Charge-mode curves: the polarisation adds to the OCV instead of
        # subtracting, so the whole curve sits v_shift higher (hysteresis).
        c = replace(c, voltage=c.voltage + v_shift)
    return signature(c)


def _residual(sig: Signature, m: Signature, w_shape: float) -> np.ndarray:
    rq = np.nan_to_num(sig.q - m.q)                        # Ah
    rs = np.nan_to_num(sig.dqdv - m.dqdv) * w_shape        # Ah/V, scaled
    return np.concatenate([rq, rs, [5 * (sig.q_total - m.q_total)]])


DESIGN_FIELDS = ("c_pe", "c_ne", "li", "sigma0", "gr_w_scale")


def fit_design(fresh: DischargeCurve, c_rate: float, v_max: float = 3.65, v_min: float = 2.5,
               base: synth.CellDesign | None = None, w_shape: float = 0.01,
               fit_shift: bool = False):
    """Fit the cell design to a fresh curve. With fit_shift=True a voltage
    offset (charge-mode hysteresis, see _model) is fitted too and the result
    is (design, rms, v_shift); otherwise (design, rms)."""
    base = base or synth.CellDesign()
    sig = signature(fresh)
    x0 = [getattr(base, f) for f in DESIGN_FIELDS] + list(base.gr_du) + ([0.02] if fit_shift else [])
    lo = [0.8, 0.8, 0.8, 0.0005, 0.3, -0.03, -0.03] + ([-0.06] if fit_shift else [])
    hi = [1.6, 1.8, 1.4, 0.03, 2.0, 0.03, 0.03] + ([0.06] if fit_shift else [])

    def make(x):
        return replace(base, **dict(zip(DESIGN_FIELDS, x[:5])), gr_du=(x[5], x[6]))

    def f(x):
        return _residual(sig, _model(make(x), c_rate, v_max, v_min,
                                     v_shift=x[7] if fit_shift else 0.0), w_shape)

    r = least_squares(f, np.clip(x0, lo, hi), bounds=(lo, hi), diff_step=1e-3, max_nfev=400)
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    return (make(r.x), rms, float(r.x[7])) if fit_shift else (make(r.x), rms)


def model_error(fresh: DischargeCurve, design, c_rate, v_max=3.65, v_min=2.5,
                v_shift: float = 0.0) -> Signature:
    """Measured fresh minus fitted fresh model: the part of the curve the
    model can't reproduce. fit_mechanisms subtracts it from aged curves so
    that this constant mismatch isn't mistaken for degradation."""
    a, m = signature(fresh), _model(design, c_rate, v_max, v_min, v_shift=v_shift)
    return Signature(a.q - m.q, a.dqdv - m.dqdv, a.q_total - m.q_total)


def fit_mechanisms(aged: DischargeCurve, design: synth.CellDesign, c_rate: float,
                   v_max: float = 3.65, v_min: float = 2.5, w_shape: float = 0.01,
                   x0=(0.05, 0.05, 0.05), error: Signature | None = None,
                   v_shift: float = 0.0) -> tuple[dict, float]:
    sig = signature(aged)
    if error is not None:
        sig = Signature(sig.q - np.nan_to_num(error.q), sig.dqdv - np.nan_to_num(error.dqdv),
                        sig.q_total - error.q_total)

    def f(x):
        return _residual(sig, _model(design, c_rate, v_max, v_min, v_shift=v_shift,
                                     lli=x[0], lam_pe=x[1], lam_ne=x[2]), w_shape)

    best = None
    for start in (x0, (0.2, 0.02, 0.02), (0.02, 0.2, 0.02), (0.02, 0.02, 0.2), (0.4, 0.1, 0.05)):
        r = least_squares(f, start, bounds=([0, 0, 0], [0.8, 0.8, 0.8]), diff_step=1e-3,
                          max_nfev=200)
        if best is None or r.cost < best.cost:
            best = r
    mech = dict(zip(("LLI", "LAM_PE", "LAM_NE"), map(float, best.x)))
    return mech, float(np.sqrt(np.mean(best.fun ** 2)))
