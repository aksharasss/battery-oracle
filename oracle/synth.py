"""Synthetic LFP/graphite full-cell discharges with known degradation.

There is no teardown ground truth for the public datasets, so this module
is the ground truth: a half-cell ("alawa"-style) model of an A123-like
LFP/graphite 18650 where we dial in exactly how much of each mechanism a
cell has and get back a discharge curve in the same format the loaders
produce.

Model
-----
Each electrode has an open-circuit potential (OCP) as a function of its
lithiation x in [0, 1] and a capacity in Ah:

- graphite (NE): Multi-Species Multi-Reaction (MSMR) model, Verbrugge et
  al., J. Electrochem. Soc. 164 (2017) E3243. Six galleries, each a
  Fermi-Dirac step x_j(U) = X_j / (1 + exp(f (U - U0_j) / w_j)). The
  staging steps are what make the full-cell dQ/dV peaks.
- LFP (PE): one flat MSMR gallery at 3.42 V plus the steep end terms of
  the Prada et al. (2013) OCP so the voltage collapses when LFP is full.

Cyclable lithium (Ah) is conserved: Li = C_pe * x_pe + C_ne * x_ne.
A cell is charged CC-CV to 3.6 V, so at the top the OCV is exactly 3.6 V;
that fixes where both electrodes start. Discharging q Ah moves
x_ne -> x_ne - q/C_ne and x_pe -> x_pe + q/C_pe, and the terminal voltage is

    V(q) = U_pe(x_pe) - U_ne(x_ne) - I * R  - (distributed overpotential)

where R changes linearly with depth of discharge and the distributed
overpotential is a zero-mean Gaussian spread of local overpotentials whose
width grows with current. Together they smear the graphite staging peaks
into the single broad ~3.13 V peak Severson sees at 4C; they were fitted to
real Severson cycle-10 discharges (see CellDesign).

Degradation modes (Dubarry et al., J. Power Sources 219 (2012) 204;
Birkl et al., J. Power Sources 341 (2017) 373):

- LLI     cyclable lithium shrinks: Li -> Li - lli * Q_fresh
          (lli is a fraction of the fresh cell's C/20 capacity)
- LAM_PE  C_pe -> C_pe * (1 - lam_pe)   (delithiated LFP lost)
- LAM_NE  C_ne -> C_ne * (1 - lam_ne)   (delithiated graphite lost)
- r_ohm   extra series resistance in ohms (kinetic / contact loss)

What each mode does to the curve (see scripts/synth_validation.py):
LLI removes the highest-voltage graphite peak first (the NE never gets
fully lithiated), LAM_PE cuts the end of discharge (LFP fills up early,
removing the lowest-voltage peak), LAM_NE squeezes every graphite peak
together, and r_ohm slides everything to lower voltage without changing
areas.

Public API
----------
generate(lli, lam_pe, lam_ne, r_ohm, c_rate, n, ...) -> DischargeCurve
generate_dataset(...) -> (list[DischargeCurve], labels DataFrame)
save_dataset / load_dataset
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import exponnorm

from oracle.schema import DischargeCurve

F_RT = 96485.0 / (8.314 * 298.15)    # 1/V at 25 C
NOMINAL_AH = 1.1                     # A123 APR18650M1A rating; 1C = 1.1 A
V_MAX, V_MIN = 3.6, 2.0              # Severson charge and discharge cut-offs
MODES = ("lli", "lam_pe", "lam_ne")

# Graphite MSMR parameters (Verbrugge et al. 2017), high to low lithiation.
GRAPHITE_U0 = np.array([0.08843, 0.12799, 0.14331, 0.16984, 0.21446, 0.36325])
GRAPHITE_X = np.array([0.43336, 0.23963, 0.15018, 0.05462, 0.06744, 0.03476])
GRAPHITE_W = np.array([0.08611, 0.08009, 0.72469, 2.53277, 0.09470, 5.97354])

LFP_U0, LFP_W = 3.422, 0.35   # w < 1 keeps the plateau within ~30 mV


# ─────────────────────────────────────────────────────────────────────────
# Half-cell OCPs
# ─────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _graphite_table(w_scale: float = 1.0, du: tuple = (0.0, 0.0), n: int = 8000):
    """x(U) on a fine U grid. w_scale scales the two main staging galleries'
    widths, du shifts their potentials (both fitted to C/25 data)."""
    u0, w = GRAPHITE_U0.copy(), GRAPHITE_W.copy()
    u0[:2] += np.asarray(du)
    w[:2] *= w_scale
    u = np.linspace(0.0, 1.2, n)
    x = np.sum(GRAPHITE_X[:, None] * expit(-F_RT * (u[None, :] - u0[:, None]) / w[:, None]), axis=0)
    return x[::-1], u[::-1]          # x ascending, U descending


def ocp_graphite(x: np.ndarray, w_scale: float = 1.0, du: tuple = (0.0, 0.0)) -> np.ndarray:
    """Graphite OCP vs Li/Li+ (V) at lithiation x."""
    gx, gu = _graphite_table(float(w_scale), tuple(float(v) for v in du))
    return np.interp(x, gx, gu)


def ocp_lfp(x: np.ndarray, w: float = LFP_W) -> np.ndarray:
    """LFP OCP vs Li/Li+ (V) at lithiation x."""
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return (LFP_U0 + w / F_RT * np.log((1 - x) / x)
            + 0.5 * np.exp(-150 * x) - 0.9 * np.exp(-30 * (1 - x)))


# ─────────────────────────────────────────────────────────────────────────
# Cell
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CellDesign:
    """Fresh-cell balance, calibrated to Severson cycle-10 discharges at 4 A."""
    c_pe: float = 1.0997      # LFP capacity (Ah)
    c_ne: float = 1.3722      # graphite capacity (Ah)
    li: float = 1.0900        # cyclable lithium (Ah)
    r0: float = 0.04437       # effective resistance at the start of discharge (ohm)
    r_q: tuple = (-0.01744, -0.00053, 0.07829)  # resistance change with depth of
                              # discharge z in [0, 1]: r0 + r_q[0] z + r_q[1] z^2
                              # + r_q[2] z^3 (ohm). It first falls as the cell
                              # self-heats, then rises at the end (solid diffusion).
    sigma0: float = 0.00258   # overpotential spread at rest (V)
    sigma_per_a: float = 0.00429  # Gaussian spread per amp of current (V/A)
    tau_per_a: float = 0.00492    # one-sided overpotential tail per amp (V/A)
    # c_pe, c_ne, li, r0, r_q, sigma_per_a and tau_per_a were fitted
    # (Nelder-Mead) to the mean of 24 Severson cells at cycle 10 (4 A), on top
    # of the C/25 material fit below: dQ/dV on 2.9-3.35 V, V(Q), peak position,
    # FWHM and capacity. dQ/dV RMSE 0.29 Ah/V on a 6.5 Ah/V peak, position
    # within 1 mV, FWHM within 3 mV, capacity within 1 mAh, V(Q) RMSE 24 mV.
    # Both fits put li close to c_pe: LFP is nearly the limiting electrode.
    r_rc: float = 0.0629      # relaxing resistance: extra voltage at the start of
    q_rc: float = 0.0259      # discharge, I * r_rc * exp(-q / q_rc) (ohm, Ah). The
                              # overpotential takes ~25 mAh to build up; fitted to
                              # Severson's dQ/dV area above 3.2 V (0.036 Ah).
    lfp_w: float = 0.1444     # LFP plateau width parameter (MSMR w)
    gr_w_scale: float = 0.3133  # width scale of graphite stage 1 and 2 galleries
    gr_du: tuple = (0.00263, -0.00374)  # potential shift of graphite stage 1, 2 (V)
    # sigma0, lfp_w, gr_w_scale and gr_du (material properties) were fitted to
    # the C/25 discharge of Wheeler et al. (2025) cell 6a, checkup 0 (same A123
    # cell, 50 degC): peak positions within 1 mV, FWHM 10/11 mV vs 10/9 mV,
    # heights 17.3/35.3 vs 15.7/35.9 Ah/V for the top two peaks.
    # li, r0, r_q, sigma_per_a and tau_per_a were fitted (Nelder-Mead) to the
    # mean of 24 Severson cells at cycle 10 (4 A): dQ/dV on 2.9-3.35 V, V(Q),
    # peak position, FWHM and capacity. Result: dQ/dV RMSE 0.14 Ah/V on a
    # 6.5 Ah/V peak, peak position and FWHM within 1 mV, capacity within
    # 1 mAh, V(Q) RMSE 43 mV (mostly the last 10 % of discharge).

    @classmethod
    def sample(cls, rng: np.random.Generator, spread: float = 0.015) -> "CellDesign":
        """A fresh cell with manufacturing scatter (relative sd = spread)."""
        d = cls()
        j = lambda v, s=spread: float(v * (1 + s * rng.standard_normal()))
        return cls(c_pe=j(d.c_pe), c_ne=j(d.c_ne), li=j(d.li, spread / 2),
                   r0=j(d.r0, 3 * spread), r_q=tuple(j(c, 3 * spread) for c in d.r_q), sigma0=d.sigma0,
                   sigma_per_a=j(d.sigma_per_a, 3 * spread),
                   tau_per_a=j(d.tau_per_a, 3 * spread),
                   r_rc=j(d.r_rc, 3 * spread), q_rc=j(d.q_rc, 3 * spread))


@dataclass(frozen=True)
class Degradation:
    lli: float = 0.0
    lam_pe: float = 0.0
    lam_ne: float = 0.0
    r_ohm: float = 0.0
    het: float = 0.0
    r_diff: float = 0.0

    def apply(self, d: CellDesign, q_fresh: float) -> CellDesign:
        return replace(d, c_pe=d.c_pe * (1 - self.lam_pe), c_ne=d.c_ne * (1 - self.lam_ne),
                       li=d.li - self.lli * q_fresh, r0=d.r0 + self.r_ohm,
                       r_q=(d.r_q[0] + self.r_diff, *d.r_q[1:]),
                       sigma_per_a=d.sigma_per_a * (1 + self.het),
                       tau_per_a=d.tau_per_a * (1 + self.het))


def ocv_curve(d: CellDesign, n: int = 3000, v_max: float = V_MAX, v_min: float = V_MIN):
    """Full-cell OCV vs discharged capacity, from v_max to the v_min cut-off.

    Returns (q, ocv, x_ne_top, x_pe_top).
    """
    U_pe = lambda x: ocp_lfp(x, d.lfp_w)
    U_ne = lambda x: ocp_graphite(x, d.gr_w_scale, d.gr_du)

    def top(x_ne):
        x_pe = (d.li - d.c_ne * x_ne) / d.c_pe
        return U_pe(x_pe) - U_ne(x_ne) - v_max

    lo = max(1e-6, (d.li - d.c_pe * (1 - 1e-6)) / d.c_ne)
    hi = min(1 - 1e-6, (d.li - d.c_pe * 1e-6) / d.c_ne)
    if lo >= hi:
        raise ValueError(f"electrode balance leaves no room to charge to {v_max} V")
    if top(hi) < 0:
        # Graphite fills before the cell reaches v_max (heavy LAM_NE): the
        # CV step plates the surplus lithium, which is lost to cycling.
        x_ne0 = hi
        target = v_max + float(U_ne(hi))
        x_pe0 = brentq(lambda x: U_pe(x) - target, 1e-9, 1 - 1e-9, xtol=1e-12)
    else:
        x_ne0 = brentq(top, lo, hi, xtol=1e-12)
        x_pe0 = (d.li - d.c_ne * x_ne0) / d.c_pe
    q_lim = min(x_ne0 * d.c_ne, (1 - x_pe0) * d.c_pe) * (1 - 1e-7)
    q = np.linspace(0, q_lim, n)
    ocv = U_pe(x_pe0 + q / d.c_pe) - U_ne(x_ne0 - q / d.c_ne)
    end = np.argmax(ocv < v_min) if (ocv < v_min).any() else n
    q, ocv = q[:max(end, 2)], ocv[:max(end, 2)]
    return q, np.minimum.accumulate(ocv), x_ne0, x_pe0


def _loaded_q_of_v(q, ocv, current, d: CellDesign, v_grid):
    """Q discharged when the terminal voltage reaches each v in v_grid.

    Local overpotentials follow an exponentially modified Gaussian: a
    Gaussian of width sigma plus a one-sided exponential tail of mean tau
    (parts of the electrode that lag behind). The tail gives the real 4C
    peak its sharp high-voltage edge and long low-voltage tail.
    """
    sigma = d.sigma0 + d.sigma_per_a * current
    tau = d.tau_per_a * current
    # Dense uniform quadrature: Gauss-Hermite nodes would leave ripples on
    # the sharp graphite steps that ICA picks up as fake peaks.
    eps = np.linspace(-4 * sigma, 4 * sigma + 8 * tau, 401)
    w = exponnorm.pdf(eps, tau / sigma, scale=sigma) if tau > 0 else np.exp(-0.5 * (eps / sigma) ** 2)
    w = w / w.sum()
    # np.interp needs ascending x: ocv falls with q, so flip both.
    e_asc, q_asc = ocv[::-1], q[::-1]
    shifted = v_grid[:, None] + current * d.r0 + eps[None, :]
    q_of = np.interp(shifted, e_asc, q_asc, left=q[-1], right=0.0)
    return q_of @ w


def generate(
    lli: float = 0.0,
    lam_pe: float = 0.0,
    lam_ne: float = 0.0,
    r_ohm: float = 0.0,
    het: float = 0.0,
    r_diff: float = 0.0,
    c_rate: float = 4.0 / NOMINAL_AH,
    v_max: float = V_MAX,
    v_min: float = V_MIN,
    n: int = 500,
    window: tuple[float, float] | None = None,
    noise_v: float = 0.0002,
    v_resolution: float = 1e-6,
    design: CellDesign | None = None,
    seed: int | None = None,
    cell_id: str = "synth",
    cycle: int = 0,
) -> DischargeCurve:
    """One constant-current discharge from a full (3.6 V) cell.

    lli, lam_pe, lam_ne : fractions (0.1 = 10 %), see module docstring.
    r_ohm   : added resistance in ohms.
    het     : growth of the local-overpotential spread (0.5 = 50 % wider), i.e.
              the electrode becoming kinetically less uniform with age. At
              4C it lowers and broadens the peak without moving its area.
    r_diff  : polarisation that grows through the discharge, in ohms at the
              end of discharge (transport loss: electrolyte, particle
              contact). At 4C it stretches the peak to lower voltage and
              lowers it while keeping its sharp high-voltage edge, which is
              how real aged Severson cells look.
    c_rate  : discharge rate in C (1C = 1.1 A). Default matches Severson's 4 A.
    v_max, v_min : charge and discharge cut-offs (Severson 3.6/2.0 V,
              Wheeler 2025 3.65/2.5 V).
    n       : number of samples, evenly spaced in time.
    window  : (v_high, v_low) to return only a partial discharge between
              those voltages, with capacity re-zeroed at v_high. None = full
              3.6 -> 2.0 V discharge.
    noise_v : voltage noise sd (V); v_resolution: cycler quantisation (V; Severson is ~1 uV).

    The true mode amounts are in curve.meta.
    """
    rng = np.random.default_rng(seed)
    base = design or CellDesign()
    deg = Degradation(lli, lam_pe, lam_ne, r_ohm, het, r_diff)
    d = deg.apply(base, fresh_capacity(base))
    current = c_rate * NOMINAL_AH

    q_ocv, ocv, x_ne0, x_pe0 = ocv_curve(d, v_max=v_max, v_min=v_min)
    # Polarisation that changes linearly with depth of discharge. Fitted to
    # Severson it falls (self-heating), which squeezes the graphite staging
    # peaks into one.
    z = q_ocv / q_ocv[-1]
    ocv = ocv + current * d.r_rc * np.exp(-q_ocv / d.q_rc)
    ocv = np.minimum.accumulate(ocv - current * z * (d.r_q[0] + z * (d.r_q[1] + z * d.r_q[2])))
    v_grid = np.linspace(v_max, v_min, 8000)
    q_v = np.maximum.accumulate(_loaded_q_of_v(q_ocv, ocv, current, d, v_grid))
    started = q_v > 1e-6
    v_grid, q_v = v_grid[started], q_v[started]
    v_grid = np.concatenate(([v_max - current * d.r0], v_grid))
    q_v = np.concatenate(([0.0], q_v))

    # Sample evenly in time like a cycler, i.e. evenly in Q at constant current.
    keep = np.concatenate(([True], np.diff(q_v) > 0))
    q = np.linspace(0, q_v[keep][-1], n)
    v = np.interp(q, q_v[keep], v_grid[keep])
    v = v + noise_v * rng.standard_normal(n)
    if v_resolution:
        v = np.round(v / v_resolution) * v_resolution

    if window is not None:
        v_hi, v_lo = window
        mask = (v <= v_hi) & (v >= v_lo)
        if mask.sum() < 10:
            raise ValueError(f"fewer than 10 samples between {v_lo} and {v_hi} V")
        first, last = np.argmax(mask), len(mask) - np.argmax(mask[::-1])
        v, q = v[first:last], q[first:last] - q[first]

    return DischargeCurve(
        voltage=v,
        capacity=q,
        cell_id=cell_id,
        cycle=cycle,
        source="synthetic",
        current=np.full(v.size, -current),
        c_rate=c_rate,
        meta=dict(lli=lli, lam_pe=lam_pe, lam_ne=lam_ne, r_ohm=r_ohm, het=het, r_diff=r_diff,
                  dominant_mode=dominant_mode(lli, lam_pe, lam_ne),
                  q_full=float(q_v[-1]), x_ne_top=float(x_ne0), x_pe_top=float(x_pe0),
                  limiting_electrode=_limiting(d, x_ne0, x_pe0, q_ocv[-1]),
                  time_s=q / current * 3600.0, window=window),
    )


def fresh_capacity(design: CellDesign | None = None) -> float:
    """C/20-like (OCV) capacity of the undegraded cell, Ah."""
    q, *_ = ocv_curve(design or CellDesign())
    return float(q[-1])


def _limiting(d, x_ne0, x_pe0, q_end):
    """Which electrode runs out first at the end of discharge."""
    ne_left = x_ne0 * d.c_ne - q_end
    pe_left = (1 - x_pe0) * d.c_pe - q_end
    return "NE" if ne_left < pe_left else "PE"


def dominant_mode(lli: float, lam_pe: float, lam_ne: float, mixed_ratio: float = 0.6) -> str:
    """Label used for scoring: the largest mode, or 'mixed' if no mode holds
    at least `mixed_ratio` of the total, or 'fresh' if nothing is degraded."""
    amounts = dict(lli=lli, lam_pe=lam_pe, lam_ne=lam_ne)
    total = sum(amounts.values())
    if total < 1e-6:
        return "fresh"
    top = max(amounts, key=amounts.get)
    return top if amounts[top] / total >= mixed_ratio else "mixed"


# Grade hint that follows the project's mapping (LLI -> A, LAM -> B, mixed -> C).
GRADE_HINT = {"fresh": "A", "lli": "A", "lam_pe": "B", "lam_ne": "B", "mixed": "C"}


# ─────────────────────────────────────────────────────────────────────────
# Labelled dataset
# ─────────────────────────────────────────────────────────────────────────

# Per-profile end-of-test amounts (fraction) for (lli, lam_pe, lam_ne):
# (low, high) for the dominant mode, (0, minor) for the others.
PROFILES = {
    "lli": dict(lli=(0.08, 0.25), lam_pe=(0, 0.03), lam_ne=(0, 0.03)),
    "lam_pe": dict(lli=(0, 0.03), lam_pe=(0.15, 0.35), lam_ne=(0, 0.03)),
    "lam_ne": dict(lli=(0, 0.03), lam_pe=(0, 0.03), lam_ne=(0.10, 0.25)),
    "mixed": dict(lli=(0.06, 0.18), lam_pe=(0.08, 0.25), lam_ne=(0.05, 0.18)),
}


def generate_dataset(
    n_cells_per_profile: int = 25,
    n_cycles: int = 1000,
    cycle_step: int = 50,
    c_rate: float = 4.0 / NOMINAL_AH,
    window: tuple[float, float] | None = None,
    r_growth: float = 0.0,
    r_diff_range: tuple[float, float] = (0.0, 0.15),
    het_range: tuple[float, float] = (0.0, 0.25),
    seed: int = 0,
    n: int = 500,
    noise_v: float = 0.0002,
) -> tuple[list[DischargeCurve], pd.DataFrame]:
    """Labelled ageing trajectories, one profile per cell.

    Each cell gets a fresh design with manufacturing scatter and end-of-test
    amounts drawn from PROFILES. Mode amounts grow along a power law
    amount(k) = end * (k / n_cycles) ** p with p in [0.6, 1.4], so some
    cells fade early and some late. r_growth is extra ohms at end of test.

    Every cell, whatever its profile, also gets kinetic ageing drawn from
    r_diff_range (ohm) and het_range. These are not mechanism labels; they
    are the nuisance that makes real aged 4C peaks slide down, shrink and
    broaden. The defaults come from fitting the model to aged Severson cells
    (cycles at 95-78 % capacity): r_diff reached 0.02-0.13 ohm, het 0-0.2.

    Returns (curves, labels), where labels has one row per curve with
    cell_id, cycle, profile, true lli/lam_pe/lam_ne/r_ohm, dominant_mode,
    grade_hint and q_full.
    """
    rng = np.random.default_rng(seed)
    curves, rows = [], []
    cycles = np.arange(0, n_cycles + 1, cycle_step)
    for profile, ranges in PROFILES.items():
        for c in range(n_cells_per_profile):
            cell_id = f"syn_{profile}_{c:03d}"
            design = CellDesign.sample(rng)
            end = {m: rng.uniform(*ranges[m]) for m in MODES}
            power = {m: rng.uniform(0.6, 1.4) for m in MODES}
            r_end = r_growth * rng.uniform(0.5, 1.5)
            rd_end, het_end = rng.uniform(*r_diff_range), rng.uniform(*het_range)
            for k in cycles:
                frac = k / n_cycles
                amt = {m: end[m] * frac ** power[m] for m in MODES}
                kin = dict(r_diff=rd_end * frac ** power["lli"], het=het_end * frac)
                curve = generate(**amt, r_ohm=r_end * frac, **kin, c_rate=c_rate, n=n,
                                 window=window, noise_v=noise_v, design=design,
                                 seed=int(rng.integers(2**31)), cell_id=cell_id, cycle=int(k))
                curves.append(curve)
                rows.append(dict(cell_id=cell_id, cycle=int(k), profile=profile, **amt,
                                 r_ohm=r_end * frac, **kin, dominant_mode=curve.meta["dominant_mode"],
                                 grade_hint=GRADE_HINT[curve.meta["dominant_mode"]],
                                 q_full=curve.meta["q_full"],
                                 limiting_electrode=curve.meta["limiting_electrode"]))
    return curves, pd.DataFrame(rows)


def save_dataset(curves: list[DischargeCurve], labels: pd.DataFrame, path: str | Path) -> Path:
    """Write curves to <path>.npz and labels to <path>_labels.csv."""
    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.array([c.voltage.size for c in curves])
    np.savez_compressed(
        path.with_suffix(".npz"),
        voltage=np.concatenate([c.voltage for c in curves]),
        capacity=np.concatenate([c.capacity for c in curves]),
        current=np.concatenate([c.current for c in curves]),
        lengths=lengths,
        cell_id=np.array([c.cell_id for c in curves]),
        cycle=np.array([c.cycle for c in curves]),
        c_rate=np.array([c.c_rate for c in curves]),
    )
    labels.to_csv(path.parent / f"{path.name}_labels.csv", index=False)
    return path.with_suffix(".npz")


def load_dataset(path: str | Path) -> tuple[list[DischargeCurve], pd.DataFrame]:
    """Inverse of save_dataset. Labels are also copied into each curve.meta."""
    path = Path(path).with_suffix("")
    z = np.load(path.with_suffix(".npz"))
    labels = pd.read_csv(path.parent / f"{path.name}_labels.csv")
    lab = labels.set_index(["cell_id", "cycle"])
    splits = np.cumsum(z["lengths"])[:-1]
    curves = []
    for v, q, i, cid, cyc, cr in zip(np.split(z["voltage"], splits), np.split(z["capacity"], splits),
                                     np.split(z["current"], splits), z["cell_id"], z["cycle"], z["c_rate"]):
        meta = lab.loc[(str(cid), int(cyc))].to_dict() if (str(cid), int(cyc)) in lab.index else {}
        curves.append(DischargeCurve(v, q, str(cid), int(cyc), "synthetic", current=i,
                                     c_rate=float(cr), meta=meta))
    return curves, labels
