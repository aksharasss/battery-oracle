"""Peak features from an ICA curve: the degradation fingerprint.

For an LFP/graphite cell the LFP side is one flat plateau, so the dQ/dV
peaks seen on discharge come mostly from graphite staging transitions.
How they move tells us the mechanism (the Phase 2 synthetic model is what
confirms this mapping for our cells):

- peak shifts in voltage, total area roughly constant  -> resistance / kinetics
- peak areas shrink together, positions fixed          -> loss of lithium (LLI)
- one peak's area shrinks more than the others         -> loss of active
                                                          material (LAM_NE/PE)

Peaks are named by the voltage window they fall in, so the same physical
peak lines up across cycles and cells. The default windows are a starting
guess and must be tuned per dataset and C-rate in Phase 1, because high
discharge rates push every peak to lower voltage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from oracle.ica import ICAConfig, ICACurve


@dataclass
class PeakConfig:
    # name -> (v_low, v_high). Ordered high voltage to low.
    windows: dict = field(default_factory=lambda: {
        "p1": (3.28, 3.42),
        "p2": (3.15, 3.28),
        "p3": (3.00, 3.15),
    })
    min_prominence_frac: float = 0.03   # of the curve's max dQ/dV
    # fixed voltage bands whose dQ/dV area is reported whether or not a
    # peak is resolved there (needed at 4C, where peaks merge)
    bands: dict = field(default_factory=dict)
    # how grading.Reference regroups the bands around the anchor peak:
    # "fwhm" (edges in fresh-FWHM units) or "volt" (edges in volts)
    band_scale: str = "fwhm"
    band_edges: tuple = ()


def extract_peaks(ica: ICACurve, cfg: PeakConfig | None = None) -> dict:
    """Return a flat dict of features for one ICA curve.

    Per named peak: position (V), height (Ah/V), area (Ah), fwhm (V).
    Missing peaks are NaN, which the grading step treats as information.
    """
    cfg = cfg or PeakConfig()
    v, y = ica.voltage, ica.dqdv
    dv = float(np.mean(np.diff(v)))

    idx, _ = find_peaks(y, prominence=cfg.min_prominence_frac * y.max())
    widths, _, left, right = peak_widths(y, idx, rel_height=0.5)

    feats = {
        "cell_id": ica.cell_id,
        "cycle": ica.cycle,
        "source": ica.source,
        "q_window": float(ica.capacity.max() - ica.capacity.min()),
        "total_area": float(np.trapezoid(y, v)),
    }
    for name, (lo, hi) in cfg.bands.items():
        m = (v >= lo) & (v < hi)
        feats[f"{name}_band"] = float(np.trapezoid(y[m], v[m])) if m.sum() > 1 else np.nan
    for name, (lo, hi) in cfg.windows.items():
        in_win = [k for k, i in enumerate(idx) if lo <= v[i] < hi]
        if not in_win:
            for key in ("pos", "height", "area", "fwhm", "asym"):
                feats[f"{name}_{key}"] = np.nan
            continue
        k = max(in_win, key=lambda k: y[idx[k]])  # tallest peak in the window
        i = idx[k]
        # area between the valleys separating this peak from its neighbours,
        # so neighbouring peak areas add up to the total instead of overlapping
        lo_i = idx[k - 1] if k > 0 else 0
        hi_i = idx[k + 1] if k + 1 < len(idx) else len(y) - 1
        lb = lo_i + int(np.argmin(y[lo_i:i + 1]))
        rb = i + int(np.argmin(y[i:hi_i + 1]))
        feats[f"{name}_pos"] = float(v[i])
        feats[f"{name}_height"] = float(y[i])
        feats[f"{name}_area"] = float(np.trapezoid(y[lb:rb + 1], v[lb:rb + 1]))
        feats[f"{name}_fwhm"] = float(widths[k] * dv)
        # share of the peak's area above its centre: a merged 4C peak still
        # shows which side (high- or low-voltage transition) is losing area
        feats[f"{name}_asym"] = float(np.trapezoid(y[i:rb + 1], v[i:rb + 1])
                                      / max(feats[f"{name}_area"], 1e-12))
    return feats


# Per-dataset windows. Severson discharges at 4C, which merges the graphite
# staging peaks into one broad peak near 3.13 V (FWHM ~0.11 V), so it gets a
# single wide window.
PRESETS = {
    "severson": (ICAConfig(v_min=2.85, v_max=3.35),
                 PeakConfig(windows={"p1": (2.90, 3.30)},
                            # 10 mV bins of dQ/dV area: the shape of the merged peak.
                            # grading.Reference regroups them relative to the fresh peak.
                            bands={f"v{int(round(1000 * lo))}": (lo, round(lo + 0.01, 3))
                                   for lo in np.round(np.arange(2.85, 3.35, 0.01), 3)})),
    # slow-rate checkups (C/25, e.g. Wheeler et al. 2025): the three graphite
    # staging peaks are resolved (synthetic model: 3.19, 3.29, 3.34 V)
    "slow": (ICAConfig(v_min=3.05, v_max=3.42, dv=0.001, savgol_window_v=0.015),
             PeakConfig(windows={"p1": (3.32, 3.38), "p2": (3.255, 3.32), "p3": (3.15, 3.255)},
                        bands={f"v{int(round(1000 * lo))}": (lo, round(lo + 0.01, 3))
                               for lo in np.round(np.arange(3.05, 3.42, 0.01), 3)},
                        band_scale="volt",
                        band_edges=tuple(np.round(np.arange(-0.16, 0.081, 0.02), 3)))),
    "default": (ICAConfig(), PeakConfig()),
}


def preset(source: str):
    return PRESETS.get(source, PRESETS["default"])


def relative_to_reference(df: pd.DataFrame, ref_cycles: int = 10) -> pd.DataFrame:
    """Add per-cell deltas against that cell's early-life average.

    Absolute peak heights vary cell to cell; changes from each cell's own
    fresh state are what carry the mechanism signal, and they also make
    datasets with different cells comparable (needed for cross-dataset
    validation).
    """
    out = []
    value_cols = [c for c in df.columns if c.endswith(("_pos", "_height", "_area", "_fwhm", "_asym"))
                  or c in ("q_window", "total_area")]
    for _, g in df.sort_values("cycle").groupby(["source", "cell_id"]):
        ref = g.head(ref_cycles)[value_cols].mean()
        g = g.copy()
        for c in value_cols:
            if c.endswith("_pos"):
                g[f"d_{c}"] = g[c] - ref[c]          # voltage shift (V)
            else:
                g[f"r_{c}"] = g[c] / ref[c]          # ratio to fresh
        out.append(g)
    return pd.concat(out, ignore_index=True)
