"""Map ICA features to degradation mechanisms: LLI, LAM_PE, LAM_NE.

The calibrated path fits a regressor on synthetic curves with known
mechanism labels (oracle.synth), pushed through the same compute_ica /
extract_peaks code as real data. Until that exists, a heuristic based on
the usual LFP/graphite ICA signatures (Dubarry et al. 2012) is used, and
every result says so in its notes.

Heuristic, on features relative to a fresh reference:
- LLI     : peak area shrinks with height and width roughly unchanged
            (the electrodes slip, part of the graphite staging is never used)
- LAM_NE  : height falls faster than area (peak broadens), because graphite
            sites are lost unevenly and the staging transitions smear
- LAM_PE  : the low-voltage side of the peak loses area first (asymmetry
            moves up), because the LFP plateau ends early on discharge
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MECHANISMS = ("LLI", "LAM_PE", "LAM_NE")


# At slow rates the peaks are resolved but the synthetic model's peaks are
# wider and ~13 mV further apart than real ones, so fixed-voltage bands and
# width/asymmetry features don't transfer; per-peak areas and heights do.
# A vanished peak (NaN) means a ratio of 0, which is how fit/predict fill it.
SLOW_FEATURES = r"^(r_p\d_(area|height)|n_p\d_shift)$"


def transferable(cols, pattern: str | None = None) -> list[str]:
    """Relative features that mean the same thing on synthetic and real
    cells: peak-aligned shape bands, shift in FWHM units, and ratios of the
    peak itself. Window capacity is left out because the voltage window
    cuts real and synthetic tails at different points."""
    import re

    keep = re.compile(pattern or r"^(r_p\d_(height|area|fwhm)|d_p\d_asym|n_p\d_shift|u.*_band)$")
    return [c for c in cols if keep.match(c)]


@dataclass
class MechanismModel:
    feature_cols: list[str] = field(default_factory=list)
    model: object | None = None      # sklearn regressor, multi-output
    calibrated: bool = False
    # out-of-distribution check: distance to the synthetic training cloud
    _scaler: object | None = None
    _nn: object | None = None
    ood_threshold: float = np.inf    # 99th percentile of training-point distances

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, groups=None) -> "MechanismModel":
        """Fit on synthetic features X and their known labels y[MECHANISMS]."""
        from sklearn.ensemble import RandomForestRegressor

        self.feature_cols = list(X.columns)
        self.model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0)
        self.model.fit(X.fillna(0).to_numpy(), y[list(MECHANISMS)].to_numpy())
        self.calibrated = True

        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        Z = StandardScaler().fit(X.fillna(0).to_numpy())
        self._scaler = Z
        Zx = Z.transform(X.fillna(0).to_numpy())
        self._nn = NearestNeighbors(n_neighbors=6).fit(Zx)
        if groups is None:
            d = self._nn.kneighbors(Zx)[0][:, 1:].mean(1)   # skip self
        else:
            # leave-cell-out: a curve's neighbours from its own synthetic cell
            # (the same trajectory a few cycles away) would make the spread
            # look far tighter than it is between cells
            g = np.asarray(groups)
            dist, idx = self._nn.kneighbors(Zx, n_neighbors=min(80, len(Zx)))
            d = np.array([dist[i][g[idx[i]] != g[i]][:5].mean() for i in range(len(Zx))])
        self.ood_threshold = float(np.quantile(d, 0.99))
        return self

    def ood_ratio(self, rel: dict) -> float:
        """Mean distance to the 5 nearest synthetic training curves, divided
        by the 99th percentile of the same distance for training curves
        (counting only other cells' curves when fit() got groups).
        Above 1 the mechanism estimate is an extrapolation."""
        if self._nn is None:
            return np.nan
        x = np.nan_to_num(np.array([[rel.get(c, 0.0) for c in self.feature_cols]], dtype=float))
        d = self._nn.kneighbors(self._scaler.transform(x), n_neighbors=5)[0].mean()
        return float(d / self.ood_threshold)

    def predict(self, rel: dict) -> dict[str, float]:
        if self.calibrated:
            x = np.array([[rel.get(c, 0.0) for c in self.feature_cols]], dtype=float)
            out = np.clip(self.model.predict(np.nan_to_num(x))[0], 0, 1)
            return dict(zip(MECHANISMS, map(float, out)))
        return heuristic(rel)

    def predict_trees(self, rel: dict) -> list[dict[str, float]] | None:
        """One estimate per tree of the forest (None if uncalibrated); the
        spread is used as the grade confidence."""
        if not self.calibrated:
            return None
        x = np.nan_to_num(np.array([[rel.get(c, 0.0) for c in self.feature_cols]], dtype=float))
        return [dict(zip(MECHANISMS, map(float, np.clip(t.predict(x)[0], 0, 1))))
                for t in self.model.estimators_]


def heuristic(rel: dict) -> dict[str, float]:
    area = _get(rel, "r_p1_area", 1.0)
    height = _get(rel, "r_p1_height", 1.0)
    d_asym = _get(rel, "d_p1_asym", 0.0)
    loss = max(0.0, 1.0 - area)
    lam_ne = float(np.clip(area - height, 0.0, loss))            # broadening share
    lam_pe = float(np.clip(2.0 * max(d_asym, 0.0) * loss, 0.0, loss - lam_ne))
    lli = max(0.0, loss - lam_ne - lam_pe)
    return {"LLI": lli, "LAM_PE": lam_pe, "LAM_NE": lam_ne}


def _get(d: dict, key: str, default: float) -> float:
    v = d.get(key, default)
    return default if v is None or not np.isfinite(v) else float(v)
