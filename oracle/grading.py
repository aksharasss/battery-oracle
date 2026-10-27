"""Phase 3: unsupervised grading.

Relative ICA features -> standardise -> PCA -> GMM (k picked by BIC).
Clusters are then mapped to Grade A/B/C by the median mechanism mix of
their members, not by capacity: lithium inventory loss alone is the benign
mode (A), loss of one electrode's active material is B, several modes at
once (or very severe loss) is C. See grade_of().

Features are taken relative to a population fresh reference (the median of
every training cell's early cycles), not each cell's own fresh state,
because a retired cell arrives with no history. The same reference is used
at prediction time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from oracle.ica import ICACurve
from oracle.mechanisms import MECHANISMS, MechanismModel

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Severity = weighted mechanism loss. LAM_NE is weighted highest because
# losing graphite sites raises the local current density on the rest and
# leads to lithium plating, the safety-relevant failure for second life.
SEVERITY_WEIGHTS = {"LLI": 1.0, "LAM_PE": 1.5, "LAM_NE": 2.0}
FRESH_TOTAL = 0.15   # below this total loss is estimator noise (3 modes x synthetic 4C MAE 0.03-0.037, with ageing nuisance terms): grade A
MODERATE = 0.20      # severity at or above this: no better than B, even for pure LLI
SEVERE = 0.40        # severity at or above this: grade C whatever the mode
LAM_NE_MINOR = 0.2   # graphite share of total loss below which a mixed cell is B

SECOND_LIFE_USE = {
    "A": "Stationary storage (solar / grid buffering)",
    "B": "Light-duty, low-rate use (UPS, backup, small mobility)",
    "C": "Recycle: material recovery only",
}


@dataclass
class Reference:
    values: dict              # fresh median of every raw feature
    ica: ICACurve | None      # median fresh dQ/dV curve, for plotting
    anchor: str = "p1"        # peak the shape bands are placed around
    band_scale: str = "fwhm"  # "fwhm": band edges in fresh-FWHM units; "volt": in volts
    band_edges: tuple = ()    # empty -> U_EDGES

    def relative(self, feats: dict) -> dict:
        rel = {}
        for c, ref in self.values.items():
            if c.endswith("_band"):
                continue
            x = feats.get(c, np.nan)
            if c.endswith(("_pos", "_asym")):   # signed: use differences
                rel[f"d_{c}"] = x - ref
            else:
                rel[f"r_{c}"] = x / ref if ref else np.nan
        a = getattr(self, "anchor", "p1")
        if f"{a}_pos" in self.values and f"{a}_fwhm" in self.values:
            # peak shift in fresh FWHMs, or per 10 mV for slow-rate presets (the
            # synthetic slow-rate peaks are ~2x wider than real ones, so FWHM
            # units would not transfer)
            unit = 0.01 if getattr(self, "band_scale", "fwhm") == "volt" else self.values[f"{a}_fwhm"]
            rel[f"n_{a}_shift"] = rel[f"d_{a}_pos"] / unit
            rel.update(self._shape_bands(feats))
        return rel

    def _shape_bands(self, feats: dict) -> dict:
        """Change in dQ/dV area per band, with bands placed relative to the
        fresh anchor peak: u = (V - fresh peak) / fresh FWHM, or in volts for
        slow-rate presets where the peaks are resolved.

        Different cells (and the synthetic model vs real cells) put the fresh
        peak at different voltages and widths; in u units their ageing
        signatures line up, so a mechanism model calibrated on synthetic
        curves can be applied to real ones.
        """
        a = getattr(self, "anchor", "p1")
        volt = getattr(self, "band_scale", "fwhm") == "volt"
        edges = np.asarray(getattr(self, "band_edges", ()) or U_EDGES, dtype=float)
        pos, fwhm = self.values[f"{a}_pos"], self.values[f"{a}_fwhm"]
        names = [f"u{1000 * e:+.0f}mV_band" if volt else f"u{e:+.1f}_band" for e in edges[:-1]]
        out = dict.fromkeys(names, 0.0)
        for c, ref in self.values.items():
            if not c.endswith("_band"):
                continue
            u = (band_centre(c) - pos) / (1.0 if volt else fwhm)
            k = np.searchsorted(edges, u, side="right") - 1
            if 0 <= k < len(edges) - 1:
                x = feats.get(c, np.nan)
                if np.isfinite(x) and np.isfinite(ref):   # edge bins can fall outside the data
                    out[names[k]] += x - ref
        if volt:   # slow-rate bands: as a fraction of the fresh window capacity
            return {k: v / self.values["q_window"] for k, v in out.items()}
        return out


# Band edges in fresh-FWHM units. +-1.5 FWHM is what both the synthetic
# (FWHM 0.16 V) and real Severson (0.11 V) curves cover inside 2.85-3.35 V.
U_EDGES = np.round(np.arange(-1.5, 1.51, 0.5), 1)


def band_centre(name: str) -> float:
    """'v3120_band' -> 3.125 (10 mV band starting at 3.120 V)."""
    return int(name[1:5]) / 1000 + 0.005


def raw_value_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith(("r_", "d_")) and (
            c.endswith(("_pos", "_height", "_area", "_fwhm", "_asym", "_band")) or c == "q_window")]


def build_reference(df: pd.DataFrame, fresh_cycles: int = 30, ica: ICACurve | None = None,
                    peak_cfg=None) -> Reference:
    fresh = df[df["cycle"] <= fresh_cycles]
    return make_reference(fresh[raw_value_cols(df)].median().to_dict(), ica, peak_cfg)


def make_reference(values: dict, ica: ICACurve | None = None, peak_cfg=None) -> Reference:
    """Reference with the shape bands anchored on the tallest fresh peak
    (at slow rates the highest-voltage peak vanishes with LLI, so it would
    be a poor anchor)."""
    heights = {k[:-7]: v for k, v in values.items() if k.endswith("_height") and np.isfinite(v)}
    anchor = max(heights, key=heights.get) if heights else "p1"
    scale = getattr(peak_cfg, "band_scale", "fwhm")
    edges = tuple(getattr(peak_cfg, "band_edges", ()) or ())
    return Reference(values, ica, anchor, scale, edges)


def severity(mech: dict) -> float:
    return float(sum(SEVERITY_WEIGHTS[m] * mech[m] for m in MECHANISMS))


def grade_of(mech: dict) -> str:
    """Grade from the mechanism mix (the project's mapping, shared with
    oracle.synth.GRADE_HINT): LLI-dominant -> A, one LAM mode dominant -> B,
    no single dominant mode -> C, except LLI + LAM_PE with little LAM_NE -> B. Large loss caps the grade: severity
    >= MODERATE is at best B, >= SEVERE is C."""
    from oracle.synth import GRADE_HINT, dominant_mode

    if sum(mech[m] for m in MECHANISMS) < FRESH_TOTAL:
        return "A"
    sev = severity(mech)
    if sev >= SEVERE:
        return "C"
    mode = dominant_mode(mech["LLI"], mech["LAM_PE"], mech["LAM_NE"])
    grade = GRADE_HINT[mode]
    # Lithium and cathode loss together, with little graphite loss, is not a
    # plating risk: B rather than C. Mixed modes involving graphite stay C.
    total = sum(mech[m] for m in MECHANISMS)
    if mode == "mixed" and mech["LAM_NE"] < LAM_NE_MINOR * total:
        grade = "B"
    return "B" if grade == "A" and sev >= MODERATE else grade


@dataclass
class GradingModel:
    feature_cols: list[str]
    reference: Reference
    mechanisms: MechanismModel
    scaler: StandardScaler
    pca: PCA
    gmm: GaussianMixture
    cluster_grade: dict[int, str]
    cluster_summary: pd.DataFrame
    trained_on: str
    meta: dict = field(default_factory=dict)

    # ── fitting ─────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, df: pd.DataFrame, trained_on: str, mechanisms: MechanismModel | None = None,
            k_range=range(2, 8), n_pca: int = 3, ref_ica: ICACurve | None = None,
            seed: int = 0, peak_cfg=None, fresh_cycles: int = 30) -> "GradingModel":
        ref = build_reference(df, fresh_cycles, ica=ref_ica, peak_cfg=peak_cfg)
        rel = relative_table(df, ref)
        mech = mechanisms or MechanismModel()
        feature_cols = [c for c in rel.columns if rel[c].notna().mean() > 0.9]
        X = rel[feature_cols].fillna(rel[feature_cols].median())

        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        pca = PCA(n_components=min(n_pca, Z.shape[1]), random_state=seed).fit(Z)
        P = pca.transform(Z)
        fits = [GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(P)
                for k in k_range]
        bic = [g.bic(P) for g in fits]
        gmm = fits[int(np.argmin(bic))]
        labels = gmm.predict(P)

        m = pd.DataFrame([mech.predict(r) for r in rel.to_dict("records")])
        m["severity"] = [severity(r) for r in m.to_dict("records")]
        m["cluster"] = labels
        m["r_q_window"] = rel["r_q_window"].to_numpy()
        summary = m.groupby("cluster").agg(n=("severity", "size"), **{
            c: (c, "median") for c in [*MECHANISMS, "severity", "r_q_window"]})
        cluster_grade = {int(c): grade_of(row) for c, row in summary[list(MECHANISMS)].iterrows()}
        summary["grade"] = summary.index.map(cluster_grade)

        return cls(feature_cols, ref, mech, scaler, pca, gmm, cluster_grade,
                   summary.sort_values("severity"), trained_on,
                   {"bic": dict(zip(k_range, map(float, bic))),
                    "pca_explained": pca.explained_variance_ratio_.tolist()})

    # ── inference ───────────────────────────────────────────────────────
    def embed(self, rel: pd.DataFrame) -> np.ndarray:
        X = rel.reindex(columns=self.feature_cols)
        X = X.fillna(pd.Series(self.scaler.mean_, index=self.feature_cols))
        return self.pca.transform(self.scaler.transform(X))

    def grade(self, rel: pd.DataFrame) -> pd.DataFrame:
        """Cluster, grade and grade confidence for each row of relative features."""
        P = self.embed(rel)
        post = self.gmm.predict_proba(P)
        grades = np.array([self.cluster_grade[c] for c in range(post.shape[1])])
        out = pd.DataFrame({"cluster": post.argmax(1)}, index=rel.index)
        for g in "ABC":
            out[f"p_{g}"] = post[:, grades == g].sum(1)
        out["grade"] = out[["p_A", "p_B", "p_C"]].to_numpy().argmax(1)
        out["grade"] = out["grade"].map(dict(enumerate("ABC")))
        out["confidence"] = out[["p_A", "p_B", "p_C"]].max(1)
        return out

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or MODEL_DIR / f"grading_{self.trained_on}.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "GradingModel":
        return joblib.load(path)


def relative_table(df: pd.DataFrame, ref: Reference) -> pd.DataFrame:
    return pd.DataFrame([ref.relative(r) for r in df.to_dict("records")], index=df.index)
