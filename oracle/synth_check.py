"""Phase 2 check: do ICA features separate LLI, LAM_PE and LAM_NE?

Runs the synthetic half-cell model (oracle.synth) through the same
compute_ica / extract_peaks code used on real data and reports:

1. single-mode sweeps: how each feature moves as one mechanism grows,
2. a labelled ageing dataset (4 profiles: lli, lam_pe, lam_ne, mixed) and
   - supervised upper bound: cross-validated LDA accuracy on the profile,
   - unsupervised: GMM clusters vs the profile (adjusted Rand index),
   - mode amounts: cross-validated R^2 of a ridge regression per mode,
   each with features relative to the cell's own fresh curve and with
   absolute features from a single test (no fresh reference, which is
   the real second-life situation),
at Severson's 4C and at C/20, where the graphite staging peaks are sharp.

Usage:  python -m oracle.synth_check --out data/processed/synth
Writes figures, the datasets (npz + labels csv) and summary.json there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oracle import synth
from oracle.features import PeakConfig, extract_peaks
from oracle.ica import ICAConfig, compute_ica

# ICA window and peak windows per rate. 4C pushes everything ~0.2 V lower.
RATES = {
    "4C": dict(
        c_rate=4.0 / synth.NOMINAL_AH,
        ica=ICAConfig(v_min=2.85, v_max=3.40),
        peaks=PeakConfig(windows={"p1": (3.13, 3.30), "p2": (3.04, 3.13), "p3": (2.85, 3.04)}),
        bands=[2.85, 2.95, 3.03, 3.08, 3.12, 3.16, 3.22, 3.40],
    ),
    "C/20": dict(
        c_rate=0.05,
        ica=ICAConfig(v_min=3.00, v_max=3.45, savgol_window_v=0.02),
        peaks=PeakConfig(),
        bands=[3.00, 3.12, 3.20, 3.26, 3.31, 3.45],
    ),
}
COLORS = {"lli": "tab:blue", "lam_pe": "tab:green", "lam_ne": "tab:orange",
          "r_ohm": "tab:red", "mixed": "tab:purple"}


def features(curve, rate: dict) -> dict:
    """Peak features (extract_peaks) plus capacity in fixed voltage bands.

    Band capacities are peak areas that need no peak finding, so they still
    work when 4C merges the peaks.
    """
    ica = compute_ica(curve, rate["ica"])
    f = extract_peaks(ica, rate["peaks"])
    b = rate["bands"]
    for lo, hi in zip(b[:-1], b[1:]):
        m = (ica.voltage >= lo) & (ica.voltage < hi)
        f[f"band_{lo:.2f}_{hi:.2f}"] = float(np.trapezoid(ica.dqdv[m], ica.voltage[m]))
    f["q_discharge"] = float(curve.capacity.max())   # whole discharge, if available
    return f


def value_cols(df):
    return [c for c in df.columns
            if c.endswith(("_pos", "_height", "_area", "_fwhm")) or c.startswith("band_")
            or c in ("q_window", "total_area", "q_discharge")]


def relative(df: pd.DataFrame) -> pd.DataFrame:
    """Each cell's features minus its cycle-0 (fresh) features."""
    cols = value_cols(df)
    ref = df[df.cycle == 0].set_index("cell_id")[cols]
    out = df.copy()
    out[cols] = df[cols].to_numpy() - ref.loc[df.cell_id].to_numpy()
    return out


# ─────────────────────────────────────────────────────────────────────────
# 1. single-mode sweeps
# ─────────────────────────────────────────────────────────────────────────

def sweeps(out: Path, levels=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)) -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey="row")
    table = {}
    for r, (rname, rate) in enumerate(RATES.items()):
        for k, mode in enumerate(("lli", "lam_pe", "lam_ne", "r_ohm")):
            ax = axes[r, k]
            rows = []
            for i, lvl in enumerate(levels):
                amt = lvl * 0.1 if mode == "r_ohm" else lvl     # 0..30 mOhm
                c = synth.generate(**{mode: amt}, c_rate=rate["c_rate"], seed=i)
                ica = compute_ica(c, rate["ica"])
                ax.plot(ica.voltage, ica.dqdv, color=plt.cm.viridis(i / (len(levels) - 1)),
                        label=f"{amt * 1000:.0f} mOhm" if mode == "r_ohm" else f"{lvl:.0%}")
                rows.append(dict(level=amt, **features(c, rate)))
            df = pd.DataFrame(rows)
            base = df.iloc[0]
            table[f"{rname}:{mode}"] = {
                c: float(df[c].iloc[-1] - base[c]) for c in value_cols(df)
                if np.isfinite(df[c].iloc[-1]) and np.isfinite(base[c])}
            ax.set_title(f"{mode.upper()} at {rname}")
            ax.set_xlabel("V")
            if k == 0:
                ax.set_ylabel("-dQ/dV (Ah/V)")
            ax.legend(fontsize=7)
    fig.suptitle("Synthetic LFP/graphite: dQ/dV as one mechanism grows (others zero)")
    fig.tight_layout()
    fig.savefig(out / "sweeps_dqdv.png", dpi=110)
    plt.close(fig)
    return table


# ─────────────────────────────────────────────────────────────────────────
# 2. labelled dataset
# ─────────────────────────────────────────────────────────────────────────

def score(df: pd.DataFrame, cols: list[str], label: str = "profile") -> dict:
    X = df[cols].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    X = X[:, np.nanstd(X, axis=0) > 0]
    y = df[label].to_numpy()
    groups = df.cell_id.to_numpy()
    cv = GroupKFold(n_splits=5)
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    acc = cross_val_score(lda, X, y, cv=cv, groups=groups).mean()
    Z = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(n_components=df[label].nunique(), n_init=5, random_state=0).fit(Z)
    ari = adjusted_rand_score(y, gmm.predict(Z))
    r2 = {}
    for m in synth.MODES:
        t = df[m].to_numpy()
        p = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
                              X, t, cv=cv, groups=groups)
        r2[m] = float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
    return dict(n=len(df), lda_cv_accuracy=float(acc), gmm_ari=float(ari), ridge_cv_r2=r2)


def dataset(out: Path, n_cells: int, cycle_step: int, seed: int) -> dict:
    res = {}
    for rname, rate in RATES.items():
        tag = rname.replace("/", "_")
        curves, labels = synth.generate_dataset(n_cells_per_profile=n_cells, cycle_step=cycle_step,
                                                c_rate=rate["c_rate"], seed=seed)
        synth.save_dataset(curves, labels, out / f"synth_{tag}")
        feats = pd.DataFrame([features(c, rate) for c in curves])
        df = labels.merge(feats.drop(columns=["source"]), on=["cell_id", "cycle"])
        df.to_csv(out / f"features_{tag}.csv", index=False)
        cols = value_cols(df)
        rel = relative(df)
        late = lambda d: d[d.cycle >= d.cycle.max() * 0.5]     # clearly aged half
        res[rname] = {
            "relative_to_fresh": score(late(rel), cols),
            "single_test_no_reference": score(late(df), cols),
            "relative_to_fresh_window_only": score(late(rel), [c for c in cols if c != "q_discharge"]),
        }
        _scatter(late(rel), rname, out / f"separation_{tag}.png")
    return res


def _scatter(df: pd.DataFrame, rname: str, path: Path):
    cols = [c for c in value_cols(df) if c != "q_discharge"]
    X = df[cols].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    X = X[:, np.nanstd(X, axis=0) > 0]
    Z = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2).fit(Z, df.profile)
    P = lda.transform(Z)
    fig, ax = plt.subplots(figsize=(6, 5))
    for prof in synth.PROFILES:
        m = (df.profile == prof).to_numpy()
        ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.6, color=COLORS[prof], label=prof)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title(f"{rname}: ICA window features vs fresh, aged half of life")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/synth")
    ap.add_argument("--cells", type=int, default=25, help="cells per profile")
    ap.add_argument("--cycle-step", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"sweep_change_at_30pct": sweeps(out),
               "dataset": dataset(out, a.cells, a.cycle_step, a.seed)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["dataset"], indent=2))


if __name__ == "__main__":
    main()
