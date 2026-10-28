"""Dubarry synthetic graphite//LFP curves are C/25 charge curves. Compare the
twin fitted to them as discharges (v_shift = 0, the original treatment) with
the twin given a charge-mode voltage offset (twin.fit_design(fit_shift=True)).
Same curves, same labels, files 2 and 3.

Input is a compact .npz made on the machine that holds the .mat files:
Qnorm, pathinfo (all paths, file 1), and for i in 1..3 n<i>, idx<i> (column
indices) and V<i> (voltage matrix columns at idx<i>).

  python scripts/dubarry_charge_check.py data/raw/dubarry_subset.npz --per-mode 8
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from oracle import twin
from oracle.grading import grade_of, severity
from oracle.mechanisms import MECHANISMS
from oracle.schema import DischargeCurve
from oracle.synth import dominant_mode

AH = 1.1
_S = {}


def curve(q_pct, v, k):
    ok = np.isfinite(v)
    q, v = q_pct[ok] / 100 * AH, v[ok]
    return DischargeCurve(voltage=v[::-1], capacity=q.max() - q[::-1], cell_id=f"dubarry_{k}",
                          cycle=int(k), source="dubarry")


def init(designs):
    _S.update(designs)


def fit(item):
    k, q, v, truth, qloss, file = item
    c = curve(q, v, k)
    row = {"file": file, "path": int(k), **{f"true_{m}": truth[m] for m in MECHANISMS},
           "true_qloss": qloss, "grade_true": grade_of(truth)}
    for tag in ("discharge", "charge"):
        d = _S[tag]
        try:
            mech, rms = twin.fit_mechanisms(c, d["design"], 0.04, error=d["error"], v_shift=d["shift"])
        except ValueError:
            return None
        row.update({f"{tag}_{m}": x for m, x in mech.items()})
        row.update({f"{tag}_rms": rms, f"{tag}_severity": severity(mech), f"{tag}_grade": grade_of(mech)})
    print(file, k, flush=True)
    return row


def metrics(d, tag):
    T = d[[f"true_{m}" for m in MECHANISMS]].to_numpy()
    E = d[[f"{tag}_{m}" for m in MECHANISMS]].to_numpy()
    out = {"n": len(d),
           "r2": {m: float(r2_score(T[:, j], E[:, j])) for j, m in enumerate(MECHANISMS)},
           "pearson": {m: float(pearsonr(T[:, j], E[:, j])[0]) for j, m in enumerate(MECHANISMS)},
           "mae": {m: float(mean_absolute_error(T[:, j], E[:, j])) for j, m in enumerate(MECHANISMS)},
           "dominant_mode_agreement": float((T.argmax(1) == E.argmax(1)).mean()),
           "grade_agreement": float((d["grade_true"] == d[f"{tag}_grade"]).mean()),
           "spearman_severity_vs_true_qloss": float(spearmanr(d[f"{tag}_severity"], d["true_qloss"])[0]),
           "median_fit_rms": float(d[f"{tag}_rms"].median())}
    d = d.assign(tm=[MECHANISMS[j] for j in T.argmax(1)])
    out["dominant_agreement_by_true_mode"] = {
        m: float((s[[f"{tag}_{x}" for x in MECHANISMS]].to_numpy().argmax(1)
                  == s[[f"true_{x}" for x in MECHANISMS]].to_numpy().argmax(1)).mean())
        for m, s in d.groupby("tm")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--per-mode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/processed/dubarry_charge")
    a = ap.parse_args()
    z = np.load(a.npz)
    q, P = z["Qnorm"], z["pathinfo"]
    rng = np.random.default_rng(a.seed)

    fresh = curve(q, z["V1"][:, 0].astype(float), 0)
    des0, rms0 = twin.fit_design(fresh, 0.04)
    des1, rms1, shift = twin.fit_design(fresh, 0.04, fit_shift=True)
    print(f"fresh fit rms: discharge treatment {rms0:.4f}, charge mode {rms1:.4f} (v_shift {1000 * shift:+.1f} mV)")
    designs = {"discharge": dict(design=des0, error=twin.model_error(fresh, des0, 0.04), shift=0.0),
               "charge": dict(design=des1, error=twin.model_error(fresh, des1, 0.04, v_shift=shift), shift=shift)}

    items, off = [], 0
    for i in (2, 3):
        off += int(z[f"n{i - 1}"])
        n, idx, V = int(z[f"n{i}"]), z[f"idx{i}"], z[f"V{i}"]
        lab = P[off + idx]
        qmax = np.array([np.nanmax(np.where(np.isfinite(V[:, j]), q, np.nan)) for j in range(len(idx))])
        r = pearsonr(100 - qmax, lab[:, 3])[0]
        print(f"file {i}: offset {off}, label/curve alignment r={r:.3f}")
        if r < 0.95:
            continue
        modes = np.array([dominant_mode(*l[:3]) for l in lab])
        ok = (lab[:, 3] > 1) & (lab[:, 3] < 50) & (lab[:, :3].max(1) <= 60)
        for mode in ("lli", "lam_pe", "lam_ne", "mixed"):
            pool = np.where(ok & (modes == mode))[0]
            for j in rng.choice(pool, min(a.per_mode, len(pool)), replace=False):
                truth = dict(zip(MECHANISMS, lab[j, :3] / 100))
                items.append((off + idx[j], q, V[:, j].astype(float), truth, lab[j, 3] / 100, f"file{i}"))
    print(len(items), "curves")
    with Pool(a.workers, initializer=init, initargs=(designs,)) as p:
        rows = [r for r in p.map(fit, items, chunksize=1) if r]
    d = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    d.to_csv(Path(a.out) / "dubarry_charge_rows.csv", index=False)
    res = {"fresh_rms": {"discharge": rms0, "charge": rms1}, "v_shift_V": shift,
           "discharge": metrics(d, "discharge"), "charge": metrics(d, "charge")}
    (Path(a.out) / "dubarry_charge.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
