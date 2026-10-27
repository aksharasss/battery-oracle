"""Independent check of the mechanism estimates on Dubarry's synthetic
graphite//LFP set (Mendeley doi:10.17632/bs2j56pn7y; A123 LFP/graphite
half-cell data, C/25, labelled %LLI / %LAM_PE / %LAM_NE). The curves come
from Dubarry's own 'alawa model, not ours, so this tests whether our twin
(oracle.twin) recovers mechanisms from a model it was not built on.

  python scripts/validate_dubarry.py --n 150
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import mean_absolute_error, r2_score

from oracle import twin
from oracle.grading import grade_of
from oracle.mechanisms import MECHANISMS
from oracle.schema import DischargeCurve

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "dubarry_lfp_synth"
NOMINAL_AH = 1.1


def to_curve(q_pct, v, k) -> DischargeCurve:
    """Dubarry stores V against capacity in % of the fresh cell, V rising
    with Q. Flip it into a discharge from the top of charge."""
    ok = np.isfinite(v)
    q, v = q_pct[ok] / 100 * NOMINAL_AH, v[ok]
    return DischargeCurve(voltage=v[::-1], capacity=q.max() - q[::-1], cell_id=f"dubarry_{k}",
                          cycle=k, source="dubarry")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-loss", type=float, default=50.0, help="max %% capacity loss sampled")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = sio.loadmat(RAW / "Gr-LFP_5000paths_1.mat",
                    variable_names=["Qnorm", "pathinfo", "volt_part1"])
    q, P, V = m["Qnorm"].ravel(), m["pathinfo"][: m["volt_part1"].shape[1]], m["volt_part1"]
    fresh = to_curve(q, V[:, 0].astype(float), 0)
    design, rms0 = twin.fit_design(fresh, c_rate=0.04)
    err = twin.model_error(fresh, design, 0.04)

    rng = np.random.default_rng(args.seed)
    pool = np.where((P[:, 3] > 1) & (P[:, 3] < args.max_loss) & (P[:, :3].max(1) <= 60))[0]
    rows = []
    for k in rng.choice(pool, args.n, replace=False):
        try:
            mech, rms = twin.fit_mechanisms(to_curve(q, V[:, k].astype(float), k), design, 0.04,
                                            error=err)
        except ValueError:
            continue
        truth = dict(zip(MECHANISMS, P[k, :3] / 100))
        rows.append({**{f"true_{m}": v for m, v in truth.items()}, **mech,
                     "qloss": P[k, 3] / 100, "fit_rms": rms,
                     "grade_true": grade_of(truth), "grade_est": grade_of(mech)})
        print(k, {m: round(truth[m], 3) for m in MECHANISMS}, {m: round(v, 3) for m, v in mech.items()},
              flush=True)

    d = pd.DataFrame(rows)
    d.to_csv("data/processed/dubarry_twin.csv", index=False)
    res = {"n": len(d), "fresh_fit_rms": rms0,
           "r2": {m: float(r2_score(d[f"true_{m}"], d[m])) for m in MECHANISMS},
           "mae": {m: float(mean_absolute_error(d[f"true_{m}"], d[m])) for m in MECHANISMS},
           "grade_agreement": float((d["grade_true"] == d["grade_est"]).mean()),
           "dominant_mode_agreement": float(
               (d[[f"true_{m}" for m in MECHANISMS]].to_numpy().argmax(1)
                == d[list(MECHANISMS)].to_numpy().argmax(1)).mean())}
    Path("data/processed/validation_dubarry.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
