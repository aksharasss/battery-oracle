"""Slow-path confidence check on the Wheeler 50 degC C/25 checkups, app path.

Grades every checkup of the given cells with predict's slow twin and
reports whether the confidence flags grades that disagree with measured
SOH (A >85%, B 65-85%, C <65%). Mechanisms and grades don't depend on the
confidence rule; only the confidence column changes.

  python scripts/confidence_check.py data/raw/c25 --cells 1b 2b 3b 4b 5a 6a 7a --out data/processed/conf_tuning.csv
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from oracle.loaders import wheeler
from oracle.predict import SLOW_POOR_FIT_RMS, _predict_slow


def one(curve):
    r = _predict_slow(curve)
    raw = r.confidence * (2 if r.features["fit_rms"] > SLOW_POOR_FIT_RMS else 1)
    old = raw * (0.5 if r.features["fit_rms"] > 0.02 else 1)
    return dict(cell=curve.cell_id[-2:], cycle=curve.cycle, q_total=curve.total_capacity, grade=r.grade,
                fit_rms=r.features["fit_rms"], conf_jitter=raw, conf_old=old, conf_new=r.confidence,
                **r.mechanisms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    curves = [c for cell in a.cells for c in wheeler.iter_npz(Path(a.root) / f"Cell{cell}.npz")]
    with Pool(a.workers) as p:
        rows = p.map(one, curves, chunksize=1)
    d = pd.DataFrame(rows)
    q0 = d[d.cycle == 0].set_index("cell").q_total
    d["soh"] = d.q_total / d.cell.map(q0)
    d.to_csv(a.out, index=False)
    exp = np.where(d.soh > 0.85, "A", np.where(d.soh > 0.65, "B", "C"))
    off = exp != d.grade
    from sklearn.metrics import roc_auc_score
    print(f"{len(d)} checkups, {off.sum()} grades off the SOH band")
    for k in ("conf_old", "conf_new", "conf_jitter"):
        print(k, "median", round(d[k].median(), 3), "share<0.5", round((d[k] < 0.5).mean(), 3),
              "AUC(low conf -> off-band)", round(roc_auc_score(off, 1 - d[k]), 3) if off.any() else None)
    f = d[d.cycle == 0]
    print("fresh checkups: median conf old", round(f.conf_old.median(), 3), "new", round(f.conf_new.median(), 3))


if __name__ == "__main__":
    main()
