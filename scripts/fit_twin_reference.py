"""Fit the slow-rate twin reference that predict() uses for C/25 uploads.

The fresh reference is the mean Q(V) of every Wheeler cell's cycle-0 C/25
checkup (a retired cell arrives with no history, so the reference has to be
a population one). The half-cell design is fitted to it, and the remaining
model error is stored so it can be subtracted from uploads.

  python scripts/fit_twin_reference.py data/raw/wheeler2025
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from oracle import twin
from oracle.features import preset
from oracle.grading import MODEL_DIR
from oracle.ica import compute_ica
from oracle.loaders import wheeler
from oracle.schema import DischargeCurve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()

    fresh = []
    for cell_dir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        c0 = [c for c in wheeler.iter_discharges(cell_dir) if c.cycle == 0]
        fresh += c0
    grid_q = np.linspace(0, min(c.total_capacity for c in fresh), 4000)
    v = np.mean([np.interp(grid_q, c.capacity, c.voltage) for c in fresh], axis=0)
    ref = DischargeCurve(v, grid_q, cell_id="wheeler_fresh_mean", cycle=0, source="wheeler")

    design, rms = twin.fit_design(ref, c_rate=0.04)
    error = twin.model_error(ref, design, 0.04)
    ica = compute_ica(ref, preset("slow")[0])
    out = MODEL_DIR / "twin_slow.joblib"
    joblib.dump({"design": design, "error": error, "reference_ica": ica, "c_rate": 0.04,
                 "fit_rms": rms, "n_fresh": len(fresh)}, out)
    print(f"{len(fresh)} fresh checkups, design fit rms {rms:.4f} -> {out}")
    print(design)


if __name__ == "__main__":
    main()
