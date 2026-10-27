"""Write a few synthetic partial-discharge CSVs for trying the app.

    python app/make_samples.py

Each file is a discharge from 3.45 V to 2.95 V with a known degradation
applied to the fresh peaks, so the app's output can be checked by eye.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FRESH = [(3.34, 0.30, 0.012), (3.22, 0.45, 0.015), (3.08, 0.20, 0.020)]

# name -> per-peak area multipliers (p1, p2, p3)
CASES = {
    "mild_lli": (0.93, 0.93, 0.93),
    "lam_pe": (0.92, 0.90, 0.55),
    "severe": (0.45, 0.62, 0.40),
}


def discharge(scale, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.linspace(3.45, 2.95, n)
    q = sum(a * s / (1 + np.exp((v - v0) / w)) for (v0, a, w), s in zip(FRESH, scale))
    q = q - q[0]
    v_meas = np.round(v + rng.normal(0, 0.0005, n), 3)
    t = np.linspace(0, 3600 * q[-1] / 1.1, n)  # roughly a 1.1 A discharge
    return pd.DataFrame({"time_s": t, "voltage_V": v_meas, "capacity_Ah": q})


if __name__ == "__main__":
    out = Path(__file__).parent / "samples"
    out.mkdir(exist_ok=True)
    for i, (name, scale) in enumerate(CASES.items()):
        discharge(scale, seed=i).to_csv(out / f"{name}.csv", index=False)
        print("wrote", out / f"{name}.csv")
