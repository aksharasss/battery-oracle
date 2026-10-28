"""Cut the Wheeler 2025 raw CSVs (4.7 GB) down to the C/25 steps only, one
compact .npz per cell (about 10 MB), read by oracle.loaders.wheeler.iter_npz.
Keeps the longest |I| < 0.08 A discharge ('__dis') and charge ('__chg') step
of every characterisation file, every k-th row so each step has ~20k points.
Comma decimals (some 25 degC files) are converted.

  python scripts/extract_wheeler_c25.py data/raw/wheeler2025 data/raw/c25 [Cell1a ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["Step_Index", "Test_Time", "Current", "Voltage", "Discharge_Capacity", "Charge_Capacity"]


def read(f):
    d = pd.read_csv(f, sep=";", usecols=COLS)
    for c in COLS:
        if d[c].dtype == object:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return d


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cells = sys.argv[3:] or sorted(p.name for p in root.iterdir() if p.is_dir())
    for cell in cells:
        arrs = {}
        for f in sorted((root / cell).rglob("*charac*.csv")):
            d = read(f)
            st = d.groupby("Step_Index")["Current"].agg(["median", "size"])
            for kind, sel in (("dis", (st["median"] < -0.01) & (st["median"] > -0.08)),
                              ("chg", (st["median"] > 0.01) & (st["median"] < 0.08))):
                s = st[sel]
                if s.empty:
                    continue
                seg = d[d["Step_Index"] == s["size"].idxmax()]
                if len(seg) < 500:
                    continue
                seg = seg.iloc[::max(1, len(seg) // 20000)]
                arrs[f"{f.stem}__{kind}"] = seg[COLS[1:]].to_numpy(np.float64)
            print(cell, f.name, flush=True)
        np.savez_compressed(out / f"{cell}.npz", **arrs)


if __name__ == "__main__":
    main()
