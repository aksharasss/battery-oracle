"""Phase 1 entry point: raw dataset -> per-cycle ICA feature table.

Examples:
  python scripts/build_features.py severson data/raw/2017-05-12_batchdata_updated_struct_errorcorrect.mat
  python scripts/build_features.py sandia data/raw/SNL_18650_LFP_25C_0-100_0.5-1C_a_timeseries.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from oracle.features import extract_peaks, preset, relative_to_reference
from oracle.ica import compute_ica
from oracle.loaders import batteryarchive, severson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["severson", "sandia"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--cycle-step", type=int, default=10, help="Severson: use every Nth cycle")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ica_cfg, peak_cfg = preset(args.dataset)
    rows, skipped = [], 0
    for p in args.paths:
        curves = (severson.iter_discharges(p, cycle_step=args.cycle_step)
                  if args.dataset == "severson" else batteryarchive.iter_discharges(p))
        for c in curves:
            try:
                f = extract_peaks(compute_ica(c, ica_cfg), peak_cfg)
            except ValueError:
                skipped += 1
                continue
            f["c_rate"] = c.c_rate
            f["q_total"] = c.total_capacity
            f.update({k: v for k, v in c.meta.items() if k in ("policy", "cycle_life")})
            rows.append(f)

    df = relative_to_reference(pd.DataFrame(rows))
    out = Path(args.out or f"data/processed/features_{args.dataset}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out) if out.suffix == ".parquet" else df.to_csv(out, index=False)
    print(f"{len(df)} cycles from {df.cell_id.nunique()} cells -> {out} ({skipped} skipped)")


if __name__ == "__main__":
    main()
