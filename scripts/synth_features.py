"""Synthetic labelled set -> feature table with true mechanism labels.

  python scripts/synth_features.py --c-rate 3.636 --r-growth 0.02 --out data/processed/features_synth_4C.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from oracle import synth
from oracle.features import extract_peaks, preset
from oracle.ica import compute_ica


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="severson", help="ICA/peak preset to extract with")
    ap.add_argument("--c-rate", type=float, default=4.0 / 1.1)
    ap.add_argument("--r-growth", type=float, default=0.02)
    ap.add_argument("--cells", type=int, default=25, help="cells per profile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500, help="points per synthetic discharge")
    ap.add_argument("--range-scale", type=float, default=1.6,
                    help="stretch synth.PROFILES end-of-test amounts; real Severson cells "
                         "age past the default ranges (LLI <= 0.25) and a forest can't extrapolate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # widen the profiles locally (oracle/synth.py itself is unchanged)
    synth.PROFILES = {p: {m: (lo * args.range_scale, hi * args.range_scale) for m, (lo, hi) in r.items()}
                      for p, r in synth.PROFILES.items()}

    curves, labels = synth.generate_dataset(n_cells_per_profile=args.cells, c_rate=args.c_rate,
                                            r_growth=args.r_growth, seed=args.seed, n=args.n)
    ica_cfg, peak_cfg = preset(args.preset)
    rows = []
    for c in curves:
        try:
            f = extract_peaks(compute_ica(c, ica_cfg), peak_cfg)
        except ValueError:
            continue
        f["q_total"] = c.total_capacity
        rows.append(f)
    df = pd.DataFrame(rows).merge(labels, on=["cell_id", "cycle"])
    df = df.rename(columns={m.lower(): m for m in ("LLI", "LAM_PE", "LAM_NE")})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)}/{len(curves)} curves -> {args.out}")


if __name__ == "__main__":
    main()
