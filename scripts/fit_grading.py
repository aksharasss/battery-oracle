"""Phase 3: fit the grading model on one dataset's feature table.

Example:
  python scripts/fit_grading.py severson data/processed/features_severson.csv \
      --mat ../2017-06-30_batchdata_updated_struct_errorcorrect.mat
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from oracle.features import preset
from oracle.grading import MODEL_DIR, GradingModel
from oracle.ica import ICACurve, compute_ica
from oracle.loaders import severson


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop glitch cycles: lost peak, or a peak far narrower/smaller than the cell's norm."""
    df = df[np.isfinite(df["p1_pos"])]
    med = df.groupby("cell_id")[["p1_fwhm", "p1_area"]].transform("median")
    ok = (df["p1_fwhm"] > 0.6 * med["p1_fwhm"]) & (df["p1_area"] > 0.4 * med["p1_area"])
    return df[ok]


def median_fresh_ica(curves, dataset: str) -> ICACurve:
    cfg, _ = preset(dataset)
    grid = np.arange(cfg.v_min, cfg.v_max + cfg.dv / 2, cfg.dv)
    ys, qs = [], []
    for c in curves:
        ica = compute_ica(c, cfg)
        ys.append(np.interp(grid, ica.voltage, ica.dqdv, left=np.nan, right=np.nan))
        qs.append(np.interp(grid, ica.voltage, ica.capacity, left=np.nan, right=np.nan))
    keep = ~np.isnan(np.vstack(ys)).any(0)
    return ICACurve(grid[keep], np.nanmedian(ys, 0)[keep], np.nanmedian(qs, 0)[keep],
                    cell_id="fresh-median", cycle=0, source=dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["severson", "sandia"])
    ap.add_argument("features")
    ap.add_argument("--mat", help="Severson .mat, for the fresh reference ICA curve")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    n0 = len(df)
    df = clean(df)
    print(f"{len(df)}/{n0} cycles after glitch filter, {df.cell_id.nunique()} cells")

    ref_ica = None
    if args.mat:
        # cycle_step larger than any cell's life -> only cycle 1 of each cell
        ref_ica = median_fresh_ica(severson.iter_discharges(args.mat, cycle_step=10**6), "severson")

    mech_path = MODEL_DIR / f"mechanisms_{args.dataset}.joblib"
    mech = joblib.load(mech_path) if mech_path.exists() else None
    print("mechanisms:", f"calibrated ({mech_path.name})" if mech else "heuristic")
    gm = GradingModel.fit(df, trained_on=args.dataset, ref_ica=ref_ica, mechanisms=mech)
    path = gm.save()
    print(f"saved {path}")
    print("features:", gm.feature_cols)
    print("PCA explained:", np.round(gm.meta["pca_explained"], 3))
    print("BIC by k:", {k: round(v) for k, v in gm.meta["bic"].items()})
    print(gm.cluster_summary.round(3).to_string())

    rel = pd.DataFrame([gm.reference.relative(r) for r in df.to_dict("records")], index=df.index)
    out = pd.concat([df[["cell_id", "cycle", "q_total"]], gm.grade(rel)], axis=1)
    out.to_csv(Path(args.features).with_name(f"grades_{args.dataset}.csv"), index=False)
    print(out.grade.value_counts().to_dict())


if __name__ == "__main__":
    main()
