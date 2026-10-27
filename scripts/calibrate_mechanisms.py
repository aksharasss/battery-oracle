"""Fit the mechanism model on synthetic curves with known LLI / LAM_PE / LAM_NE.

Synthetic cells (oracle.synth) are pushed through the same ICA and feature
code as real data, with features taken relative to the synthetic fresh
median, the same way real data is taken relative to its own fresh median.

  python scripts/calibrate_mechanisms.py --features data/processed/features_synth_4C.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from oracle.features import preset
from oracle.grading import MODEL_DIR, make_reference, raw_value_cols, relative_table
from oracle.mechanisms import MECHANISMS, SLOW_FEATURES, MechanismModel, transferable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="from scripts/synth_features.py")
    ap.add_argument("--name", default="severson", help="saved as models/mechanisms_<name>.joblib")
    ap.add_argument("--preset", default="severson", help="feature preset the table was built with")
    args = ap.parse_args()

    d = pd.read_csv(args.features)
    ref = make_reference(d[d["cycle"] == 0][raw_value_cols(d)].median().to_dict(), None,
                         preset(args.preset)[1])
    rel = relative_table(d, ref)
    pattern = SLOW_FEATURES if args.preset == "slow" else None
    X = rel[[c for c in transferable(rel.columns, pattern) if rel[c].notna().mean() > 0.5]]
    y = d[list(MECHANISMS)]

    pred = np.zeros(y.shape)
    for tr, te in GroupKFold(5).split(X, y, d["cell_id"]):
        m = MechanismModel().fit(X.iloc[tr], y.iloc[tr])
        pred[te] = [list(m.predict(r).values()) for r in X.iloc[te].to_dict("records")]
    cv = {mn: {"r2": round(float(r2_score(y[mn], pred[:, i])), 3),
               "mae": round(float(mean_absolute_error(y[mn], pred[:, i])), 4)}
          for i, mn in enumerate(MECHANISMS)}

    model = MechanismModel().fit(X, y, groups=d["cell_id"])
    path = MODEL_DIR / f"mechanisms_{args.name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    summary = {"n_curves": len(d), "n_cells": int(d["cell_id"].nunique()),
               "features": list(X.columns), "grouped_cv_by_cell": cv}
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(cv, indent=2))
    print("->", path)


if __name__ == "__main__":
    main()
