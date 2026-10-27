"""Cross-dataset validation on Wheeler et al. 2025 (C/25 checkups), no refit.

Nothing is fitted on Wheeler data. The mechanism model comes from synthetic
C/25 curves only (scripts/calibrate_mechanisms.py --preset slow --name slow),
and the grade rule is oracle.grading.grade_of. Features are taken relative to
the population fresh reference: the median of every cell's cycle-0 checkup.

Checks:
- severity vs measured C/25 capacity (Spearman, all checkups)
- per cell, share of consecutive checkups where the grade gets better
- final-checkup severity by ageing group (6 and 7 aged fastest in the paper)
- teardown consistency: post-mortems of cells from this batch found graphite
  (negative electrode) loss and an intact LFP electrode, so at end of test
  LAM_NE should exceed LAM_PE
- share of checkups outside the synthetic calibration range (ood_ratio > 1)

  python scripts/validate_wheeler.py data/raw/wheeler2025
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oracle.features import extract_peaks, preset
from oracle.grading import MODEL_DIR, build_reference, grade_of, relative_table, severity
from oracle.ica import compute_ica
from oracle.loaders import wheeler
from oracle.mechanisms import MECHANISMS

RANK = {"A": 0, "B": 1, "C": 2}


def feature_table(root: Path) -> pd.DataFrame:
    ica_cfg, peak_cfg = preset("slow")
    rows = []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Cell")):
        for c in wheeler.iter_discharges(cell_dir):
            f = extract_peaks(compute_ica(c, ica_cfg), peak_cfg)
            f.update(q_total=c.total_capacity, group=c.meta["group"])
            rows.append(f)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder holding Cell<id>/ subfolders")
    ap.add_argument("--model", default="slow", help="models/mechanisms_<model>.joblib")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    df = feature_table(Path(args.root))
    out = Path(args.out)
    df.to_csv(out / "features_wheeler.csv", index=False)
    mech = joblib.load(MODEL_DIR / f"mechanisms_{args.model}.joblib")
    ref = build_reference(df, fresh_cycles=0, peak_cfg=preset("slow")[1])
    rel = relative_table(df, ref)

    est = pd.DataFrame([mech.predict(r) for r in rel.to_dict("records")], index=df.index)
    g = pd.concat([df[["cell_id", "cycle", "group", "q_total"]], est], axis=1)
    g["severity"] = [severity(r) for r in est.to_dict("records")]
    g["grade"] = [grade_of(r) for r in est.to_dict("records")]
    g["ood_ratio"] = [mech.ood_ratio(r) for r in rel.to_dict("records")]
    g["soh"] = g["q_total"] / g.groupby("cell_id")["q_total"].transform("first")
    g.sort_values(["cell_id", "cycle"]).to_csv(out / "grades_wheeler.csv", index=False)

    def improves(cell):
        r = cell.sort_values("cycle")["grade"].map(RANK).to_numpy()
        return float((np.diff(r) < 0).mean()) if len(r) > 1 else 0.0

    final = g.sort_values("cycle").groupby("cell_id").tail(1)
    rho, p = spearmanr(g["severity"], g["q_total"])
    res = {
        "n_checkups": len(g), "n_cells": int(g["cell_id"].nunique()),
        "cycles": [int(g["cycle"].min()), int(g["cycle"].max())],
        "grade_counts": g["grade"].value_counts().to_dict(),
        "spearman_severity_vs_capacity": {"rho": float(rho), "p": float(p)},
        "median_soh_by_grade": g.groupby("grade")["soh"].median().round(3).to_dict(),
        "median_share_steps_grade_improves": float(
            g.groupby("cell_id").apply(improves, include_groups=False).median()),
        "share_ood": float((g["ood_ratio"] > 1).mean()),
        "final_checkup": final[["cell_id", "group", "cycle", "soh", *MECHANISMS, "severity",
                                "grade", "ood_ratio"]].round(3).to_dict("records"),
        "teardown_consistent_share": float((final["LAM_NE"] > final["LAM_PE"]).mean()),
    }
    (out / "validation_wheeler.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
