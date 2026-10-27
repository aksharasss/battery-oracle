"""Digital-twin degradation analysis of the Wheeler 2025 C/25 checkups.

Per cell: fit the half-cell design to its cycle-0 checkup, then fit
(LLI, LAM_PE, LAM_NE) to every later checkup with the design frozen and the
fresh-state model mismatch removed (oracle.twin). Nothing is trained on
Wheeler data; grades come from oracle.grading.grade_of.

  python scripts/twin_wheeler.py data/raw/wheeler2025
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oracle import twin
from oracle.grading import grade_of, severity
from oracle.loaders import wheeler
from oracle.mechanisms import MECHANISMS

RANK = {"A": 0, "B": 1, "C": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    out = Path(args.out)

    rows = []
    for cell_dir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        curves = sorted(wheeler.iter_discharges(cell_dir), key=lambda c: c.cycle)
        if not curves or curves[0].cycle != 0:
            print(f"{cell_dir.name}: no cycle-0 checkup, skipped", flush=True)
            continue
        fresh = curves[0]
        design, rms0 = twin.fit_design(fresh, c_rate=0.04)
        err = twin.model_error(fresh, design, 0.04)
        for c in curves:
            mech, rms = twin.fit_mechanisms(c, design, 0.04, error=err) if c.cycle else (
                dict.fromkeys(MECHANISMS, 0.0), 0.0)
            rows.append(dict(cell_id=c.cell_id, group=c.meta["group"], cycle=c.cycle,
                             q_total=c.total_capacity, soh=c.total_capacity / fresh.total_capacity,
                             **mech, fit_rms=rms, fresh_fit_rms=rms0,
                             severity=severity(mech), grade=grade_of(mech)))
            print(cell_dir.name, c.cycle, {k: round(v, 3) for k, v in mech.items()}, flush=True)

    g = pd.DataFrame(rows)
    g.to_csv(out / "twin_wheeler.csv", index=False)

    def improves(cell):
        r = cell.sort_values("cycle")["grade"].map(RANK).to_numpy()
        return float((np.diff(r) < 0).mean()) if len(r) > 1 else 0.0

    final = g.sort_values("cycle").groupby("cell_id").tail(1)
    aged = g[g["cycle"] > 0]
    rho, p = spearmanr(g["severity"], g["q_total"])
    res = {
        "method": "per-cell half-cell twin fit to C/25 checkups (oracle.twin), no training on Wheeler",
        "n_checkups": len(g), "n_cells": int(g["cell_id"].nunique()),
        "spearman_severity_vs_capacity": {"rho": float(rho), "p": float(p)},
        "spearman_total_loss_vs_soh": float(spearmanr(aged[list(MECHANISMS)].sum(1), aged["soh"])[0]),
        "grade_counts": g["grade"].value_counts().to_dict(),
        "median_soh_by_grade": g.groupby("grade")["soh"].median().round(3).to_dict(),
        "median_share_steps_grade_improves": float(
            g.groupby("cell_id").apply(improves, include_groups=False).median()),
        "dominant_mode_share_aged": aged[list(MECHANISMS)].idxmax(1).value_counts(normalize=True).round(3).to_dict(),
        "median_fit_rms": float(aged["fit_rms"].median()),
        "final_checkup": final[["cell_id", "group", "cycle", "soh", *MECHANISMS, "severity", "grade",
                                "fit_rms"]].round(3).to_dict("records"),
        "teardown_LAM_NE_gt_LAM_PE_share": float((final["LAM_NE"] > final["LAM_PE"]).mean()),
    }
    (out / "validation_wheeler_twin.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
