"""Phase 3/4: validation of a fitted grading model, with no refitting.

apply_frozen(model, df) grades another table with the model as fitted.
report(model, df) then checks, on that table:
- capacity ordering: Spearman of severity against measured capacity
  (should be clearly negative even though capacity is not a grading input)
- grade order: median capacity should fall from A to B to C
- ageing consistency: per cell, the share of consecutive snapshots where
  the grade gets better (should be near zero)
- stability: adjusted Rand index between the frozen model's clusters and a
  model refit on this table (same features, same k range)
- if the table has known mechanism labels (synthetic), mean absolute error
  of the mechanism estimates and grade agreement with the label severity
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from oracle.grading import GradingModel, build_reference, grade_of, relative_table, severity
from oracle.mechanisms import MECHANISMS

RANK = {"A": 0, "B": 1, "C": 2}


def apply_frozen(gm: GradingModel, df: pd.DataFrame, own_reference: bool = False) -> pd.DataFrame:
    """own_reference: take features relative to this table's own fresh median
    instead of the model's (for synthetic data, whose absolute peak position
    differs from the real cells it was calibrated to)."""
    ref = build_reference(df) if own_reference else gm.reference
    rel = relative_table(df, ref)
    out = gm.grade(rel)
    mech = pd.DataFrame([gm.mechanisms.predict(r) for r in rel.to_dict("records")], index=df.index)
    out = pd.concat([df, out, mech.add_prefix("est_")], axis=1)
    est = [{m: r[f"est_{m}"] for m in MECHANISMS} for _, r in out.iterrows()]
    out["severity"] = [severity(e) for e in est]
    out = out.rename(columns={"grade": "cluster_grade"})
    out["grade"] = [grade_of(e) for e in est]   # what predict() reports
    return out


def report(gm: GradingModel, df: pd.DataFrame, cap_col: str = "q_total",
           own_reference: bool = False) -> dict:
    g = apply_frozen(gm, df, own_reference)
    res = {"n_rows": len(g), "n_cells": int(g["cell_id"].nunique()),
           "grade_counts": g["grade"].value_counts().to_dict()}

    rho, p = spearmanr(g["severity"], g[cap_col], nan_policy="omit")
    res["spearman_severity_vs_capacity"] = {"rho": float(rho), "p": float(p)}
    res["median_capacity_by_grade"] = g.groupby("grade")[cap_col].median().round(4).to_dict()

    # grades flicker at cluster borders, so count backward steps rather than
    # demanding a strictly monotone sequence
    def backward_share(cell):
        r = cell.sort_values("cycle")["grade"].map(RANK).to_numpy()
        return float((np.diff(r) < 0).mean()) if len(r) > 1 else 0.0
    res["median_share_steps_grade_improves"] = float(
        g.groupby("cell_id").apply(backward_share, include_groups=False).median())

    refit = GradingModel.fit(df, trained_on="refit", mechanisms=gm.mechanisms)
    labels_refit = refit.grade(relative_table(df, refit.reference))
    res["ari_clusters_vs_refit"] = float(adjusted_rand_score(g["cluster"], labels_refit["cluster"]))
    res["ari_cluster_grades_vs_refit"] = float(
        adjusted_rand_score(g["cluster_grade"], labels_refit["grade"]))

    if all(m in g.columns for m in MECHANISMS):
        res["mechanism_mae"] = {m: float((g[f"est_{m}"] - g[m]).abs().mean()) for m in MECHANISMS}
        true_grade = [grade_of({m: r[m] for m in MECHANISMS}) for _, r in g.iterrows()]
        res["grade_agreement_with_labels"] = float((g["grade"] == np.array(true_grade)).mean())
        res["cluster_grade_agreement_with_labels"] = float(
            (g["cluster_grade"] == np.array(true_grade)).mean())
        if "grade_hint" in g.columns:   # synth's mode-based hint: LLI A, LAM B, mixed C
            res["grade_agreement_with_hint"] = float((g["grade"] == g["grade_hint"]).mean())
        res["grade_confusion"] = pd.crosstab(pd.Series(true_grade, name="true"),
                                             g["grade"].rename("pred")).to_dict()
    return res


def holdout_report(df: pd.DataFrame, trained_on: str, test_frac: float = 0.33, seed: int = 0,
                   **fit_kw) -> dict:
    """Fit on some cells, report on the rest. A within-dataset stand-in for
    cross-dataset validation when only one dataset is available."""
    cells = df["cell_id"].unique()
    rng = np.random.default_rng(seed)
    test = set(rng.choice(cells, int(round(test_frac * len(cells))), replace=False))
    tr, te = df[~df["cell_id"].isin(test)], df[df["cell_id"].isin(test)]
    gm = GradingModel.fit(tr, trained_on=trained_on, **fit_kw)
    return {"train_cells": int(tr["cell_id"].nunique()), **report(gm, te)}
