"""Unsupervised check of the grade rule on real cells.

Clusters the twin's (LLI, LAM_PE, LAM_NE) estimates of every Wheeler C/25
checkup with a Gaussian mixture (k picked by BIC), fitted on the seven
tuning cells only, then applied unchanged to the six blind cells. Clusters
are named by their own median mechanisms, never by the rule, and then
compared with the rule's grades and with measured SOH.

  python scripts/cluster_mechanisms.py --tuning data/processed/conf_tuning.csv \
      --blind data/processed/heldout/heldout_wheeler.csv
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
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from oracle.grading import MODEL_DIR
from oracle.mechanisms import MECHANISMS

RANK = {"A": 0, "B": 1, "C": 2}


def load(path):
    d = pd.read_csv(path)
    if "temperature" in d:
        d = d[d["temperature"] == "50C"]
    d = d.dropna(subset=["grade"]).copy()
    d["cell"] = d["cell"].astype(str)
    q0 = d[d.cycle == 0].set_index("cell").q_total
    d["soh"] = d.q_total / d.cell.map(q0)
    return d


def fit_gmm(X, ks, seed=0):
    fits = [GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed).fit(X) for k in ks]
    bic = [g.bic(X) for g in fits]
    return fits[int(np.argmin(bic))], dict(zip(ks, map(float, bic)))


def relabel(gmm, X):
    """Order clusters by total mechanism loss so cluster 0 is the least worn."""
    order = np.argsort(gmm.means_.sum(1))
    return np.argsort(order)[gmm.predict(X)], order


def describe(d, lab):
    d = d.assign(cluster=lab)
    s = d.groupby("cluster").agg(n=("soh", "size"), soh_median=("soh", "median"),
                                 **{m: (m, "median") for m in MECHANISMS})
    s["rule_grades"] = [d[d.cluster == c].grade.value_counts().to_dict() for c in s.index]
    s["majority_grade"] = [d[d.cluster == c].grade.mode().iloc[0] for c in s.index]
    return s


def scores(d, lab, cmap):
    pred = pd.Series(lab, index=d.index).map(cmap)
    dd = d.assign(cluster=lab)
    return {"n": len(d),
            "ari_cluster_vs_rule_grade": float(adjusted_rand_score(d.grade, lab)),
            "cluster_majority_grade_agrees_with_rule": float((pred == d.grade).mean()),
            "spearman_cluster_vs_soh": float(spearmanr(lab, d.soh)[0]),
            "spearman_rule_grade_vs_soh": float(spearmanr(d.grade.map(RANK), d.soh)[0]),
            "cluster_goes_back_with_age": improves_sum(dd),
            "steps": int((dd.groupby("cell").size() - 1).sum())}


def improves_sum(dd):
    return int(sum((np.diff(g.sort_values("cycle")["cluster"].to_numpy()) < 0).sum()
                   for _, g in dd.groupby("cell")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--blind", required=True)
    ap.add_argument("--out", default="data/processed/clusters_wheeler.json")
    a = ap.parse_args()
    tu, bl = load(a.tuning), load(a.blind)
    Xt, Xb = tu[list(MECHANISMS)].to_numpy(), bl[list(MECHANISMS)].to_numpy()

    gmm, bic = fit_gmm(Xt, range(1, 8))
    lt, order = relabel(gmm, Xt)
    lb = relabel(gmm, Xb)[0]
    st = describe(tu, lt)
    cmap = st["majority_grade"].to_dict()
    sb = describe(bl, lb)

    # stability: refit on the blind cells alone and compare partitions of the blind cells
    gmm_b, _ = fit_gmm(Xb, [gmm.n_components])
    ari_refit = float(adjusted_rand_score(lb, gmm_b.predict(Xb)))
    # bootstrap over tuning cells (resample whole cells)
    rng = np.random.default_rng(0)
    cells = tu.cell.unique()
    boot = []
    for _ in range(50):
        pick = rng.choice(cells, len(cells), replace=True)
        Xs = np.vstack([Xt[tu.cell.to_numpy() == c] for c in pick])
        g = GaussianMixture(gmm.n_components, covariance_type="full", n_init=3, random_state=0).fit(Xs)
        boot.append(adjusted_rand_score(lb, g.predict(Xb)))

    res = {"k_by_bic": int(gmm.n_components), "bic": bic,
           "cluster_means_ordered": gmm.means_[order].round(3).tolist(),
           "tuning_clusters": st.round(3).astype(object).reset_index().to_dict("records"),
           "blind_clusters": sb.round(3).astype(object).reset_index().to_dict("records"),
           "cluster_to_grade_from_tuning": cmap,
           "tuning": scores(tu, lt, cmap), "blind_no_refit": scores(bl, lb, cmap),
           "blind_refit_ari_vs_frozen": ari_refit,
           "bootstrap_blind_ari_median": float(np.median(boot)),
           "bootstrap_blind_ari_p10_p90": [float(np.quantile(boot, .1)), float(np.quantile(boot, .9))]}
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    joblib.dump({"gmm": gmm, "order": order, "cluster_grade": cmap,
                 "summary": st.reset_index().to_dict("records")}, MODEL_DIR / "clusters_wheeler.joblib")
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
