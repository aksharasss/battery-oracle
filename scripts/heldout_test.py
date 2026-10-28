"""Held-out test of the frozen grader (protocol: docs/heldout_protocol.md).

Nothing here is fitted. Every curve goes through the same code the app
uses, against the models committed at the freeze, and the results are
written as they come out. Three sets, from fairest to least fair:

1. Wheeler 2025 cells never used anywhere in the project (every Cell<id>
   folder except the seven used for tuning). App path: predict()'s slow
   twin against the frozen population reference, no history of the cell.
2. Dubarry synthetic graphite//LFP curves from files 2 and 3 (file 1 was
   used before). Labelled LLI / LAM_PE / LAM_NE, so mechanisms can be
   scored. Same method as scripts/validate_dubarry.py.
3. The 25 degC initial/final characterisations of the seven tuning cells
   (files never used, other temperature, but cells already seen).

  python scripts/heldout_test.py --wheeler-root data/raw/wheeler2025 \
      --dubarry-root data/raw/dubarry_lfp_synth --workers 6
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from oracle.grading import grade_of, severity
from oracle.mechanisms import MECHANISMS
from oracle.schema import DischargeCurve

TUNING_CELLS = {"1b", "2b", "3b", "4b", "5a", "6a", "7a"}
FROZEN = {   # sha256 at the freeze (docs/heldout_protocol.md)
    "models/twin_slow.joblib": "ac5dc761876b38f9052ed42de7a27f6faa3ff4fe0fd8885cf04b7e2a2dc16857",
    "oracle/grading.py": "3ddaea1e385f9777b37d0a0c5f6d5c4f1ec99b825bdb7536be840f21a220d855",
    "oracle/twin.py": "cf2b7d2d7b1c7e84d8aeb9381914b0c5e6f2e0da9d657f0af58383eee8ae57a5",
    "oracle/synth.py": "f911165beee5059a2fddf042bf4b08465035fbca10e2688769b21e93e743e0f8",
    "oracle/predict.py": "4a1e1fda2c35dbd48cb996cd432784223ae165ad8345781bb0ed2b55ff7a6197",
}
RANK = {"A": 0, "B": 1, "C": 2}
DUBARRY_AH = 1.1


def frozen_check() -> dict:
    out = {}
    for rel, want in FROZEN.items():
        p = ROOT / rel
        # git may check text files out with CRLF on Windows; hash with LF
        b = p.read_bytes()
        if rel.endswith(".py"):
            b = b.replace(b"\r\n", b"\n")
        out[rel] = hashlib.sha256(b).hexdigest() == want
    try:
        out["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip()
    except OSError:
        out["git_head"] = None
    return out


# ── 1 and 3: Wheeler, app path ─────────────────────────────────────────────
def _grade_curve(curve: DischargeCurve) -> dict:
    from oracle.predict import _predict_slow, is_slow_rate

    t0 = time.time()
    row = dict(cell=curve.cell_id.replace("wheeler_", ""), cycle=curve.cycle,
               temperature=curve.meta.get("temperature"), q_total=curve.total_capacity,
               routed_slow=is_slow_rate(curve))
    try:
        r = _predict_slow(curve)
        row.update(r.mechanisms, grade=r.grade, confidence=r.confidence,
                   severity=r.features["severity"], fit_rms=r.features["fit_rms"])
    except ValueError as exc:
        row.update(grade=None, error=str(exc))
    row["seconds"] = time.time() - t0
    print(row["cell"], row["cycle"], row["temperature"], row.get("grade"),
          {m: round(row.get(m, np.nan), 3) for m in MECHANISMS}, flush=True)
    return row


def wheeler_curves(root: Path, cells: set[str] | None, temperature: str) -> list[DischargeCurve]:
    from oracle.loaders import wheeler

    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Cell")):
        cid = d.name.replace("Cell", "")
        if cells is not None and cid not in cells:
            continue
        if cells is None and cid in TUNING_CELLS:
            continue
        out += [c for c in wheeler.iter_discharges(d, include_25c=True)
                if c.meta["temperature"] == temperature]
    return out


def wheeler_metrics(g: pd.DataFrame) -> dict:
    g = g.dropna(subset=["grade"]).copy()
    if g.empty:
        return {"n": 0}
    fresh_q = g[g["cycle"] == 0].set_index("cell")["q_total"]
    g["soh"] = g["q_total"] / g["cell"].map(fresh_q)

    def improves(c):
        r = c.sort_values("cycle")["grade"].map(RANK).to_numpy()
        return int((np.diff(r) < 0).sum())

    def pair_order(hi, lo):
        a, b = g.loc[g["grade"] == hi, "soh"].dropna(), g.loc[g["grade"] == lo, "soh"].dropna()
        if a.empty or b.empty:
            return None
        return float((a.to_numpy()[:, None] > b.to_numpy()[None, :]).mean())

    aged = g[g["cycle"] > 0]
    bins = pd.cut(g["soh"], [0, 0.8, 0.9, 1.01], labels=["SOH<80%", "80-90%", ">90%"])
    rho, p = spearmanr(g["severity"], g["soh"], nan_policy="omit")
    rho_g, p_g = spearmanr(g["grade"].map(RANK), g["soh"], nan_policy="omit")
    return {
        "n_checkups": len(g), "cells": sorted(g["cell"].unique()),
        "routed_to_slow_path": float(g["routed_slow"].mean()),
        "spearman_severity_vs_soh": {"rho": float(rho), "p": float(p)},
        "spearman_grade_vs_soh": {"rho": float(rho_g), "p": float(p_g)},
        "grade_counts": g["grade"].value_counts().to_dict(),
        "median_soh_by_grade": g.groupby("grade")["soh"].median().round(3).to_dict(),
        "share_pairs_ordered": {"A_above_B": pair_order("A", "B"), "B_above_C": pair_order("B", "C"),
                                "A_above_C": pair_order("A", "C")},
        "grade_by_soh_band": pd.crosstab(bins, g["grade"]).to_dict(),
        "grade_improvements_with_age": int(g.groupby("cell").apply(improves, include_groups=False).sum()),
        "grade_steps": int((g.groupby("cell").size() - 1).sum()),
        "dominant_mode_aged": aged[list(MECHANISMS)].idxmax(1).value_counts().to_dict(),
        "median_fit_rms": float(g["fit_rms"].median()),
        "final_checkup": g.sort_values("cycle").groupby("cell").tail(1)[
            ["cell", "cycle", "soh", *MECHANISMS, "grade"]].round(3).to_dict("records"),
    }


# ── 2: Dubarry, labelled ───────────────────────────────────────────────────
def _dubarry_curve(q_pct, v, k) -> DischargeCurve:
    ok = np.isfinite(v)
    q, v = q_pct[ok] / 100 * DUBARRY_AH, v[ok]
    return DischargeCurve(voltage=v[::-1], capacity=q.max() - q[::-1], cell_id=f"dubarry_{k}",
                          cycle=int(k), source="dubarry")


_DUB = {}


def _dub_init(design, err):
    _DUB["design"], _DUB["err"] = design, err


def _dub_fit(item):
    from oracle import twin

    tag, k, q, v, truth, qloss = item
    try:
        mech, rms = twin.fit_mechanisms(_dubarry_curve(q, v, k), _DUB["design"], 0.04, error=_DUB["err"])
    except ValueError as exc:
        print(tag, k, "failed", exc, flush=True)
        return None
    print(tag, k, {m: round(truth[m], 3) for m in MECHANISMS}, {m: round(x, 3) for m, x in mech.items()},
          flush=True)
    return {"file": tag, "path": int(k), **{f"true_{m}": x for m, x in truth.items()}, **mech,
            "true_qloss": qloss, "fit_rms": rms, "severity": severity(mech),
            "grade_true": grade_of(truth), "grade_est": grade_of(mech)}


def dubarry_items(root: Path, n_per_mode: int, seed: int, report: dict):
    import scipy.io as sio
    from oracle.synth import dominant_mode

    rng = np.random.default_rng(seed)
    items = []
    prev_n = [next(sh[1] for nm, sh, _ in sio.whosmat(root / "Gr-LFP_5000paths_1.mat")
                   if nm.startswith("volt"))]
    for i in (2, 3):
        path = root / f"Gr-LFP_5000paths_{i}.mat"
        names = [n for n, *_ in sio.whosmat(path)]
        vname = next(n for n in names if n.startswith("volt"))
        m = sio.loadmat(path, variable_names=["Qnorm", "pathinfo", vname])
        q, P_all, V = m["Qnorm"].ravel(), m["pathinfo"], m[vname]
        n = V.shape[1]
        report[f"file{i}"] = {"variables": names, "pathinfo": list(P_all.shape), "volt": list(V.shape)}
        # Label/curve alignment check: the capacity loss in pathinfo must match
        # the curve's own end capacity (fresh = 100%). pathinfo may hold this
        # file's paths only, or every path with this file's at an offset.
        qmax = np.array([np.nanmax(np.where(np.isfinite(V[:, k]), q, np.nan)) for k in range(n)])
        offsets = sorted({0, *np.cumsum([0, *prev_n]).tolist()})
        fits = {o: pearsonr(100 - qmax, P_all[o:o + n, 3])[0] for o in offsets if o + n <= len(P_all)}
        off = max(fits, key=fits.get)
        r, P = fits[off], P_all[off:off + n]
        prev_n.append(n)
        report[f"file{i}"].update(label_alignment_r=float(r), pathinfo_offset=int(off))
        if r < 0.95:
            print(f"file {i}: labels don't line up with curves (r={r:.2f}); skipped", flush=True)
            continue
        modes = np.array([dominant_mode(*P[k, :3]) for k in range(len(P))])
        ok = (P[:, 3] > 1) & (P[:, 3] < 50) & (P[:, :3].max(1) <= 60)
        for mode in ("lli", "lam_pe", "lam_ne", "mixed"):
            pool = np.where(ok & (modes == mode))[0]
            report[f"file{i}"][f"pool_{mode}"] = int(len(pool))
            for k in rng.choice(pool, min(n_per_mode, len(pool)), replace=False):
                truth = dict(zip(MECHANISMS, P[k, :3] / 100))
                items.append((f"file{i}", k, q, V[:, k].astype(float), truth, P[k, 3] / 100))
        del V
    return items


def dubarry_fresh(root: Path):
    import scipy.io as sio
    from oracle import twin

    m = sio.loadmat(root / "Gr-LFP_5000paths_1.mat", variable_names=["Qnorm", "volt_part1"])
    fresh = _dubarry_curve(m["Qnorm"].ravel(), m["volt_part1"][:, 0].astype(float), 0)
    design, rms0 = twin.fit_design(fresh, c_rate=0.04)
    return design, twin.model_error(fresh, design, 0.04), rms0


def dubarry_metrics(d: pd.DataFrame) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    T = d[[f"true_{m}" for m in MECHANISMS]].to_numpy()
    E = d[list(MECHANISMS)].to_numpy()
    out = {"n": len(d),
           "r2": {m: float(r2_score(T[:, j], E[:, j])) for j, m in enumerate(MECHANISMS)},
           "pearson": {m: float(pearsonr(T[:, j], E[:, j])[0]) for j, m in enumerate(MECHANISMS)},
           "mae": {m: float(mean_absolute_error(T[:, j], E[:, j])) for j, m in enumerate(MECHANISMS)},
           "dominant_mode_agreement": float((T.argmax(1) == E.argmax(1)).mean()),
           "grade_agreement": float((d["grade_true"] == d["grade_est"]).mean()),
           "grade_confusion_true_rows": pd.crosstab(d["grade_true"], d["grade_est"]).to_dict("index"),
           "spearman_severity_vs_true_qloss": float(spearmanr(d["severity"], d["true_qloss"])[0]),
           "spearman_total_vs_true_total": float(spearmanr(E.sum(1), T.sum(1))[0]),
           "median_fit_rms": float(d["fit_rms"].median())}
    d = d.assign(true_mode=[MECHANISMS[j] for j in T.argmax(1)])
    out["by_true_dominant_mode"] = {
        k: {"n": len(s), "dominant_mode_agreement": float((s[list(MECHANISMS)].to_numpy().argmax(1)
                                                            == s[[f"true_{m}" for m in MECHANISMS]].to_numpy().argmax(1)).mean()),
            "grade_agreement": float((s["grade_true"] == s["grade_est"]).mean())}
        for k, s in d.groupby("true_mode")}
    return out


# ── demo files ─────────────────────────────────────────────────────────────
def write_demo(root: Path, rows: pd.DataFrame, out: Path, per_cell=3, max_cells=4):
    from oracle.loaders import wheeler

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for cell in sorted(rows["cell"].unique())[:max_cells]:
        cyc = sorted(rows.loc[(rows["cell"] == cell) & (rows["temperature"] == "50C"), "cycle"])
        pick = sorted({cyc[0], cyc[len(cyc) // 2], cyc[-1]})[:per_cell]
        for c in wheeler.iter_discharges(root / f"Cell{cell}"):
            if c.cycle in pick and c.meta["temperature"] == "50C":
                name = f"HELDOUT_wheeler_{cell}_cycle{c.cycle}_C25.csv"
                pd.DataFrame({"voltage_V": np.round(c.voltage, 5),
                              "capacity_Ah": np.round(c.capacity, 6)}).to_csv(out / name, index=False)
                r = rows[(rows["cell"] == cell) & (rows["cycle"] == c.cycle) & (rows["temperature"] == "50C")].iloc[0]
                written.append({"file": name, "grade_at_freeze": r["grade"],
                                **{m: round(float(r[m]), 3) for m in MECHANISMS}})
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheeler-root", type=Path, default=ROOT / "data/raw/wheeler2025")
    ap.add_argument("--dubarry-root", type=Path, default=ROOT / "data/raw/dubarry_lfp_synth")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/heldout")
    ap.add_argument("--demo-out", type=Path, default=ROOT / "data/samples/heldout")
    ap.add_argument("--sets", nargs="*", default=["wheeler", "dubarry", "wheeler25"])
    ap.add_argument("--dubarry-per-mode", type=int, default=15, help="per true mode per file")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    res = {"frozen": frozen_check(), "protocol": "docs/heldout_protocol.md"}
    print(json.dumps(res["frozen"]), flush=True)
    if not all(v for k, v in res["frozen"].items() if k != "git_head"):
        print("WARNING: frozen files differ from the freeze; results are NOT the frozen test", flush=True)

    if "wheeler" in args.sets:
        curves = wheeler_curves(args.wheeler_root, None, "50C")
        print(f"held-out Wheeler: {len(curves)} checkups", flush=True)
        with Pool(args.workers) as pool:
            rows = pd.DataFrame(pool.map(_grade_curve, curves, chunksize=1))
        rows.to_csv(args.out / "heldout_wheeler.csv", index=False)
        res["wheeler_heldout_cells"] = wheeler_metrics(rows)
        if len(rows):
            res["demo_files"] = write_demo(args.wheeler_root, rows, args.demo_out)
        (args.out / "heldout_results.json").write_text(json.dumps(res, indent=2, default=str))

    if "dubarry" in args.sets:
        rep = {}
        items = dubarry_items(args.dubarry_root, args.dubarry_per_mode, args.seed, rep)
        design, err, rms0 = dubarry_fresh(args.dubarry_root)
        print(f"Dubarry: {len(items)} labelled curves, fresh fit rms {rms0:.4f}", flush=True)
        with Pool(args.workers, initializer=_dub_init, initargs=(design, err)) as pool:
            d = pd.DataFrame([r for r in pool.map(_dub_fit, items, chunksize=1) if r])
        d.to_csv(args.out / "heldout_dubarry.csv", index=False)
        res["dubarry_files_2_3"] = {"files": rep, "fresh_fit_rms": rms0,
                                    **(dubarry_metrics(d) if len(d) else {"n": 0})}
        (args.out / "heldout_results.json").write_text(json.dumps(res, indent=2, default=str))

    if "wheeler25" in args.sets:
        curves = wheeler_curves(args.wheeler_root, TUNING_CELLS, "25C")
        print(f"25 degC files of tuning cells: {len(curves)}", flush=True)
        with Pool(args.workers) as pool:
            rows = pd.DataFrame(pool.map(_grade_curve, curves, chunksize=1))
        rows.to_csv(args.out / "heldout_wheeler_25C.csv", index=False)
        res["wheeler_25C_tuning_cells"] = wheeler_metrics(rows)
        (args.out / "heldout_results.json").write_text(json.dumps(res, indent=2, default=str))

    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
