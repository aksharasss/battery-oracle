"""Stable entry point for the UI: one discharge file in, a grade out.

    from oracle.predict import predict
    result = predict("data/samples/severson_c0_cycle300.csv")

Two paths, picked automatically from the curve itself (model="auto"):
- slow (C/25-like, graphite staging peaks resolved): the half-cell twin is
  fitted directly to the upload (oracle.twin) against a population fresh
  reference built from Wheeler et al. 2025 (models/twin_slow.joblib, from
  scripts/fit_twin_reference.py). This is the validated path.
- fast (Severson-like 4C, one merged peak): features -> mechanism forest
  calibrated on synthetic 4C curves (models/grading_severson.joblib, from
  scripts/fit_grading.py). Less reliable; notes say why.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from oracle.features import extract_peaks, preset
from oracle.grading import (FRESH_TOTAL, MODEL_DIR, SECOND_LIFE_USE, GradingModel, grade_of,
                            severity)
from oracle.ica import ICACurve, compute_ica
from oracle.loaders.upload import load_csv

DEFAULT_MODEL = "severson"
# Allowed main-peak shift vs the fresh reference (V). Ageing pushes the
# peak down (Severson reaches -78 mV at end of life); it never moves up much.
PEAK_SHIFT_RANGE = (-0.10, 0.03)


@dataclass
class GradeResult:
    grade: str                       # "A" | "B" | "C"
    confidence: float                # 0..1: share of forest trees (4C) or jittered fits (slow) giving that grade
    mechanisms: dict[str, float]     # LLI, LAM_PE, LAM_NE -> fractional loss vs fresh
    dominant: str                    # largest mechanism, or "none" if total loss < FRESH_TOTAL
    ica: ICACurve                    # ICA curve of the upload
    reference: ICACurve | None       # fresh-cell overlay
    features: dict                   # extract_peaks() output + relative features
    second_life_use: str
    notes: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_twin() -> dict:
    path = MODEL_DIR / "twin_slow.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no twin reference at {path}; run scripts/fit_twin_reference.py first")
    return joblib.load(path)


def is_slow_rate(curve) -> bool:
    """Slow discharges put a sharp (FWHM < 25 mV) staging peak above 3.25 V;
    at 4C the peaks merge into one ~110 mV wide peak near 3.1 V."""
    try:
        ica = compute_ica(curve, preset("slow")[0])
    except ValueError:
        return False
    from scipy.signal import peak_widths

    i = int(np.argmax(ica.dqdv))
    width = peak_widths(ica.dqdv, [i], rel_height=0.5)[0][0] * float(np.mean(np.diff(ica.voltage)))
    return bool(ica.voltage[i] > 3.25 and width < 0.025)


def _predict_slow(curve) -> GradeResult:
    from oracle import twin

    t = load_twin()
    try:
        mech, rms = twin.fit_mechanisms(curve, t["design"], t["c_rate"], error=t["error"])
    except ValueError as exc:
        raise ValueError(f"could not fit the slow-rate model to this discharge ({exc})") from exc
    grade = grade_of(mech)
    # confidence: share of grades unchanged when each mode is jittered by the
    # twin's error on synthetic checks (~0.02)
    rng = np.random.default_rng(0)
    jit = [dict(zip(mech, np.clip(np.array(list(mech.values())) + rng.normal(0, 0.02, 3), 0, 1)))
           for _ in range(300)]
    confidence = float(np.mean([grade_of(j) == grade for j in jit]))
    ica = compute_ica(curve, preset("slow")[0])
    feats = extract_peaks(ica, preset("slow")[1])
    notes = ["Slow-rate path: half-cell model fitted directly to this discharge "
             "(validated on Wheeler et al. 2025 C/25 checkups, Spearman -0.84 vs capacity)."]
    if rms > 0.02:
        confidence *= 0.5
        notes.append(f"Poor model fit (rms {rms:.3f}); the mechanism split is uncertain.")
    notes.append("Graphite loss (LAM_NE) is a lower bound: it only shows once graphite "
                 "limits the capacity.")
    return GradeResult(
        grade=grade, confidence=confidence, mechanisms=mech,
        dominant=max(mech, key=mech.get) if sum(mech.values()) >= FRESH_TOTAL else "none",
        ica=ica, reference=t["reference_ica"],
        features={**feats, "fit_rms": rms, "severity": severity(mech), "path": "slow"},
        second_life_use=SECOND_LIFE_USE[grade], notes=notes)


@lru_cache(maxsize=4)
def load_model(name: str = DEFAULT_MODEL) -> GradingModel:
    path = MODEL_DIR / f"grading_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no fitted model at {path}; run scripts/fit_grading.py first")
    return GradingModel.load(path)


def predict(file: str | Path | io.IOBase, model: str = "auto") -> GradeResult:
    """Grade one discharge (CSV path or file-like).

    model: "auto" (slow twin path if the staging peaks are resolved, else the
    4C model), "slow", or the name of a fitted grading model ("severson").
    Raises ValueError with a readable message if the file can't be used.
    """
    try:
        curve = load_csv(file)
    except pd.errors.ParserError as exc:
        raise ValueError(f"could not read the file as a CSV ({exc})") from exc
    except ValueError:
        raise
    except Exception as exc:  # wrong encoding, binary file, ...
        raise ValueError(f"could not read the file as a CSV ({exc})") from exc

    if model == "slow" or (model == "auto" and is_slow_rate(curve)):
        return _predict_slow(curve)
    gm = load_model(DEFAULT_MODEL if model == "auto" else model)
    ica_cfg, peak_cfg = preset(gm.trained_on)
    try:
        ica = compute_ica(curve, ica_cfg)
    except ValueError as exc:
        raise ValueError(
            f"the discharge doesn't cover the {ica_cfg.v_min:.2f}-{ica_cfg.v_max:.2f} V "
            f"window this model needs ({exc})") from exc
    feats = extract_peaks(ica, peak_cfg)
    if not np.isfinite(feats.get("p1_pos", np.nan)):
        raise ValueError("no dQ/dV peak found in the voltage window; is this an LFP discharge?")

    rel = gm.reference.relative(feats)
    mech = gm.mechanisms.predict(rel)
    g = gm.grade(pd.DataFrame([rel])).iloc[0]

    # The grade comes from this cell's own mechanism mix. The GMM cluster is
    # reported alongside: on held-out synthetic cells the per-cell grade
    # matched the true grade 60% of the time, the Severson-fitted cluster
    # grade 28% (scripts/validate.py).
    notes = []
    grade = grade_of(mech)
    trees = gm.mechanisms.predict_trees(rel)
    if trees is not None:
        confidence = float(np.mean([grade_of(t) == grade for t in trees]))
    else:
        confidence = float(g[f"p_{grade}"])
    shift = rel.get("d_p1_pos", 0.0)
    if not PEAK_SHIFT_RANGE[0] <= shift <= PEAK_SHIFT_RANGE[1]:
        # far outside anything seen in training: likely a different C-rate,
        # temperature or chemistry, so the grade is extrapolation
        confidence *= 0.3
        notes.append(f"Out of range: the main peak is at {feats['p1_pos']:.3f} V, "
                     f"{1000 * shift:+.0f} mV from the {gm.trained_on} fresh reference "
                     f"({gm.reference.values['p1_pos']:.3f} V). The grade is unreliable for this file.")
    ood = gm.mechanisms.ood_ratio(rel)
    if np.isfinite(ood) and ood > 1:
        # every real Severson cell below ~90% capacity lands here: the
        # synthetic model doesn't yet reproduce how real aged 4C peaks shrink
        confidence *= 0.5
        notes.append(f"Outside the synthetic calibration range (distance {ood:.1f}x the "
                     "training spread): the mechanism split is extrapolated.")
    if not gm.mechanisms.calibrated:
        notes.append("Mechanism split is a provisional ICA heuristic; it will be "
                     "calibrated on the synthetic half-cell model.")
    else:
        notes.append("Mechanism split calibrated on synthetic 4C curves (held-out R2 "
                     "0.62 / 0.70 / 0.54 for LLI / LAM_PE / LAM_NE); at 4C the split is "
                     "indicative only, a slow (C/25) discharge grades far more reliably.")
    if grade != g["grade"]:
        notes.append(f"Its cluster of similar Severson cells (cluster {int(g['cluster'])}) "
                     f"is grade {g['grade']}.")
    if gm.trained_on == "severson":
        notes.append("Model fitted on Severson 4C discharges; other C-rates shift the peak "
                     "and are less reliable.")

    return GradeResult(
        grade=grade,
        confidence=confidence,
        mechanisms=mech,
        dominant=max(mech, key=mech.get) if sum(mech.values()) >= FRESH_TOTAL else "none",
        ica=ica,
        reference=gm.reference.ica,
        features={**feats, **rel, "cluster": int(g["cluster"]), "cluster_grade": g["grade"],
                  "severity": severity(mech), "ood_ratio": ood, "path": "fast"},
        second_life_use=SECOND_LIFE_USE[grade],
        notes=notes,
    )
