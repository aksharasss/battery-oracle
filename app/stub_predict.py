"""Stand-in for oracle.predict until the real grading pipeline lands.

Same interface as the agreed contract:

    predict(file) -> GradeResult

It runs the real ICA and peak extraction on the uploaded file, but the
mechanism breakdown and grade come from simple hand rules on peak-area
ratios against a synthetic fresh cell, not from the fitted clusters. The
app shows a banner whenever this stub is in use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oracle.features import extract_peaks
from oracle.ica import ICACurve, compute_ica
from oracle.loaders.upload import load_csv
from oracle.schema import DischargeCurve

IS_STUB = True

# Fresh-cell peaks (V, Ah, width), the same shape the ICA tests use.
FRESH_PEAKS = [(3.34, 0.30, 0.012), (3.22, 0.45, 0.015), (3.08, 0.20, 0.020)]

SECOND_LIFE = {
    "A": "Stationary storage (solar / wind buffering)",
    "B": "Light-duty backup / UPS",
    "C": "Recycle: material recovery only",
}


@dataclass
class GradeResult:
    grade: str
    confidence: float
    mechanisms: dict[str, float]
    dominant: str
    ica: ICACurve
    reference: ICACurve | None
    features: dict
    second_life_use: str
    notes: list[str] = field(default_factory=list)


def _fresh_reference() -> ICACurve:
    v = np.linspace(3.45, 2.95, 4000)
    q = sum(a / (1 + np.exp((v - v0) / w)) for v0, a, w in FRESH_PEAKS)
    curve = DischargeCurve(v, q - q[0], cell_id="fresh reference", cycle=0, source="synthetic")
    return compute_ica(curve)


def _ratio(feats: dict, ref: dict, name: str) -> float:
    a, r = feats.get(f"{name}_area"), ref.get(f"{name}_area")
    if a is None or r is None or not np.isfinite(a) or not np.isfinite(r) or r <= 0:
        return 0.0  # a missing peak counts as fully lost
    return float(np.clip(a / r, 0.0, 1.0))


def predict(file) -> GradeResult:
    curve = load_csv(file)
    ica = compute_ica(curve)
    ref = _fresh_reference()
    feats = extract_peaks(ica)
    ref_feats = extract_peaks(ref)

    notes = []
    r1, r2, r3 = (_ratio(feats, ref_feats, p) for p in ("p1", "p2", "p3"))
    for name, r in (("p1", r1), ("p2", r2), ("p3", r3)):
        if r == 0.0:
            notes.append(f"Peak {name} was not found in the uploaded window, so it is treated as fully lost.")

    # Loss common to every peak reads as lithium inventory loss; extra loss
    # on the high-voltage peak as LAM on the graphite side, on the low-voltage
    # peak as LAM on the LFP side.
    common = float(np.median([r1, r2, r3]))
    mech = {
        "LLI": 1.0 - common,
        "LAM_PE": max(common - r3, 0.0),
        "LAM_NE": max(common - r1, 0.0),
    }
    mech = {k: float(np.clip(v, 0, 1)) for k, v in mech.items()}
    dominant = max(mech, key=mech.get)

    total = 1.0 - (r1 * 0.30 + r2 * 0.45 + r3 * 0.20) / 0.95
    if total < 0.15 and dominant == "LLI":
        grade = "A"
    elif total < 0.30:
        grade = "B"
    else:
        grade = "C"
    # distance from the nearest grade boundary, squashed to 0.5..1
    edge = min(abs(total - 0.15), abs(total - 0.30))
    confidence = float(0.5 + 0.5 * np.tanh(edge / 0.05))
    if confidence < 0.7:
        notes.append("This cell sits close to a grade boundary; treat the grade as provisional.")

    return GradeResult(
        grade=grade,
        confidence=confidence,
        mechanisms=mech,
        dominant=dominant,
        ica=ica,
        reference=ref,
        features=feats,
        second_life_use=SECOND_LIFE[grade],
        notes=notes,
    )
