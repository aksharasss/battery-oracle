# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: examiners watching a live demo at the end-semester review of an academic mathematics project. The student drives the app, uploads or picks a discharge file, and talks the examiners through the result. The examiners are technically literate but did not build the model, and they need to see within seconds what grade a cell got and why.

Secondary, as the product's stated purpose: an engineer at a battery recycling or second-life facility grading retired cells. This is the framing the demo argues for, not a user present today.

## Product Purpose

Battery Oracle 2.0 grades a retired LFP/graphite cell for second life from one partial discharge. It reads the degradation mechanism from the incremental capacity (dQ/dV) curve instead of predicting remaining life with a black-box model, and maps the mechanism mix to a grade:

- **A**: lithium-inventory loss (LLI) dominant, electrodes intact. Stationary storage.
- **B**: one active-material loss mode (LAM_PE or LAM_NE), or a lithium + cathode mix. Light-duty, low-rate use.
- **C**: mixed modes involving graphite, or severe loss. Recycle for materials.

Success in the demo: the examiner sees the grade, the mechanism behind it and the dQ/dV evidence, and believes the grade is explained rather than asserted.

## Positioning

A mechanistic grade, not a remaining-life prediction. Every grade is explained by an estimated LLI / LAM_PE / LAM_NE breakdown and shown against the measured dQ/dV curve and a fresh-cell reference. It needs only one partial discharge, with no cycling history or capacity label.

## Operating Context

- Shown on a projector or shared screen during the presentation, and also on the student's laptop up close. Both must read well.
- Run locally with `streamlit run app/streamlit_app.py`.
- Input is a CSV with a voltage column plus a capacity column, or current and time columns. Bundled samples cover real Severson 4C cells and Wheeler C/25 checkups.
- Two grading paths are picked automatically from the curve: a slow-rate half-cell twin fit (C/25 files, 15 to 50 s) and a fast 4C feature model (under a second).

## Capabilities and Constraints

- Stack: Streamlit (Python), with Plotly charts. The app calls one function, `oracle.predict.predict(file) -> GradeResult`.
- GradeResult carries grade, confidence, mechanisms (LLI, LAM_PE, LAM_NE), dominant mode, the ICA curve, a fresh reference curve, features (including `path`, `fit_rms`, `ood_ratio`), second-life use and notes.
- Must keep: the grade, the mechanism breakdown, the dQ/dV plot against the fresh reference, the warnings (out of range, outside calibration range, 4C caveats), both grading paths, and a spinner that says a slow file can take up to a minute.
- Falls back to a placeholder grader when no fitted model exists, and must say so.
- Terminology: LLI (loss of lithium inventory), LAM_PE (loss of active material, LFP cathode), LAM_NE (loss of active material, graphite anode), SOH (state of health), dQ/dV / ICA.

## Evidence on Hand

- Real samples in `data/samples/`: Severson cells (4C) and Wheeler cell 6a at cycles 0, 1,500 and 3,300 (C/25).
- Synthetic samples in `app/samples/`.
- Validation results in `data/processed/` and the project report. No customers, deployments or field results exist; none may be implied.

## Product Principles

1. Explain, don't assert: the mechanism and the curve are always beside the grade.
2. Honest uncertainty: confidence, out-of-range and calibration caveats are visible, never buried.
3. Legible at a distance: the grade and its reason must read from the back of a room.
4. One file in, one answer out: no setup or configuration in the way of the demo.

## Accessibility & Inclusion

Projector use requires high contrast and large type for the grade and key numbers. Grade must not be conveyed by colour alone.
