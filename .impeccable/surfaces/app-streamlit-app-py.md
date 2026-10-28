---
version: 1
slug: "app-streamlit-app-py"
primary_target: "app/streamlit_app.py"
related_targets: []
---

# Surface brief: grading screen (app/streamlit_app.py)

Mode: Operate. Build path: code-led.
Audience: examiners at the end-semester demo, on a projector and on the student's laptop. Job: see the grade, why, and the evidence within seconds.
Must keep: grade, mechanism breakdown, dQ/dV vs fresh reference, all caveats, two grading paths, slow-file spinner, placeholder-grader notice.
Chosen by the user: engineering graph paper, dark first with a light/dark switch.

## Direction contract

THESIS: The result is a lab-notebook page, not a dashboard of cards: one ruled sheet where the grade, the mechanism and the curve are written in ink and the change from fresh is annotated by hand. It refuses the grey-card analytics layout.
OWN-WORLD: Dark: green-black ground with a faint millimetre grid, near-white graphite ink, one red pen. Light: pale green-white sheet, graphite ink, same red pen. One-pixel rules, square corners, no shadows, no fills except ink bars. Atkinson Hyperlegible for all UI, Archivo Black-weight for the grade letter only, Caveat for red-pen notes only.
STORY: The examiner sees the letter, reads the one-line use, sees which mechanism caused it, then sees the red-pen note on the curve pointing at the peak that changed. Caveats are stamped on the sheet and stay visible.
FIRST VIEWPORT: Sidebar: input (upload, sample), mode switch, file format. Main: cell name with a method stamp at right; row of grade block (letter ~120px) and mechanism bars; below, the dQ/dV plot on graph paper with the red-pen annotation; stamped notes under the grade.
FORM: Engineering graph paper, candidate 6 of 7 (assigned), seed:2a9f1be77cbd0e5f. Raises: hierarchy by scale contrast alone (from the type specimen); every state is a stamp that stays on the record (from the ticket wallet); three ink values, rules at one pixel (from the reference setting).
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
