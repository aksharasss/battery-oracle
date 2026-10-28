# Held-out test protocol (frozen 2026-10-28, before any held-out data was looked at)

**Frozen code and models:** main at commit `0848e1b`. `scripts/heldout_test.py` checks the
sha256 of `models/twin_slow.joblib`, `oracle/grading.py`, `oracle/twin.py`, `oracle/synth.py`
and `oracle/predict.py` against that commit and flags any difference. The grade rule
(`grading.grade_of`, thresholds FRESH_TOTAL 0.15 / MODERATE 0.20 / SEVERE 0.40 / LAM_NE_MINOR 0.2)
and the slow-rate twin reference are not changed after looking at these results.
Changes made later are reported separately as post-freeze.

**Sets, fairest first**
1. **Wheeler 2025 cells never used** (all cells except 1b 2b 3b 4b 5a 6a 7a, which tuned the
   rule and built the fresh reference). Each C/25 checkup (every 300 cycles, 50 degC) is
   graded through the app's own path (`predict` slow twin, population reference, no history
   of the cell). Real cells, same make and protocol as the tuning cells.
2. **Dubarry synthetic graphite//LFP, files 2 and 3** (file 1 was used before). Stratified
   sample per true dominant mode, seed 2026. Labelled, so mechanisms are scored. Known caveat:
   these are C/25 charge curves treated as discharges, as in the earlier Dubarry check.
3. **25 degC initial/final characterisations of the tuning cells.** Files never used, other
   temperature, but cells already seen: only partly held out.

**Reported as they come out, including failures:** Spearman of severity and of grade vs
measured SOH; median SOH per grade and share of A>B>C pairs ordered by SOH; grade
improvements with age; dominant mode; Dubarry R2 / MAE / dominant-mode and grade agreement.
