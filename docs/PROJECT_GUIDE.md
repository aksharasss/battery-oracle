# Battery Oracle 2.0: A-to-Z project guide

For teammates who know nothing about batteries or the code. Every claim here comes from the repo (`README.md`, `docs/phase2_synthetic.md`, `PRODUCT.md`) or the project notes. Repo: https://github.com/aksharasss/battery-oracle

## 1. What is the project?
A tool that looks at one **partial discharge** of a used lithium-iron-phosphate (LFP) battery cell and says **why** it aged and which **grade** it deserves for second life:
- **Grade A**: stationary storage (e.g. solar).
- **Grade B**: light duty, low rate.
- **Grade C**: recycle.

Most research predicts "Remaining Useful Life" with a black box. We instead estimate the physical ageing mechanisms, then grade from those.

## 2. Battery basics
- A cell has a **positive electrode (cathode)**, here LFP, and a **negative electrode (anode)**, here graphite. Lithium ions move between them. Discharge = lithium goes from graphite to LFP.
- **Capacity** (Ah) is how much charge the cell holds. **SOH** (state of health) = capacity now / capacity when new.
- **C-rate**: discharge speed. 1C empties the cell in 1 hour; C/25 takes 25 hours (slow); 4C takes 15 minutes (fast).
- LFP has a very flat voltage, so the voltage curve alone hides ageing. That is why we use ICA.

## 3. ICA (Incremental Capacity Analysis), dQ/dV
Plot how much charge (dQ) comes out per small voltage step (dV). Flat voltage regions become **peaks**. Peaks correspond to graphite "staging" steps. When the cell ages, peaks shrink, shift or vanish, and *which* peak changes tells you the mechanism. Code: `oracle/ica.py` (window, uniform grid, Savitzky-Golay smoothing, derivative) and `oracle/features.py` (peak position, height, area, width).

## 4. The three mechanisms
| Name | Meaning | Effect on the C/20 dQ/dV curve (synthetic model) |
|---|---|---|
| **LLI** | Loss of lithium inventory: lithium is lost (e.g. to SEI growth) | highest-voltage peak (~3.34 V) disappears |
| **LAM_PE** | Loss of active material, positive electrode (LFP) | hidden at first (about 10% spare LFP capacity), then the lowest peak (~3.19 V) is cut |
| **LAM_NE** | Loss of active material, negative electrode (graphite) | middle peak (~3.29 V) shrinks, peaks squeeze together |

Resistance growth is separate: at 4C it shifts the whole peak to lower voltage.

## 5. Plan and phases
1. ICA pipeline on public LFP data. 2. Synthetic half-cell validation (mid-sem). 3. Unsupervised grading plus cross-dataset validation. 4. Streamlit grading app and report (end-sem).

## 6. The synthetic model (Phase 2, `oracle/synth.py`)
There is no teardown data for public cells, so we built a half-cell model of an A123-like graphite/LFP cell where we **choose** the LLI / LAM_PE / LAM_NE values. This gives labelled "ground truth" curves. Its fresh peak matches real Severson cells within 1 mV, and its C/25 shape was fitted to Wheeler cell 6a. On 100 synthetic cells the features separate mechanisms well (LDA accuracy 0.94 at 4C, 0.93 at C/20); slow discharge is clearly better.

## 7. Datasets
- **Severson** (MIT/Stanford, A123 LFP, 4C discharge): used to build clusters. At 4C the graphite peaks merge into one broad peak (~3.15 V), so mechanisms are hard to separate.
- **Wheeler et al. 2025** (Sci Data 12, 392; doi 10.57745/OLBXKT): same A123 1.1 Ah cells, aged at 50 degC to 30-60% SOH, C/25 checkups every 300 cycles. Used for cross-dataset validation. Sandia raw files could no longer be downloaded.
- **Dubarry synthetic set** (Mendeley doi 10.17632/bs2j56pn7y): 72 labelled curves used to check the twin. They are really C/25 *charge* curves that we treat as discharges.
Datasets (~2 GB) are not in git; `scripts/fetch_data.py` downloads them.

## 8. The digital twin (`oracle/twin.py`)
A learned model trained on 4C synthetic data did not transfer to C/25. So for slow data we **fit the half-cell model directly to the measured curve**: first the electrode design from a fresh checkup, then LLI / LAM_PE / LAM_NE for each later curve. Nothing is trained on Wheeler data. On synthetic curves it recovers 0.20 / 0.30 / 0.20 to within 0.002.

## 9. Grading rule (`oracle/grading.py`, `grade_of`)
- Total loss under 0.15: **A**.
- Severity 0.40 or more: **C**. Severity 0.20 or more: at best **B**.
- Otherwise by dominant mode: LLI dominant = A, one LAM mode dominant = B, mixed = C.
- Exception (approved 2026-10-27): LLI + LAM_PE mixed with graphite loss under 20% of the total is **B**, because neither drives lithium plating.
- Unsupervised side: PCA (3 components) and a Gaussian mixture (k by BIC, 7 on Severson) show the structure; HDBSCAN finds no clusters because ageing is a continuum.

## 10. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest
streamlit run app/streamlit_app.py
```
In the app, upload a CSV (voltage + capacity, or current + time), or use files in `data/samples/` (Severson 4C cells; Wheeler 6a at cycles 0, 1500, 3300, graded A, B, C). Slow C/25 files take 15 to 50 s; 4C files under a second. From code: `from oracle.predict import predict; predict(file)` returns grade, confidence, mechanisms, dQ/dV curve, notes. Rebuild commands are in the README.

## 11. Results
- Wheeler (7 cells, 103 checkups, no refit): fit rms 0.011; severity vs capacity Spearman -0.84; total loss vs SOH -0.87; 0% of steps improve in grade with age; LLI dominant at end in 6 of 7 cells; grades A 7, B 11, C 85.
- Dubarry: LAM_PE Pearson 0.95 (R2 0.78); LAM_NE not detected; LLI not scorable in that sample.
- Severson 4C forest model: cross-validated R2 0.62 / 0.70 / 0.54; synthetic hold-out grade agreement was 82% in the earlier version (before recalibration). 4C mechanism split is indicative only.
- Held-out test (rule and models frozen at commit 0848e1b, protocol in `docs/heldout_protocol.md`):
  - 6 unseen Wheeler cells, 93 checkups: severity vs SOH Spearman -0.96, grade vs SOH -0.89; median SOH A 0.93, B 0.77, C 0.58; no C above 80% SOH; 1 of 87 steps improved with age.
  - Unseen Dubarry (73 curves): severity Spearman 0.91, but dominant mode right only 33% (0 of 20 for LAM_NE), grade agreement 58%, per-mode R2 at most 0.09. The mechanism split does not hold up there.
  - Limits: B spans a wide SOH range; same cell make as tuning; 25 degC check not run.

## 12. Honest limitations
1. Grade rule was tuned on Wheeler after seeing it; held-out test is the fair check.
2. Dubarry curves are charge curves fitted as discharge.
3. LAM_NE is a lower bound (seen only when graphite becomes limiting).
4. LLI and LAM_PE trade off mid-life; their sum is well determined, the split is not (not fixed even with charge curves on synthetic data).
5. Mechanism split is weak on unseen Dubarry data (see section 11).
6. One chemistry, small sample (7 tuning cells).
7. Grades are not safety-validated; no teardown ground truth.

## 13. Novelty
All parts are standard (ICA, half-cell twin, ML on synthetic curves, ICA clustering). Closest prior work: Beatty et al. 2026 (Batteries 12:55), which clusters IC features including Severson but has no LLI/LAM split and no cross-dataset test. Our claim, "to our knowledge": grading by fitted mechanism, plus a no-refit test on another dataset, plus an upload tool.

## 14. Likely reviewer questions
- **Why not predict RUL?** It says how long, not why; second-life use depends on the mechanism.
- **Where is ground truth?** None public; we use a synthetic half-cell model and the Dubarry labelled set. Wheeler has teardown info only partly.
- **Is the grade rule overfit?** Partly tuned on Wheeler; we froze it and tested on unseen cells (held-out test).
- **Why Wheeler?** Same A123 cells as Severson, slow checkups, teardown; Sandia was unavailable.
- **Why does 4C fail?** Peaks merge; slow partial discharge (C/20 to C/5) is recommended.
- **Other chemistries?** Not covered.
- **Is it safe to deploy?** No; grades are not safety-validated.
- **What if the upload is unlike the training data?** The app flags out-of-range curves and lowers confidence.
