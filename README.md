# Battery Oracle 2.0

A mechanistic digital twin for grading retired LFP cells for second life.
Incremental Capacity Analysis (ICA) on a partial discharge gives a physical
degradation fingerprint. Unsupervised clustering of those fingerprints then
assigns each cell to Grade A (stationary storage), B (light duty) or C
(recycle).

## Roadmap

| Phase | What | Due |
|---|---|---|
| 1 | ICA pipeline on public LFP data. Loaders, dQ/dV, peak features | Mid-sem |
| 2 | Synthetic LFP/graphite half-cell model with known LLI / LAM_PE / LAM_NE, used to check which features separate which mechanism (replaces teardown ground truth) | Mid-sem |
| 3 | Unsupervised grading (PCA/UMAP + GMM/HDBSCAN) fitted on **Severson**, then **cross-dataset validation** on Sandia (fallback CALCE) with no refitting | End-sem |
| 4 | **Streamlit UI**: upload a partial discharge CSV and get Grade A/B/C plus a mechanism breakdown and ICA plot. Final report | End-sem |

The UI uses Streamlit because a file uploader and interactive Plotly charts
take a few lines each, and it runs in a browser with no packaging work.
Gradio would also work. Tkinter would cost more effort for a worse-looking
result.

## Layout

```
oracle/
  schema.py              DischargeCurve: common format every loader produces
  loaders/
    severson.py          MIT/Stanford .mat (HDF5) batches  -> training set
    batteryarchive.py    Sandia CSVs from batteryarchive.org -> validation set
    calce.py             CALCE A123 (fallback validation, TODO)
    upload.py            single CSV from the UI
  ica.py                 window -> Q(V) on uniform grid -> Savitzky-Golay dQ/dV
  features.py            named peaks: position, height, area, FWHM; deltas vs fresh
  synth.py               Phase 2 (stub)
  grading.py             Phase 3 (stub)
  validate.py            Phase 3 cross-dataset validation (stub)
app/streamlit_app.py     Phase 4 UI (stub)
scripts/build_features.py  raw data -> data/processed/features_<dataset>.parquet
tests/test_ica.py        ICA recovers known peaks from synthetic curves
data/raw, data/processed (git-ignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Phase 1: next steps

1. **Get the data.**
   - Severson: download the three batch `.mat` files from data.matr.io (the
     "Data-driven prediction of battery cycle life" project) into `data/raw/`.
     Batch 1 alone (~3 GB) is enough to start.
   - Sandia: download the LFP time-series CSVs from batteryarchive.org
     (search "SNL LFP"). Start with one cell per temperature and C-rate.
2. **Check the loaders against real files.** They follow the published
   formats but haven't been run on a real file yet. Load one cell, plot V
   against Q for a few cycles, and fix field or column names if needed.
3. **Tune the ICA window and peak windows.** Severson discharges at 4C,
   which smears peaks and shifts them to lower voltage compared with Sandia
   at 0.5 to 3C. Plot dQ/dV for cycle 10 and a late cycle of a few cells,
   then set `ICAConfig.v_min/v_max`, the Savitzky-Golay width, and the
   `PeakConfig.windows` for each dataset. **Go/no-go:** if Severson's 4C
   peaks are too smeared to track, consider fitting on Sandia's slow cells
   and validating on Severson instead.
4. **Build the feature tables.**
   ```bash
   python scripts/build_features.py severson data/raw/<batch1>.mat --cycle-step 10
   python scripts/build_features.py sandia data/raw/SNL_*LFP*.csv
   ```
5. **Sanity-check the features (notebook).** Plot peak area and position
   against cycle number for several cells. Peak areas should fall as the
   cell's capacity fades, and the feature trends should look similar across
   the two datasets once expressed relative to each cell's fresh state.

Deliverable for the mid-sem review: steps 1 to 5, plus the Phase 2
synthetic model showing which feature responds to which mechanism.

## Phase 1 and 3 status (Severson batch 2017-06-30)

- **Go/no-go result:** at 4C the graphite staging peaks merge into one broad
  peak near 3.15 V (FWHM ~0.11 V) on every cycle checked. Severson therefore
  uses a single-peak preset plus 10 mV dQ/dV bands (`oracle.features.PRESETS`).
  The bands are regrouped relative to the fresh peak, in FWHM units
  (`oracle.grading.Reference`), so synthetic and real curves line up.
  Clean mechanism separation needs a slower-rate dataset (Sandia).
- **Mechanisms:** a random forest calibrated on 2,100 synthetic 4C curves
  from `oracle.synth`, after the synthetic thread refitted it to the real
  cells (fresh peak within 1 mV). The ageing ranges are 1.6x the default
  profiles (`scripts/synth_features.py --range-scale`). Grouped-by-cell CV
  R² is 0.88 / 0.91 / 0.74 for LLI / LAM_PE / LAM_NE.
- **Grade:** taken from each cell's own mechanism mix (`grading.grade_of`).
  Total loss under 0.10 is A; otherwise LLI-dominant is A, one LAM mode is
  B, and mixed modes are C. Severity ≥ 0.20 caps the grade at B, and
  ≥ 0.40 makes it C. Confidence is the share of forest trees giving the
  same grade.
- **Unsupervised structure:** PCA (3) and a GMM with k picked by BIC (7 on
  Severson). The cluster and its grade are reported next to the grade.
  HDBSCAN finds no density structure, because Severson ageing is a continuum.
- **Scores without refitting, on an independent synthetic set** (60 cells,
  seed 1): the per-cell grade matches the true mechanism grade 82% of the
  time. The cluster grade matches 49%. Mechanism MAE is 0.018 / 0.024 / 0.027.
- **Main open problem: real aged cells are outside the synthetic range.**
  Every real Severson snapshot below 90% capacity, and 69% of those at
  90–95%, is farther from the synthetic training curves than 99% of
  synthetic curves are from each other. Real aged cells lose much more
  peak height than the model does for the same shift. So the real-cell
  mechanism split is an extrapolation. It flips between LLI-dominant and
  LAM_PE-dominant depending on the synthetic ranges. `predict()` measures
  this distance (`features["ood_ratio"]`), adds a note above 1 and halves
  the confidence. The fix is in the synthetic model (peak broadening or
  height loss under 4C ageing), or a slow-rate dataset.
- **UI entry point:** `oracle.predict.predict(file) -> GradeResult`. Samples
  are in `data/samples/`.
- **Rebuild everything:**
  ```bash
  python scripts/build_features.py severson <batch>.mat --cycle-step 10 --out data/processed/features_severson.csv
  python scripts/synth_features.py --out data/processed/features_synth_4C.csv
  python scripts/calibrate_mechanisms.py --features data/processed/features_synth_4C.csv
  python scripts/fit_grading.py severson data/processed/features_severson.csv --mat <batch>.mat
  python scripts/synth_features.py --seed 1 --cells 15 --out data/processed/features_synth_4C_test.csv
  python scripts/validate.py severson data/processed/features_synth_4C_test.csv --own-reference
  python scripts/validate.py severson data/processed/features_severson.csv --holdout
  ```

## Cross-dataset validation (Wheeler et al. 2025, slow C/25 checkups)

**Data.** Wheeler, Bultel, Venet & Sari, "An ageing study of twenty 18650
lithium-ion Graphite/LFP cells in first and second life use", *Sci Data* 12,
392 (2025), doi:10.1038/s41597-025-04712-7. Data: Recherche Data Gouv,
doi:10.57745/OLBXKT, CC BY 4.0. These are the same A123 1.1 Ah cells as
Severson, aged at 50 degC down to 30-60% SOH. One cell per ageing group was
used (1b, 2b, 3b, 4b, 5a, 6a, 7a), with a C/25 checkup every 300 cycles,
103 checkups in all. `scripts/fetch_data.py wheeler-checkups` pulls only the
checkup files out of each zip by byte range (the server gives ~40 KB/s).

**Method: mechanistic digital twin (`oracle/twin.py`).** The learned
feature→mechanism model doesn't transfer at C/25: the synthetic peaks are 2x
wider than real ones, and 2-6x outside the training spread. So the half-cell
model is fitted directly to each measured curve instead. For each cell, the
electrode design is fitted to its cycle-0 checkup. LLI / LAM_PE / LAM_NE are
then fitted to every later checkup, after subtracting the fresh-state model
error. Nothing is trained on Wheeler data. On synthetic curves the twin
recovers 0.20 LLI / 0.30 LAM_PE / 0.20 LAM_NE to within 0.002.

**Results** (`data/processed/validation_wheeler_twin.json`, `twin_wheeler.csv`):

| Check | Result |
|---|---|
| Fit quality | median rms 0.011 (Q in Ah, dQ/dV scaled) |
| Severity vs measured capacity | Spearman −0.84 (p ≈ 6e-28); total loss vs SOH −0.87 |
| Grade never improves with age | 0% of steps go back to a better grade |
| Dominant mode at end of test | LLI in 6 of 7 cells (0.44–0.70); 5a is LAM_PE 0.55 vs LLI 0.50 |
| Teardown (graphite loss, LFP intact) | LAM_NE > LAM_PE in 3/7 final checkups; the two deepest-aged cells (7a 32% SOH, 6a 36%) have LAM_NE 0.30 / 0.19 |
| Grades | A 7 (fresh), B 11 (median SOH 0.88), C 85 (median SOH 0.63); B only above 80% SOH |

**Independent model check (Dubarry synthetic graphite//LFP set,** Mendeley
doi:10.17632/bs2j56pn7y, 72 random labelled curves, `validation_dubarry.json`):
LAM_PE is tracked well (Pearson 0.95, R² 0.78). LAM_NE is not seen at all
(estimated mean 0.02 against a true 0.19), and LLI is near zero in that
sample, so it isn't scored meaningfully. Our twin only notices graphite loss
once the graphite becomes the limiting electrode, so **LAM_NE is a lower
bound**. That likely explains why the Wheeler teardown's graphite loss only
shows up at the end of life.

**Open points.**
1. Mid-life LLI and LAM_PE come out nearly equal and partly trade off. Their
   sum is well determined; the split is not.
2. Grade rule (approved 2026-10-27): mixed LLI + LAM_PE with graphite under
   20% of the total loss is B, not C, because neither mode drives lithium
   plating. Mixed modes that involve graphite stay C.
3. Severson 4C stays limited by the peak merge (see above).

## Update 2026-10-27: recalibration and two-path predict()

- **4C recalibration** on the refitted synthetic model, which adds the
  start-of-discharge relaxation, in-discharge resistance growth (r_diff) and
  spread growth (het) as nuisance terms. Grouped-CV R² is now
  0.62 / 0.70 / 0.54 (lower, because of the nuisance terms, but honest).
  The out-of-range check now uses a leave-cell-out threshold. Only 1.2% of
  real Severson snapshots fall outside (9-14% below 85% capacity), down from
  every snapshot below 90%. Hold-out Spearman is −0.85 to −0.88 over
  3 seeds. FRESH_TOTAL went from 0.10 to 0.15 to match the new noise floor
  (3 modes × MAE 0.03-0.037).
- **Still weak at 4C:** many near-fresh Severson snapshots come out as mixed
  LAM and grade C. At 4C, treat the mechanism split as indicative only.
- **`predict(file, model="auto")`** now picks the path from the curve itself:
  - **slow path:** if the tallest dQ/dV peak is above 3.25 V and narrower
    than 25 mV (a C/25-like discharge), the twin is fitted to the upload
    against a population fresh reference built from the 7 Wheeler cycle-0
    checkups (`models/twin_slow.joblib`, from `scripts/fit_twin_reference.py`).
    It takes about 15-50 s per file. Confidence is the share of grades that
    stay the same when each mode is jittered by ±0.02.
  - **fast path:** otherwise the 4C forest path, as before.
  - Both paths set `features["path"]` to `"slow"` or `"fast"`.
- **Samples:** `data/samples/wheeler_6a_cycle{0,1500,3300}_C25.csv` grade A,
  B (LLI 0.17) and C (LLI 0.40).

## Held-out test (2026-10-28, frozen at 0848e1b)

Protocol: `docs/heldout_protocol.md`; script: `scripts/heldout_test.py`; results:
`data/processed/heldout/`. Grade rule and models unchanged since the freeze.

- **Unseen Wheeler cells (1a 2a 3a 4a 5b 6b, 93 C/25 checkups, app path).** Severity vs SOH
  Spearman −0.96, grade vs SOH −0.89. Median SOH A 0.93 / B 0.77 / C 0.58; pairs ordered by
  SOH: A>B 99.8%, B>C 95%. No C above 80% SOH; 1 of 87 steps improved with age. Limits: same
  cell make and ageing protocol as the tuning cells; B spans a wide SOH range; LLI dominates
  almost everywhere, so this tests the grade more than the mechanism split.
- **Unseen Dubarry curves (files 2 and 3, 73 labelled).** Severity vs true capacity loss
  Spearman 0.91, but dominant mode correct only 33% (0/20 when the true mode is LAM_NE),
  grade agreement 58%, per-mode R² ≤ 0.09. The mechanism split does not hold up here. Caveat:
  Dubarry curves are C/25 charge curves treated as discharges.
- 25 °C files of the tuning cells: not run (a file with comma decimals broke the loader).
- Demo uploads: `data/samples/heldout/HELDOUT_wheeler_*` are fair held-out. The older
  `data/samples/wheeler_6a_*` and `app/samples/*` are not (tuning cell / synthetic).

## Post-freeze improvements (2026-10-28)

Everything below changed after the held-out test, so the blind cells are no
longer blind for these changes. The grade rule (`grade_of` and its
thresholds) is **unchanged**; mechanisms and grades on the slow path are
unchanged too.

**1. Slow-path confidence (`oracle/predict.py`).** The old rule halved the
confidence when the fit rms was above 0.02. But fresh cells fit at
0.020–0.046 against the population fresh reference (cell-to-cell spread,
not ageing), while aged, LLI-dominated cells fit at ~0.015: on the blind
cells rms rises with SOH (Spearman +0.86). So the halving hit healthy cells,
e.g. every blind cycle-0 checkup showed 50%. The cut-off is now 0.05, just
above the worst fresh cell of all 13. A per-cell fresh reference is not an
option for the app, because a retired cell arrives with no fresh record.
Checked on the 7 tuning cells through the app path (`scripts/confidence_check.py`,
`data/processed/conf_tuning.csv`); "off-band" means the grade disagrees with
measured SOH (A >85%, B 65–85%, C <65%):

| | old (0.02) | new (0.05) |
|---|---|---|
| Tuning cells, 103 checkups: median confidence | 0.95 | 0.97 |
| share of checkups under 50% | 20% | 1% |
| AUC, low confidence flags an off-band grade | 0.68 | 0.75 |
| Fresh checkups, median confidence | 0.50 | 1.00 |
| Blind cells, 93 checkups (same comparison) | AUC 0.73, 27% under 50% | AUC 0.82, 4% under 50% |
| Demo samples 2a c0 / 5b c600 / 6b c1800 | 50 / 36 / 46% | 100 / 71 / 93% |

**2. Dubarry as charge curves (`oracle/twin.py`, `fit_design(fit_shift=True)`).**
Dubarry's curves are C/25 charges. The twin now fits a voltage offset
(hysteresis, +28 mV) to the fresh curve and keeps it for the aged fits.
Same 36 labelled curves from files 2 and 3 for both treatments
(`scripts/dubarry_charge_check.py`, `data/processed/dubarry_charge/`):

| | as discharge (old) | charge mode (new) |
|---|---|---|
| Fresh fit rms | 0.029 | 0.013 |
| R² LLI / LAM_PE / LAM_NE | −0.47 / −0.11 / −0.62 | 0.58 / −0.28 / 0.53 |
| MAE LLI / LAM_PE / LAM_NE | 0.079 / 0.080 / 0.110 | 0.032 / 0.083 / 0.049 |
| Dominant mode correct | 22% | 53% |
| ... when the true mode is graphite loss | 0 / 12 | 9 / 12 |
| Grade agreement | 67% | 69% |
| Severity vs true capacity loss (Spearman) | 0.87 | 0.90 |

So most of the "graphite loss is never found" result came from fitting
charge curves as discharges. LAM_PE is still poorly separated (it trades off
with LLI), and n = 36 is small. The app is unaffected: Wheeler and uploads
are discharges.

**3. Unsupervised clusters on real cells (`scripts/cluster_mechanisms.py`,
`models/clusters_wheeler.joblib`, `data/processed/clusters_wheeler.json`).**
A Gaussian mixture on the twin's (LLI, LAM_PE, LAM_NE) for the 7 tuning
cells (103 checkups); BIC picks k = 3. Clusters are named by their own
medians, never by the rule, then applied unchanged to the 6 blind cells:

| Cluster | Median LLI / LAM_PE / LAM_NE | Tuning: n, median SOH, rule grades | Blind: n, median SOH, rule grades |
|---|---|---|---|
| 0: lithium loss, mild to moderate | 0.22 / 0.01 / 0.02 | 65, 0.79, A19 B30 C16 | 75, 0.76, A17 B37 C21 |
| 1: heavy lithium loss + some LAM | 0.45 / 0.07 / 0.06 | 29, 0.55, B1 C28 | 18, 0.56, B2 C16 |
| 2: lithium + cathode loss | 0.55 / 0.57 / 0.05 | 9, 0.44, C9 | 0 |

- The clusters find **mechanism type**, not wear level: cluster 2 is cell 5a's
  LLI + LAM_PE path. Within the big LLI cluster ageing is a continuum, so the
  GMM does not split A from B.
- Agreement with the rule is weak: ARI 0.13 (tuning) / 0.07 (blind); the
  cluster's majority grade matches the rule grade 65% / 57%. Cluster index
  vs SOH: Spearman −0.73 / −0.46, against −0.89 / −0.89 for the rule grade.
- Stable: refitting on 50 bootstrap resamples of the tuning cells gives a
  median ARI of 0.86 (10–90%: 0.69–0.96) on the blind cells.
- Reading: unsupervised structure confirms the mechanism groups, and the rule
  adds what clustering can't, the severity thresholds. The app now reports the
  cluster as a note on the slow path; the grade still comes from the rule.
- A composition feature set (total loss + mode shares) orders by SOH better
  (blind Spearman −0.84) but BIC then splits 7 fresh checkups into 3 tiny
  clusters, so it is not used.

**Data handling.** `scripts/extract_wheeler_c25.py` cuts the 4.7 GB of raw
Wheeler CSVs to ~10 MB per cell; `wheeler.iter_npz` reads it. On 1a it
reproduces the frozen run to within 0.0001 in rms and 0.005 in mechanisms.
