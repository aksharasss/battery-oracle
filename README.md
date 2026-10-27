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
