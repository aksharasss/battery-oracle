# Battery Oracle 2.0: the A-to-Z guide for the final review

This guide is for teammates who have not followed the project. It explains, in plain words, everything the final slides and the live demo show, in the same order as the deck (`final-review/Battery_Oracle_Final_Review_Deck.pptx`). It only covers work we actually did. Numbers come from the repo (github.com/aksharasss/battery-oracle) and from re-running the app on 28 Oct 2026.

---

## A. The one-sentence version

We take **one slow partial discharge** of a used battery cell, work out **what kind of ageing happened inside it**, and give it a **second-life grade**: A (reuse for storage), B (light duty) or C (recycle).

## B. Why anyone needs this (slide 2)

- An electric-car battery is usually retired when it drops to about 70–80% of its original capacity. That does not mean it is useless: many cells can still store energy for years in lighter jobs, such as storing solar power.
- Every retired cell has to be sorted. Today that needs slow, expensive lab tests.
- Most research predicts **remaining useful life**: a single number for how long the cell will last. It does not say **why** the cell aged. Two cells with the same capacity can have aged in completely different ways, and the reason matters for whether reuse is a good idea.
- Our tool gives the reason as well as the grade, from one file, with no cycling history.

## C. Battery basics you need

- A cell has two electrodes. The **cathode** (positive side) is LFP, lithium iron phosphate. The **anode** (negative side) is graphite. Lithium moves between them; on discharge it moves from graphite to LFP.
- **Capacity** (in Ah) is how much charge the cell can hold. **State of health (SOH)** = capacity now ÷ capacity when new. 0.93 means 93% of new.
- **C-rate** is how fast the cell is discharged. 1C empties it in one hour. **4C** is fast (15 minutes). **C/25** is very slow (25 hours). Slow is important for us; see section I.
- **Partial discharge** means we only need part of the discharge curve, not a full cycle from 100% to 0%.

## D. The data (slide 3)

All three datasets are public and use the same type of cell (A123 LFP/graphite).

| Dataset | What it is | What we used it for |
|---|---|---|
| **Severson et al. 2019** (Nature Energy, MIT/Stanford) | 124 cells discharged fast, at 4C. We processed the 2017-06-30 batch. | Our first model. It showed that at 4C the peaks we need blur into one, so you cannot read the ageing type reliably. **No longer in the demo.** |
| **Wheeler et al. 2025** (Sci Data 12, 392; data doi 10.57745/OLBXKT) | The same A123 1.1 Ah cells, aged at 50 °C down to 30–60% health, with a slow C/25 check-up every 300 cycles. | Our main dataset. **7 cells** (103 check-ups) were used to tune the grading rule. **6 other cells** (93 check-ups) were kept back for a blind test. |
| **Dubarry synthetic set** (Mendeley) | Computer-generated curves where the true amount of each ageing type is known. | Checks whether we name the **right** ageing type. 72 curves were used early; 73 others were kept unseen for the blind test. |

Why not Sandia? It was in the original plan, but its raw files could no longer be downloaded, so Wheeler replaced it.

## E. What has been done before, and what is new (slide 4)

- Severson 2019 predicts **when** a cell dies, not what to do with it afterwards.
- Remaining-life models give one number and no ageing type.
- Lab tests (impedance, opening the cell) need equipment and hours per cell.
- Grading by capacity alone (for example "above 80% = good") misses that two cells at the same capacity can be different inside.
- The closest published work, **Beatty et al. 2026**, groups cells by their dQ/dV features (including Severson cells), but does not split ageing into types and does not test on a second dataset.

**Our contribution, "to our knowledge":** the combination of (1) a grade based on the fitted ageing type rather than capacity bands, (2) a test on a second dataset with no refitting plus a blind test, and (3) an upload tool. Each piece alone is a standard technique. **Never say "first".**

## F. The three ways a cell ages (slide 5)

| Short name | Full name | Plain meaning |
|---|---|---|
| **LLI** | Loss of lithium inventory | Some lithium gets trapped (for example in a surface layer) and can no longer move. Usually gradual and stable. |
| **LAM_PE** | Loss of active material, positive electrode | Part of the LFP cathode stops working. |
| **LAM_NE** | Loss of active material, negative electrode | Part of the graphite anode stops working. This is linked to a risk called lithium plating, so we treat it as the most serious. |

The **grades** mean: **A** = stationary storage (solar, grid buffering); **B** = light duty, low current; **C** = recycle for materials.

## G. What changed since the mid-semester review (slide 6)

| Mid-sem plan | What happened |
|---|---|
| O1 Severson data loader | Done. Finding: at 4C the peaks merge, so ageing types blur. |
| O2 dQ/dV + Gaussian peak fitting | dQ/dV done. Peaks are described by position, height, area and width. |
| O3 21-number "lifetime trajectory" per cell | Changed. A retired cell arrives with no history, so we grade **one** discharge. |
| O4 Bayesian GMM clustering | A Gaussian mixture was tried (it picked 7 groups); Severson ageing has no natural groups, it is a smooth continuum. The grade now comes from each cell's own mix of ageing types. |
| O5 Bootstrap ARI validation | Replaced by stronger checks: synthetic curves with known answers, Wheeler with no refitting, a blind test, and Dubarry. |
| O6 Certificate UI | Done: the Streamlit app in the demo. |

## H. How the system works, step by step (slide 7)

1. **Upload a CSV** with voltage plus capacity, or voltage plus current and time.
2. **Build the dQ/dV curve** (section I).
3. **Check the speed.** If there is a sharp peak above 3.25 V, the file is a slow discharge.
4. **Slow files** go to the **digital twin** (section J). This takes 15–50 seconds and is the path behind every result in the deck. (Fast 4C files go to an older quick model, under 1 second, which we treat as indicative only. The demo samples are all slow files.)
5. The twin gives three numbers: **LLI, LAM_PE, LAM_NE**, each as a fraction of loss (0.10 = 10%).
6. The **grade rule** (section K) turns them into A, B or C, with a confidence and notes.

## I. Maths 1: incremental capacity analysis, "ICA" (slide 8)

- LFP's voltage is almost flat during discharge, so ageing hardly shows in the plain voltage curve.
- Instead we plot **dQ/dV**: how much charge comes out for each small drop in voltage. Every flat part of the voltage curve becomes a **sharp peak**. The peaks come from phase changes inside the graphite.
- Steps in the code (`oracle/ica.py`):
  1. Put capacity Q on an evenly spaced voltage grid (1 mV steps between 3.05 and 3.42 V). Real logs have uneven points, which would create fake spikes.
  2. Smooth Q with a **Savitzky–Golay filter** (15 mV window). It removes noise but keeps peak position and height.
  3. Differentiate: dQ/dV ≈ ΔQ / ΔV.
- As a cell ages, peaks shrink, move or disappear, and each ageing type does this in its own way.
- This needs a slow discharge. At 4C all peaks merge into one broad bump near 3.15 V.

## J. Maths 2: the half-cell "digital twin" (slide 9)

- A **digital twin** here is a physics model of the cell, not a trained black box.
- Cell voltage = cathode voltage − anode voltage: **V = U_LFP(x_PE) − U_graphite(x_NE)**. Each electrode has a known voltage curve depending on how full of lithium it is.
- The three ageing types change the model like this: LAM_PE shrinks the cathode's capacity by a factor (1 − LAM_PE); LAM_NE shrinks the anode's by (1 − LAM_NE); LLI removes lithium, which slides the two electrode curves against each other.
- **Fitting:** the computer searches for the three numbers (each between 0 and 0.8) that make the model's Q(V) and dQ/dV shape match the measured curve as closely as possible (least squares, `oracle/twin.py`).
- The "fresh" starting point is the **average of all Wheeler cycle-0 check-ups**, because a retired cell arrives without its own fresh record.
- Sanity check: on computer-made curves with known answers 0.20 / 0.30 / 0.20, the fit gets them back to within 0.002.
- The figure on the slide shows our model's curves at C/20 as each ageing type grows: each one changes the peaks differently, which is why the fit can tell them apart.
- Nothing in the twin is trained on Wheeler data.

## K. From ageing numbers to a grade (slide 10)

First, **severity** = LLI + 1.5 × LAM_PE + 2 × LAM_NE. Anode loss counts most because of the plating risk.

Then the rule is checked in order (`grade_of` in `oracle/grading.py`):

1. Total loss under 0.15 → **A** (looks fresh).
2. Severity 0.40 or more → **C** (too worn, whatever the cause).
3. Which type holds at least 60% of the loss? LLI → **A**. One LAM type → **B**. None ("mixed") → **C**.
4. Exception: mixed, but graphite loss under 20% of the total → **B** (lithium + cathode loss is not a plating risk). Added on 27 Oct 2026 after seeing Wheeler results.
5. Severity 0.20 or more → an A becomes **B** at best.

**Confidence**, as the app computes it: the three ageing numbers are each nudged randomly by about ±0.02 (the twin's typical error), 300 times. Confidence = the share of those runs that give the **same grade**. If the model fits the curve poorly (fit error, "rms", above 0.02) the confidence is **halved**. So confidence means "how stable is this grade", not "the probability the grade is correct".

## L. Result 1: Wheeler tuning cells (slide 11)

7 cells, 103 check-ups, model never trained on Wheeler.

- Severity vs measured capacity: Spearman **−0.84** (more severe = less capacity, as it should be).
- Total loss vs state of health: Spearman **−0.87**.
- Typical fit error: **0.011**.
- **0%** of steps where a cell got a *better* grade as it aged.
- Grades: A 7, B 11, C 85 check-ups (these cells were aged hard). Median health of B = 0.88, of C = 0.63.
- Lithium loss was the main type in 6 of 7 cells.
- **Caveat:** the rule was adjusted after seeing these cells, so this is not a fair test on its own. That is what the blind test is for.

*Spearman* is a rank correlation from −1 to +1. −0.96 means that as one goes up, the other goes down almost perfectly in order.

## M. Result 2: the blind test (slide 12)

Before looking at any results, the code was frozen (commit 0848e1b). Then it was run on **6 Wheeler cells it had never seen** (1a, 2a, 3a, 4a, 5b, 6b; 93 check-ups).

- Severity vs state of health: Spearman **−0.96**.
- Median health per grade: **A 0.93, B 0.77, C 0.58**.
- Pick a random A check-up and a random B one: the A is healthier **99.8%** of the time. B vs C: **95%**.
- **No** grade C above 80% health. Only 1 of 87 steps improved with age.
- **Limits:** same cell make and ageing method as the tuning cells; grade B covers a wide health range; lithium loss dominates almost everywhere, so this tests the grade more than the split into ageing types.

## N. Result 3: where it does not work yet (slide 13)

On 73 unseen Dubarry curves where the true answer is known:

- **Works:** severity vs true capacity loss, Spearman **0.91**. We can tell **how worn** a cell is.
- **Does not hold up:** the main ageing type was named correctly only **33%** of the time; **0 of 20** when the real cause was graphite loss; grade agreement **58%**; the per-type fit quality (R²) is at most 0.09.
- Why: graphite loss only shows once graphite becomes the limiting electrode, so the twin under-counts it. Also, these Dubarry curves are charge curves that we fitted as if they were discharges.
- Bottom line: trust the grade order and severity more than the exact mechanism breakdown.
- **Update 28 Oct:** we now fit Dubarry's curves as what they are, charge curves (the twin gets a +28 mV hysteresis offset). On 36 labelled curves: graphite loss found **9 of 12** times (was 0 of 12), main type right **53%** (was 22%), R² LLI 0.58 and LAM_NE 0.53 (were negative). Cathode loss (LAM_PE) is still not separated well. So most of the graphite "blind spot" was our mistake in treating charges as discharges.

## O. The live demo (slide 14)

Run: `streamlit run app/streamlit_app.py` from the `battery-oracle` folder.

What is on screen:
- Left sidebar: **Upload** a CSV, or **pick a sample**. The sample list now only has Wheeler check-ups. Severson and synthetic samples were removed.
- Big **grade letter** and what it means (e.g. "Stationary storage").
- **Confidence**, **dominant mechanism**, **method** (half-cell twin, C/25) and **fit RMS** (fit error).
- **Mechanism breakdown**: three bars for lithium loss, cathode loss, anode loss.
- **dQ/dV plot**: this cell (solid line) against an average fresh cell (dashed).
- Boxes labelled METHOD / CAVEAT with warnings.
- The red stamp top-right says which path ran ("SLOW PATH · TWIN FIT").
- A toggle switches between the dark and light "graph paper" sheet. The deck uses the same colours.

Which samples are fair to show:
- **Blind (never used for tuning):** cell **2a** cycle 0, **5b** cycle 600, **6b** cycle 1800.
- **Not blind:** cell **6a** (cycles 0, 1500, 3300) was one of the tuning cells. Fine to show, but say so.

Suggested demo: pick 5b cycle 600 (grade A), then 6b cycle 1800 (grade B), then 6a cycle 3300 (grade C). Each slow sample takes 15–50 seconds.

## P. Why the blind cells only show 36–50% confidence (slide 15)

We re-ran the three blind samples with the current code:

| Sample | Known state | Grade | Fit error | Confidence before halving | Shown in app |
|---|---|---|---|---|---|
| 2a, cycle 0 | fresh | A | 0.022 | 100% | **50%** |
| 5b, cycle 600 | mid-life | A | 0.029 | 71% | **36%** |
| 6b, cycle 1800 | late-life | B | 0.022 | 93% | **46%** |
| 6a (tuning cell) cycles 0 / 1500 / 3300 | fresh → old | A / B / C | 0.013–0.020 | – | 100 / 84 / 99% |

Three reasons, all from how the app computes confidence (section K):
1. The app **halves** confidence when the fit error is above 0.02. All three blind cells land just above that line (0.022–0.029).
2. The app compares every upload with **one average fresh cell**. Real cells differ a little from the average, so cells the system has never seen fit a bit more loosely than cell 6a.
3. Cell 5b also sits right at a rule boundary: its total loss is 0.14, just under the 0.15 "looks fresh" cut-off, so small nudges change its grade.

**How to say it at the review:** the grades themselves make sense (fresh → A, mid-life → A, late-life → B). The low number means "the fit is a little loose, so double-check", not "the grade is wrong".

**Fixed on 28 Oct (after the freeze, grade rule unchanged).** Looking at all checkups showed the 0.02 line was simply wrong: fresh cells fit at 0.020–0.046 against the average fresh cell, while aged cells fit at about 0.015, so the halving punished *healthy* cells. A per-cell fresh reference isn't possible in the app (a retired cell has no fresh record), so instead the line moved to 0.05, just above the worst fresh cell of all 13. Now the three demo samples show **100 / 71 / 93%**. Checked on the 7 tuning cells: checkups under 50% confidence went from 20% to 1%, and low confidence now flags grades that disagree with measured health better (AUC 0.68 → 0.75). 5b at 71% is real: it sits at the 0.15 "looks fresh" boundary.

## Q. Honest limitations (slide 16)

1. The grade rule was tuned on Wheeler after seeing it; the blind test is the fair check.
2. The split into ageing types is only partly reliable. With the charge-mode fix, Dubarry: main type right 53%, graphite loss found 9/12, but cathode loss still poorly separated (n = 36).
3. Graphite loss is a lower bound. In mid-life, lithium loss and cathode loss trade off: their sum is solid, the split between them is not.
4. One chemistry (LFP/graphite), one cell make, small sample.
5. At 4C the ageing-type split is only indicative.
6. A 25 °C check was not run (a file with commas as decimal points broke the loader).
7. Grades are **not** safety-validated.

**What you can trust:** how worn a cell is, and its grade order. **What needs more work:** naming the exact ageing type, especially graphite loss.

## R. Conclusion and next steps (slide 17)

- **Main message:** a grade with a physical reason, from one slow partial discharge.
- **Shown:** ICA + a physics twin read ageing from one curve; on 6 unseen cells, severity tracks health (−0.96); a working app grades an uploaded file.
- **Next steps:** test the mechanism split against teardown data (Wheeler) and fix cathode-loss separation; other chemistries and temperatures; safety validation.

## S. Unsupervised clusters on real cells (added 28 Oct)

Latent chose to make the clustering mean something. A Gaussian mixture is fitted to the twin's three numbers (LLI, LAM_PE, LAM_NE) for the 7 tuning cells (103 checkups); BIC picks **3 clusters**, named only by their own medians:

| Cluster | What it is | Tuning cells: SOH, rule grades | Blind cells (no refit) |
|---|---|---|---|
| 0 | lithium loss, mild to moderate | 0.79; A19 B30 C16 | 0.76; A17 B37 C21 |
| 1 | heavy lithium loss + a little LAM | 0.55; C28 B1 | 0.56; C16 B2 |
| 2 | lithium + cathode loss (cell 5a) | 0.44; C9 | none |

- The clusters find **ageing type**, not wear level. Inside cluster 0, ageing is a smooth slope, so no clustering can separate A from B there.
- So the clusters agree only weakly with the rule (ARI 0.13 tuning, 0.07 blind) and track health less well (Spearman −0.73 / −0.46) than the rule grade (−0.89 / −0.89).
- They are stable: refitting on resampled cells gives a median ARI of 0.86 on the blind cells.
- **How to say it:** the unsupervised step confirms that real cells fall into mechanism groups; the rule adds the severity thresholds that clustering cannot find on a continuum. The app shows the cluster as a note; the grade comes from the rule.

## T. Three common confusions: where the AI is, Severson, and "fit" vs "twin"

**Where is AI / machine learning used?** Only in the fast (4C) path, and it is not what the main results rest on.

| Part | What it is | Trained on |
|---|---|---|
| ICA (dQ/dV) | Signal processing: smoothing + differentiation | Nothing |
| Synthetic half-cell model (`oracle/synth.py`) | Physics: graphite + LFP voltage curves | Its fresh shape was tuned to Severson cycle-10 (4C) and Wheeler cycle-0 (C/25) curves |
| Mechanism random forest (`oracle/mechanisms.py`, 4C path only) | **ML, supervised**: ICA features → LLI / LAM_PE / LAM_NE | 2,100 *synthetic* 4C curves from 100 synthetic cells, labels known because we made them |
| PCA + Gaussian mixture (`oracle/grading.py`, 4C path only) | **ML, unsupervised clustering**: k = 7 picked by BIC | Real Severson features. The cluster is shown next to the grade but does not set it |
| Digital twin fit (`oracle/twin.py`, slow path) | Physics model + least-squares curve fitting. **No ML** | Nothing; the fresh design is fitted to the average of 7 Wheeler cycle-0 curves |
| Grade rule (`grade_of`) | Hand-written rule (section K) | Nothing; thresholds chosen by us |

So: the slow path, which gives every result in sections L–N, is physics plus optimisation, not AI. That is a strength for the "not a black box" story. If someone asks "where is the unsupervised learning?", the honest answer is: we tried GMM clustering on Severson (O4); it found 7 groups, but Severson ageing is a smooth continuum, so the grade is taken from each cell's own mechanism mix instead.

**Did we use Severson (the 4C dataset)?** Yes, in the first half of the project. It is the real data behind the fast path: the fresh reference and the GMM clusters come from Severson batch 2017-06-30, and the synthetic model was tuned so its fresh peak sits within 1 mV of Severson's. At 4C the graphite peaks merge into one wide bump near 3.15 V, so we:
- describe that one peak (position, height, area, width) plus 10 mV slices of dQ/dV ("bands") placed relative to the fresh peak, in peak-width units;
- added "nuisance" terms to the synthetic model (resistance growth, relaxation, spread) so the forest learns to ignore them;
- flag files outside the synthetic range and halve their confidence.

Result: at 4C we can tell **how worn** a cell is, but the split into LLI / LAM_PE / LAM_NE is indicative only. That is why the project moved to slow C/25 data (Wheeler), where the peaks are separate. Severson is not used anywhere in the slow path.

**"Fit" vs "digital twin": what's the difference?** They are not two methods. They are the model and the act of tuning it.
- The **digital twin** is the virtual cell: cell voltage = LFP curve − graphite curve, with three knobs (LLI, LAM_PE, LAM_NE) that slide and shrink the two curves (section J).
- **Fitting** is turning those knobs until the twin's curve lies on top of the measured curve (least squares). The knob settings that match best *are* the answer.
- It happens twice: (1) once, ahead of time, the twin's *design* (electrode sizes, lithium amount, peak shapes) is fitted to an average fresh Wheeler cell (`scripts/fit_twin_reference.py` → `models/twin_slow.joblib`); (2) for each uploaded file, only the three ageing knobs are fitted (`twin.fit_mechanisms`).
- **Fit rms** is how far apart the twin's curve and the real curve still are after fitting. Small rms = the twin explains the curve well.
- On the 4C path there is no per-file fit: the random forest *guesses* the knobs from features, having learned from curves the twin produced. Both paths use the same physics model; the slow path fits it, the fast path learns from it.

**Where exactly does the grade come from? (one pipeline, two ways to get the three numbers)**
1. Upload → dQ/dV curve → speed check (sharp peak above 3.25 V = slow file).
2. Get LLI / LAM_PE / LAM_NE: **slow path** = fit the twin to the curve; **fast path** = random forest guesses them from peak features.
3. **Grade = `grade_of(LLI, LAM_PE, LAM_NE)`**, the hand-written rule in section K, identical on both paths.
4. Confidence and notes are added afterwards. Neither changes the grade.

**Why a random forest on the fast path but not the slow path?** A forest for slow files exists (`models/mechanisms_slow.joblib`, synthetic R² 0.91–0.98) but did not transfer to real Wheeler curves (synthetic C/25 peaks are ~2x too wide), so the slow path fits the twin to each real curve instead. At 4C the peaks are merged and kinetics dominate; the twin assumes a near-equilibrium slow curve and has not been validated there, so the forest is kept, labelled indicative only.

**Will the two paths grade differently?** They can: same rule, different estimators of its inputs. The slow path is the validated one; the fast path is noisier.

**Why a GMM if it does not grade?** It does not. The GMM clusters Severson features; each cluster is then labelled by running `grade_of` on the cluster's median mechanisms, and the cluster grade is shown only as a note. On synthetic checks it matched the true grade less often (28–49%) than the per-cell rule (60–82%). It is a leftover of the original "unsupervised grading" plan, so the "unsupervised" claim is weak. Options: describe it as exploratory; cluster the twin's mechanism numbers on Wheeler so the clusters mean something; or remove it.

---

## Likely questions and short answers

- **Why not just predict remaining life?** It says how long, not why. Whether a cell suits storage depends on why it aged.
- **Where is your ground truth?** There is no public teardown data, so we use computer-made curves with known answers (our own model and Dubarry's), plus measured capacity in Wheeler.
- **Isn't the rule overfit?** Partly tuned on Wheeler, which is why we froze the code and ran a blind test on 6 unseen cells.
- **Why is confidence so low on the blind cells?** See section P: the fit is just past our strictness line, mostly because we compare against an average fresh cell.
- **Why did you drop Severson from the demo?** Its 4C discharge blurs the peaks, and those cells were our early training population, so they are not a fair demo.
- **Does it work on other batteries?** Only tested on LFP/graphite A123 cells.
- **Is it safe to use for real?** Not yet. Grades are not safety-validated.

## Glossary

- **dQ/dV / ICA:** the derivative of charge with respect to voltage; turns flat voltage regions into peaks.
- **Savitzky–Golay filter:** a smoothing method that keeps peak shapes.
- **Half-cell model / digital twin:** a physics model of the cell built from the two electrodes' voltage curves.
- **Least squares:** finding the numbers that make the model curve closest to the measured curve.
- **RMS (fit error):** the typical gap between the model and the measured curve.
- **Spearman correlation:** how well two things rank together, from −1 to +1.
- **SOH:** state of health, capacity now ÷ capacity when new.
- **Held-out / blind test:** testing on data that was never used to build or tune the system.
