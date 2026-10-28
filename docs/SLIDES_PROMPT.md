# Master prompt: final-review slides for Battery Oracle 2.0

Copy everything below the line into any AI slide tool (Gamma, ChatGPT, Gemini, Copilot, etc.).
Then fill the `[[PLACEHOLDER]]` values (held-out results) before or after generating.
Attach as images: `docs/figures/synth_vs_severson.png`, `docs/figures/sweeps_dqdv.png`, `docs/figures/separation_4C.png`, plus screenshots of the Streamlit app.

NOTE: the mid-review deck file was not found in the repo or project folder. Add this line to the prompt yourself if you have it:
"Match the visual style, fonts, colours and layout of the attached mid-review deck."

---

## PROMPT START

You are making the **end-semester (final review) slide deck** for an academic maths project called **Battery Oracle 2.0**. Build **15 slides**, in the order below. Use ONLY the content given here. **Do not add any fact, number, citation, dataset, claim or result that is not written in this prompt.** If a slide seems to need more, leave it out rather than inventing. Keep text short: max 5 bullets per slide, plain English, large type. Keep the same style as the mid-review deck (title at the top, bullets left, figure right). Put speaker notes on every slide using the "Say:" line.

Where you see `[[PLACEHOLDER: ...]]`, keep it visible in a highlighted colour so the presenter fills it in. Do not guess a value.

### Slide 1: Title
Battery Oracle 2.0: A mechanistic digital twin for second-life grading of retired LFP cells.
Subtitle: Final review. Team names: [[PLACEHOLDER: names]]. Guide: [[PLACEHOLDER]].

### Slide 2: Problem
- Retired EV/storage cells still have use, but each must be graded: reuse for storage, light duty, or recycle.
- Usual approach: black-box Remaining Useful Life (RUL) prediction. It says "how long", not "why".
- Our idea: read the physical degradation mechanism from the voltage curve and grade from that.
Say: We grade by cause of ageing, not by a lifetime number.

### Slide 3: Approach in one picture (flow diagram)
Partial discharge file -> Incremental Capacity Analysis (dQ/dV) -> fit half-cell digital twin -> mechanism mix (LLI, LAM_PE, LAM_NE) -> Grade A / B / C.
Say: One partial discharge in, grade and reason out. No cycling history or capacity label needed.

### Slide 4: The three degradation mechanisms (LFP/graphite cell)
- LLI: loss of lithium inventory (lithium trapped, e.g. in the SEI layer).
- LAM_PE: loss of active material in the positive electrode (the LFP cathode).
- LAM_NE: loss of active material in the negative electrode (graphite).
Say: Each one changes the dQ/dV curve differently.

### Slide 5: What each mechanism does to dQ/dV (from our synthetic model)
Figure: `sweeps_dqdv.png`.
- LLI at C/20: the highest-voltage peak (~3.34 V) disappears.
- LAM_PE at C/20: hidden at first (LFP has about 10% spare capacity), then the lowest-voltage peak (~3.19 V) is cut.
- LAM_NE at C/20: the middle peak (~3.29 V) shrinks and peaks squeeze together.
- Resistance: shifts the peak to lower voltage at 4C and keeps its area.
Say: This contradicts our first heuristic that LLI shrinks all peaks together.

### Slide 6: Recap of mid-review (Phases 1 and 2)
- Phase 1: ICA pipeline (window -> Q(V) on a uniform grid -> Savitzky-Golay dQ/dV -> peak features).
- Phase 2: synthetic graphite/LFP half-cell model (MSMR open-circuit potentials) with known LLI, LAM_PE, LAM_NE, used instead of teardown ground truth.
- Model fresh peak matches real Severson cells within 1 mV (3.148 V, FWHM 0.110 V). Figure: `synth_vs_severson.png`.
Say: There is no teardown data for public cells, so synthetic curves are our ground truth.

### Slide 7: Datasets
- Severson (MIT/Stanford): A123 LFP cells, discharged at 4C. Used to build the grading clusters.
- Wheeler et al. 2025 (Sci Data 12, 392): same A123 1.1 Ah cells aged at 50 degC to 30-60% SOH, C/25 checkups every 300 cycles. Used for cross-dataset validation. 7 cells, 103 checkups.
- Dubarry synthetic graphite//LFP set (Mendeley, 72 labelled curves): independent check of the twin.
- Finding: at 4C the graphite peaks merge into one broad peak near 3.15 V, so clean mechanism separation needs a slow rate.
Say: Sandia raw files were no longer downloadable, so Wheeler was chosen.

### Slide 8: Can features tell mechanisms apart? (synthetic test)
Figure: `separation_4C.png`. Table (100 synthetic cells, grouped cross-validation):
- 4C, change from fresh: LDA accuracy 0.94, R2 LLI/LAM_PE/LAM_NE 0.93/0.89/0.84.
- C/20, single test: LDA accuracy 0.93, R2 0.94/0.97/0.89.
- After adding kinetic ageing nuisance, 4C random forest R2 is 0.71/0.85/0.58 (single test).
Say: Slow discharge reads mechanisms much more reliably, so a grading station should use C/20 to C/5.

### Slide 9: Why a digital twin for slow-rate data
- A learned feature-to-mechanism model did not transfer to C/25 (synthetic peaks 2x wider than real).
- So the half-cell model is fitted directly to each measured curve: electrode design from the fresh checkup, then LLI / LAM_PE / LAM_NE per checkup.
- Nothing trained on Wheeler data. On synthetic curves the twin recovers 0.20 / 0.30 / 0.20 to within 0.002.
Say: The twin is physics fitting, not a black box.

### Slide 10: Grading rule
- Total loss under 0.15: Grade A (fresh-like).
- LLI dominant: A. One LAM mode dominant: B. Mixed modes involving graphite: C.
- Lithium + cathode loss together with graphite under 20% of total loss: B (neither drives lithium plating).
- Severity 0.20 or more caps at B; 0.40 or more is C.
- Grade A = stationary storage, B = light duty / low rate, C = recycle.
Say: The rule was approved by the team on 2026-10-27, after seeing Wheeler results (see caveats).

### Slide 11: Cross-dataset results (Wheeler, no refit)
- Fit quality: median rms 0.011.
- Severity vs measured capacity: Spearman -0.84; total loss vs SOH -0.87.
- 0% of steps move to a better grade with age.
- Dominant mode at end of test: LLI in 6 of 7 cells; cell 5a LAM_PE 0.55 vs LLI 0.50.
- Grades over 103 checkups: A 7, B 11 (median SOH 0.88), C 85 (median SOH 0.63). B only above 80% SOH.
Say: Grades follow measured health without ever seeing capacity.

### Slide 12: Independent check and held-out test
- Earlier Dubarry check (72 curves): LAM_PE tracked well (Pearson 0.95, R2 0.78). LAM_NE not seen (estimated mean 0.02 vs true 0.19). LLI near zero in that sample, not scored.
- Held-out test, rule and models frozen at commit 0848e1b before looking at results:
  - 6 unseen Wheeler cells (1a 2a 3a 4a 5b 6b), 93 C/25 checkups: severity vs SOH Spearman -0.96; grade vs SOH -0.89.
  - Median SOH: A 0.93, B 0.77, C 0.58. Pairs ordered by SOH: A>B 99.8%, B>C 95%.
  - No C above 80% SOH; 1 of 87 steps improved with age.
  - Unseen Dubarry curves (73 labelled): severity vs true capacity loss Spearman 0.91, BUT dominant mode correct only 33% (0 of 20 when the true mode is LAM_NE), grade agreement 58%, per-mode R2 at most 0.09. The mechanism split does not hold up here.
  - Limits: same cell make and protocol as tuning cells; grade B spans a wide SOH range; LLI dominates almost everywhere, so this tests the grade more than the mechanism split; 25 degC check was not run (loader broke on a comma-decimal file).
Say: This is the fairest test we have; we report it as it came out.

### Slide 13: The app (live demo)
Figure: Streamlit screenshot [[PLACEHOLDER: screenshot]].
- Upload a partial discharge CSV (voltage + capacity, or current + time).
- Output: grade, confidence, LLI / LAM_PE / LAM_NE breakdown, dQ/dV plot against a fresh reference, warnings.
- Two paths chosen automatically: slow twin fit for C/25-like files (15 to 50 s), fast 4C feature model (under 1 s).
- Demo files: fair held-out = data/samples/heldout/HELDOUT_wheeler_*. NOT fair (tuning cell or synthetic) = data/samples/wheeler_6a_* and app/samples/*.
Say: Run `streamlit run app/streamlit_app.py`.

### Slide 14: Honest limitations (must appear in full)
- The grade rule was tuned on Wheeler after seeing it; the held-out test is the fair check.
- Dubarry curves are C/25 charge curves that we fitted as discharge, so those numbers carry that caveat.
- Graphite loss (LAM_NE) is a lower bound: the twin only notices it once graphite becomes the limiting electrode.
- Mid-life LLI and LAM_PE trade off: their sum is well determined, the split is not. Even adding charge curves on synthetic data did not separate them.
- One chemistry (LFP/graphite), small sample (7 Wheeler cells in tuning).
- At 4C the mechanism split is only indicative: many near-fresh Severson snapshots come out as mixed LAM, grade C.
- Grades are not safety-validated.
Say: We would rather state these than have them found.

### Slide 15: Novelty, conclusion, future work
- Novelty (to our knowledge): the combination of grading by fitted degradation mechanism instead of SOH bands, plus a no-refit test on a second dataset, plus an upload tool.
- Each part alone (ICA, half-cell twin, ML on synthetic curves, ICA clustering) is standard. Closest prior work: Beatty et al. 2026, Batteries 12:55, which clusters IC features including Severson but has no LLI/LAM split and no cross-dataset test.
- Conclusion: a mechanism-explained grade from one partial discharge.
- Future work: teardown validation, other chemistries, safety validation.
Say: Never say "first". Say "to our knowledge".

## RULES FOR YOU
1. Do not add facts, numbers, citations or datasets beyond this prompt.
2. Do not round or change numbers. Do not fill placeholders.
3. Keep the limitations slide complete; do not soften it.
4. Same visual format as the mid-review deck; plain, readable from the back of a room; grade colours must not be the only grade indicator (also print the letter).

## PROMPT END
