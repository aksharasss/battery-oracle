# Handover prompt: Battery Oracle 2.0 (for Antigravity or any coding assistant)

## Part 1. What to upload or open with this prompt

1. **Open the repo folder** (best: clone fresh, see Part 2 "Getting the code"). The repo holds the code, models, samples and docs.
2. **Datasets, only if you will re-run validation**: `data/raw/` and `data/processed/` from Latent's PC copy
   (`C:\Users\aksha\OneDrive\Desktop\Sem-3\Maths\project\battery-oracle`). About 2 GB, gitignored. Not needed for UI work.
3. **If it exists on the PC:** any uncommitted Dubarry "charge mode" code from the branch `claude/project-thread-tn9elr` (it was never pushed to GitHub).
4. **Optional, for the report and slides:** `docs/PROJECT_GUIDE.md`, `docs/SLIDES_PROMPT.md` (on branch `claude/project-thread-lbuayo`), and app screenshots `desktop.png`, `desktop-light.png`.
5. **Optional:** export of the final report draft (Claude Doc, https://claude.ai/artifact/JK3kB9coZJpe9YKHbmYYam) and the related-work doc (https://claude.ai/artifact/KCUL4X4Gb9wdUf5X6AZT7r). Antigravity cannot open these links, so paste or export them.

Then paste everything below the line into the assistant.

---

## Part 2. Context prompt

You are taking over **Battery Oracle 2.0**, an academic maths project (end-semester review) by Latent. Read this, then verify everything against the code before you change anything.

### Goal
Grade a retired **LFP/graphite** cell for second life from **one partial discharge**. Incremental Capacity Analysis (dQ/dV) gives a physical fingerprint; a mechanistic half-cell model estimates **LLI** (lithium inventory loss), **LAM_PE** (cathode loss) and **LAM_NE** (graphite loss), and a rule maps that mix to **Grade A** (stationary storage), **B** (light duty) or **C** (recycle). The point is a grade explained by mechanism, not a black-box remaining-life prediction. Public data only; no own test rig.

### Agreed plan
1. ICA pipeline on public LFP data. 2. Synthetic half-cell model as stand-in ground truth. 3. Unsupervised grading built on Severson, cross-validated on another LFP set with no refit. 4. Streamlit app: upload a partial discharge, get grade + mechanism breakdown + dQ/dV plot; final report. All four are built; the open items are at the end.

### Getting the code
Repo: https://github.com/aksharasss/battery-oracle (public). `main` is at `0848e1b`.
Two unmerged branches matter:
- `claude/project-thread-wc6dvm`: the **held-out test** (`scripts/heldout_test.py`, `docs/heldout_protocol.md`, results in `data/processed/heldout/`, demo files `data/samples/heldout/HELDOUT_wheeler_*_C25.csv` (12 files), README section). Only adds files; no change to `oracle/` or `models/`.
- `claude/project-thread-lbuayo`: adds `docs/PROJECT_GUIDE.md` and `docs/SLIDES_PROMPT.md`.

```bash
git clone https://github.com/aksharasss/battery-oracle && cd battery-oracle
git fetch origin
git merge origin/claude/project-thread-wc6dvm   # held-out test
git merge origin/claude/project-thread-lbuayo   # guide + slide prompt
git push origin main                            # only once Latent agrees
```
(Or just `git checkout claude/project-thread-wc6dvm` to look first.)

### Layout (verify by reading the files)
- `oracle/schema.py`: `DischargeCurve`, the common format all loaders produce.
- `oracle/loaders/`: `severson.py` (MIT/Stanford .mat, 4C), `wheeler.py` (Wheeler 2025 C/25 checkups), `batteryarchive.py`, `calce.py` (stub), `upload.py` (single CSV from the UI).
- `oracle/ica.py`: window, uniform Q(V) grid, Savitzky-Golay dQ/dV. `oracle/features.py`: peak position/height/area/FWHM, presets per dataset.
- `oracle/synth.py`, `synth_check.py`: synthetic graphite/LFP half-cell model with chosen LLI/LAM_PE/LAM_NE.
- `oracle/twin.py`: mechanistic digital twin; fits the half-cell model directly to a measured slow (C/25) curve.
- `oracle/mechanisms.py`: random-forest mechanism model calibrated on synthetic 4C curves (fast path).
- `oracle/grading.py`: `grade_of` rule, PCA + GMM structure, `Reference`.
- `oracle/predict.py`: `predict(file, model="auto") -> GradeResult`. Picks **slow path** (twin, 15-50 s) if the tallest peak is above 3.25 V and narrower than 25 mV, else **fast path** (4C forest, under 1 s). Sets `features["path"]`.
- `oracle/validate.py`: cross-dataset validation helpers.
- `app/streamlit_app.py`: the UI. `app/make_samples.py` + `app/samples/*.csv` are 3 **synthetic** samples; `app/stub_predict.py` is an old stub.
- `scripts/`: `fetch_data.py`, `build_features.py`, `synth_features.py`, `calibrate_mechanisms.py`, `fit_grading.py`, `fit_twin_reference.py`, `twin_wheeler.py`, `validate*.py`, and (branch) `heldout_test.py`. Rebuild commands are in `README.md`.
- `models/`: committed models (`twin_slow.joblib`, `mechanisms_*.joblib/json`, `grading_severson.joblib`).
- `data/samples/`: Severson 4C and Wheeler 6a C/25 demo files. `docs/`: `phase2_synthetic.md`, figures.
- `tests/`: `test_ica.py`, `test_synth.py`, `test_predict.py`, `test_app_contract.py`.

### Setup and run
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest                                  # 28 passed at the last run; re-run and report the real count
streamlit run app/streamlit_app.py
```

### UI design decisions (keep unless Latent says otherwise)
- Graph-paper "notebook sheet" look, with **dark and light** themes (commit `0848e1b`).
- Design context lives in `PRODUCT.md` (users = examiners in a live demo; they must see the grade and why within seconds), `DESIGN.md`, `.impeccable/` (design tokens and the app surface notes) and `.streamlit/config.toml`. Read these before changing the UI.
- `tests/test_app_contract.py` pins what the app expects from `predict()`; keep it passing.

### Frozen rule: do not change silently
The grade rule and models are **frozen at `0848e1b`** for the held-out test: `grading.grade_of` with FRESH_TOTAL 0.15, MODERATE 0.20, SEVERE 0.40, LAM_NE_MINOR 0.2, and `models/twin_slow.joblib`. Rule in words: total loss < 0.15 is A; severity ≥ 0.40 is C, ≥ 0.20 caps at B; otherwise LLI-dominant A, one LAM mode B, mixed C, except LLI + LAM_PE with graphite < 20% of the loss is B (approved 2026-10-27).
**Do not change `oracle/grading.py`, `oracle/twin.py`, `oracle/synth.py`, `oracle/predict.py` logic or anything in `models/` without telling Latent first.** Any later change must be reported as "post-freeze" and the held-out numbers no longer apply to it. UI-only changes are fine.

### Key results (from README; quote them, don't improve them)
- Severson 4C: graphite peaks merge into one broad peak near 3.15 V, so the mechanism split at 4C is indicative only; many near-fresh cells come out mixed LAM / grade C.
- Wheeler (tuning cells 1b 2b 3b 4b 5a 6a 7a): twin fit median rms 0.011; severity vs capacity Spearman −0.84; grades never improve with age.
- **Held-out Wheeler** (unseen cells 1a 2a 3a 4a 5b 6b, 93 checkups, app path): severity vs SOH Spearman −0.96, grade vs SOH −0.89; median SOH A 0.93 / B 0.77 / C 0.58; no C above 80% SOH. Same make and protocol as tuning cells; LLI dominates, so this tests the grade more than the mechanism split.
- **Held-out Dubarry** (synthetic, files 2 and 3, 73 curves): severity Spearman 0.91, but dominant mode correct only 33% (0/20 for LAM_NE), grade agreement 58%, per-mode R² ≤ 0.09. **The mechanism split does not hold up there.**

### Honest caveats (keep them in any report or UI text)
- LAM_NE (graphite loss) is a **lower bound**: the twin only sees it once graphite limits the cell.
- Mid-life LLI and LAM_PE trade off; their sum is reliable, the split is not.
- Dubarry curves are C/25 **charge** curves that were treated as discharges (fresh-fit rms 0.029 vs 0.011 on Wheeler).
- Grades are not safety-validated. Novelty claims: say "to our knowledge", never "first". Closest prior work: Beatty et al. 2026, Batteries 12:55.
- Fair demo uploads are `data/samples/heldout/HELDOUT_wheeler_*`. `data/samples/wheeler_6a_*` is a tuning cell and `app/samples/*` is synthetic; say so if they are shown.

### Datasets (gitignored, ~2 GB, on Latent's PC under `data/raw/`, `data/processed/`)
Severson 2017-06-30 batch (data.matr.io); Wheeler et al. 2025, Sci Data 12:392, data doi 10.57745/OLBXKT (C/25 checkups pulled by `scripts/fetch_data.py wheeler-checkups`); Dubarry synthetic graphite//LFP, Mendeley doi 10.17632/bs2j56pn7y. Sandia raw files are no longer downloadable.

### Open to-do list (in this order unless Latent says otherwise)
1. **UI decisions pending (ask Latent):** remove or rename the 3 synthetic samples (`app/samples/`, labelled "Synthetic slow-rate: ..." in `streamlit_app.py`); choose the default sample (suggest a `HELDOUT_wheeler_*` file once the held-out branch is merged); optional trajectory view (grade/mechanisms across a cell's checkups).
2. **25 °C check not run:** a file with comma decimals broke the loader. Fix the loader (accept `,` decimals) and run set 3 of `docs/heldout_protocol.md`. Loader fixes are harness changes, not rule changes, but report them.
3. **Dubarry charge mode:** a twin "charge" mode fitting hysteresis (v_hyst ≈ 22 mV) brought Dubarry fresh rms from 0.029 to 0.013. The code was left uncommitted on branch `claude/project-thread-tn9elr` on the PC; recover or re-implement it, and report as post-freeze.
4. **Graphite-loss and mechanism improvements** (LAM_NE detection, LLI/LAM_PE split). Post-freeze; report separately from held-out results.
5. **Report sections:** fill the cross-dataset section, abstract and conclusion with the held-out numbers above; check references.
6. **Last step:** extra cross-validation on LFP cells from other makers (the teacher's main suggestion is cross-validation across different batteries).

### Rules for you
- **Verify against the code** before stating or changing anything; this prompt can be out of date. If code and prompt disagree, trust the code and say so.
- **Keep claims honest:** report numbers exactly as the scripts produce them, including failures. Don't invent results, citations or datasets.
- **Never change grading rules, thresholds or models without saying so** (see "Frozen rule").
- **Commit and push after each finished change**, one change per commit, with a clear message. Never commit `data/raw/`, `data/processed/` bulk data, or files over 50 MB. OneDrive can lock `.git` files; retry if git errors.
- Run `pytest` before each commit and say the real result.
