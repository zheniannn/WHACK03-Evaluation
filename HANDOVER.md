# WHACK03-Discrimination — Handover

Last updated: 2026-07-13. Status: **complete and runs end-to-end**, but the
results are over-framed as written — read "Known issues" before quoting any
number. This file is the honest state of the project, not a summary of the
README.

## 1. Where this sits

Three-repo pipeline, all sharing one data root (`data/` beside the repos,
or `$WHACK_DATA_ROOT`):

```
WHACK01-Preprocessing  ADS-B -> clean GA trajectories (stages 1-4)   [GitHub: main]
WHACK02-Radar          trajectories -> simulated 2D radar detections (stages 5-9)  [GitHub: main]
WHACK03-Discrimination detections -> candidate tracks -> discrimination (stages 10-12)  [local git only]
```

WHACK03 reads WHACK02's `active/radar/stage09/radar_detections_<date>.csv`
and writes under `active/discrimination/`.

## 2. Current state

- All three stages written, run on all 4 days, committed (`d4ebb39`).
- **Not pushed to GitHub.** No remote set. `gh` CLI is not installed;
  auth is SSH-only. To push: create the repo, then
  `git remote add origin git@github.com:zheniannn/WHACK03-Discrimination.git && git push -u origin master`
  (note: local default branch is `master`, unlike WHACK01/02 which are
  `main` — rename with `git branch -m master main` before first push if you
  want consistency).
- Outputs on disk (NOT in git — `data/` is gitignored):
  - `active/discrimination/tracks/` — `tracks_<date>.csv`, `track_points_<date>.csv`
  - `active/discrimination/scores/` — `scores_<date>.csv`
  - `active/discrimination/eval/evaluation_report.json`
  - `data/plot/stage12_*.png` (3 figures)

## 3. How to run (order matters)

```bash
python scripts/10_tracking.py         # ~5 min/day, ~20 min total; streams to disk
python scripts/11_discriminators.py   # ~12 min (feature extract + GBM + GRU + score)
python scripts/12_evaluation.py       # seconds; ROC/AUC + figures + report
```

Train days: 06-06/13/20. Held-out test day: **06-27** (`utils/io.py::TEST_DATE`).

## 4. Environment gotchas (these bit me — save yourself the time)

- **Memory: 7 GB total on this machine.** Stage 10 was OOM-killed at 7.2 GB
  on the first attempt because the tracker held all ~265k `Track` objects
  in RAM. Fixed by streaming completed tracks to disk in
  `tracker.run_day` (the `on_complete` callback). **Do not** revert to
  accumulating tracks in a list. Also: don't run a second memory-heavy
  Python process concurrently with stage 10.
- Stage 11 holds all 4 days' features+sequences in memory (~500 MB) — fine,
  but if days grow, cache to disk per day instead.
- `torch` is CUDA-built (`2.12+cu130`) but the GRU ran on CPU here; it's
  fast enough (~1 min/epoch on ~66k sequences). No GPU dependency.
- `lightgbm`/`xgboost` are NOT installed; the GBM uses sklearn's
  `HistGradientBoostingClassifier` (equivalent). Don't add a dependency.

## 5. Results as they stand (held-out 06-27, 263,571 tracks, 0.67% true)

| Method | AUC | AP | TPR @ baseline FPR |
|---|---|---|---|
| M-of-N baseline | — | — | 0.993 at FPR **0.409** |
| SPRT (LLR) | 0.975 | 0.596 | 0.975 |
| IPDA | 0.689 | 0.084 | 0.670 |
| Grad-boosted trees | 1.000 | 0.966 | 1.000 |
| GRU sequence | 1.000 | 0.949 | 1.000 |

Clutter-track survival at the baseline operating point (lower = better):
M-of-N 0.99, GRU 0.72, LLR 0.55, GBM 0.27, IPDA 0.23.

## 6. Known issues — READ BEFORE QUOTING RESULTS

Ranked by severity. These are real; an examiner will find them.

1. **AUC is the wrong headline metric** at 0.7% prevalence. Lead with AP or
   precision-at-recall / false-tracks-per-true-track. AUC 1.000 reads as a
   leak. (AP already computed and in the report JSON.)

2. **The tracker did most of the discrimination.** GNN association with a
   tight gate pre-filters noise; the label (>=60% detections from one real
   trajectory) correlates almost perfectly with track length (`n_det`,
   `span_scans`), which is a feature. So the ML mostly learns "long =
   real," which is why AUC pins to 1.0. Reframe around the HARD subset
   (persistent clutter chains vs short/gappy true tracks) where length does
   not give the answer away. **The clutter-survival result is the least
   circular and most defensible — make it the headline.**

3. **Positive-class selection bias.** A true track exists only if >=3 target
   detections formed one candidate. Distant low-Pd aircraft that never form
   a clean track are absent from the positive set, so every method is
   evaluated on the true tracks that were easy to build. TODO: report, per
   range bin, how many real trajectories in coverage produced NO labelable
   track (the tracker's miss, currently invisible).

4. **Held-out day is weakly independent.** Same radar, same airspace, and
   the 25 clutter patches are at IDENTICAL positions across all 4 days
   (frozen in scenario.json). Features are mostly translation-invariant, so
   patch-location memorization is limited — but verify via GBM feature
   importances (confirm range/position aren't top features) and disclose.

5. **Matched-model optimism + single seed.** SPRT/IPDA use the exact
   Pd/Swerling likelihoods that generated the data. And it's ONE Monte-Carlo
   realization — no error bars. WHACK02 stage 9 supports `--seed`; run ~5
   realizations, report mean +/- std.

6. **IPDA is likely mis-implemented.** AP 0.084 is too low for a method
   built for this. `classical.ipda_existence` approximates PDA with the
   single GNN-associated measurement per scan, breaking IPDA's clutter
   model. Either implement it properly (all gated measurements, its own
   association) or drop the "classical SOTA" claim.

7. **The stated thesis isn't directly tested.** The problem statement is
   about the aircraft's physically plausible path. The WHACK01 stage-4 GA
   motion prior (measured speed/turn/accel distributions) is never
   explicitly injected as a likelihood — it's only implicit in hand-picked
   features. The cleanest contribution would be an explicit motion-prior
   likelihood-ratio test.

## 7. Recommended next steps (in priority order)

1. **Re-run stage 12 with AP-based figures** and a hard-subset evaluation
   (exclude length-trivial negatives; evaluate at a fixed LOW FPR, not the
   baseline's 0.409). The current `stage12_recovery_vs_range.png` is weak
   because it evaluates at FPR 0.409 where everything is recovered.
2. **Add a multi-seed loop** (regenerate WHACK02 stage 9 with 5 seeds; rerun
   10-12; report variance).
3. **Fix IPDA** or remove it from the SOTA comparison.
4. **Report tracker recall** (real trajectories with no candidate track) per
   range bin — closes the selection-bias gap.
5. **Add GBM feature-importance figure** (`plots.plot_importance` exists but
   is not called by stage 12) to show what the model actually uses and
   check for scenario leakage.
6. Optional: explicit GA-motion-prior likelihood test to directly realize
   the thesis.

## 8. Code map

- `utils/tracker.py` — KF + GNN + streaming labelling. Params at top
  (DT, SIGMA_A, GATE_CHI2, MAX_COAST=5, MIN_KEEP=3, PURITY_THRESHOLD=0.6).
- `utils/classical.py` — M-of-N, SPRT LLR, IPDA. `Physics` computes Pd(range),
  Pfa, clutter density from the scenario.
- `utils/features.py` — 31 track features (FEATURE_NAMES) + per-scan sequence.
- `utils/ml.py` — GBM (sklearn HGB) + GRU (torch). Negative subsample =
  80k/train set for the GRU.
- `utils/plots.py` — ROC, AUC-vs-range, recovery-vs-range, importance.
- `utils/io.py` — paths, DATES, TEST_DATE, TRAIN_DATES.

The knobs that most shape the results: MIN_KEEP, MAX_COAST, GATE_CHI2
(tracker → candidate population), PURITY_THRESHOLD (labels), and the range
bin edges in `12_evaluation.py`. None have been sensitivity-tested.
