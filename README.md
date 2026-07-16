# WHACK03-Evaluation

Track-based **target-vs-clutter discrimination** on the simulated radar
detections from [WHACK02-Radar](https://github.com/zheniannn/WHACK02-Radar). At a
low CFAR threshold the detection stream is ~99% false alarms and clutter; this
repo forms candidate tracks and then scores each one *"how target-like"* with
seven discriminators — a classical baseline, two more classical methods, two
supervised-ML methods, and two label-free one-class motion priors — evaluated on
a held-out day.

This tests the central claim of the problem statement: *a valid target follows a
physically plausible path and exhibits consistent motion across scans*, so motion
analysis over time should separate real aircraft from clutter and noise that
single-scan detection cannot.

## Structure

Same layout as WHACK01/WHACK02: `scripts/` are numbered pipeline stages,
`utils/` holds the shared modules, data resolves relative to the repo.

```
WHACK03-Evaluation/
├── requirements.txt
├── scripts/
│   ├── 10_tracking.py       # detections -> candidate tracks + labels (KF + GNN)
│   ├── 11_M-of-N.py         # classical baseline (M-of-N confirmation)
│   ├── 12_SPRT.py           # classical: sequential log-likelihood ratio
│   ├── 13_GBM.py            # supervised ML: gradient-boosted trees (31 features)
│   ├── 14_GRU.py            # supervised ML: recurrent sequence model
│   ├── 15_IMM.py            # classical motion: IMM cv-fraction
│   ├── 16_VAE.py            # one-class: Trajectory-VAE (motion manifold)
│   ├── 17_Latent-SDE.py     # one-class: latent neural SDE (gap-aware)
│   └── 18_Evaluation.py     # compare all methods (P/R/F1, AUC, clutter survival)
└── utils/
    ├── io.py                # paths, DATES, train/test split
    ├── data.py              # per-track point loaders
    ├── tracker.py           # converted-measurement KF + GNN association + labelling
    ├── classical.py         # M-of-N, SPRT
    ├── features.py          # 31 track features + per-scan sequence
    ├── ml.py                # gradient-boosted trees + GRU
    ├── motion.py            # IMM + one-class VAE + Latent SDE
    └── plots.py             # ROC / range figures
```

## Requirements

Python ≥ 3.10 with numpy, pandas, scipy, scikit-learn, torch, matplotlib:

```bash
pip install -r requirements.txt
```

## Data layout

Reads WHACK02's stage-9 detections; writes under `active/discrimination/`. Data
root defaults to `data/` next to the repo (override `WHACK_DATA_ROOT`).

```
<data root>/active/
├── radar/stage09/                 # WHACK02 output (this repo's input)
├── trajectories_10s/              # WHACK01 real GA motion (VAE/SDE training)
└── discrimination/
    ├── tracks/                    # stage 10: tracks + per-scan points
    ├── scores/                    # stages 11-17: <method>_<date>.csv, + all_methods_<date>.csv
    └── eval/                      # stage 18: evaluation_report.json
```

## Usage

```bash
python scripts/10_tracking.py        # detections -> candidate tracks (all four days)
# each method writes scores/<method>_2022-06-27.csv for the held-out day:
python scripts/11_M-of-N.py
python scripts/12_SPRT.py
python scripts/13_GBM.py
python scripts/14_GRU.py
python scripts/15_IMM.py
python scripts/16_VAE.py
python scripts/17_Latent-SDE.py
python scripts/18_Evaluation.py      # merge + compare -> scorecard + report
```

Days 06-06/13/20 train; **06-27 is held out** (`utils/io.py::TEST_DATE`).

## The seven methods

| Stage | Method | Type | How it decides |
|---|---|---|---|
| 11 | **M-of-N** | classical | confirmed iff ≥3 detections in some 5-scan window |
| 12 | **SPRT** | classical | sequential target-vs-clutter log-likelihood ratio, using Pd(range) |
| 13 | **GBM** | supervised ML | HistGradientBoosting on 31 track features |
| 14 | **GRU** | supervised ML | recurrent net on the per-scan sequence |
| 15 | **IMM** | classical motion | CV + coordinated-turn Kalman bank; score = constant-velocity mode fraction |
| 16 | **Traj-VAE** | one-class ML | reconstruction error under a VAE trained only on real GA motion |
| 17 | **Latent-SDE** | one-class ML | reconstruction error under a gap-aware latent neural SDE |

Classical scores use only the tracker's estimated range and the scenario's own Pd
model; the one-class methods train only on real WHACK01 motion — **none reads the
tracker's labels** (used only for evaluation in stage 18).

## Notes

- `HANDOVER.md` documents the original study's known methodology caveats — read it
  before quoting numbers (AUC is misleading at 0.7% prevalence; the tracker's
  length prior does much of the work; scores span two 8 dB scenario realizations).
- An earlier **IPDA** discriminator was dropped: its existence probability
  anti-correlated with track length and existence is weakly matched to
  aircraft-vs-clutter. The classical family is now M-of-N + SPRT.
- Exploratory extensions (flow/ensemble, SNR/RCS channel, Tri-PC, CFDR) and the
  documented negatives live separately in `../SPAM/`, which imports `utils.motion`
  and `utils.tracker` from here.
```
