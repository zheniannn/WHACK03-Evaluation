"""Stage 18 -- Latent neural SDE (one-class, continuous-time).

The gap-aware version of the VAE: a latent SDE (learned drift + diffusion) rolled
with Euler-Maruyama over the actual per-scan dt. Trained on CLEAN TARGET TRACKS
(the tracker run on target-only detections; KF-smoothed, no clutter/noise), using
each track's REAL per-scan dt from its Pd gaps -- so training matches the scoring
domain (closes the ADS-B train/test gap, like stage 16) AND exercises the SDE's
gap-awareness on the real gap distribution rather than synthetic augmentation.
Score = reconstruction error (higher = target-like).

Writes  scores/sde_<TEST_DATE>.csv  [track_id, score_sde].
Usage:  python scripts/18_Latent-SDE.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from utils.io import TRAIN_DATES, TEST_DATE, get_method_score_path
from utils.motion import (DEVICE, load_clean_target_motion, train_latent_sde,
                          load_track_points, track_speed_turn, windows_with_dt)
from utils.data import report_gated_f1

REPS = 3


def main() -> None:
    print("training latent SDE on clean target tracks (real per-scan dt) ...")
    seqs = load_clean_target_motion(TRAIN_DATES, with_scan=True)
    Xs, Ds = [], []
    for sp, tr, scan in seqs:
        w, d = windows_with_dt(sp, tr, scan=scan)   # real dt from the track's scan gaps
        Xs.append(w); Ds.append(d)
    X = np.concatenate(Xs); D = np.concatenate(Ds)
    rng = np.random.default_rng(0)
    if len(X) > 300_000:
        sel = rng.choice(len(X), 300_000, replace=False); X, D = X[sel], D[sel]
    print(f"  {len(seqs):,} clean target tracks -> {len(X):,} windows")
    sde = train_latent_sde(X, D, epochs=12)

    print("scoring candidate tracks ...")
    pts = load_track_points(TEST_DATE)
    tids = list(pts.keys())
    all_w, all_d, owner = [], [], []
    for i, t in enumerate(tids):
        sp, tr = track_speed_turn(pts[t]["e"], pts[t]["n"], pts[t]["scan"])
        w, d = windows_with_dt(sp, tr, scan=pts[t]["scan"])
        all_w.append(w); all_d.append(d); owner.append(np.full(len(w), i))
    owner = np.concatenate(owner); W = np.concatenate(all_w); Dd = np.concatenate(all_d)
    errs = np.zeros(len(W), np.float64)
    with torch.no_grad():
        for _ in range(REPS):
            for b in range(0, len(W), 8192):
                rec, _, _ = sde(torch.from_numpy(W[b:b + 8192]).to(DEVICE),
                                torch.from_numpy(Dd[b:b + 8192]).to(DEVICE))
                errs[b:b + 8192] += rec.cpu().numpy()
    errs /= REPS
    serr = np.zeros(len(tids)); cnt = np.zeros(len(tids))
    np.add.at(serr, owner, errs); np.add.at(cnt, owner, 1)
    serr /= np.maximum(cnt, 1)

    path = get_method_score_path("sde", TEST_DATE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"track_id": tids, "score_sde": -serr}).to_csv(path, index=False)
    print(f"Latent-SDE: {len(tids):,} tracks scored -> {path}")
    report_gated_f1(tids, -serr, TEST_DATE, label="Latent-SDE")


if __name__ == "__main__":
    main()
