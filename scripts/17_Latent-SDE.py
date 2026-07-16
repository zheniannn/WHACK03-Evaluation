"""Stage 17 -- Latent neural SDE (one-class, continuous-time).

The gap-aware version of the VAE: a latent SDE (learned drift + diffusion) rolled
with Euler-Maruyama over the actual per-scan dt, trained on WHACK01 real GA
motion with gap augmentation. Score = reconstruction error (higher = target-like).
Handling missed scans as real dt is its edge over the VAE on long/gappy tracks.

Writes  scores/sde_<TEST_DATE>.csv  [track_id, score_sde].
Usage:  python scripts/17_Latent-SDE.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from utils.io import TRAIN_DATES, TEST_DATE, get_method_score_path
from utils.motion import (DEVICE, load_real_motion, build_sde_training_windows,
                          train_latent_sde, load_track_points, track_speed_turn, windows_with_dt)

REPS = 3


def main() -> None:
    print("training latent SDE on real GA motion (gap-augmented) ...")
    seqs = load_real_motion(TRAIN_DATES)
    X, D = build_sde_training_windows(seqs, per_traj=6)
    rng = np.random.default_rng(0)
    if len(X) > 300_000:
        sel = rng.choice(len(X), 300_000, replace=False); X, D = X[sel], D[sel]
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


if __name__ == "__main__":
    main()
