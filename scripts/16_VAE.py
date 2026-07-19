"""Stage 16 -- one-class Trajectory-VAE (domain-matched training).

Train a sequence VAE on [speed, turn] windows of CLEAN TARGET TRACKS -- real
aircraft detections (source=='target', no clutter/noise) run through the same
tracker that forms the candidate tracks, so training and scoring share the
KF-smoothed radar/tracker domain. Score each candidate track by its
reconstruction error (higher score = lower error = motion on the real-flight
manifold). The VAE never sees the tracker's labels; the input is speed+turn
(translation/rotation-invariant), so it carries no track-length or position leak.

Why not ADS-B motion: an ADS-B-trained VAE penalises real aircraft for the
measurement/KF jitter it never saw at training time (a train/test domain gap).
Training on clean target tracks in the tracker domain closes that gap and lifts
the honest clutter-vs-aircraft AUC from ~0.949 to ~0.969. (See TESTTEST/.)

Writes  scores/vae_<TEST_DATE>.csv  [track_id, score_vae].
Usage:  python scripts/16_VAE.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch

from utils.io import TRAIN_DATES, TEST_DATE, get_method_score_path
from utils.motion import (DEVICE, windows_from_seq, train_vae, load_clean_target_motion,
                          load_track_points, track_speed_turn)
from utils.data import report_gated_f1


def main() -> None:
    print("training VAE on clean target tracks (tracker domain) ...")
    seqs = load_clean_target_motion(TRAIN_DATES)
    wins = np.concatenate([windows_from_seq(sp, tr) for sp, tr in seqs])
    rng = np.random.default_rng(0)
    if len(wins) > 300_000:
        wins = wins[rng.choice(len(wins), 300_000, replace=False)]
    print(f"  {len(seqs):,} clean target tracks -> {len(wins):,} windows")
    vae = train_vae(wins, epochs=10)

    print("scoring candidate tracks ...")
    pts = load_track_points(TEST_DATE)
    tids = list(pts.keys())
    all_w, owner = [], []
    for i, t in enumerate(tids):
        sp, tr = track_speed_turn(pts[t]["e"], pts[t]["n"], pts[t]["scan"])
        w = windows_from_seq(sp, tr)
        all_w.append(w); owner.append(np.full(len(w), i))
    owner = np.concatenate(owner); W = np.concatenate(all_w)
    errs = np.empty(len(W), np.float32)
    with torch.no_grad():
        for b in range(0, len(W), 8192):
            errs[b:b + 8192] = vae.recon_error(torch.from_numpy(W[b:b + 8192]).to(DEVICE)).cpu().numpy()
    verr = np.zeros(len(tids)); cnt = np.zeros(len(tids))
    np.add.at(verr, owner, errs); np.add.at(cnt, owner, 1)
    verr /= np.maximum(cnt, 1)

    path = get_method_score_path("vae", TEST_DATE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"track_id": tids, "score_vae": -verr}).to_csv(path, index=False)
    print(f"VAE: {len(tids):,} tracks scored -> {path}")
    report_gated_f1(tids, -verr, TEST_DATE, label="Traj-VAE")


if __name__ == "__main__":
    main()
