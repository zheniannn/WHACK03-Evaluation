"""Stage 16 -- one-class Trajectory-VAE.

Train a sequence VAE on WHACK01 real GA motion ([speed, turn] windows) ONLY --
it never sees the tracker's labels. Score each candidate track by its
reconstruction error (higher score = lower error = motion on the real-flight
manifold). Because the input is speed+turn (translation/rotation-invariant),
the VAE carries no track-length or position leak.

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
from utils.motion import (DEVICE, load_real_motion, windows_from_seq, train_vae,
                          load_track_points, track_speed_turn)


def main() -> None:
    print("training VAE on real GA motion ...")
    seqs = load_real_motion(TRAIN_DATES)
    wins = np.concatenate([windows_from_seq(sp, tr) for sp, tr in seqs])
    rng = np.random.default_rng(0)
    if len(wins) > 300_000:
        wins = wins[rng.choice(len(wins), 300_000, replace=False)]
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


if __name__ == "__main__":
    main()
