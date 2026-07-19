"""Stage 14 -- GRU sequence model (supervised deep learning).

Read each track's per-scan sequence, train a small recurrent net on the three
training days (false tracks subsampled, positives up-weighted), predict a
target-probability on the held-out day.

Writes  scores/gru_<TEST_DATE>.csv  [track_id, score_gru].
Usage:  python scripts/15_GRU.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from utils.io import TRAIN_DATES, TEST_DATE, get_method_score_path
from utils.data import load_points_by_track, load_track_meta
from utils.features import track_sequence
from utils.ml import train_gru, predict_gru

NEG_SUBSAMPLE = 80_000
SEED = 20220606


def seqs_for(date):
    tids, c, (starts, ends) = load_points_by_track(date)
    seqs = [track_sequence(c["est_e"][s:e], c["est_n"][s:e], c["est_range_m"][s:e],
                           c["snr_db"][s:e], c["miss"][s:e], c["scan_idx"][s:e])
            for s, e in zip(starts, ends)]
    lab = load_track_meta(date).set_index("track_id").loc[tids, "label"].to_numpy()
    return tids, seqs, lab


def main() -> None:
    rng = np.random.default_rng(SEED)
    tr_seqs, tr_y = [], []
    for d in TRAIN_DATES:
        _, seqs, lab = seqs_for(d)
        pos = np.flatnonzero(lab == 1)
        neg = rng.choice(np.flatnonzero(lab == 0),
                         min(NEG_SUBSAMPLE, int((lab == 0).sum())), replace=False)
        for i in np.concatenate([pos, neg]):
            tr_seqs.append(seqs[i]); tr_y.append(lab[i])
    print(f"training GRU on {len(tr_seqs):,} sequences ...")
    gru = train_gru(tr_seqs, np.array(tr_y), tr_seqs[0].shape[1], seed=SEED)

    tids, seqs, _ = seqs_for(TEST_DATE)
    score = predict_gru(gru, seqs)
    path = get_method_score_path("gru", TEST_DATE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"track_id": tids, "score_gru": score}).to_csv(path, index=False)
    print(f"GRU: {len(tids):,} tracks scored -> {path}")


if __name__ == "__main__":
    main()
