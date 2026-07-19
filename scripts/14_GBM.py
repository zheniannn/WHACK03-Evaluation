"""Stage 13 -- gradient-boosted trees (supervised ML).

Extract the 31 track features per candidate, train HistGradientBoosting on the
three training days (class-weighted for the ~0.7% positive rate), predict a
target-probability on the held-out day.

Writes  scores/gbm_<TEST_DATE>.csv  [track_id, score_gbm].
Usage:  python scripts/14_GBM.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from utils.io import TRAIN_DATES, TEST_DATE, get_method_score_path
from utils.data import load_points_by_track, load_track_meta
from utils.features import FEATURE_NAMES, track_features
from utils.ml import train_gbm, predict_gbm

SEED = 20220606


def features_for(date):
    """(track_ids, feature-matrix, labels) for one day."""
    tids, c, (starts, ends) = load_points_by_track(date)
    X = np.empty((len(tids), len(FEATURE_NAMES)), np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        f = track_features(c["est_e"][s:e], c["est_n"][s:e], c["est_range_m"][s:e],
                           c["snr_db"][s:e], c["nis"][s:e], c["miss"][s:e], c["scan_idx"][s:e])
        X[i] = [f[k] for k in FEATURE_NAMES]
    lab = load_track_meta(date).set_index("track_id").loc[tids, "label"].to_numpy()
    return tids, X, lab


def main() -> None:
    print("extracting features on training days ...")
    Xtr, ytr = [], []
    for d in TRAIN_DATES:
        _, X, y = features_for(d)
        Xtr.append(X); ytr.append(y)
    print("training GBM ...")
    gbm = train_gbm(np.vstack(Xtr), np.concatenate(ytr), SEED)

    tids, Xte, _ = features_for(TEST_DATE)
    score = predict_gbm(gbm, Xte)
    path = get_method_score_path("gbm", TEST_DATE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"track_id": tids, "score_gbm": score}).to_csv(path, index=False)
    print(f"GBM: {len(tids):,} tracks scored -> {path}")


if __name__ == "__main__":
    main()
