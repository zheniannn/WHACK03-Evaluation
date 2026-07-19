"""Stage 13 -- IMM cv-fraction (classical motion, no training).

A constant-velocity + coordinated-turn Kalman-filter bank with Markov mode
switching, run over each track's position estimates. Score = the fraction of
scans spent in the constant-velocity mode: a real flight cruises coherently in
one mode, clutter/noise thrash between modes.

Writes  scores/imm_<TEST_DATE>.csv  [track_id, score_imm].
Usage:  python scripts/13_IMM.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from utils.io import TEST_DATE, get_method_score_path
from utils.motion import load_track_points, IMM
from utils.data import report_gated_f1


def main() -> None:
    pts = load_track_points(TEST_DATE)          # detected scans, per track
    imm = IMM()
    tids = list(pts.keys())
    score = [imm.score(pts[t]["e"], pts[t]["n"], pts[t]["rng"], pts[t]["scan"])["cv_frac"]
             for t in tids]

    path = get_method_score_path("imm", TEST_DATE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"track_id": tids, "score_imm": score}).to_csv(path, index=False)
    print(f"IMM: {len(tids):,} tracks scored -> {path}")
    report_gated_f1(tids, score, TEST_DATE, label="IMM")


if __name__ == "__main__":
    main()
