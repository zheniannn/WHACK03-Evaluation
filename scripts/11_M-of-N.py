"""Stage 11 -- M-of-N (classical baseline, no ML, no training).

Confirm a candidate track if it holds >= M detections in any window of N scans
(M=3, N=5). Pure track-confirmation logic -- what threshold-only tracking gives
you, and the operating point every other method must beat.

Writes  scores/mofn_<TEST_DATE>.csv  [track_id, score_mofn].
Usage:  python scripts/11_M-of-N.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from utils.io import TEST_DATE, get_method_score_path
from utils.data import load_points_by_track
from utils.classical import mofn_confirmed


def main() -> None:
    tids, cols, (starts, ends) = load_points_by_track(TEST_DATE)
    scan, miss = cols["scan_idx"], cols["miss"]
    score = np.array([mofn_confirmed(scan[s:e], miss[s:e]) for s, e in zip(starts, ends)])

    path = get_method_score_path("mofn", TEST_DATE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"track_id": tids, "score_mofn": score}).to_csv(path, index=False)
    print(f"M-of-N: {len(tids):,} tracks, {int(score.sum()):,} confirmed -> {path}")


if __name__ == "__main__":
    main()
