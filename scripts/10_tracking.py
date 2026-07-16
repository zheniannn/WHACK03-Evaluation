"""Stage 10: form candidate tracks from WHACK02 stage-9 detections with a
permissive multi-target tracker (converted-measurement KF + GNN
association), then label each candidate true/false via truth linkage.

Outputs per day:
  tracks_<date>.csv        -- one row per candidate track (+ label)
  track_points_<date>.csv  -- per-scan estimated state (input to the sequence model)

Usage:
    python scripts/10_tracking.py
    python scripts/10_tracking.py --dates 2022-06-06
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import (
    DATES,
    get_detections_path,
    get_scenario_path,
    get_track_points_path,
    get_tracks_path,
)
from utils.tracker import run_day


def parse_args():
    p = argparse.ArgumentParser(description="Stage 10: candidate-track formation.")
    p.add_argument("--dates", nargs="*", default=DATES, help="Dates to process (default: all four).")
    p.add_argument("--stage", type=int, default=9, help="WHACK02 detection stage to read (default 9).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(get_scenario_path()) as f:
        sc = json.load(f)
    sigma_r, sigma_az = sc["sigma_range_m"], sc["sigma_azimuth_deg"]

    for date in args.dates:
        det_path = get_detections_path(date, args.stage)
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Detections not found: {det_path} (run WHACK02 stage 9)")
        det = pd.read_csv(det_path, dtype={"trajectory_id": str, "icao24": str})

        n_det = len(det)
        os.makedirs(os.path.dirname(get_tracks_path(date)), exist_ok=True)
        tracks = run_day(date, det, sigma_r, sigma_az, get_track_points_path(date))
        tracks.to_csv(get_tracks_path(date), index=False)
        del det

        n = len(tracks)
        n_true = int(tracks["label"].sum())
        print(f"--- {date} ---")
        print(f"detections in:      {n_det}")
        print(f"candidate tracks:   {n}  ({n_true} true, {n - n_true} false)")
        print(f"true-track purity:  median {tracks.loc[tracks['label']==1,'purity'].median():.2f}")
        print(f"output: {get_tracks_path(date)}")

    print("\n10_tracking completed.")


if __name__ == "__main__":
    main()
