"""Shared loaders for the discrimination scripts.

`load_points_by_track` returns the stage-10 per-scan track points grouped for
O(1) per-track slicing (all scans, detected + coasted) -- used by the classical
and supervised-ML scripts. Motion methods use `utils.motion.load_track_points`
(detected scans only). `load_track_meta` gives per-track labels / metadata.
"""
import numpy as np
import pandas as pd

from utils.io import get_track_points_path, get_tracks_path

POINT_COLS = ["scan_idx", "est_e", "est_n", "est_range_m", "snr_db", "nis", "logdet_s", "miss"]


def load_points_by_track(date):
    """(track_ids, cols_dict, (starts, ends)) — slice arrays with cols[c][s:e].
    track_points sorted by (track_id, scan_idx)."""
    pts = pd.read_csv(get_track_points_path(date),
                      usecols=["track_id"] + POINT_COLS).sort_values(
        ["track_id", "scan_idx"]).reset_index(drop=True)
    tid = pts["track_id"].to_numpy()
    bounds = np.flatnonzero(np.diff(tid)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(tid)]))
    cols = {c: pts[c].to_numpy() for c in POINT_COLS}
    return tid[starts], cols, (starts, ends)


def load_track_meta(date):
    """tracks table (label, track_source, n_det, range_median_m) per track_id."""
    return pd.read_csv(get_tracks_path(date),
                       usecols=["track_id", "label", "track_source", "n_det", "range_median_m"])
