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


def report_gated_f1(track_ids, score, date, gate=12, label="method"):
    """Print a per-track score's best-F1 at a length gate (recall vs ALL true
    tracks). Lets a raw motion stage self-report its realistic operating point
    (~0.6 gated) instead of only the ungated full-population F1 (~0.01) the
    stage-19 scorecard shows -- the ungated number is dominated by the ~94%
    short-noise false tracks a pure motion prior cannot reject."""
    meta = load_track_meta(date).set_index("track_id").reindex(np.asarray(track_ids))
    y = (meta["label"].to_numpy() == 1).astype(int)
    nd = meta["n_det"].to_numpy(float)
    s = np.where(np.isfinite(score), score, -1e18).astype(float)
    s = np.where(nd >= gate, s, -1e18)
    P = max(int(y.sum()), 1)
    o = np.argsort(-s)
    tp = np.cumsum(y[o] == 1)
    k = np.arange(1, len(s) + 1)
    prec = tp / k
    rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    print(f"  {label} gated best-F1 @ n_det>={gate}: F1 {f1[i]:.3f}  (P {prec[i]:.3f}  R {rec[i]:.3f})  "
          f"[stage-19 scorecard shows the ungated full-population F1]")

