"""Stage 10 rules: a multi-target tracking front-end over WHACK02 detections.

Converted-measurement Kalman filter (constant-velocity state in local ENU
metres), global-nearest-neighbour association per scan, one-point
initiation with a large velocity covariance, and coast-K deletion. It is
deliberately PERMISSIVE -- it forms tracks from noise and clutter as
readily as from targets -- because the whole point of stages 11-12 is to
discriminate the resulting true tracks from the false ones. Aggressive
confirmation/deletion here would throw away the very false tracks we test on.

Every track that ever reaches MIN_KEEP detections is emitted as a candidate,
with per-scan points (for the sequence model) and truth linkage (for
labelling), but never any per-detection truth inside the tracker itself.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# --- Model constants ---
DT = 10.0                    # scan period (s)
SIGMA_A = 3.0                # process white-acceleration intensity (m/s^2), covers GA manoeuvres
GATE_CHI2 = 9.21             # 2-dof chi-square, 99% gate
COARSE_RADIUS_M = 4000.0     # KD-tree pre-gate radius before the exact Mahalanobis test
SIGMA_V0 = 100.0             # initial velocity std for one-point initiation (m/s)
MAX_COAST = 5                # delete a track after this many consecutive misses
MIN_KEEP = 3                 # emit a candidate track only if it reached this many detections
BIG = 1e9                    # non-gated assignment cost

# Constant-velocity transition and process noise (discrete white-accel model).
F = np.array([[1, DT, 0, 0], [0, 1, 0, 0], [0, 0, 1, DT], [0, 0, 0, 1]], float)
_G = np.array([[DT ** 2 / 2, 0], [DT, 0], [0, DT ** 2 / 2], [0, DT]])
Q = _G @ (SIGMA_A ** 2 * np.eye(2)) @ _G.T
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)


@dataclass
class Track:
    tid: int
    x: np.ndarray                       # state [E, vE, N, vN]
    P: np.ndarray                       # 4x4 covariance
    scans: List[int] = field(default_factory=list)
    ts: List[float] = field(default_factory=list)
    exs: List[float] = field(default_factory=list)   # estimated E per scan
    nys: List[float] = field(default_factory=list)   # estimated N per scan
    ranges: List[float] = field(default_factory=list)
    snrs: List[float] = field(default_factory=list)
    innov2: List[float] = field(default_factory=list)  # normalised innovation^2 (NIS)
    logdetS: List[float] = field(default_factory=list)  # log|innovation covariance| (for track score)
    miss: List[int] = field(default_factory=list)      # 1 = coasted (no detection) this scan
    det_idx: List[int] = field(default_factory=list)   # source detection row, -1 on a miss
    coast: int = 0


def polar_to_enu(range_m, az_deg):
    """Compass azimuth (0=N, 90=E) -> local ENU east/north metres."""
    a = np.radians(az_deg)
    return range_m * np.sin(a), range_m * np.cos(a)


def converted_R(range_m, az_deg, sigma_r, sigma_az_deg):
    """Converted-measurement covariance (Jacobian linearisation) for one detection."""
    a = np.radians(az_deg)
    sa = np.radians(sigma_az_deg)
    J = np.array([[np.sin(a), range_m * np.cos(a)],
                  [np.cos(a), -range_m * np.sin(a)]])
    return J @ np.diag([sigma_r ** 2, sa ** 2]) @ J.T


class MultiTargetTracker:
    def __init__(self, sigma_r: float, sigma_az_deg: float, on_complete=None):
        self.sigma_r = sigma_r
        self.sigma_az_deg = sigma_az_deg
        self.tracks: List[Track] = []
        self.done: List[Track] = []
        # on_complete(track) is called as each candidate finishes, so callers
        # can stream it to disk instead of holding every track in memory.
        self.on_complete = on_complete if on_complete is not None else self.done.append
        self._next = 0

    def _new_track(self, e, n, R, scan, t, rng_m, snr, det_i) -> None:
        x = np.array([e, 0.0, n, 0.0])
        P = np.diag([R[0, 0], SIGMA_V0 ** 2, R[1, 1], SIGMA_V0 ** 2])
        tr = Track(self._next, x, P)
        tr.scans.append(scan); tr.ts.append(t); tr.exs.append(e); tr.nys.append(n)
        tr.ranges.append(rng_m); tr.snrs.append(snr); tr.innov2.append(0.0)
        tr.logdetS.append(float(np.log(np.linalg.det(R))))   # no innovation yet; R as the scale proxy
        tr.miss.append(0); tr.det_idx.append(det_i)
        self.tracks.append(tr)
        self._next += 1

    def step(self, scan: int, t: float, det_e, det_n, det_r, det_az, det_snr,
             det_gid) -> None:
        """Advance one scan: predict, associate (GNN), update, manage tracks.
        det_gid holds the global detection-row id for truth linkage later."""
        m = len(det_e)
        assigned_det = np.zeros(m, dtype=bool)

        if self.tracks:
            # --- Predict every live track ---
            X = np.array([tr.x for tr in self.tracks])
            Pn = np.array([tr.P for tr in self.tracks])
            X = X @ F.T
            Pn = np.einsum("ij,njk,lk->nil", F, Pn, F) + Q
            pred = X[:, [0, 2]]                                   # predicted (E, N)

            if m:
                tree = cKDTree(np.column_stack([det_e, det_n]))
                # Per detection converted-measurement covariance.
                Rd = np.array([converted_R(det_r[j], det_az[j], self.sigma_r, self.sigma_az_deg)
                               for j in range(m)])
                n_tr = len(self.tracks)
                cost = np.full((n_tr, m), BIG)
                for i in range(n_tr):
                    Pz = Pn[i][np.ix_([0, 2], [0, 2])]
                    for j in tree.query_ball_point(pred[i], COARSE_RADIUS_M):
                        S = Pz + Rd[j]
                        d = np.array([det_e[j] - pred[i, 0], det_n[j] - pred[i, 1]])
                        Si = np.linalg.inv(S)
                        d2 = float(d @ Si @ d)
                        if d2 < GATE_CHI2:
                            cost[i, j] = d2 + np.log(np.linalg.det(S))
                rows, cols = linear_sum_assignment(cost)
            else:
                rows, cols = np.array([], int), np.array([], int)

            paired = {r: c for r, c in zip(rows, cols) if cost[r, c] < BIG}

            for i, tr in enumerate(self.tracks):
                tr.x, tr.P = X[i], Pn[i]
                if i in paired:
                    j = paired[i]
                    assigned_det[j] = True
                    z = np.array([det_e[j], det_n[j]])
                    Pz = tr.P[np.ix_([0, 2], [0, 2])]
                    S = Pz + Rd[j]
                    innov = z - tr.x[[0, 2]]
                    K = tr.P[:, [0, 2]] @ np.linalg.inv(S)
                    tr.x = tr.x + K @ innov
                    tr.P = tr.P - K @ H @ tr.P
                    tr.coast = 0
                    Sinv = np.linalg.inv(S)
                    nis = float(innov @ Sinv @ innov)
                    tr.scans.append(scan); tr.ts.append(t)
                    tr.exs.append(tr.x[0]); tr.nys.append(tr.x[2])
                    tr.ranges.append(det_r[j]); tr.snrs.append(det_snr[j])
                    tr.innov2.append(nis); tr.logdetS.append(float(np.log(np.linalg.det(S))))
                    tr.miss.append(0); tr.det_idx.append(int(det_gid[j]))
                else:
                    tr.coast += 1
                    tr.scans.append(scan); tr.ts.append(t)
                    tr.exs.append(tr.x[0]); tr.nys.append(tr.x[2])
                    tr.ranges.append(float(np.hypot(tr.x[0], tr.x[2])))
                    tr.snrs.append(np.nan); tr.innov2.append(np.nan); tr.logdetS.append(np.nan)
                    tr.miss.append(1); tr.det_idx.append(-1)

            # --- Retire coasted-out tracks ---
            live = []
            for tr in self.tracks:
                if tr.coast > MAX_COAST:
                    self._trim_and_store(tr)
                else:
                    live.append(tr)
            self.tracks = live

        # --- One-point initiation from every unassigned detection ---
        for j in range(m):
            if not assigned_det[j]:
                R = converted_R(det_r[j], det_az[j], self.sigma_r, self.sigma_az_deg)
                self._new_track(det_e[j], det_n[j], R, scan, t,
                                det_r[j], det_snr[j], int(det_gid[j]))

    def _trim_and_store(self, tr: Track) -> None:
        """Drop the trailing coasted misses (they carry no detection) and keep
        the track if it reached MIN_KEEP real detections."""
        while tr.miss and tr.miss[-1] == 1:
            for lst in (tr.scans, tr.ts, tr.exs, tr.nys, tr.ranges, tr.snrs,
                        tr.innov2, tr.logdetS, tr.miss, tr.det_idx):
                lst.pop()
        if sum(1 for md in tr.miss if md == 0) >= MIN_KEEP:
            self.on_complete(tr)

    def finish(self) -> None:
        for tr in self.tracks:
            self._trim_and_store(tr)
        self.tracks = []


# --- Per-day driver + truth labelling ---

PURITY_THRESHOLD = 0.6      # a candidate is a TRUE track if this fraction of its
                            # detections come from one real trajectory


POINT_HEADER = ["date", "track_id", "scan_idx", "t", "est_e", "est_n",
                "est_range_m", "snr_db", "nis", "logdet_s", "miss"]


def run_day(date: str, det_df: pd.DataFrame, sigma_r: float, sigma_az_deg: float,
            points_path: str) -> pd.DataFrame:
    """Track one day's detections, streaming per-scan track points to
    points_path and returning the compact tracks summary (with truth labels).

    Completed tracks are converted and written as they finish, so peak memory
    is bounded by the live-track set, not the whole day's candidate tracks.
    Truth columns are used ONLY for labelling, never inside the tracker.
    """
    import csv

    e, n = polar_to_enu(det_df["range_m"].to_numpy(), det_df["azimuth_deg"].to_numpy())
    det = det_df.assign(_e=e, _n=n).reset_index(drop=True)
    src = det["source"].to_numpy()
    tid_arr = det["trajectory_id"].fillna("").astype(str).to_numpy()

    track_rows = []
    state = {"k": 0}
    pf = open(points_path, "w", newline="")
    writer = csv.writer(pf)
    writer.writerow(POINT_HEADER)

    def on_complete(tr: Track) -> None:
        k = state["k"]; state["k"] += 1
        gids = [d for d in tr.det_idx if d >= 0]
        srcs = src[gids]; tids = tid_arr[gids]; n_det = len(gids)

        if n_det:
            s_vals, s_counts = np.unique(srcs, return_counts=True)
            track_source = s_vals[s_counts.argmax()]
        else:
            track_source = "noise"

        target_tids = tids[srcs == "target"]
        if target_tids.size:
            vals, counts = np.unique(target_tids, return_counts=True)
            dom, dom_count = vals[counts.argmax()], int(counts.max())
        else:
            dom, dom_count = "", 0
        purity = dom_count / n_det if n_det else 0.0
        label = int(purity >= PURITY_THRESHOLD and dom_count >= MIN_KEEP)

        det_ranges = [r for r, md in zip(tr.ranges, tr.miss) if md == 0]
        track_rows.append({
            "date": date, "track_id": k, "n_det": n_det, "n_scan": len(tr.scans),
            "t_start": tr.ts[0], "t_end": tr.ts[-1],
            "range_median_m": float(np.median(det_ranges)) if det_ranges else np.nan,
            "label": label, "purity": round(purity, 3),
            "matched_trajectory_id": dom if label else "", "track_source": track_source,
        })
        writer.writerows(
            [date, k, tr.scans[i], tr.ts[i], tr.exs[i], tr.nys[i], tr.ranges[i],
             tr.snrs[i], tr.innov2[i], tr.logdetS[i], tr.miss[i]]
            for i in range(len(tr.scans)))

    trk = MultiTargetTracker(sigma_r, sigma_az_deg, on_complete=on_complete)
    for scan, g in det.groupby("scan_idx", sort=True):
        trk.step(int(scan), float(g["t"].iloc[0]),
                 g["_e"].to_numpy(), g["_n"].to_numpy(),
                 g["range_m"].to_numpy(), g["azimuth_deg"].to_numpy(),
                 g["snr_db"].to_numpy(), g.index.to_numpy())
    trk.finish()
    pf.close()
    return pd.DataFrame(track_rows)
