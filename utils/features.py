"""Stage 11 feature extraction: a fixed-length descriptor per candidate track
for the gradient-boosted model, plus the per-scan sequence for the GRU.

Features are motion/kinematic and signal statistics -- exactly the quantities
that separate a physically plausible GA flight (steady speed 30-90 m/s, low
turn rate, high straightness, high net displacement) from noise coincidences
(erratic speed, near-zero displacement) and persistent clutter (many
detections pinned to one spot -> tiny spatial spread). No truth is used.
"""

import numpy as np

DT = 10.0

FEATURE_NAMES = [
    "n_det", "n_scan", "span_scans", "det_rate", "miss_frac", "longest_miss_run",
    "speed_mean", "speed_std", "speed_max", "speed_min",
    "accel_absmean", "accel_absmax", "accel_std",
    "turn_absmean", "turn_absmax", "turn_std", "heading_circ_std",
    "straightness", "net_disp_m", "path_len_m", "spatial_spread_m", "bbox_diag_m",
    "snr_mean", "snr_std", "snr_min", "snr_max",
    "nis_mean", "nis_max", "range_mean", "range_min", "range_max",
]


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def track_features(e, n, r, snr, nis, miss, scan) -> dict:
    """Fixed descriptor from one track's per-scan arrays (already scan-sorted)."""
    det = miss == 0
    n_det = int(det.sum())
    n_scan = len(scan)
    span = int(scan.max() - scan.min() + 1)

    ed, nd = e[det], n[det]                       # detected positions (KF estimates)
    snr_d = snr[det]
    nis_d = nis[det]
    r_d = r[det]

    # Kinematics from consecutive detected points (dt scaled by scan gap).
    f = {name: 0.0 for name in FEATURE_NAMES}
    f["n_det"] = n_det
    f["n_scan"] = n_scan
    f["span_scans"] = span
    f["det_rate"] = n_det / n_scan if n_scan else 0.0
    f["miss_frac"] = float(miss.mean())
    f["longest_miss_run"] = _longest_run(miss == 1)

    if n_det >= 2:
        dt = np.diff(scan[det]) * DT
        dt = np.where(dt <= 0, DT, dt)
        seg = np.hypot(np.diff(ed), np.diff(nd))
        speed = seg / dt
        f["speed_mean"] = float(speed.mean()); f["speed_std"] = float(speed.std())
        f["speed_max"] = float(speed.max()); f["speed_min"] = float(speed.min())
        heading = np.arctan2(np.diff(ed), np.diff(nd))       # compass-like
        R_bar = min(np.abs(np.mean(np.exp(1j * heading))), 1.0)   # clamp for the log
        f["heading_circ_std"] = float(np.sqrt(-2 * np.log(R_bar + 1e-12)))
        net = np.hypot(ed[-1] - ed[0], nd[-1] - nd[0])
        path = seg.sum()
        f["net_disp_m"] = float(net); f["path_len_m"] = float(path)
        f["straightness"] = float(net / path) if path > 0 else 0.0
        f["spatial_spread_m"] = float(np.hypot(ed.std(), nd.std()))
        f["bbox_diag_m"] = float(np.hypot(ed.max() - ed.min(), nd.max() - nd.min()))

    if n_det >= 3:
        dt = np.diff(scan[det]) * DT
        dt = np.where(dt <= 0, DT, dt)
        speed = np.hypot(np.diff(ed), np.diff(nd)) / dt
        accel = np.diff(speed) / dt[:-1]
        f["accel_absmean"] = float(np.abs(accel).mean())
        f["accel_absmax"] = float(np.abs(accel).max())
        f["accel_std"] = float(accel.std())
        heading = np.arctan2(np.diff(ed), np.diff(nd))
        dh = (np.diff(heading) + np.pi) % (2 * np.pi) - np.pi
        turn = np.degrees(dh) / dt[:-1]
        f["turn_absmean"] = float(np.abs(turn).mean())
        f["turn_absmax"] = float(np.abs(turn).max())
        f["turn_std"] = float(turn.std())

    if n_det:
        f["snr_mean"] = float(np.nanmean(snr_d)); f["snr_std"] = float(np.nanstd(snr_d))
        f["snr_min"] = float(np.nanmin(snr_d)); f["snr_max"] = float(np.nanmax(snr_d))
        f["nis_mean"] = float(np.nanmean(nis_d)); f["nis_max"] = float(np.nanmax(nis_d))
    f["range_mean"] = float(np.nanmean(r)); f["range_min"] = float(np.nanmin(r))
    f["range_max"] = float(np.nanmax(r))
    return f


# Per-scan channels for the sequence model (GRU).
SEQ_CHANNELS = ["de", "dn", "speed", "snr_norm", "miss", "range_norm"]


def track_sequence(e, n, r, snr, miss, scan) -> np.ndarray:
    """(T, len(SEQ_CHANNELS)) per-scan sequence for the GRU. Position deltas
    (translation-invariant), instantaneous speed, normalised SNR, miss flag,
    normalised range."""
    T = len(scan)
    de = np.concatenate(([0.0], np.diff(e))) / 1000.0        # km step
    dn = np.concatenate(([0.0], np.diff(n))) / 1000.0
    gaps = np.concatenate(([1.0], np.maximum(np.diff(scan), 1))) * DT
    speed = np.hypot(de, dn) * 1000.0 / gaps / 100.0          # ~O(1)
    snr_norm = np.nan_to_num((snr - 8.0) / 20.0)              # 0 at floor, ~1 at 28 dB
    range_norm = r / 80000.0
    return np.column_stack([de, dn, speed, snr_norm, miss.astype(float), range_norm]).astype(np.float32)
