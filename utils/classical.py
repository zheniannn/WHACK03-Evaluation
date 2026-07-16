"""Stage 11 classical discriminators: two model-based scores per candidate
track, computed from its per-scan sequence and the radar physics.

  M-of-N      -- the baseline. Binary: confirmed iff some window of N scans
                 holds >= M detections. This is what threshold-only tracking
                 achieves; every continuous method must beat its operating point.
  SPRT / LLR  -- sequential log-likelihood ratio, target vs clutter hypothesis.
                 Per scan: detected -> ln(Pd * N(y;y_hat,S) / lambda);
                 missed -> ln(1 - Pd). Folds in Pd(range) from the radar
                 equation and the clutter density lambda.

Both take the estimated range per scan (never truth) and the scenario's
own Pd model, so nothing here cheats.

(An IPDA existence-probability discriminator was dropped: its score
anti-correlated with track length and existence-probability is weakly matched
to aircraft-vs-clutter. SPRT is the credible classical method here.)
"""

import numpy as np

# M-of-N window.
MOFN_M = 3
MOFN_N = 5
LN_2PI = np.log(2 * np.pi)


class Physics:
    """Radar detection physics from the WHACK02 scenario, evaluated at the
    tracker's estimated range (not truth)."""

    def __init__(self, sc: dict):
        self.tau = 10 ** (sc["threshold_min_db"] / 10)
        self.snr_ref_lin = 10 ** (sc["snr_ref_db"] / 10)
        self.range_ref = sc["range_ref_m"]
        n_range = int((sc["range_max_m"] - sc["range_min_m"]) / sc["range_resolution_m"])
        n_az = int(round(360.0 / sc["azimuth_beamwidth_deg"]))
        area = np.pi * (sc["range_max_m"] ** 2 - sc["range_min_m"] ** 2)
        self.pfa = float(np.exp(-self.tau))
        self.lam = n_range * n_az * self.pfa / area     # clutter density (per m^2)

    def pd(self, range_m):
        snr = self.snr_ref_lin * (self.range_ref / np.maximum(range_m, 1.0)) ** 4
        return np.exp(-self.tau / (1.0 + snr))


def per_point_delta_llr(miss, nis, logdet_s, pd, lam):
    """SPRT per-scan log-likelihood increment (vectorised over all points).

    Detected: ln(Pd) - ln(lambda) - ln(2pi) - 0.5 ln|S| - 0.5 d^2
    Missed:   ln(1 - Pd)
    """
    pd = np.clip(pd, 1e-6, 1 - 1e-6)
    detected = miss == 0
    d = np.where(
        detected,
        np.log(pd) - np.log(lam) - LN_2PI - 0.5 * np.nan_to_num(logdet_s) - 0.5 * np.nan_to_num(nis),
        np.log(1 - pd),
    )
    return d


def mofn_confirmed(scan_idx: np.ndarray, miss: np.ndarray) -> int:
    """1 if some window of MOFN_N consecutive scans holds >= MOFN_M detections."""
    if scan_idx.size == 0:
        return 0
    lo, hi = scan_idx.min(), scan_idx.max()
    hit = np.zeros(hi - lo + 1, dtype=int)
    hit[scan_idx[miss == 0] - lo] = 1
    if hit.size < MOFN_N:
        return int(hit.sum() >= MOFN_M)
    win = np.convolve(hit, np.ones(MOFN_N, int), "valid")
    return int(win.max() >= MOFN_M)
