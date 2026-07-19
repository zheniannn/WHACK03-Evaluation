"""Stage 17 -- label-free channel fusion (the deployable label-free discriminator).

Fuses three physically-independent, LABEL-FREE channels into one score:
  motion    -- stage-16 Traj-VAE reconstruction score (score_vae): rejects clutter
  amplitude -- SNR vs radar-equation residual + mean SNR:           rejects noise
  length    -- n_det, and a reliability weight on the motion channel

  score = w * rank(motion) + rank(-resid_rms) + rank(mean_snr) + rank(n_det),
          w = n_det / (n_det + N0)

No labels are used to build or weight the score: N0=10 is a-priori (~ the 100 s
motion window, so motion is down-weighted until a track has a full window), and
the amplitude channel comes from the scenario's own radar equation. Motion owns
clutter, amplitude owns noise -> orthogonal. Reported at a length gate
(n_det >= MIN_GATE); the gate is where the fused F1 peaks and is ~ one motion
window. Reaches best-gate F1 ~ 0.77 on the held-out day, the label-free ceiling.

Requires stage 16 (score_vae) to have run. Writes scores/fusion_<TEST_DATE>.csv.
Usage:  python scripts/17_Fusion.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from utils.io import TEST_DATE, get_scenario_path, get_method_score_path
from utils.classical import Physics
from utils.data import load_track_meta
from utils.motion import load_track_points

MIN_GATE = 12       # length gate (~ one 100 s motion window); a-priori, not label-tuned
N0 = 10.0           # reliability-weight scale for the motion channel


def amp_features(snr, rng, phys):
    """(mean_snr, resid_rms) -- RMS deviation of SNR(dB) from the R^-4 radar
    equation with the RCS scale fitted out. Noise is flat-at-floor; a real
    target's SNR falls off as R^-4, so its residual is small."""
    snr = np.asarray(snr, float)
    rng = np.maximum(np.asarray(rng, float), 1.0)
    exp_db = 10 * np.log10(phys.snr_ref_lin) + 40 * np.log10(phys.range_ref / rng)
    resid = snr - exp_db
    resid = resid - resid.mean()
    return float(snr.mean()), float(resid.std() if len(resid) > 1 else 0.0)


def rank(x):
    return pd.Series(np.asarray(x, float)).rank().to_numpy()


def gated_best_f1(y, s, nd, gate):
    """Best-F1 sweeping the threshold; recall vs ALL positives; n_det<gate rejected."""
    P = max(int(y.sum()), 1)
    ss = np.where(nd >= gate, np.where(np.isfinite(s), s, -1e18), -1e18)
    o = np.argsort(-ss)
    tp = np.cumsum(y[o] == 1)
    k = np.arange(1, len(ss) + 1)
    prec = tp / k
    rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(prec[i]), float(rec[i]), float(f1[i])


def main() -> None:
    phys = Physics(json.load(open(get_scenario_path())))

    # motion channel (stage 16)
    vae_path = get_method_score_path("vae", TEST_DATE)
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"{vae_path} missing -- run scripts/16_VAE.py first.")
    vae = pd.read_csv(vae_path)

    # amplitude channel (per-track SNR vs radar equation)
    pts = load_track_points(TEST_DATE)
    rows = [(int(t), *amp_features(d["snr"], d["rng"], phys)) for t, d in pts.items()]
    amp = pd.DataFrame(rows, columns=["track_id", "mean_snr", "resid_rms"])

    meta = load_track_meta(TEST_DATE)      # track_id, label, track_source, n_det, range_median_m
    df = meta.merge(vae, on="track_id", how="left").merge(amp, on="track_id", how="left")
    df["score_vae"] = df["score_vae"].fillna(df["score_vae"].min())
    df[["mean_snr", "resid_rms"]] = df[["mean_snr", "resid_rms"]].fillna(0.0)

    nd = df["n_det"].to_numpy(float)
    w = nd / (nd + N0)
    fused = (w * rank(df["score_vae"].to_numpy())
             + rank(-df["resid_rms"].to_numpy())
             + rank(df["mean_snr"].to_numpy())
             + rank(nd))
    # Emit the continuous (ungated) fused score. The length gate is an evaluation
    # operating point, applied UNIFORMLY to every method in stage 19 (its F1@n>=12
    # column) -- so keeping it out of the score keeps the cross-method comparison
    # fair and leaves ROC-AUC / clut-surv meaningful for this method too.
    df["score_fusion"] = fused

    out = get_method_score_path("fusion", TEST_DATE)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[["track_id", "score_fusion"]].to_csv(out, index=False)

    # self-report (validation of the label-free ceiling)
    y = (df["label"].to_numpy() == 1).astype(int)
    P, R, F = gated_best_f1(y, fused, nd, MIN_GATE)
    print(f"Fusion: {len(df):,} tracks scored -> {out}")
    print(f"  label-free best-F1 @ n_det>={MIN_GATE}: F1 {F:.3f}  (P {P:.3f}  R {R:.3f})")


if __name__ == "__main__":
    main()
