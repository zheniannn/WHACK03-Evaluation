"""Stage 18 -- compare every discriminator on the held-out day.

Merges the per-method score files (stages 11-17) with the track metadata, then
reports, per method: precision / recall / F1 at the best-F1 operating point,
overall AUC, clutter-vs-aircraft AUC (the honest, length-neutralised metric),
and persistent-clutter survival at the baseline operating point. Writes a
combined score table, a JSON report, and a scorecard figure.

Usage:  python scripts/18_Evaluation.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

from utils.io import TEST_DATE, get_method_score_path, get_all_methods_path, get_eval_dir, get_plot_dir
from utils.data import load_track_meta

# (score-file key, display name, score column, type)
METHODS = [
    ("mofn", "M-of-N",     "score_mofn", "classical"),
    ("sprt", "SPRT",       "score_llr",  "classical"),
    ("gbm",  "GBM",        "score_gbm",  "supervised ML"),
    ("gru",  "GRU",        "score_gru",  "supervised ML"),
    ("imm",  "IMM",        "score_imm",  "classical motion"),
    ("vae",  "Traj-VAE",   "score_vae",  "one-class ML"),
    ("sde",  "Latent-SDE", "score_sde",  "one-class ML"),
]


def best_f1(y, s):
    """Precision/recall/F1 at the threshold that maximises F1."""
    s = np.where(np.isfinite(s), s, -1e18)
    o = np.argsort(-s)
    tp = np.cumsum(y[o] == 1)
    k = np.arange(1, len(s) + 1)
    prec = tp / k
    rec = tp / max(int(y.sum()), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(prec[i]), float(rec[i]), float(f1[i])


def main() -> None:
    meta = load_track_meta(TEST_DATE)
    df = meta.copy()
    present = []
    for key, name, col, typ in METHODS:
        p = get_method_score_path(key, TEST_DATE)
        if os.path.exists(p):
            df = df.merge(pd.read_csv(p), on="track_id", how="left")
            present.append((key, name, col, typ))
    if not present:
        raise SystemExit("No per-method score files found -- run stages 11-17 first.")

    os.makedirs(os.path.dirname(get_all_methods_path(TEST_DATE)), exist_ok=True)
    df.to_csv(get_all_methods_path(TEST_DATE), index=False)

    y = df["label"].to_numpy()
    cvt = np.isin(df["track_source"].to_numpy(), ["target", "clutter"])
    is_clut = (df["track_source"].to_numpy() == "clutter")
    base_fpr = df["score_mofn"].to_numpy()[y == 0].mean() if "score_mofn" in df else 0.4

    print(f"Held-out {TEST_DATE}: {len(df):,} tracks, {int(y.sum())} true "
          f"({100*y.mean():.2f}%)\n")
    print(f"{'method':12s} {'type':16s} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6} {'AUCclt/ac':>9} {'clut-surv':>9}")
    rows = []
    for key, name, col, typ in present:
        s = np.nan_to_num(df[col].to_numpy(), nan=-1e18)
        P, R, F = best_f1(y, s)
        auc = roc_auc_score(y, s)
        auc_cvt = roc_auc_score(y[cvt], s[cvt])
        thr = np.quantile(s[y == 0], 1 - base_fpr)
        clut_surv = float((s[is_clut] >= thr).mean())
        rows.append(dict(method=name, type=typ, precision=P, recall=R, f1=F,
                         auc=float(auc), auc_clut_vs_aircraft=float(auc_cvt),
                         clutter_survival=clut_surv))
        print(f"{name:12s} {typ:16s} {P:6.3f} {R:6.3f} {F:6.3f} {auc:6.3f} {auc_cvt:9.3f} {clut_surv:9.3f}")

    # ---- report ----
    os.makedirs(get_eval_dir(), exist_ok=True)
    json.dump({"test_date": TEST_DATE, "n_tracks": int(len(df)), "n_true": int(y.sum()),
               "baseline_fpr": float(base_fpr), "methods": rows},
              open(os.path.join(get_eval_dir(), "evaluation_report.json"), "w"), indent=2)

    # ---- scorecard figure ----
    R = pd.DataFrame(rows)
    COLS = [("Precision", "precision", True), ("Recall", "recall", True), ("F1", "f1", True),
            ("AUC\nclut/ac", "auc_clut_vs_aircraft", True), ("Clutter\nsurvival", "clutter_survival", False)]
    V = np.array([[R.iloc[i][c[1]] for c in COLS] for i in range(len(R))])
    G = np.zeros_like(V)
    for j, (_, _, hib) in enumerate(COLS):
        c = V[:, j]; g = (c - c.min()) / (np.ptp(c) + 1e-12); G[:, j] = g if hib else 1 - g
    cmap = LinearSegmentedColormap.from_list("gr", ["#c65b4e", "#f0efe8", "#3a8f5b"])
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(R) + 1.5))
    ax.imshow(G, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[0] for c in COLS], fontsize=9)
    ax.set_yticks(range(len(R))); ax.set_yticklabels(R.method, fontsize=10)
    ax.tick_params(length=0)
    for i in range(len(R)):
        for j in range(len(COLS)):
            ax.text(j, i, f"{V[i,j]:.3f}", ha="center", va="center", fontsize=9.5,
                    color="#0b0b0b" if 0.2 < G[i, j] < 0.85 else "#fcfcfb")
    for i, typ in enumerate(R.type):
        ax.text(len(COLS) - 0.4, i, typ, ha="left", va="center", fontsize=8, color="#898781", style="italic")
    ax.set_xlim(-0.5, len(COLS) + 1.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Discriminator comparison -- held-out {TEST_DATE} "
                 f"({len(df):,} tracks, {int(y.sum())} true)\n"
                 "green = best in column - clutter survival: lower is better",
                 fontsize=11, loc="left", pad=10)
    fig.tight_layout()
    os.makedirs(get_plot_dir(), exist_ok=True)
    out = os.path.join(get_plot_dir(), "18_scorecard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nreport  -> {os.path.join(get_eval_dir(), 'evaluation_report.json')}")
    print(f"scorecard -> {out}")
    print(f"combined -> {get_all_methods_path(TEST_DATE)}")


if __name__ == "__main__":
    main()
