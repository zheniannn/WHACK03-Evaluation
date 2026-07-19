"""Compare WHACK03 discrimination at 8 dB vs 13 dB CFAR (main WHACK02 radar).
8 dB = data/active/discrimination; 13 dB = data/main_cfar13 (snr>=13 filter)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = {8: "/home/ian/working/WHACK/data/active/discrimination/eval/evaluation_report.json",
           13: "/home/ian/working/WHACK/data/main_cfar13/active/discrimination/eval/evaluation_report.json"}
OUTDIR = "/home/ian/working/WHACK/data/plot/WHACK03-Evaluation"
os.makedirs(OUTDIR, exist_ok=True)
R = {db: json.load(open(p)) for db, p in REPORTS.items()}
methods = [m["method"] for m in R[8]["methods"]]
colors = {"M-of-N": "#b0b0b0", "SPRT": "#8a8a8a", "GBM": "#7d5ba6", "GRU": "#5b7da6",
          "IMM": "#c65b4e", "Traj-VAE": "#d99a2b", "Latent-SDE": "#c98a1b", "Fusion": "#3a8f5b"}


def val(db, method, key):
    return next(m[key] for m in R[db]["methods"] if m["method"] == method)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
xs = [13, 8]
for ax, key, title in [(axes[0], "f1", "Ungated F1 (full population)"),
                       (axes[1], "f1_gate", "F1 @ n_det>=12 (gated)")]:
    for method in methods:
        ax.plot(xs, [val(db, method, key) for db in xs], "o-", label=method, color=colors.get(method), lw=2)
    ax.axvspan(-1, 3, color="#c65b4e", alpha=0.08)
    ax.text(1.5, 0.06, "0 dB\ninfeasible", ha="center", fontsize=8, color="#c65b4e")
    ax.set_xlim(15, -1); ax.set_ylim(0, 1); ax.set_xticks([13, 8, 0])
    ax.set_xlabel("CFAR floor (dB)  —  lower = more false alarms")
    ax.set_ylabel(key); ax.set_title(title); ax.grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="lower left", ncol=2)
prev = {db: 100 * R[db]["n_true"] / R[db]["n_tracks"] for db in xs}
fig.suptitle("WHACK03-Evaluation (main WHACK02 radar) — discrimination vs CFAR floor\n"
             f"true-track prevalence: 13 dB {prev[13]:.0f}%  ->  8 dB {prev[8]:.2f}%  ->  0 dB (untrackable)",
             fontsize=12)
fig.tight_layout()
out = f"{OUTDIR}/cfar_sweep.png"
fig.savefig(out, dpi=150)
print(f"figure -> {out}")
