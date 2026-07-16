"""Stage 12 evaluation figures (PNG, light surface, project palette)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
# One fixed hue per method (categorical slots), baseline in ink.
METHOD_COLORS = {
    "llr": "#1baf7a", "gbm": "#2a78d6", "gru": "#e34948",
    "mofn": "#0b0b0b",
}
METHOD_LABELS = {
    "llr": "SPRT (LLR)", "gbm": "Grad-boosted trees",
    "gru": "GRU sequence", "mofn": "M-of-N baseline",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12,
})


def _legend(ax, loc="lower right"):
    leg = ax.legend(loc=loc, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)


def plot_roc(rocs: dict, mofn_point, auc: dict, title: str, out_path: str) -> None:
    """ROC curves for continuous methods + the M-of-N operating point."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    for m, (fpr, tpr) in rocs.items():
        ax.plot(fpr, tpr, color=METHOD_COLORS[m], lw=2, zorder=3,
                label=f"{METHOD_LABELS[m]}  (AUC {auc[m]:.3f})")
    if mofn_point is not None:
        ax.plot([mofn_point[0]], [mofn_point[1]], marker="*", ms=16,
                color=METHOD_COLORS["mofn"], zorder=5, label=METHOD_LABELS["mofn"])
    ax.set_xlabel("false-track rate (FPR)"); ax.set_ylabel("true-track rate (TPR)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    _legend(ax)
    ax.set_title(title, color=INK)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_auc_vs_range(range_mid_km, auc_by_method: dict, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.5, color=GRID, lw=1)
    for m, aucs in auc_by_method.items():
        ax.plot(range_mid_km, aucs, color=METHOD_COLORS[m], lw=2, marker="o", ms=4,
                label=METHOD_LABELS[m])
    ax.set_xlabel("track median range (km)"); ax.set_ylabel("AUC in range bin")
    ax.set_ylim(0.45, 1.01)
    _legend(ax, "lower left")
    ax.set_title(title, color=INK)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_tpr_vs_range(range_mid_km, tpr_by_method: dict, mofn_tpr, fpr_target: float,
                      title: str, out_path: str) -> None:
    """At a false-track rate matched to M-of-N, the fraction of true tracks
    recovered vs range -- the 'how far can we still hold tracks' plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.5, color=GRID, lw=1)
    for m, tprs in tpr_by_method.items():
        ax.plot(range_mid_km, tprs, color=METHOD_COLORS[m], lw=2, marker="o", ms=4,
                label=METHOD_LABELS[m])
    if mofn_tpr is not None:
        ax.plot(range_mid_km, mofn_tpr, color=METHOD_COLORS["mofn"], lw=2, ls="--",
                marker="s", ms=4, label=METHOD_LABELS["mofn"])
    ax.set_xlabel("track median range (km)")
    ax.set_ylabel(f"true tracks recovered (TPR at FPR={fpr_target:.3f})")
    ax.set_ylim(0, 1.02)
    _legend(ax, "lower left")
    ax.set_title(title, color=INK)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_importance(names, importances, title: str, out_path: str, top: int = 15) -> None:
    order = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order)), np.array(importances)[order], color=METHOD_COLORS["gbm"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("permutation importance (AUC drop)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, color=INK)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
