"""Auxiliary figures visualising the stage-10 tracker output -- the
discrimination problem the rest of WHACK03 solves. Reads tracks_<date>.csv +
track_points_<date>.csv and writes to plot/WHACK03-Evaluation/:

  1.1_tracks_scene_true_<date>.png   true tracks (real aircraft) on a PPI
  1.2_tracks_scene_false_<date>.png  false tracks (noise/clutter) on a PPI
  1.3_tracks_gallery_<date>.png      example true vs false tracks, coloured by time
  1.4_tracks_features_<date>.png     track length vs mean NIS, coloured by class --
                                     what the discriminators key on
  1.5_tracks_imbalance.png           candidate-track yield: true vs false (~0.7% true)

Usage:
    python scripts/make_track_figures.py
    python scripts/make_track_figures.py --date 2022-06-13
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import load_points_by_track, load_track_meta
from utils.io import DATES, TEST_DATE, get_plot_dir

# Project palette.
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
C_TRUE = "#2a78d6"; C_FALSE = "#898781"
START_C = "#1baf7a"; END_C = "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12,
})
RANGE_MAX_KM = 200.0
FALSE_SCENE_CAP = 12000     # cap false tracks drawn in the scene (keeps confetti readable)


def _save(fig, name):
    out = os.path.join(get_plot_dir(), name)
    os.makedirs(get_plot_dir(), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")
    return out


def _ppi_frame(ax):
    for r in (40, 80, 120, 160, 200):
        if r <= RANGE_MAX_KM:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color=GRID, lw=0.8, zorder=1))
            ax.annotate(f"{r} km", (0, r), color=MUTED, fontsize=8, ha="center", va="bottom", zorder=2)
    lim = RANGE_MAX_KM * 1.1
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.plot(0, 0, marker="^", color=INK, ms=9, zorder=6)


def _segments(e, n, idxs, starts, ends):
    """NaN-separated path arrays (km) for the tracks in idxs -- one plot call."""
    xs, ys = [], []
    for i in idxs:
        s, en = starts[i], ends[i]
        xs.append(e[s:en]); xs.append([np.nan])
        ys.append(n[s:en]); ys.append([np.nan])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def load_day(date):
    """Per-track points + labels, joined. Returns (labels, e_km, n_km, starts,
    ends, cols, tid_to_idx)."""
    tids, cols, (starts, ends) = load_points_by_track(date)
    meta = load_track_meta(date)
    label_of = dict(zip(meta["track_id"].to_numpy(), meta["label"].to_numpy()))
    labels = np.array([int(label_of.get(t, 0)) for t in tids])
    return labels, cols["est_e"] / 1000.0, cols["est_n"] / 1000.0, starts, ends, cols, tids


def _scene(e, n, idxs, starts, ends, color, lw, alpha, title, name):
    """One PPI of the given tracks -- shared by the true / false scene figures."""
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    _ppi_frame(ax)
    x, y = _segments(e, n, idxs, starts, ends)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, rasterized=True)
    ax.annotate("radar", (0, -RANGE_MAX_KM * 0.05), color=INK2, fontsize=8, ha="center", va="top")
    ax.set_title(title, color=INK)
    return _save(fig, name)


def fig_scene(date, labels, e, n, starts, ends, rng):
    """Two separate PPIs: true tracks (coherent arcs), false tracks (confetti)."""
    true_i = np.flatnonzero(labels == 1)
    false_i = np.flatnonzero(labels == 0)
    n_false = false_i.size
    shown_false = false_i if n_false <= FALSE_SCENE_CAP else rng.choice(false_i, FALSE_SCENE_CAP, replace=False)

    _scene(e, n, true_i, starts, ends, C_TRUE, 0.6, 0.75,
           f"Stage-10 TRUE tracks — {true_i.size:,} real aircraft ({date})\n"
           "coherent arcs, concentrated inside the detection horizon",
           f"1.1_tracks_scene_true_{date}.png")
    _scene(e, n, shown_false, starts, ends, C_FALSE, 0.5, 0.35,
           f"Stage-10 FALSE tracks — {n_false:,} from noise / clutter "
           f"({len(shown_false):,} shown, {date})\n"
           "the confetti the discriminators must reject",
           f"1.2_tracks_scene_false_{date}.png")


def _panel(ax, e_seg, n_seg):
    t = np.linspace(0, 1, len(e_seg))
    ax.scatter(e_seg, n_seg, c=t, cmap="viridis", s=6, lw=0, zorder=3)
    ax.plot(e_seg[0], n_seg[0], marker="o", color=START_C, ms=5)
    ax.plot(e_seg[-1], n_seg[-1], marker="s", color=END_C, ms=5)
    r = max(np.ptp(e_seg), np.ptp(n_seg), 0.5) * 0.6
    cx, cy = (e_seg.max() + e_seg.min()) / 2, (n_seg.max() + n_seg.min()) / 2
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)


def fig_gallery(date, labels, e, n, starts, ends):
    """Example true (top rows) vs false (bottom rows) tracks, coloured by time."""
    lens = ends - starts
    true_i = np.flatnonzero(labels == 1)
    false_i = np.flatnonzero(labels == 0)
    # True: 6 longest (clearest aircraft). False: 6 LONGEST false -- the hardest
    # confusers, the ones most target-like.
    pick_true = true_i[np.argsort(lens[true_i])[::-1][:6]]
    pick_false = false_i[np.argsort(lens[false_i])[::-1][:6]]

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
    for col, i in enumerate(pick_true):
        _panel(axes[0, col], e[starts[i]:ends[i]], n[starts[i]:ends[i]])
        axes[0, col].set_title(f"{lens[i]} scans", fontsize=8, color=INK2)
    for col, i in enumerate(pick_false):
        _panel(axes[1, col], e[starts[i]:ends[i]], n[starts[i]:ends[i]])
        axes[1, col].set_title(f"{lens[i]} scans", fontsize=8, color=INK2)
    axes[0, 0].set_ylabel("TRUE\n(real aircraft)", color=C_TRUE, fontsize=11, rotation=0,
                          ha="right", va="center", labelpad=30)
    axes[1, 0].set_ylabel("FALSE\n(noise / clutter)", color=INK2, fontsize=11, rotation=0,
                          ha="right", va="center", labelpad=30)
    fig.suptitle(f"Example tracks — true aircraft vs the longest false tracks ({date})\n"
                 "green = start, red = end; even the worst false tracks lack coherent motion",
                 color=INK, y=0.99)
    fig.subplots_adjust(top=0.82)
    return _save(fig, f"1.3_tracks_gallery_{date}.png")


def fig_imbalance():
    """Candidate-track yield across all days: true vs false, with prevalence."""
    days, n_true, n_false = [], [], []
    for d in DATES:
        m = load_track_meta(d)
        days.append(d[5:]); n_true.append(int((m["label"] == 1).sum()))
        n_false.append(int((m["label"] == 0).sum()))
    n_true = np.array(n_true); n_false = np.array(n_false)
    tot = n_true + n_false
    prev = 100.0 * n_true.sum() / tot.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(days)); w = 0.38
    ax.bar(x - w / 2, n_false, w, color=C_FALSE, label=f"false ({n_false.sum():,})")
    ax.bar(x + w / 2, n_true, w, color=C_TRUE, label=f"true ({n_true.sum():,})")
    ax.set_yscale("log"); ax.set_ylim(300, n_false.max() * 4)
    for i in range(len(days)):
        ax.annotate(f"{100*n_true[i]/tot[i]:.1f}%\ntrue", (x[i] + w / 2, n_true[i]),
                    ha="center", va="bottom", color=C_TRUE, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_ylabel("candidate tracks (log scale)"); ax.set_xlabel("survey day (2022-06-…)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(f"Stage-10 track yield — only {prev:.1f}% of candidate tracks are real\n"
                 "~140 false tracks per true one: the imbalance that makes AP, not AUC, the honest metric",
                 color=INK)
    return _save(fig, "1.5_tracks_imbalance.png")


def fig_features(date, labels, cols, starts, ends):
    """Track length vs mean NIS, coloured by class -- the separability the
    discriminators exploit."""
    nis = cols["nis"]
    lens = ends - starts
    mean_nis = np.array([np.nanmean(nis[s:e]) if e > s else np.nan
                         for s, e in zip(starts, ends)])
    ok = np.isfinite(mean_nis)
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)
    for lab, color, name, z, a in ((0, C_FALSE, "false", 2, 0.25), (1, C_TRUE, "true", 4, 0.7)):
        sel = ok & (labels == lab)
        # jitter length a touch for visibility on the discrete axis
        jl = lens[sel] + rng.uniform(-0.3, 0.3, sel.sum())
        ax.scatter(jl, mean_nis[sel], s=6, color=color, alpha=a, lw=0, zorder=z,
                   rasterized=True, label=f"{name} ({int(sel.sum()):,})")
    ax.axhline(2.0, color=INK, lw=1.0, ls=":", zorder=3)
    ax.annotate("χ²(2) expectation = 2", (ax.get_xlim()[1], 2.0), color=INK, fontsize=8,
                ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("track length (detections)"); ax.set_ylabel("mean NIS (innovation consistency)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(f"What separates the classes ({date})\n"
                 "true tracks are long with well-modelled motion; false tracks short and erratic",
                 color=INK)
    return _save(fig, f"1.4_tracks_features_{date}.png")


def main():
    p = argparse.ArgumentParser(description="Stage-10 track visualisation figures.")
    p.add_argument("--date", default=TEST_DATE,
                   help="Day for the per-day figures (default: the held-out test day, 2022-06-27).")
    args = p.parse_args()
    date = args.date
    rng = np.random.default_rng(20220606)

    print(f"loading stage-10 tracks for {date} ...")
    labels, e, n, starts, ends, cols, tids = load_day(date)
    print(f"  {len(tids):,} candidate tracks ({int((labels==1).sum()):,} true)")
    print(f"writing figures -> {get_plot_dir()}")
    fig_scene(date, labels, e, n, starts, ends, rng)
    fig_gallery(date, labels, e, n, starts, ends)
    fig_features(date, labels, cols, starts, ends)
    fig_imbalance()
    print("make_track_figures completed.")


if __name__ == "__main__":
    main()
