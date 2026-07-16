"""Filesystem path helpers for the discrimination pipeline.

Same convention as WHACK01/02: the data root defaults to `data/` next to
the repository (override with WHACK_DATA_ROOT). This repo reads WHACK02's
stage-9 detections and writes tracks, scores, and evaluation under
`active/discrimination/`.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The four data days; the last is held out for testing.
DATES = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]
TEST_DATE = "2022-06-27"
TRAIN_DATES = [d for d in DATES if d != TEST_DATE]


def get_data_root() -> str:
    """$WHACK_DATA_ROOT if set, else `data/` beside the repository."""
    return os.environ.get("WHACK_DATA_ROOT") or os.path.join(os.path.dirname(_REPO_ROOT), "data")


def get_scenario_path() -> str:
    """WHACK02 stage-5 scenario JSON (radar physics + geometry)."""
    return os.path.join(get_data_root(), "active", "radar", "scenario.json")


def get_detections_path(date: str, stage: int = 9) -> str:
    """WHACK02 per-day detection CSV (default stage 9: radar-equation + clutter/noise)."""
    return os.path.join(get_data_root(), "active", "radar", f"stage{stage:02d}",
                        f"radar_detections_{date}.csv")


def get_discrimination_dir() -> str:
    """Root for everything this repo writes."""
    return os.path.join(get_data_root(), "active", "discrimination")


def get_tracks_path(date: str) -> str:
    """Per-day candidate-track table (stage 10 output)."""
    return os.path.join(get_discrimination_dir(), "tracks", f"tracks_{date}.csv")


def get_track_points_path(date: str) -> str:
    """Per-day per-scan track points (stage 10 output; input to sequence model)."""
    return os.path.join(get_discrimination_dir(), "tracks", f"track_points_{date}.csv")


def get_method_score_path(method: str, date: str) -> str:
    """Per-method, per-day score file. Each method script (11-17) writes one:
    scores/<method>_<date>.csv with columns [track_id, score_<method>]."""
    return os.path.join(get_discrimination_dir(), "scores", f"{method}_{date}.csv")


def get_all_methods_path(date: str) -> str:
    """Combined per-track scores for every method (stage-18 merges into this)."""
    return os.path.join(get_discrimination_dir(), "scores", f"all_methods_{date}.csv")


def get_real_trajectories_path(date: str, dt_s: float = 10.0) -> str:
    """WHACK01 stage-4 clean GA trajectories (VAE/SDE one-class training source)."""
    tag = f"{dt_s:g}".replace(".", "p") + "s"
    return os.path.join(get_data_root(), "active", f"trajectories_{tag}",
                        f"states_{date}_conventionalGA_trajectories_{tag}.csv")


def get_eval_dir() -> str:
    """Evaluation outputs (stage 18)."""
    return os.path.join(get_discrimination_dir(), "eval")


def get_plot_dir() -> str:
    """Figures directory (isolated under plot/WHACK03-Evaluation/)."""
    return os.path.join(get_data_root(), "plot", "WHACK03-Evaluation")
