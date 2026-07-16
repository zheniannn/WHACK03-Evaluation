"""Stage 12 -- SPRT / sequential log-likelihood ratio (classical, no training).

Per scan, add a target-vs-clutter log-likelihood-ratio increment using the
radar's own Pd(range) and clutter density (from the scenario). The summed LLR is
the score. Model-based statistics -- nothing learned.

Writes  scores/sprt_<TEST_DATE>.csv  [track_id, score_llr].
Usage:  python scripts/12_SPRT.py     (needs the WHACK02 scenario.json)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from utils.io import TEST_DATE, get_scenario_path, get_method_score_path
from utils.data import load_points_by_track
from utils.classical import Physics, per_point_delta_llr


def main() -> None:
    with open(get_scenario_path()) as f:
        physics = Physics(json.load(f))

    tids, cols, (starts, ends) = load_points_by_track(TEST_DATE)
    pd_pt = physics.pd(cols["est_range_m"])
    dllr = per_point_delta_llr(cols["miss"], cols["nis"], cols["logdet_s"], pd_pt, physics.lam)
    score = np.array([float(dllr[s:e].sum()) for s, e in zip(starts, ends)])

    path = get_method_score_path("sprt", TEST_DATE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"track_id": tids, "score_llr": score}).to_csv(path, index=False)
    print(f"SPRT: {len(tids):,} tracks scored -> {path}")


if __name__ == "__main__":
    main()
