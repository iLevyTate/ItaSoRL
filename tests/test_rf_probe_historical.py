"""The historical random-forest probe artifact (hidden-8 dumps of fullruns/l3_n10).

Aggregates recompute from the stored cells, every linear value equals the committed fold
re-score of the same dumps, and the reproduction-attempt summary follows from its cells. One
cell is re-scored live when the gitignored dumps are present; the test skips cleanly without
them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "artifacts" / "agent_free" / "rf_probe_l3_n10_historical.json"


@pytest.fixture(scope="module")
def doc():
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def test_complete_and_labeled_historical(doc):
    assert doc["complete"] and doc["n_cells"] == 60 == len(doc["cells"])
    assert doc["label"].startswith("HISTORICAL")
    keys = {(c["drift"], c["seed"], c["agent"]) for c in doc["cells"]}
    assert keys == {(d, s, a) for d in ("0.00", "0.45") for s in range(10)
                    for a in ("survival", "predictor", "untrained")}


def _diff(a, b, path="") -> list:
    """Paths where a and b differ: exact for keys, strings, booleans and integers, within 1e-9
    for floats, whose last bits (t quantiles) depend on the scipy and numpy build."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        return [d for k in a for d in _diff(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length differs"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and not isinstance(a, bool):
        if np.isnan(a) and np.isnan(b):
            return []
        return [] if abs(a - b) <= 1e-9 else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def test_aggregate_recomputes(doc):
    import rf_probe_historical_h8 as rf
    diff = _diff(rf.aggregate(doc["cells"]), doc["aggregate"])
    assert not diff, diff[:5]
    assert rf.attempts_summary(doc["cells"]) == doc["reproduction_attempts"]["summary"]


def test_linear_matches_committed_fold_rescore(doc):
    with open(ROOT / "artifacts" / "fold_rescore" / "l3_n10.json", encoding="utf-8") as fh:
        fr = {(c["drift"], c["seed"], c["agent"]): c for c in json.load(fh)["cells"]}
    for c in doc["cells"]:
        r = fr[(c["drift"], c["seed"], c["agent"])]
        for scheme in ("legacy", "explicit"):
            assert c[scheme]["linear"] == r[scheme]["target"], (c["drift"], c["seed"], c["agent"])
    rep = doc["reproduction"]
    assert rep["vs_fold_rescore"]["max_abs_diff_legacy"] == 0.0
    assert rep["vs_run_record"]["max_abs_diff_linear_target"] == 0.0
    assert rep["vs_run_record"]["max_abs_diff_linear_shuffled_first"] == 0.0


def test_published_headline_is_the_legacy_linear_mean(doc):
    agg = doc["aggregate"]["d=0.45 survival"]["legacy"]["linear"]
    assert round(agg["mean"], 3) == 0.752 and agg["n_ge_065"] == 8


def test_live_cell_when_dumps_present(doc):
    import rf_probe_historical_h8 as rf
    path = os.path.join(rf.PRIMARY_FULLRUNS, "l3_n10", "states", "states_d0.45_s0_untrained.npz")
    if not os.path.exists(path):
        pytest.skip("gitignored dump fullruns/l3_n10/states not present")
    from itasorl import folds
    from itasorl.experiment_b import episode_features
    with np.load(path) as z:
        Ha, Hs = np.asarray(z["Ha"]), np.asarray(z["Hs"])
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    X = episode_features(np.concatenate([Ha, Hs]))
    cell = next(c for c in doc["cells"]
                if (c["drift"], c["seed"], c["agent"]) == ("0.45", 0, "untrained"))
    with folds.fold_scheme("legacy"):
        assert rf.rf_auroc(X, y) == cell["legacy"]["rf"]
