"""The off-design (T-qd) and failed-gates (hidden 4) promotions are internally consistent and
carry the values the record states in prose. The full promotion is re-run when the
gitignored run bundles are present; the test skips that part cleanly without them."""

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

ARTS = {"tqd": ROOT / "artifacts" / "texture" / "T_qd_l3_h8_wm.json",
        "h4": ROOT / "artifacts" / "expB2" / "failed_gates_l3_h4_traces.json"}


def _load(which):
    with open(ARTS[which], encoding="utf-8") as fh:
        return json.load(fh)


def _same(a, b, path="") -> list:
    """Paths where a and b differ: exact for keys, strings, booleans and integers, and within
    1e-9 for floats, whose last bits (t quantiles, bootstrap means) depend on the scipy and
    numpy build that wrote the artifact."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(a)} != {sorted(b)}"]
        return [d for k in a for d in _same(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _same(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
            return [] if a == b else [f"{path}: {a!r} != {b!r}"]
        if np.isnan(a) and np.isnan(b):
            return []
        return [] if abs(a - b) <= 1e-9 else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


@pytest.mark.parametrize("which", sorted(ARTS))
def test_summaries_recompute_from_per_seed(which):
    import promote_off_design_runs as po
    doc = _load(which)
    assert doc["rule_met"] is False and doc["gates_failed"]
    for drift, arms in doc["pooled_target"].items():
        for arm, e in arms.items():
            assert not _same(po._agg(e["per_seed"]), e), (drift, arm, _same(po._agg(e["per_seed"]), e))
    pools = doc["speed_pools"]
    sp = doc["gates"]["speed_positive_control"]
    assert len(pools) == sp["n_pools"] == 60
    assert sum(p["speed"] < 0.75 for p in pools) == sp["n_below"]
    assert min(p["speed"] for p in pools) == sp["min_all_pools"]


def test_tqd_matches_the_record():
    doc = _load("tqd")
    assert doc["status"] == "OFF-DESIGN" and doc["label"].startswith("OFF-DESIGN")
    rec = doc["recorded_in_prose"]
    s = doc["pooled_target"]["0.45"]
    assert round(s["survival"]["mean"], 3) == rec["survival_mean"]
    assert np.round(s["survival"]["t90"], 3).tolist() == rec["survival_t90"]
    assert np.round(s["survival"]["seed_boot90"], 3).tolist() == rec["survival_boot90"]
    assert round(s["predictor"]["mean"], 3) == rec["predictor_mean"]
    assert np.round(s["predictor"]["seed_boot90"], 3).tolist() == rec["predictor_boot90"]
    assert round(s["untrained"]["mean"], 3) == rec["untrained_mean"]
    assert np.round(s["untrained"]["seed_boot90"], 3).tolist() == rec["untrained_boot90"]
    sp = doc["gates"]["speed_positive_control"]
    assert sp["n_below"] == rec["speed_pools_below"] and round(sp["min_all_pools"], 4) == rec["speed_worst"]
    assert doc["gates"]["engagement"]["n_engaged"] == rec["engaged_cells"]
    assert doc["trunk_hidden"] == rec["trunk"] and doc["trunk_check"]["n_agents"] == 60
    assert round(doc["gates"]["l0_tost"]["mean"], 3) == rec["l0_mean"]
    assert not doc["gates"]["gate0_oracle_band"]["any_in_band"]
    assert doc["run"]["git_commit_at_run"] == ["2606e7f"]


def test_h4_matches_the_record():
    doc = _load("h4")
    assert doc["status"] == "FAILED GATES" and "UNINFORMATIVE" in doc["label"]
    rec = doc["recorded_in_prose"]
    assert round(doc["gates"]["untrained_floor"]["mean"], 3) == rec["untrained_floor"]
    assert doc["gates"]["reward_leakage"]["seeds_clean_survival"] == rec["reward_leak_clean_seeds"]
    e45 = doc["gates"]["engagement"]["engaged_by_drift"]["0.45"]
    assert e45["engaged"] / e45["cells"] == rec["engagement_share"]
    assert doc["trunk_hidden"] == 96 and doc["trunk_source"].startswith("dumped")
    g0 = doc["gates"]["gate0_oracle_band"]
    assert g0["in_band"] and not g0["pass"]
    with open(ROOT / g0["source"], encoding="utf-8") as fh:
        inst = next(i for i in json.load(fh)["instances"] if i["id"] == g0["instance"])
    assert inst["schemes"]["legacy"]["oracle_auroc"] == g0["oracle_auroc_legacy"]


@pytest.mark.parametrize("which", sorted(ARTS))
def test_live_promotion_when_bundle_present(which):
    import promote_off_design_runs as po
    run = os.path.join(po.PRIMARY_FULLRUNS, po.RUNS[which]["run_dir"], "expB2_results.json")
    if not os.path.exists(run):
        pytest.skip("gitignored run bundle not present")
    pytest.importorskip("torch")
    again = json.loads(json.dumps(po.promote(which)))
    doc = _load(which)
    skip = {"git_commit_at_promotion", "tracked_changes_at_generation", "stack", "generated_by"}
    assert {k: v for k, v in again.items() if k not in skip} == \
           {k: v for k, v in doc.items() if k not in skip}
