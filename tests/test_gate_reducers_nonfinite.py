"""Gate reducers fail closed on a non-finite input, wherever it sits.

Python's builtin min and max keep the running value whenever a comparison with NaN is False,
so a NaN anywhere but the first position used to drop out and the gate could read as a pass.
Each reducer below must read a NaN in the first pool, a NaN in a later pool, an all-NaN input
and an empty input as not passed, and record how many inputs were non-finite.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import build_corrected_verdicts as bcv  # noqa: E402
import build_gate_table as bgt  # noqa: E402
import promote_bv3_gates as pb  # noqa: E402
import promote_texture_trained as pt  # noqa: E402
from test_promote_bv3_gates import _bundle as _bv3_bundle  # noqa: E402

from itasorl import experiment_b2 as b2  # noqa: E402

NAN = float("nan")
FINITE = [0.52, 0.50, 0.49, 0.51, 0.53, 0.48, 0.50, 0.52, 0.49, 0.51]
POSITIONS = {"first": 0, "later": 6}


def _with_nan(vals, where):
    v = list(vals)
    v[where] = NAN
    return v


# --- itasorl.experiment_b2.leakage_audit_b2 --------------------------------------------------

def _ep(label, rsum, pellets):
    return {"H": np.zeros((3, 8), np.float32), "label": label, "speed": 0.0,
            "reward_sum": rsum, "length": 3, "lifetime": 1, "pellets_consumed": pellets}


def _balanced_eps():
    rng = np.random.default_rng(0)
    auth = [_ep(0, float(rng.random()), float(rng.random())) for _ in range(12)]
    surr = [_ep(1, float(rng.random()), float(rng.random())) for _ in range(12)]
    return auth, surr


@pytest.mark.parametrize("where", sorted(POSITIONS))
def test_leakage_audit_is_not_clean_with_one_nan_channel(monkeypatch, where):
    # grouped_auroc returns NaN for every channel or for none on real inputs (the folds depend
    # only on the groups); forcing one channel pins that the reducer itself is order-free.
    keys = ["reward_sum", "length", "lifetime", "pellets_consumed"]
    vals = dict(zip(keys, [0.52, 0.50, 0.50, 0.51]))
    vals[{"first": "reward_sum", "later": "pellets_consumed"}[where]] = NAN
    order = iter(keys)
    monkeypatch.setattr(b2, "grouped_auroc", lambda v, y, g: vals[next(order)])
    audit = b2.leakage_audit_b2(*_balanced_eps())
    assert audit["clean"] is False
    assert math.isnan(audit["max_abs_dev"])
    assert audit["n_nonfinite"] == 1


def test_leakage_audit_single_class_input_is_all_nan_and_not_clean():
    auth, surr = _balanced_eps()
    for e in surr:
        e["label"] = 0            # one class only: no fold can score, every channel is NaN
    audit = b2.leakage_audit_b2(auth, surr)
    assert audit["clean"] is False
    assert math.isnan(audit["max_abs_dev"])
    assert audit["n_nonfinite"] == 4


def test_leakage_audit_finite_channels_keep_their_old_reading():
    audit = b2.leakage_audit_b2(*_balanced_eps())
    devs = [abs(audit[k] - 0.5) for k in ("reward_sum", "length", "lifetime", "pellets_consumed")]
    assert all(math.isfinite(d) for d in devs) and "n_nonfinite" not in audit
    assert audit["max_abs_dev"] == max(devs)
    assert audit["clean"] is (max(devs) < audit["margin"])


# --- scripts/build_corrected_verdicts.py ------------------------------------------------------

@pytest.mark.parametrize("where", sorted(POSITIONS))
def test_pool_gate_fails_on_a_nan_anywhere(where):
    speeds = _with_nan([0.9] * 10, POSITIONS[where])
    g = bcv._pool_gate(speeds, np.min, "min", lambda x: x >= 0.75)
    assert g["pass"] is False and math.isnan(g["min"]) and g["n_nonfinite"] == 1
    devs = _with_nan([0.01] * 10, POSITIONS[where])
    g = bcv._pool_gate(devs, np.max, "max_abs_dev", lambda x: x < 0.1)
    assert g["pass"] is False and math.isnan(g["max_abs_dev"]) and g["n_nonfinite"] == 1


def test_pool_gate_all_nan_and_empty_do_not_pass():
    g = bcv._pool_gate([NAN] * 4, np.min, "min", lambda x: x >= 0.75)
    assert g["pass"] is False and g["n_nonfinite"] == 4
    g = bcv._pool_gate([], np.min, "min", lambda x: x >= 0.75)
    assert g["pass"] is False and math.isnan(g["min"]) and "n_nonfinite" not in g


def test_pool_gate_finite_input_has_no_nonfinite_key():
    assert bcv._pool_gate([0.9, 0.8], np.min, "min", lambda x: x >= 0.75) == {"min": 0.8,
                                                                               "pass": True}


@pytest.fixture(scope="module")
def historical_cells():
    # The historical C1 cells pass every gate (test_corrected_verdicts pins that), so any
    # failure below comes from the injected NaN alone.
    cells = bcv.load_cells("artifacts/reviewer_gaps_runs/l3_h8_wm_cpu")
    assert bcv.gates(cells)["failed"] == []
    return cells


CELLS = [(d, s, a) for d in bcv.DRIFTS for s in bcv.SEEDS for a in bcv.ARMS]


@pytest.mark.parametrize("key,gate", [("speed", "speed"), ("pool_reward_leak", "reward_leak")])
@pytest.mark.parametrize("cell", CELLS, ids=[f"d{d}-s{s}-{a}" for d, s, a in CELLS])
def test_a_nan_pool_in_any_cell_fails_the_corrected_gate(historical_cells, key, gate, cell):
    d, s, a = cell
    cells = copy.deepcopy(historical_cells)
    cells[(d, s)]["agents"][a]["pool"][key] = NAN
    g = bcv.gates(cells)
    assert g[gate]["pass"] is False and g[gate]["n_nonfinite"] == 1
    assert gate in g["failed"]


# --- scripts/build_gate_table.py --------------------------------------------------------------

@pytest.mark.parametrize("where", sorted(POSITIONS))
def test_l0_block_with_a_nan_seed_is_not_testable(where):
    b = bgt.l0_block(_with_nan(FINITE, POSITIONS[where]))
    assert b["status"] == "not testable (1 of 10 seeds non-finite)"
    assert b["n"] == 9 and b["n_nonfinite"] == 1 and math.isnan(b["tost_p"])


def test_l0_block_all_nan_and_empty():
    b = bgt.l0_block([NAN] * 10)
    assert b["status"] == "not testable (10 of 10 seeds non-finite)" and math.isnan(b["mean"])
    b = bgt.l0_block([])
    assert b["status"] == "not testable (0 of 10 seeds)" and "n_nonfinite" not in b


def test_l0_block_short_finite_list_is_not_testable():
    # Nine finite seeds where ten are registered: TOST on the short sample would print a pass.
    assert bgt.l0_block(FINITE)["status"] == "pass"
    b = bgt.l0_block(FINITE[:9])
    assert b["status"] == "not testable (9 of 10 seeds)" and b["n"] == 9
    assert math.isnan(b["tost_p"]) and "equivalent" not in b
    # A list that matches `expected` is tested: the row carries a real TOST p and a verdict.
    b3 = bgt.l0_block(FINITE[:3], expected=3)
    assert b3["status"] in ("pass", "inconclusive (not shown equivalent)") and b3["n"] == 3
    assert math.isfinite(b3["tost_p"]) and "equivalent" in b3 and "n_nonfinite" not in b3
    assert bgt.l0_block(FINITE[:4], expected=3)["status"] == "not testable (4 of 3 seeds)"


@pytest.mark.parametrize("where", sorted(POSITIONS))
def test_floor_block_with_a_nan_seed_is_not_a_pass(where):
    b = bgt.floor_block(_with_nan(FINITE, POSITIONS[where]))
    assert b["ok"] is False and b["status"] == "not testable (1 of 10 seeds non-finite)"
    assert b["n"] == 9 and b["n_nonfinite"] == 1


def test_floor_block_all_nan_and_empty():
    b = bgt.floor_block([NAN] * 3)
    assert b["ok"] is False and b["n_nonfinite"] == 3 and math.isnan(b["mean"])
    b = bgt.floor_block([])
    assert b["ok"] is False and b["status"] == "not testable (0 of 10 seeds)"


def test_floor_block_short_finite_list_is_not_a_pass():
    b = bgt.floor_block(FINITE[:9])
    assert b["ok"] is False and b["status"] == "not testable (9 of 10 seeds)" and b["n"] == 9
    assert bgt.floor_block(FINITE[:3], expected=3)["status"] == "pass"


def test_floor_block_finite_input_keeps_its_old_shape():
    assert bgt.floor_block(FINITE) == {"mean": pytest.approx(0.505), "ok": True, "status": "pass"}


@pytest.mark.parametrize("vals", [[NAN, 0.9, 0.8], [0.9, NAN, 0.8], [NAN, NAN], []])
def test_speed_min_is_nan_when_any_pool_is_nonfinite(vals):
    assert math.isnan(bgt._min_or_nan(vals))


def test_speed_min_of_finite_pools():
    assert bgt._min_or_nan({"0.0": 0.9, "0.45": 0.8}.values()) == 0.8


def _summary_artifact(speed_min, n_nonfinite):
    """A promoted expB2 summary in the shape promote_reviewer_gaps_runs.py writes, with the
    speed minimum taken over the scored pools only."""
    per_seed = {"per_seed": FINITE}
    arm = {a: {"pool_target": per_seed} for a in ("survival", "untrained")}
    return {"dmax": "0.45", "arms": {"0.0": arm, "0.45": arm},
            "gates": {"engagement": {"0.0": {"pass": 10, "n": 10}},
                      "speed_min": speed_min, "speed_n_pools": 60, "speed_n_below": 0,
                      "speed_n_nonfinite": n_nonfinite, "pool_leak_clean_all": True,
                      "deaths_total": 0}}


def _summary_table(monkeypatch, art):
    monkeypatch.setattr(bgt, "_load", lambda rel: art)
    row = {**bgt._summary_row("X", "expB2/x.json", "explicit"), "status": "corrected"}
    return row, bgt.render_md({"rows": [row]})


def test_summary_row_with_an_unscored_speed_pool_is_not_a_pass(monkeypatch):
    # One unscored pool and every scored pool at 0.9: the finite minimum would print a pass.
    row, md = _summary_table(monkeypatch, _summary_artifact(0.9, 1))
    assert math.isnan(row["speed_min"]) and row["speed_n_nonfinite"] == 1
    assert "| n/a (1 unscored pool) |" in md
    assert "Speed positive control under 0.75 in its lowest pool: 0 of 1 rows." in md
    assert "Speed positive control with unscored pools: `X` explicit (1 unscored, " \
           "speed min n/a)." in md


def test_summary_row_with_an_unscored_pool_keeps_a_failing_minimum(monkeypatch):
    # A scored pool under 0.75 fails the gate whatever the unscored pool held.
    row, md = _summary_table(monkeypatch, _summary_artifact(0.7, 2))
    assert row["speed_min"] == 0.7 and row["speed_n_nonfinite"] == 2
    assert "| 0.700 (2 unscored pools) |" in md
    assert "under 0.75 in its lowest pool: 1 of 1 rows (`X` explicit 0.700)." in md
    assert "with unscored pools: `X` explicit (2 unscored, speed min 0.700)." in md


def test_summary_row_with_scored_pools_keeps_its_old_shape(monkeypatch):
    row, md = _summary_table(monkeypatch, _summary_artifact(0.9, 0))
    assert row["speed_min"] == 0.9 and "speed_n_nonfinite" not in row
    assert "unscored" not in md


# --- scripts/promote_texture_trained.py (also used by promote_off_design_runs.py) -------------

def _run(speed_by_pool):
    """Two drifts, three arms, len(speed_by_pool) // 6 seeds, filled in drift, arm, seed order."""
    n = len(speed_by_pool) // 6
    it = iter(speed_by_pool)
    return {d: {a: {"pool_speed": [next(it) for _ in range(n)]} for a in pt.ARMS}
            for d in ("0.0", "0.45")}


@pytest.mark.parametrize("where", sorted(POSITIONS))
def test_texture_speed_gate_fails_on_a_nan_pool(where):
    g = pt.speed_gate(_run(_with_nan([0.9] * 12, POSITIONS[where])))
    assert g["pass"] is False and g["n_nonfinite"] == 1 and g["n_below"] == 0
    assert g["min_all_pools"] == 0.9          # the minimum is taken over the scored pools


def test_texture_speed_gate_all_nan_and_empty():
    g = pt.speed_gate(_run([NAN] * 6))
    assert g["pass"] is False and g["n_nonfinite"] == 6 and math.isnan(g["min_all_pools"])
    g = pt.speed_gate({})
    assert g["pass"] is False and "n_nonfinite" not in g


def test_texture_speed_gate_finite_pools_have_no_nonfinite_key():
    g = pt.speed_gate(_run([0.9] * 12))
    assert g["pass"] is True and "n_nonfinite" not in g


@pytest.mark.parametrize("where", sorted(POSITIONS))
def test_texture_leak_gate_fails_on_a_nan_pool(where):
    g = pt.leak_gate(_with_nan([0.52] * 10, POSITIONS[where]))
    assert g["pass"] is False and g["n_nonfinite"] == 1 and math.isnan(g["max_abs_dev"])


def test_texture_leak_gate_all_nan_and_empty():
    g = pt.leak_gate([NAN] * 3)
    assert g["pass"] is False and g["n_nonfinite"] == 3
    g = pt.leak_gate([])
    assert g["pass"] is False and math.isnan(g["max_abs_dev"]) and "n_nonfinite" not in g


def test_texture_leak_gate_finite_pools_keep_their_old_shape():
    g = pt.leak_gate([0.52, 0.45, 0.55])
    assert g == {"tolerance": pt.LEAK_TOL, "max_abs_dev": pytest.approx(0.05), "pass": True}


# --- scripts/promote_bv3_gates.py -------------------------------------------------------------

BV3_SPEED = {"first": [NAN, 0.91, 0.92], "later": [0.9, NAN, 0.95], "all": [NAN, NAN, NAN]}


def _bv3_gates(tmp_path, pool_speed=None):
    run_dir = tmp_path / "fullruns" / "07062026"
    _bv3_bundle(run_dir, leak_seed=None)
    if pool_speed is not None:
        p = run_dir / "artifacts" / "expB2_results.json"
        res = json.loads(p.read_text(encoding="utf-8"))
        res["0.45"]["predictor"]["pool_speed"] = pool_speed
        p.write_text(json.dumps(res), encoding="utf-8")
    out = tmp_path / "out.json"
    assert pb.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))["gates"]


@pytest.mark.parametrize("where", sorted(BV3_SPEED))
def test_bv3_speed_gate_fails_on_a_nan_pool(tmp_path, where):
    g = _bv3_gates(tmp_path, BV3_SPEED[where])
    sp = g["speed_positive_control"]
    assert sp["pass"] is False and g["all_pass"] is False
    assert sp["min_speed"]["d=0.45 predictor"] is None
    assert sp["n_nonfinite"]["d=0.45 predictor"] == (3 if where == "all" else 1)
    assert sum(sp["n_nonfinite"].values()) == sp["n_nonfinite"]["d=0.45 predictor"]


def test_bv3_finite_bundle_writes_no_nonfinite_key(tmp_path):
    g = _bv3_gates(tmp_path)
    assert g["speed_positive_control"]["pass"] is True and g["all_pass"] is True
    assert "n_nonfinite" not in g["speed_positive_control"]
