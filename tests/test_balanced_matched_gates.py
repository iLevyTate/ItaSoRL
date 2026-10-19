"""Matched-world speed and leakage gates, and the refusal to call MET without them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from itasorl.l0_audit import matched_population_gates  # noqa: E402
from promote_balanced_confirmation import decision_from_gates  # noqa: E402


def _passing_gates(**overrides):
    gates = {
        "engagement_ok": True,
        "matched_speed_ok": True,
        "matched_leak_ok": True,
        "matched_speed_collected": True,
        "matched_leak_collected": True,
        "identical_first_state_all": True,
        "shuffled_ok": True,
        "untrained_floor_ok": True,
        "untrained_dev": 0.02,
        "targets_finite": True,
        "dropped_pairs_total": 0,
        "matched_reconstruction_ok": True,
    }
    gates.update(overrides)
    return gates


def test_matched_gates_use_the_retained_pairs_only():
    rng = np.random.default_rng(0)
    n, steps, hidden = 20, 4, 2
    pair_ids = np.arange(n)
    spa = np.linspace(0.2, 1.4, n)
    sps = spa + 0.05
    ha = rng.normal(size=(n, steps, hidden)).astype(np.float32)
    hs = rng.normal(size=(n, steps, hidden)).astype(np.float32)
    ha[:, -1, 0] = spa
    hs[:, -1, 0] = sps
    out = matched_population_gates(
        ha, hs, spa, sps, np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), pair_ids, steps)
    assert out["population"] == "matched pairs" and out["n_pairs"] == n
    assert out["speed"] >= 0.75
    assert abs(out["leakage"]["reward_sum"] - 0.5) >= 0.1
    assert abs(out["leakage"]["length"] - 0.5) < 1e-9


def test_a_flat_speed_channel_is_not_a_finite_pass():
    n, steps, hidden = 20, 4, 2
    ha = np.zeros((n, steps, hidden), np.float32)
    out = matched_population_gates(
        ha, ha.copy(), np.ones(n), np.ones(n), np.zeros(n), np.zeros(n),
        np.zeros(n), np.zeros(n), np.arange(n), steps)
    assert not np.isfinite(out["speed"])


def test_met_requires_the_matched_gates():
    decision, reason = decision_from_gates(_passing_gates(), True)
    assert decision == "MET" and "every required gate" in reason
    decision, reason = decision_from_gates(
        _passing_gates(matched_speed_ok=False, matched_speed_collected=False), True)
    assert decision == "UNINFORMATIVE"
    assert "not collected" in reason
    decision, reason = decision_from_gates(
        _passing_gates(matched_speed_ok=False, untrained_floor_ok=False, untrained_dev=0.114), True)
    assert decision == "UNINFORMATIVE"
    assert reason == ("gates not held: matched speed does not hold or is not finite; "
                      "untrained floor (0.114 from 0.5, outside the 0.1 floor)")
    # Both failing gates are named, not only the first (H78).
    decision, reason = decision_from_gates(
        _passing_gates(matched_speed_ok=False, matched_speed_min=0.7493, untrained_floor_ok=False,
                       untrained_dev=0.114), True)
    assert decision == "UNINFORMATIVE"
    assert "untrained floor" in reason and "matched speed" in reason
    assert reason == ("gates not held: matched speed (min 0.749 under 0.75); "
                      "untrained floor (0.114 from 0.5, outside the 0.1 floor)")
    decision, _ = decision_from_gates(_passing_gates(matched_leak_ok=False), True)
    assert decision == "UNINFORMATIVE"
    decision, reason = decision_from_gates(_passing_gates(), False)
    assert decision == "NOT MET" and "interval clause" in reason


# --- C03 (2026-10-14 review): population contract, reconstruction, finiteness ------------

import copy  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

import promote_balanced_confirmation as pbc  # noqa: E402

SEEDS = list(range(10, 20))
TARGETS = {"survival": 0.80, "predictor": 0.60, "untrained": 0.50}


def _cell(drift, seed):
    cell = {"drift": drift, "seed": seed, "objective": "survival", "mortal": True,
            "stack": {"device": "cpu"}, "adv_norm": True, "eng": {"engaged": True}, "agents": {}}
    for arm in pbc.ARMS:
        block = {"pool": {"speed": 0.9, "pool_leakage": {k: 0.5 for k in pbc.LEAK_KEYS}}}
        if drift == 0.45:
            block["balanced"] = {"drift": 0.45, "seed_base": 910000, "n_pairs": 110, "dropped_pairs": 0,
                                 "target": TARGETS[arm] + 0.002 * (seed - 14), "identical_first_state": True,
                                 "shuffled_within_pair": 0.5}
        cell["agents"][arm] = block
    return cell


def _row(seed, arm, cell):
    return {"seed": seed, "arm": arm, "drift": 0.45, "seed_base": 910000, "n_pairs": 110,
            "dropped_pairs": 0, "target": cell["agents"][arm]["balanced"]["target"],
            "matched_gates": {"population": "matched pairs", "n_pairs": 110, "speed": 0.9,
                              "leakage": {k: 0.5 for k in pbc.LEAK_KEYS}}}


def _write_run(tmp_path, alter_cells=None, alter_doc=None, sidecar=True):
    run = tmp_path / "run"
    (run / "cells").mkdir(parents=True)
    cells = {(d, s): _cell(d, s) for d in (0.0, 0.45) for s in SEEDS}
    if alter_cells:
        alter_cells(cells)
    for (d, s), c in cells.items():
        (run / "cells" / f"cell_d{d:.2f}_s{s}.json").write_text(json.dumps(
            {"cell": c, "git_commit": "2c170f0", "fingerprint": "0123456789abcdef"}))
    if sidecar:
        doc = {"population": "retained pairs", "seed_base": 910000, "objective": "survival", "mortal": True,
               "rows": [_row(s, a, cells[(0.45, s)]) for s in SEEDS for a in pbc.ARMS]}
        if alter_doc:
            alter_doc(doc)
        (run / "matched_gates.json").write_text(json.dumps(doc))
    return str(run)


@pytest.fixture
def run_b1(monkeypatch):
    def make(tmp_path, **kw):
        path = _write_run(tmp_path, **kw)
        monkeypatch.setitem(pbc.RUNS, "B1", path)
        return path
    return make


def test_a_valid_synthetic_run_still_reads_met(tmp_path, run_b1):
    run_b1(tmp_path)
    doc = pbc.promote("B1")
    assert doc["decision"] == "MET", doc["reason"]
    # provenance is read from the cell files, never written as a literal (H77)
    assert doc["git_commit_at_run"] == "2c170f0"
    assert doc["fingerprint"] == "0123456789abcdef"
    assert doc["gates"]["matched_reconstruction"]["status"] == "checked"
    assert doc["gates"]["matched_reconstruction"]["max_abs_diff"] == 0.0


@pytest.mark.parametrize("field,alter", [
    ("seed_base", lambda d: d.update(seed_base=800000)),
    ("objective", lambda d: d.update(objective="touch")),
    ("mortal", lambda d: d.update(mortal=False)),
    ("population", lambda d: d.update(population="pooled")),
    ("seed_base", lambda d: d["rows"][3].update(seed_base=800000)),
    ("drift", lambda d: d["rows"][0].update(drift=0.3)),
    ("n_pairs", lambda d: d["rows"][5].update(n_pairs=5, dropped_pairs=105)),
    ("dropped_pairs", lambda d: d["rows"][5].update(dropped_pairs=2)),
    ("duplicate", lambda d: d["rows"].append(copy.deepcopy(d["rows"][0]))),
    ("missing", lambda d: d["rows"].pop()),
    ("seed set", lambda d: [r.update(seed=99) for r in d["rows"] if r["seed"] == 19]),
])
def test_a_sidecar_off_the_contract_is_refused_naming_the_field(tmp_path, run_b1, field, alter):
    run_b1(tmp_path, alter_doc=alter)
    with pytest.raises(ValueError, match=field):
        pbc.promote("B1")


def test_a_cell_set_of_the_right_size_with_a_wrong_member_is_refused(tmp_path, run_b1):
    def alter(cells):
        c = cells.pop((0.0, 19))
        c["seed"] = 20
        cells[(0.0, 20)] = c
    run_b1(tmp_path, alter_cells=alter)
    with pytest.raises(ValueError, match="seed"):
        pbc.promote("B1")


def test_a_reconstruction_target_off_the_balanced_target_is_refused(tmp_path, run_b1):
    run_b1(tmp_path, alter_doc=lambda d: d["rows"][4].update(target=d["rows"][4]["target"] + 0.02))
    with pytest.raises(ValueError, match="target"):
        pbc.promote("B1")


def test_a_sidecar_without_reconstruction_evidence_cannot_be_met(tmp_path, run_b1):
    run_b1(tmp_path, alter_doc=lambda d: [r.pop("target") for r in d["rows"]])
    doc = pbc.promote("B1")
    assert doc["decision"] == "UNINFORMATIVE"
    assert doc["gates"]["matched_reconstruction"]["status"].startswith("not recorded")


@pytest.mark.parametrize("pos", [0, 14, 29])
def test_a_nan_shuffle_anywhere_prevents_met(tmp_path, run_b1, pos):
    seed, arm = SEEDS[pos // 3], pbc.ARMS[pos % 3]

    def alter(cells):
        cells[(0.45, seed)]["agents"][arm]["balanced"]["shuffled_within_pair"] = float("nan")
    run_b1(tmp_path, alter_cells=alter)
    doc = pbc.promote("B1")
    assert doc["decision"] == "UNINFORMATIVE"
    assert doc["gates"]["shuffled_ok"] is False and doc["gates"]["shuffled_max_abs_dev"] is None


def test_the_collector_refuses_a_changed_configuration_resume():
    import audit_balanced_matched_gates as col
    stored = {"population": "retained pairs", "seed_base": 910000, "objective": "survival", "mortal": True,
              "config": {"drift": 0.45, "n_eps": 110, "steps": 24, "ray_steps": 5, "device": "cuda"},
              "rows": []}
    want = col.invocation_config(seed_base=910000, objective="survival", mortal=True, device="cuda")
    col.check_resume(stored, want)
    with pytest.raises(SystemExit, match="objective"):
        col.check_resume({**stored, "objective": "touch"}, want)
    with pytest.raises(SystemExit, match="device"):
        col.check_resume(stored, col.invocation_config(seed_base=910000, objective="survival",
                                                       mortal=True, device="cpu"))
    with pytest.raises(SystemExit, match="config"):
        col.check_resume({k: v for k, v in stored.items() if k != "config"}, want)


def _rewrite_cell(run, name, edit):
    p = run / "cells" / name
    doc = json.loads(p.read_text())
    edit(doc)
    p.write_text(json.dumps(doc))


def test_a_cell_with_a_different_fingerprint_is_refused_naming_the_file(tmp_path, run_b1):
    run = Path(run_b1(tmp_path))
    _rewrite_cell(run, "cell_d0.45_s13.json", lambda d: d.update(fingerprint="fedcba9876543210"))
    with pytest.raises(ValueError, match=r"fingerprint.*cell_d0\.45_s13\.json"):
        pbc.promote("B1")


def test_a_cell_without_git_commit_is_refused_naming_the_file(tmp_path, run_b1):
    run = Path(run_b1(tmp_path))
    _rewrite_cell(run, "cell_d0.00_s17.json", lambda d: d.pop("git_commit"))
    with pytest.raises(ValueError, match=r"cell_d0\.00_s17\.json.*git_commit"):
        pbc.promote("B1")
