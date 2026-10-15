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
    assert reason == "the untrained balanced mean is 0.114 from 0.5, outside the 0.1 floor"
    decision, _ = decision_from_gates(_passing_gates(matched_leak_ok=False), True)
    assert decision == "UNINFORMATIVE"
    decision, reason = decision_from_gates(_passing_gates(), False)
    assert decision == "NOT MET" and "interval clause" in reason
