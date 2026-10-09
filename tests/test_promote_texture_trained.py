"""The comparator-trained promoter must score the registered gates the way the project does.

Preregistration section 7 gate 3 carries no arm restriction, and the project's own convention,
written into scripts/promote_bv3_gates.py, is "speed probe >= 0.75 in every pool". An earlier
version of this promoter gated on the survival arm only, which understated the failure on the
white-jitter run: survival alone misses by 0.005 on one of twenty cells, while across all sixty
arm-by-cell pools the worst is 0.686 and nine are short. These tests pin the convention.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import promote_texture_trained as pt  # noqa: E402

ARMS = ("untrained", "predictor", "survival")


def _run(speeds):
    """A minimal two-drift run document; speeds is {drift: {arm: [per seed]}}."""
    doc = {}
    for drift, per_arm in speeds.items():
        doc[drift] = {}
        for arm in ARMS:
            n = len(per_arm[arm])
            doc[drift][arm] = {
                "pool_target": [0.52] * n, "pool_speed": list(per_arm[arm]),
                "pool_reward_leak": [0.5] * n,
                "pool_deaths_auth": [0] * n, "pool_deaths_surr": [0] * n,
            }
    return doc


def _flat(v):
    return {"0.0": {a: [v] * 3 for a in ARMS}, "0.45": {a: [v] * 3 for a in ARMS}}


def test_the_speed_gate_reads_every_pool_not_just_the_survival_arm():
    s = _flat(0.90)
    s["0.45"]["untrained"] = [0.90, 0.60, 0.90]      # only a baseline arm is short
    g = pt.speed_gate(_run(s))
    assert not g["pass"], "a short untrained pool must fail the gate"
    assert g["min_all_pools"] == pytest.approx(0.60)
    assert g["n_below"] == 1 and g["n_pools"] == 18
    assert g["min_survival"] == pytest.approx(0.90), "the survival minimum is still recorded"


def test_the_gate_spans_both_drift_slices():
    s = _flat(0.90)
    s["0.0"]["survival"] = [0.90, 0.70, 0.90]        # the short pool is at drift zero
    g = pt.speed_gate(_run(s))
    assert not g["pass"] and g["min_all_pools"] == pytest.approx(0.70)


def test_the_gate_passes_only_when_every_pool_clears_the_threshold():
    g = pt.speed_gate(_run(_flat(0.7501)))
    assert g["pass"] and g["n_below"] == 0
    g = pt.speed_gate(_run(_flat(0.75)))
    assert g["pass"], "the registered threshold is at least 0.75, inclusive"
    g = pt.speed_gate(_run(_flat(0.7499)))
    assert not g["pass"] and g["n_below"] == 18


def test_the_breakdown_names_which_arm_and_drift_fell_short():
    s = _flat(0.90)
    s["0.45"]["untrained"] = [0.60, 0.90, 0.90]
    s["0.0"]["untrained"] = [0.70, 0.70, 0.90]
    g = pt.speed_gate(_run(s))
    assert g["below_by_arm_drift"]["untrained"]["0.45"] == 1
    assert g["below_by_arm_drift"]["untrained"]["0.0"] == 2
    assert "survival" not in g["below_by_arm_drift"]


def test_engagement_is_read_from_the_cells_and_is_not_silently_absent(tmp_path):
    cells = tmp_path / "cells"
    cells.mkdir()
    for seed in range(3):
        (cells / f"c{seed}.json").write_text(json.dumps(
            {"cell": {"drift": 0.45, "seed": seed, "eng": {"engaged": True}}}), encoding="utf-8")
    g = pt.engagement_gate(str(cells))
    assert g["pass"] and g["n_engaged"] == 3 and g["n_cells"] == 3

    (cells / "c1.json").write_text(json.dumps(
        {"cell": {"drift": 0.45, "seed": 1, "eng": {"engaged": False}}}), encoding="utf-8")
    g = pt.engagement_gate(str(cells))
    assert not g["pass"] and g["n_engaged"] == 2


def test_a_missing_cells_directory_is_a_failed_gate_not_an_absent_one(tmp_path):
    g = pt.engagement_gate(str(tmp_path / "nope"))
    assert not g["pass"], "engagement is the gate the matrix routes on; it must never be absent"
    assert g["n_cells"] == 0


def test_the_registered_battery_is_complete_so_a_missing_gate_cannot_pass_unnoticed():
    assert set(pt.REGISTERED_GATES) == {
        "gate0_oracle_band", "engagement", "l0_tost", "speed_positive_control",
        "reward_leakage", "survivorship", "untrained_floor"}
