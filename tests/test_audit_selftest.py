"""The 1188-check scalar audit must be able to fail (revision audit item 10, finding F56).

Nothing in the suite proved that `scripts/audit_stats_recheck.py` would catch a corrupted
artifact; its sensitivity had only ever been checked by hand. These tests flip one value in a
committed artifact and assert the guard records a failure, and they pin the behavior of the
three check primitives every one of the 1188 checks is built from.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import audit_stats_recheck as asr  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_counters():
    """The audit keeps its tally in module globals; give every test a clean slate."""
    before = (asr.n_checks, list(asr.failures))
    asr.n_checks, asr.failures[:] = 0, []
    yield
    asr.n_checks, asr.failures[:] = before[0], before[1]


def test_check_records_a_failure_only_when_the_value_moves():
    asr.check("same", 0.7520, 0.752)
    assert asr.failures == []
    asr.check("moved", 0.8020, 0.752)
    assert asr.failures == ["moved"]
    assert asr.n_checks == 2


def test_check_rejects_a_non_finite_value():
    asr.check("not a number", float("nan"), 0.752)
    assert asr.failures == ["not a number"]


def test_check_int_and_check_true_record_failures():
    asr.check_int("int same", 8, 8)
    asr.check_true("true", True)
    assert asr.failures == []
    asr.check_int("int moved", 7, 8)
    asr.check_true("false", False)
    assert asr.failures == ["int moved", "false"]


def test_aggregate_consistency_passes_on_the_committed_artifact():
    doc = asr.load("behavior_audit_l3_h8_traces.json")
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures == [], asr.failures


def test_flipping_one_cell_value_makes_the_audit_fail():
    """The negative self-test F56 asked for: one artifact value moved, a FAIL recorded."""
    doc = copy.deepcopy(asr.load("behavior_audit_l3_h8_traces.json"))
    for cell in doc["cells"]:
        if "target" in cell:
            cell["target"] = float(cell["target"]) + 0.05
            break
    else:
        pytest.fail("no cell carries a 'target' metric")
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures, "a moved cell value must not reproduce the stored aggregate"
    assert all(f.startswith("h8_traces aggregate") for f in asr.failures), asr.failures


def test_flipping_a_stored_aggregate_mean_makes_the_audit_fail():
    doc = copy.deepcopy(asr.load("behavior_audit_l3_h8_traces.json"))
    arm = next(iter(doc["aggregate"]))
    metric = next(iter(doc["aggregate"][arm]))
    doc["aggregate"][arm][metric]["mean"] += 0.05
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures, "a moved stored mean must not reproduce from the cells"


def test_seed_vals_returns_the_cells_in_seed_order():
    doc = asr.load("behavior_audit_l3_h8_traces.json")
    arm = next(iter(doc["aggregate"]))
    drift = arm.split(" ", 1)[0].split("=")[1]
    agent = arm.split(" ", 1)[1]
    metric = next(iter(doc["aggregate"][arm]))
    vals = asr.seed_vals(doc, drift, agent, metric)
    seeds = sorted(c["seed"] for c in doc["cells"]
                   if c["drift"] == drift and c["agent"] == agent and metric in c)
    assert len(vals) == len(seeds) and seeds == sorted(seeds)
