"""The scalar audit (over 1400 checks; the script prints the exact count) must be able to
fail (revision audit item 10, finding F56).

Nothing in the suite proved that `scripts/audit_stats_recheck.py` would catch a corrupted
artifact; its sensitivity had only ever been checked by hand. These tests flip one value in a
committed artifact and assert the guard records a failure, and they pin the behavior of the
three check primitives every one of its checks is built from.
"""

from __future__ import annotations

import copy
import io
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import audit_stats_recheck as asr  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_counters():
    """The audit keeps its tally in module globals; give every test a clean slate."""
    before = (asr.n_checks, asr.n_skipped, list(asr.failures))
    asr.n_checks, asr.n_skipped, asr.failures[:] = 0, 0, []
    yield
    asr.n_checks, asr.n_skipped, asr.failures[:] = before[0], before[1], before[2]


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


def _first_block(doc: dict) -> tuple[str, str, str, str]:
    arm = next(iter(doc["aggregate"]))
    drift, agent = arm.split(" ", 1)
    return arm, drift.split("=")[1], agent, next(iter(doc["aggregate"][arm]))


def test_nan_stored_aggregate_fails():
    """A NaN compares unequal to everything, so it must count as a mismatch (finding c28)."""
    doc = copy.deepcopy(asr.load("behavior_audit_l3_h8_traces.json"))
    arm, _, _, metric = _first_block(doc)
    doc["aggregate"][arm][metric]["mean"] = float("nan")
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures, "a NaN stored mean must not pass as a match"


def test_all_nan_cells_fail():
    doc = copy.deepcopy(asr.load("behavior_audit_l3_h8_traces.json"))
    arm, drift, agent, metric = _first_block(doc)
    for cell in doc["cells"]:
        if cell["drift"] == drift and cell["agent"] == agent and metric in cell:
            cell[metric] = float("nan")
    doc["aggregate"][arm][metric]["n_above_bar"] = 0
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures, "an arm whose cells are all non-finite must not pass"


def test_one_nan_cell_fails_even_if_the_rest_reproduce():
    doc = copy.deepcopy(asr.load("behavior_audit_l3_h8_traces.json"))
    _, drift, agent, metric = _first_block(doc)
    for cell in doc["cells"]:
        if cell["drift"] == drift and cell["agent"] == agent and metric in cell:
            cell[metric] = float("nan")
            break
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures, "a dropped non-finite cell must be reported"


def test_aggregate_arm_without_cells_fails():
    doc = copy.deepcopy(asr.load("behavior_audit_l3_h8_traces.json"))
    doc["aggregate"]["drift=9.99 survival"] = {
        "target": {"mean": 0.9, "lo": 0.85, "hi": 0.95, "n_seeds": 0, "n_above_bar": 0}}
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures == ["h8_traces aggregate drift=9.99 survival/target: "
                            "0 finite of 0 cells"], asr.failures


def test_wrong_stored_seed_count_fails():
    doc = copy.deepcopy(asr.load("behavior_audit_l3_h8_traces.json"))
    arm, _, _, metric = _first_block(doc)
    assert "n_seeds" in doc["aggregate"][arm][metric]
    doc["aggregate"][arm][metric]["n_seeds"] = 99
    asr.verify_aggregate_consistency("h8_traces", doc)
    assert asr.failures, "a stored n_seeds that disagrees with the cells must fail"


def _load_expb2(name: str) -> dict:
    with open(os.path.join(asr.ARTROOT, "expB2", name), encoding="utf-8") as fh:
        return json.load(fh)


def test_l0_gate_reproduces_on_the_committed_goal_stakes_runs():
    for run, eq in (("T-touch", True), ("S-immortal", False)):
        e = asr.l0_gate_reproduces(run, _load_expb2(f"goal_stakes_{run}.json"))
        assert bool(e.equivalent) is eq
    assert asr.failures == [], asr.failures


def test_shifted_drift0_seeds_behind_an_unchanged_stored_gate_fail():
    """Finding c25: +0.04 on every T-touch drift-0 seed (mean 0.562, outside the band) with the
    stored gate left alone used to pass every check."""
    doc = copy.deepcopy(_load_expb2("goal_stakes_T-touch.json"))
    blk = doc["arms"]["0.0"]["survival"]["pool_target"]
    blk["per_seed"] = [x + 0.04 for x in blk["per_seed"]]
    e = asr.l0_gate_reproduces("T-touch", doc)
    assert not e.equivalent
    assert asr.failures, "a stored gate that no longer matches its per-seed values must fail"


def test_texture_l0_gate_reproduces_and_fails_when_moved():
    with open(os.path.join(asr.ARTROOT, "texture", "T_gn_l3_h96_wm.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    asr.texture_l0_gate_reproduces("T-gn-h96", doc)
    assert asr.failures == [], asr.failures
    doc["l0_control"]["survival"]["per_seed"] = [
        x - 0.06 for x in doc["l0_control"]["survival"]["per_seed"]]
    asr.texture_l0_gate_reproduces("T-gn-h96 moved", doc)
    assert asr.failures


def test_missing_guarded_artifact_fails_by_name():
    """Finding p5: a tracked artifact that a block guards must fail when it is absent, not
    silently drop the block's checks."""
    assert asr._artifact_present(os.path.join(asr.ARTROOT, "persistence", "corrected_l3_h8_wm.json"))
    assert asr.failures == []
    gone = os.path.join(asr.ARTROOT, "persistence", "renamed_corrected_l3_h8_wm.json")
    assert not asr._artifact_present(gone)
    assert asr.failures == ["artifact present: artifacts/persistence/renamed_corrected_l3_h8_wm.json"]


def test_full_audit_fails_by_name_when_a_guarded_artifact_is_hidden(monkeypatch, capsys):
    """The guard inside main(), not only the helper: hide one guarded artifact from the presence
    check and the full audit must record it as a named failure. A guard reverted to a plain
    existence test would skip the block silently and leave no such failure. Runs the whole audit
    once, about half a minute."""
    hidden = os.path.abspath(os.path.join(asr.ARTROOT, "persistence", "corrected_l3_h8_wm.json"))
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile", lambda p: False if os.path.abspath(p) == hidden else real_isfile(p))
    assert asr.main() == 1
    capsys.readouterr()
    assert "artifact present: artifacts/persistence/corrected_l3_h8_wm.json" in asr.failures, asr.failures


def test_full_audit_fails_by_name_when_the_o2_readout_is_hidden(monkeypatch, capsys):
    """The manuscript cites the O2 optional readout and the supplement ships it (2026-10-16
    review), so a missing docs/optional/o2_long_history_ridge.json is a named failure in a
    checkout and inside the archive alike, never a skip. Runs the whole audit once."""
    hidden = os.path.abspath(os.path.join(ROOT, "docs", "optional", "o2_long_history_ridge.json"))
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile", lambda p: False if os.path.abspath(p) == hidden else real_isfile(p))
    assert asr.main() == 1
    capsys.readouterr()
    assert "file present: docs/optional/o2_long_history_ridge.json" in asr.failures, asr.failures


VO_SPEC = os.path.normcase(os.path.join(ROOT, "docs", "specs", "2026-10-01-two-minds-vo-music.md"))
SUPPLEMENT_README = os.path.normcase(os.path.join(ROOT, "README_SUPPLEMENT.md"))


@pytest.mark.parametrize("where", ["checkout", "supplement"])
def test_full_audit_records_at_least_the_check_floor(where, monkeypatch, capsys):
    """A run that loses checks without failing one is caught by the count (finding p5). Inside
    the anonymized supplement the VO spec is left out and its one check is skipped; the skip is
    counted, so the extracted archive meets the same floor. The supplement case is simulated by
    making README_SUPPLEMENT.md appear and the VO spec disappear for os.path.exists only; no
    repository file is touched. It also records every file the audit opens: the audit must not
    read the LaTeX manuscript. Each case runs the whole audit once, about half a minute. Run
    inside the extracted supplement, README_SUPPLEMENT.md really exists and the VO spec really
    is absent, so there is no checkout to test and the checkout case is skipped; the supplement
    case still runs there."""
    import builtins

    if where == "checkout" and os.path.exists(SUPPLEMENT_README):
        pytest.skip("running inside the extracted supplement; the supplement case covers it")
    if where == "supplement":
        real_exists = os.path.exists

        def exists(p):
            q = os.path.normcase(os.path.abspath(p))
            if q == SUPPLEMENT_README:
                return True
            if q == VO_SPEC:
                return False
            return real_exists(p)

        monkeypatch.setattr(os.path, "exists", exists)
    opened = []
    real_open = builtins.open

    def recording_open(file, *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)):
            opened.append(os.path.normcase(os.path.abspath(os.fsdecode(file))))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(io, "open", recording_open)
    asr.main()
    capsys.readouterr()
    assert asr.n_skipped == (1 if where == "supplement" else 0), asr.n_skipped
    assert asr.n_checks + asr.n_skipped >= asr.CHECK_FLOOR, (asr.n_checks, asr.n_skipped, asr.CHECK_FLOOR)
    assert opened, "the open recorder saw no file; the manuscript check below would be empty"
    paper = os.path.normcase(os.path.join(ROOT, "docs", "paper")) + os.sep
    read_manuscript = [p for p in opened if p.startswith(paper) or os.path.basename(p) == "main.tex"]
    assert not read_manuscript, ("the audit must not read the manuscript", read_manuscript)


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
