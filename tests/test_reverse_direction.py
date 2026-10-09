"""Reverse-direction readout (spec docs/specs/2026-10-09-reverse-direction-readout-design.md).

The decision rule and the direction the pools are drawn from are frozen in that spec before the
run, so these tests pin them rather than the other way round. The direction is the whole point of
the run: the agents are the drift-0 ones and the readout drift is 0.45, which is the opposite
pairing from every other cell in the project.
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import reverse_direction as rd  # noqa: E402

BAR = 0.65
MARGIN = 0.05


# --------------------------------------------------------------- the frozen decision rule

def test_carried_when_the_level_and_the_margin_are_both_reached():
    v = rd.adjudicate(survival=0.71, untrained=0.52, bar=BAR, margin=MARGIN)
    assert v["verdict"] == "CARRIED"
    assert v["margin_over_untrained"] == pytest.approx(0.19)


def test_not_carried_when_the_level_is_missed_however_large_the_margin():
    v = rd.adjudicate(survival=0.61, untrained=0.40, bar=BAR, margin=MARGIN)
    assert v["verdict"] == "NOT CARRIED", "the bar is checked before the margin"


def test_ambiguous_when_the_level_is_reached_but_the_untrained_arm_is_close_behind():
    v = rd.adjudicate(survival=0.68, untrained=0.65, bar=BAR, margin=MARGIN)
    assert v["verdict"] == "AMBIGUOUS"


def test_the_three_verdicts_are_exhaustive_and_disjoint_over_a_grid():
    seen = set()
    for s_ in [x / 100 for x in range(40, 91, 2)]:
        for u in [x / 100 for x in range(40, 91, 2)]:
            v = rd.adjudicate(survival=s_, untrained=u, bar=BAR, margin=MARGIN)["verdict"]
            assert v in {"CARRIED", "NOT CARRIED", "AMBIGUOUS"}
            seen.add(v)
    assert seen == {"CARRIED", "NOT CARRIED", "AMBIGUOUS"}


def test_the_bar_and_the_margin_are_inclusive_exactly_as_the_registered_rule_reads():
    assert rd.adjudicate(survival=0.65, untrained=0.60, bar=BAR, margin=MARGIN)["verdict"] == "CARRIED"
    assert rd.adjudicate(survival=0.6499, untrained=0.10, bar=BAR, margin=MARGIN)["verdict"] == "NOT CARRIED"
    assert rd.adjudicate(survival=0.70, untrained=0.6501, bar=BAR, margin=MARGIN)["verdict"] == "AMBIGUOUS"
    # "at least" on both clauses, tested where the difference is exact in binary so the
    # boundary is the rule's and not floating point's.
    exact = rd.adjudicate(survival=0.75, untrained=0.5, bar=0.75, margin=0.25)
    assert exact["margin_over_untrained"] == 0.25 and exact["verdict"] == "CARRIED"


# --------------------------------------------------------------- the direction, and the gate

def test_the_scored_agents_are_the_drift_zero_ones(tmp_path):
    for d in ("0.00", "0.45"):
        for arm in ("survival", "predictor", "untrained"):
            (tmp_path / f"agent_d{d}_s0_{arm}.pt").write_bytes(b"x")
            (tmp_path / f"agent_d{d}_s1_{arm}.pt").write_bytes(b"x")
    cells = rd.select_agents(str(tmp_path), arms=["survival", "untrained"])
    assert {c[0] for c in cells} == {0, 1}
    assert {c[1] for c in cells} == {"survival", "untrained"}
    assert all("_d0.00_" in c[2] for c in cells), "a drift-0.45 agent would be the forward direction"
    assert len(cells) == 4


def test_the_readout_drift_is_045_and_is_passed_by_keyword(monkeypatch):
    """The direction guard: this fails if the runner scores the agents at their own drift."""
    seen = []

    def recorder(agent, norm, params, *args, **kwargs):
        seen.append((args, kwargs.get("drift_sigma")))
        return {"target": 0.5, "n": 4}

    monkeypatch.setattr(rd, "pooled_readout", recorder)
    rd.score_one(object(), object(), None, drift=0.45, n_eps=2, steps=2, ray_steps=4, seed=0,
                 balanced=False)
    assert seen == [((), 0.45)]
    for positional, _ in seen:
        assert positional == (), "drift must be passed by keyword, not positionally"


def test_the_integrity_gate_compares_the_drift_zero_rescore_against_the_recorded_targets():
    ref = {("survival", 0): 0.607, ("survival", 1): 0.538}
    got = {("survival", 0): 0.609, ("survival", 1): 0.534}
    rep = rd.integrity(got, ref, tol=0.01)
    assert rep["pass"] and rep["n_checked"] == 2
    assert rep["worst_abs_dev"] == pytest.approx(0.004, abs=1e-9)

    rep = rd.integrity({("survival", 0): 0.70}, ref, tol=0.01)
    assert not rep["pass"] and rep["worst_abs_dev"] > 0.01


def test_an_empty_integrity_check_is_a_failure_not_a_pass():
    assert not rd.integrity({}, {}, tol=0.01)["pass"], "no comparison is not a passed gate"


def test_the_summary_reports_both_readouts_and_the_seed_interval():
    per_seed = {"survival": [0.60, 0.62, 0.58, 0.61, 0.59, 0.63, 0.57, 0.60, 0.62, 0.58],
                "untrained": [0.50] * 10, "predictor": [0.52] * 10}
    bal = {"survival": [0.66] * 10, "untrained": [0.51] * 10, "predictor": [0.53] * 10}
    s_ = rd.summarize(per_seed, bal, bar=BAR, margin=MARGIN)
    assert s_["standard"]["survival"]["mean"] == pytest.approx(0.60)
    lo, hi = s_["standard"]["survival"]["t90"]
    assert lo < 0.60 < hi
    assert s_["balanced"]["survival"]["mean"] == pytest.approx(0.66)
    # The verdict is adjudicated on the standard readout, the registered estimand, never on the
    # balanced one, which was frozen after the registered rule.
    assert s_["verdict"] == "NOT CARRIED"
