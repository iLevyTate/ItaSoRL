"""The frozen-rule verdict builder (scripts/build_corrected_verdicts.py, revision step 4)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_corrected_verdicts as bcv  # noqa: E402

HIST_WM = "artifacts/reviewer_gaps_runs/l3_h8_wm_cpu"
HIST_NOWM = "artifacts/reviewer_gaps_runs/l3_h8_nowm"


@pytest.fixture
def historical_runs(monkeypatch):
    monkeypatch.setattr(bcv, "RUNS", {
        "C1": {"dir": HIST_WM, "replaces": "self", "historical": HIST_WM, "auxiliary": True},
        "C2": {"dir": HIST_NOWM, "replaces": "self", "historical": HIST_NOWM, "auxiliary": False},
    })
    return bcv.build()


def test_reproduces_the_published_historical_verdicts(historical_runs):
    r = historical_runs["runs"]
    assert r["C1"]["primary"]["verdict"] == "MET"          # device control 0.730
    assert r["C1"]["gates"]["failed"] == []
    assert abs(r["C1"]["primary"]["survival"]["mean"] - 0.730) < 5e-4
    assert r["C2"]["primary"]["verdict"] == "NOT MET"      # no auxiliary 0.601
    assert abs(r["C2"]["primary"]["survival"]["mean"] - 0.601) < 5e-4
    ac = historical_runs["auxiliary_comparison"]
    assert ac["holds"] and abs(ac["c1_minus_c2"] - 0.129) < 1e-3


def test_historical_cells_fail_the_successor_integrity_check(historical_runs):
    # the historical cells predate the gae_bootstrap field, so they cannot pass as corrected
    assert historical_runs["runs"]["C1"]["integrity"]["gae_bootstrap"] == ["missing"]
    assert not historical_runs["runs"]["C1"]["integrity"]["pass"]


def test_a_run_compared_with_itself_has_zero_correction_effect(historical_runs):
    eff = historical_runs["runs"]["C1"]["correction_effect"]["0.45"]["survival_target"]["difference"]
    assert eff["mean"] == 0.0


def test_inconclusive_and_gate_conditional_wording():
    cells = bcv.load_cells(HIST_WM)
    g = {"failed": ["l0"]}
    v = bcv.primary(cells, g)
    assert v["decodability"] == "MET" and not v["met"]
    assert v["verdict"].startswith("MET on the decodability clauses, conditional on open gate(s): l0")


def test_missing_run_gets_no_verdict(monkeypatch):
    monkeypatch.setattr(bcv, "RUNS", {"C1": {"dir": "artifacts/does_not_exist", "replaces": "x",
                                             "historical": HIST_WM, "auxiliary": True}})
    out = bcv.build()
    assert out["runs"]["C1"]["status"].startswith("incomplete")
    assert "auxiliary_comparison" not in out
