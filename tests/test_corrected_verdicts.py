"""The frozen-rule verdict builder (scripts/build_corrected_verdicts.py, revision step 4)."""

import os
import sys

import numpy as np
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
    assert v["verdict"].startswith("NOT MET as registered (open gate(s): l0); decodability clauses MET")


def test_integrity_fails_closed_without_historical_cells():
    # An absent historical directory used to read compared 0, no mismatches, pass True.
    cells = bcv.load_cells(bcv.RUNS["C1"]["dir"])
    r = bcv.integrity(cells, {})
    assert not r["pass"]
    assert r["historical_cells_missing"] == {"predictor": 20, "untrained": 20}
    assert not bcv.integrity(cells, bcv.load_cells("artifacts/does_not_exist"))["pass"]
    # The full comparison passes and writes no missing-cells key.
    full = bcv.integrity(cells, bcv.load_cells(bcv.RUNS["C1"]["historical"]))
    assert full["pass"] and "historical_cells_missing" not in full


def test_missing_run_gets_no_verdict(monkeypatch):
    monkeypatch.setattr(bcv, "RUNS", {"C1": {"dir": "artifacts/does_not_exist", "replaces": "x",
                                             "historical": HIST_WM, "auxiliary": True}})
    out = bcv.build()
    assert out["runs"]["C1"]["status"].startswith("incomplete")
    assert "auxiliary_comparison" not in out


# ------------------------------------------------- the committed record, read in pytest

def test_committed_verdict_artifact_regenerates_byte_for_byte():
    """The audit job checks this too; pytest checks it on its own so a local run catches a
    builder change that would move a published verdict."""
    with open(bcv.OUT, encoding="utf-8") as fh:
        assert bcv._dump(bcv.build()) == fh.read()


def test_published_corrected_verdicts():
    out = bcv.build()
    c1, c2 = out["runs"]["C1"], out["runs"]["C2"]
    assert c1["status"] == c2["status"] == "complete"
    assert c1["gates"]["failed"] == ["l0"]
    assert c1["primary"]["decodability"] == "MET" and not c1["primary"]["met"]
    assert c1["primary"]["verdict"] == "NOT MET as registered (open gate(s): l0); decodability clauses MET"
    assert c2["gates"]["failed"] == [] and c2["primary"]["verdict"] == "NOT MET"
    ac = out["auxiliary_comparison"]
    assert not ac["holds"] and ac["holds_on_decodability_clauses"]


def test_drift_zero_baselines_on_the_registered_pair():
    """On the registered pair the untrained and predictor arms, each scored on episodes
    its own actor generates, read 0.491 and 0.508 at drift zero; the survival arm reads 0.559."""
    import json

    dz = bcv.build()["runs"]["C1"]["drift_zero"]
    assert f"{dz['untrained']['mean']:.3f}" == "0.491"
    assert [f"{x:.3f}" for x in dz["untrained"]["t90"]] == ["0.466", "0.515"]
    assert f"{dz['predictor']['mean']:.3f}" == "0.508"
    assert [f"{x:.3f}" for x in dz["predictor"]["t90"]] == ["0.475", "0.541"]
    assert f"{dz['survival']['mean']:.3f}" == "0.559"
    # The same readings as the promoted run summary, which reads the same cells.
    with open(os.path.join(bcv.ROOT, "artifacts", "expB2", "corrected_l3_h8_wm.json"),
              encoding="utf-8") as fh:
        arms = json.load(fh)["arms"]["0.0"]
    for arm in bcv.ARMS:
        assert abs(dz[arm]["mean"] - arms[arm]["pool_target"]["mean"]) < 1e-9, arm


# ------------------------------------------------- fail closed

def _cells(surv, pred, untr, drift=0.45):
    return {(drift, s): {"agents": {"survival": {"pool": {"target": surv[s]}},
                                    "predictor": {"pool": {"target": pred[s]}},
                                    "untrained": {"pool": {"target": untr[s]}}}}
            for s in bcv.SEEDS}


_SHAPE = (0.9, -1.1, 0.4, -0.3, 1.3, -0.7, 0.2, -1.2, 0.8, -0.3)   # mean zero, not symmetric


def _vec(mean: float, spread: float) -> list[float]:
    return [mean + spread * x for x in _SHAPE]


def _label(h_b2: bool, zone: str) -> str:
    """decide_h_b2's verdict as a primary() label. Its inconclusive zone starts "NOT met",
    so a prefix match would misread it."""
    if h_b2:
        return "MET"
    if zone.startswith("UNINFORMATIVE"):
        return "UNINFORMATIVE"
    if "inconclusive" in zone:
        return "INCONCLUSIVE"
    return "NOT MET"


def test_primary_agrees_with_decide_h_b2_across_a_grid():
    """The registered rule is codified twice; the two must agree away from exact ties.
    primary() compares mean - mean >= 0.05 and decide_h_b2 compares s >= mean + 0.05, so the
    grid keeps clear of the 0.05 and 0.65 boundaries rather than testing float ties."""
    import run_expB2
    from itasorl.stats import t_ci90

    seen = set()
    for s_mean in np.arange(0.4013, 0.90, 0.0371):
        for b_mean in np.arange(0.4517, 0.70, 0.0433):
            for spread in (0.007, 0.031, 0.083):
                surv = _vec(float(s_mean), spread)
                pred = _vec(float(b_mean), 0.021)
                untr = _vec(float(b_mean) - 0.013, 0.017)
                lo = t_ci90(surv)[0]
                lead = min(np.mean(surv) - np.mean(pred), np.mean(surv) - np.mean(untr))
                if min(abs(lead - 0.05), abs(np.mean(surv) - 0.65), abs(lo - 0.65)) < 1e-6:
                    continue
                ok, zone, _, _ = run_expB2.decide_h_b2(surv, pred, untr)
                got = bcv.primary(_cells(surv, pred, untr), {"failed": []})["decodability"]
                assert got == _label(ok, zone), (s_mean, b_mean, spread, zone)
                seen.add(got)
    assert {"MET", "INCONCLUSIVE", "NOT MET"} <= seen


def test_a_non_finite_survival_target_reads_uninformative_not_a_clean_negative():
    surv = _vec(0.80, 0.03)
    surv[3] = float("nan")
    v = bcv.primary(_cells(surv, _vec(0.55, 0.02), _vec(0.50, 0.02)), {"failed": []})
    assert v["decodability"] == "UNINFORMATIVE"
    assert v["verdict"] == "UNINFORMATIVE"
    assert v["met"] is False
    assert v["non_finite_cells"] == 1


def test_a_non_finite_baseline_also_reads_uninformative():
    pred = _vec(0.55, 0.02)
    pred[0] = float("nan")
    v = bcv.primary(_cells(_vec(0.80, 0.03), pred, _vec(0.50, 0.02)), {"failed": []})
    assert v["decodability"] == "UNINFORMATIVE" and not v["met"]


def test_a_not_met_reading_with_an_open_gate_carries_the_condition():
    """A failed gate leaves every reading conditional on it, NOT MET included."""
    v = bcv.primary(_cells(_vec(0.61, 0.03), _vec(0.59, 0.02), _vec(0.53, 0.02)),
                    {"failed": ["deaths"]})
    assert v["decodability"] == "NOT MET" and not v["met"]
    assert v["verdict"].startswith("NOT MET as registered (open gate(s): deaths)")
    # With every gate passing the label stays bare.
    v = bcv.primary(_cells(_vec(0.61, 0.03), _vec(0.59, 0.02), _vec(0.53, 0.02)), {"failed": []})
    assert v["verdict"] == "NOT MET"
