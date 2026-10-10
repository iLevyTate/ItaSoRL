"""Round-trip of the reviewer-gap run promotion on a tiny synthetic bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_reviewer_gaps_runs as prg  # noqa: E402


def _bundle(tmp_path, surv=0.72, pred=0.55, untr=0.50, n=6, l0=0.5, speed=0.9):
    run = tmp_path / "run"
    (run / "cells").mkdir(parents=True)
    rng = np.random.default_rng(0)
    res = {}
    for d in ("0.0", "0.45"):
        res[d] = {}
        for arm, base in (("untrained", untr), ("predictor", pred), ("survival", surv)):
            lvl = base if d == "0.45" else (l0 if arm == "survival" else 0.5)
            vals = list(np.clip(lvl + 0.01 * rng.normal(size=n), 0, 1))
            res[d][arm] = {"pool_target": vals, "pool_speed": [speed] * n, "pool_reward_leak": [0.52] * n,
                           "pool_leak_clean": [True] * n, "mp_leak_clean": [True] * n,
                           "pool_deaths_auth": [0] * n, "pool_deaths_surr": [0] * n}
        for s in range(n):
            (run / "cells" / f"cell_d{float(d):.2f}_s{s}.json").write_text(json.dumps(
                {"cell": {"drift": float(d), "seed": s, "eng": {"engaged": True}}}), encoding="utf-8")
    (run / "expB2_results.json").write_text(json.dumps(res), encoding="utf-8")
    cells = [{"drift": "0.45", "seed": s, "agent": arm, "target": v, "resid_trace": v - 0.02,
              "behavior_trace_only": 0.7}
             for arm, base in (("untrained", untr), ("predictor", pred), ("survival", surv))
             for s, v in enumerate(res["0.45"][arm]["pool_target"])]
    (run / "behavior_audit.json").write_text(json.dumps({"cells": cells, "aggregate": {}}), encoding="utf-8")
    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({"g_seed": 1, "rows": [
        {"hidden": 8, "oracle_auroc": 0.93, "in_band": True, "mech_leak_pass": True, "floor": 0.66,
         "floor_ok": False, "passes_gate0": False},
        {"hidden": 10, "oracle_auroc": 0.89, "in_band": True, "mech_leak_pass": True, "floor": 0.48,
         "floor_ok": True, "passes_gate0": True}]}), encoding="utf-8")
    return run, calib


def test_promote_round_trip_positive(tmp_path):
    run, calib = _bundle(tmp_path)
    out = prg.promote(str(run), str(tmp_path / "art" / "x.json"), spec="spec.md", label="test",
                      calibration=str(calib), head="abc")
    back = json.loads((tmp_path / "art" / "x.json").read_text(encoding="utf-8"))
    assert back["decision"]["zone"].startswith("ENCODING INDUCED")
    assert back["decision"]["registered_rule_met"] is True and back["decision"]["gates_open"] == []
    assert back["gates"]["engagement"]["0.45"] == {"pass": 6, "n": 6}
    assert back["gates"]["untrained_floor_ok"] is True
    assert back["gate0_calibration"]["selected_hidden"] == 10
    assert back["behavior_audit"]["survival"]["resid_trace"]["n_seeds"] == 6
    assert back["arms"]["0.45"]["survival"]["pool_target"]["t95"][0] < back["arms"]["0.45"]["survival"]["pool_target"]["t90"][0]
    assert out["git_commit_at_promotion"] == "abc"


def test_promote_intermediate_zone(tmp_path):
    run, calib = _bundle(tmp_path, surv=0.60, pred=0.59)
    out = prg.promote(str(run), str(tmp_path / "y.json"), spec="spec.md", label="t", head="abc")
    assert out["decision"]["zone"].startswith("INTERMEDIATE")
    assert out["decision"]["pass_margin_predictor"] is False


def test_device_control_verdicts(tmp_path):
    nowm, _ = _bundle(tmp_path / "nowm", surv=0.60, pred=0.59)
    # decoder-carrying run clears the bar with room and leads the no-auxiliary run
    wm, _ = _bundle(tmp_path / "wm", surv=0.73, pred=0.59)
    out = prg.promote(str(wm), str(tmp_path / "dc.json"), spec="spec.md", label="dc", head="abc",
                      compare_run=str(nowm))
    dc = out["device_control"]
    assert dc["verdict"].startswith("DEVICE NOT THE CAUSE")
    assert dc["lead"] == pytest.approx(0.13, abs=0.02)
    assert dc["paired_lead_t90"][0] > 0.05
    assert dc["predictor_arm_identical_to_comparison"] is True
    # decoder-carrying run below the bar: the verdict is withdrawn
    low, _ = _bundle(tmp_path / "low", surv=0.62, pred=0.59)
    dc = prg.promote(str(low), str(tmp_path / "dc2.json"), spec="s", label="d", head="abc",
                     compare_run=str(nowm))["device_control"]
    assert dc["verdict"].startswith("DEVICE CONFOUND")
    # clears the bar but leads by less than the margin
    close, _ = _bundle(tmp_path / "close", surv=0.67, pred=0.59)
    base, _ = _bundle(tmp_path / "base", surv=0.64, pred=0.59)
    dc = prg.promote(str(close), str(tmp_path / "dc3.json"), spec="s", label="d", head="abc",
                     compare_run=str(base))["device_control"]
    assert dc["verdict"].startswith("INTERMEDIATE")


def test_promote_without_comparison_has_no_device_block(tmp_path):
    run, _ = _bundle(tmp_path)
    out = prg.promote(str(run), str(tmp_path / "z.json"), spec="s", label="t", head="abc")
    assert "device_control" not in out


def test_execution_field_defaults_to_cloud_and_accepts_override(tmp_path):
    run, _calib = _bundle(tmp_path, surv=0.61, pred=0.53, untr=0.52)
    default = prg.promote(str(run), str(tmp_path / "e1.json"), spec="s", label="d", head="abc")
    assert default["execution"].startswith("cloud CPU sandbox")
    local = prg.promote(str(run), str(tmp_path / "e2.json"), spec="s", label="d", head="abc",
                        execution="owner's GPU machine (RTX 4050, torch 2.7.0+cu126, 2 cuda workers)")
    assert local["execution"].startswith("owner's GPU machine")
    back = json.loads((tmp_path / "e2.json").read_text(encoding="utf-8"))
    assert back["execution"] == local["execution"]


# ---- R221: the registered verdict's name needs every gate, not only the three clauses ----

def test_open_l0_gate_withholds_the_registered_verdict(tmp_path):
    # drift-0 survival targets near 0.58 cannot be shown equivalent to chance (TOST, 0.05)
    run, _ = _bundle(tmp_path, l0=0.58)
    out = prg.promote(str(run), str(tmp_path / "x.json"), spec="s", label="t", head="abc")
    dec = out["decision"]
    assert dec["pass_bar"] and dec["pass_margin_predictor"] and dec["pass_margin_untrained"]
    assert out["gates"]["l0_tost"]["equivalent"] is False
    assert not dec["zone"].startswith("ENCODING INDUCED")
    assert dec["zone"] == "DECODABILITY CLAUSES MET, GATE OPEN: l0"
    assert dec["registered_rule_met"] is False and dec["gates_open"] == ["l0"]


def test_every_registered_gate_counts_not_only_l0(tmp_path):
    run, _ = _bundle(tmp_path, speed=0.70)
    dec = prg.promote(str(run), str(tmp_path / "x.json"), spec="s", label="t", head="abc")["decision"]
    assert dec["zone"] == "DECODABILITY CLAUSES MET, GATE OPEN: speed"
    base = {"pass_bar": True, "pass_margin_predictor": True, "pass_margin_untrained": True,
            "survival": 0.7, "predictor": 0.6, "untrained": 0.5}
    gates = {"engagement": {"0.00": {"pass": 10, "n": 10}, "0.45": {"pass": 9, "n": 10}},
             "l0_tost": {"equivalent": True}, "speed_min": 0.8, "pool_leak_clean_all": False,
             "untrained_floor_ok": False, "deaths_total": 2.0}
    z = prg.zone_for(base, gates)
    assert z["gates_open"] == ["engagement", "reward_leak", "untrained_floor", "deaths"]
    assert z["zone"].startswith("DECODABILITY CLAUSES MET, GATE OPEN: engagement, reward_leak")
    # a gate the artifact does not record cannot be counted as passed
    z = prg.zone_for(base, {**gates, "engagement": {"0.45": {"pass": 10, "n": 10}},
                            "pool_leak_clean_all": True, "untrained_floor_ok": True,
                            "deaths_total": 0.0, "speed_min": None})
    assert z["zone"] == "DECODABILITY CLAUSES MET, GATE NOT RECORDED: speed"
    assert z["registered_rule_met"] is False


def test_near_chance_with_an_open_gate_is_not_a_strengthened_negative(tmp_path):
    run, _ = _bundle(tmp_path, surv=0.50, pred=0.50, untr=0.50, l0=0.58)
    dec = prg.promote(str(run), str(tmp_path / "x.json"), spec="s", label="t", head="abc")["decision"]
    assert dec["zone"] == "ALL ARMS NEAR CHANCE, GATE OPEN: l0"


C1 = "corrected_l3_h8_wm.json"


def _promoted_artifacts():
    out = []
    for path in sorted((ROOT / "artifacts" / "expB2").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and doc.get("generated_by") == "scripts/promote_reviewer_gaps_runs.py":
            marks = ()
            if path.name == C1:
                # Relabelling C1 waits on tests/test_mechanism_readouts.py:326, which pins the
                # old zone and belongs to the goal-and-stakes work. Strict: once C1 is relabelled
                # this XPASSes and fails, so the marker cannot outlive the fix.
                marks = pytest.mark.xfail(strict=True, reason="C1 relabel pending (owner-gated test pin)")
            out.append(pytest.param(path, marks=marks, id=path.name))
    return out


@pytest.mark.parametrize("path", _promoted_artifacts())
def test_committed_artifacts_carry_the_gate_aware_zone(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    want = prg.zone_for(doc["decision"], doc["gates"])
    assert doc["decision"]["zone"] == want["zone"]
    if want["gates_open"]:
        assert not doc["decision"]["zone"].startswith("ENCODING INDUCED")


def _as_old_artifact(path):
    """Rewrite a fresh artifact the way the pre-2026-10-10 promoter wrote it."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["decision"]["zone"] = "ENCODING INDUCED (all clauses)"
    del doc["decision"]["registered_rule_met"], doc["decision"]["gates_open"]
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def test_relabel_rewrites_only_the_zone(tmp_path):
    run, _ = _bundle(tmp_path, l0=0.58)
    art = tmp_path / "art.json"
    prg.promote(str(run), str(art), spec="s", label="t", head="abc")
    before = _as_old_artifact(art)
    r = prg.relabel(str(art), on="2026-10-10")
    assert r["changed"] and r["zone"] == "DECODABILITY CLAUSES MET, GATE OPEN: l0"
    after = json.loads(art.read_text(encoding="utf-8"))
    assert after["zone_relabelled"] == {**after["zone_relabelled"], "on": "2026-10-10",
                                        "previous_zone": "ENCODING INDUCED (all clauses)"}
    assert after["git_commit_at_promotion"] == "abc"
    keep = {k: v for k, v in after.items() if k not in ("zone_relabelled", "decision")}
    assert keep == {k: v for k, v in before.items() if k != "decision"}
    for k in ("zone", "registered_rule_met", "gates_open"):
        after["decision"].pop(k)
    before["decision"].pop("zone")
    assert after["decision"] == before["decision"]
    assert prg.relabel(str(art))["changed"] is False          # a second pass writes nothing


def test_relabel_refuses_when_a_recorded_number_disagrees(tmp_path):
    run, _ = _bundle(tmp_path, l0=0.58)
    art = tmp_path / "art.json"
    prg.promote(str(run), str(art), spec="s", label="t", head="abc")
    doc = _as_old_artifact(art)
    doc["arms"]["0.45"]["survival"]["pool_target"]["mean"] += 0.01
    art.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    with pytest.raises(SystemExit, match="disagrees"):
        prg.relabel(str(art))
    assert json.loads(art.read_text(encoding="utf-8"))["decision"]["zone"] == "ENCODING INDUCED (all clauses)"


# ---- Q38: gate 3 reads every pool, not the lowest arm mean ----

def _set_pool(run, drift, arm, idx, key, value):
    p = run / "expB2_results.json"
    res = json.loads(p.read_text(encoding="utf-8"))
    res[drift][arm][key][idx] = value
    p.write_text(json.dumps(res), encoding="utf-8")


def test_one_pool_under_the_speed_threshold_opens_the_gate_even_when_every_mean_clears_it(tmp_path):
    run, _ = _bundle(tmp_path, speed=0.85)
    _set_pool(run, "0.45", "untrained", 4, "pool_speed", 0.74)
    out = prg.promote(str(run), str(tmp_path / "x.json"), spec="s", label="t", head="abc")
    g = out["gates"]
    assert g["speed_min"] == pytest.approx(0.74)
    assert g["speed_min_at"] == {"drift": "0.45", "arm": "untrained", "seed_index": 4}
    assert g["speed_n_below"] == 1 and g["speed_n_pools"] == 36 and g["speed_n_nonfinite"] == 0
    assert g["speed_min_of_arm_means"] > 0.75
    assert prg.gate_status(g)["speed"] is False
    assert "speed" in out["decision"]["gates_open"]


def test_deaths_are_a_count_and_a_missing_leak_flag_is_not_a_pass(tmp_path):
    run, _ = _bundle(tmp_path)
    _set_pool(run, "0.45", "survival", 0, "pool_deaths_auth", 2)
    p = run / "expB2_results.json"
    res = json.loads(p.read_text(encoding="utf-8"))
    del res["0.0"]["predictor"]["pool_leak_clean"]
    p.write_text(json.dumps(res), encoding="utf-8")
    g = prg.promote(str(run), str(tmp_path / "x.json"), spec="s", label="t", head="abc")["gates"]
    assert g["deaths_total"] == 2 and isinstance(g["deaths_total"], int)
    assert g["pool_leak_clean_all"] is False


def test_refresh_gates_rewrites_only_the_gate_keys_and_records_what_they_were(tmp_path):
    run, _ = _bundle(tmp_path, speed=0.85)
    _set_pool(run, "0.45", "untrained", 4, "pool_speed", 0.74)
    art = tmp_path / "art.json"
    prg.promote(str(run), str(art), spec="s", label="t", head="abc")
    doc = json.loads(art.read_text(encoding="utf-8"))
    # the pre-2026-10-10 form: the lowest arm mean, no per-pool fields, deaths as a float
    for k in ("speed_min_at", "speed_n_pools", "speed_n_below", "speed_n_nonfinite",
              "speed_min_of_arm_means"):
        del doc["gates"][k]
    doc["gates"]["speed_min"] = 0.84
    doc["gates"]["deaths_total"] = 0.0
    doc["decision"]["zone"] = "ENCODING INDUCED (all clauses)"
    art.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    r = prg.refresh_gates(str(art), on="2026-10-10")
    after = json.loads(art.read_text(encoding="utf-8"))
    assert r["changed"] and after["gates"]["speed_min"] == pytest.approx(0.74)
    assert after["gates_refreshed"]["previous"] == {"speed_min": 0.84}
    assert "speed_n_below" in after["gates_refreshed"]["added"]
    assert after["gates_refreshed"]["on"] == "2026-10-10"
    assert after["gates"]["deaths_total"] == 0.0          # equal values keep their recorded form
    assert after["decision"] == doc["decision"]          # the zone is --relabel's job
    assert {k: v for k, v in after.items() if k not in ("gates", "gates_refreshed")} == \
        {k: v for k, v in doc.items() if k != "gates"}
    assert prg.refresh_gates(str(art))["changed"] is False


# The skill-matched artifact keeps the arm-mean figure until the audit pin that quotes it
# (scripts/audit_stats_recheck.py, "pinned: skill-match speed positive control min (0.837)")
# moves to the per-pool value; then run --refresh-gates on it and drop this marker. Strict,
# so the marker cannot outlive the refresh.
U450 = "skill_matched_l3_h8_nowm_u450.json"


def _speed_params():
    out = []
    for p in _promoted_artifacts():
        path = p.values[0]
        marks = ()
        if path.name == U450:
            marks = pytest.mark.xfail(strict=True, reason="refresh waits on the audit pin (handoff)")
        out.append(pytest.param(path, marks=marks, id=path.name))
    return out


@pytest.mark.parametrize("path", _speed_params())
def test_committed_artifacts_record_the_per_pool_speed_minimum(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    res = json.loads((ROOT / doc["source_run"] / "expB2_results.json").read_text(encoding="utf-8"))
    pools = [float(x) for d in res for a in prg.ARMS for x in res[d][a]["pool_speed"]]
    assert doc["gates"]["speed_min"] == pytest.approx(min(pools), abs=1e-12)
    assert doc["gates"]["speed_n_below"] == sum(x < prg.SPEED_MIN for x in pools)
