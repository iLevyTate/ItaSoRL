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


def _bundle(tmp_path, surv=0.72, pred=0.55, untr=0.50, n=6):
    run = tmp_path / "run"
    (run / "cells").mkdir(parents=True)
    rng = np.random.default_rng(0)
    res = {}
    for d in ("0.0", "0.45"):
        res[d] = {}
        for arm, base in (("untrained", untr), ("predictor", pred), ("survival", surv)):
            lvl = base if d == "0.45" else 0.5
            vals = list(np.clip(lvl + 0.01 * rng.normal(size=n), 0, 1))
            res[d][arm] = {"pool_target": vals, "pool_speed": [0.9] * n, "pool_reward_leak": [0.52] * n,
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
