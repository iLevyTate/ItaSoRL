"""Closing item X01 (E01): scripts/run_nudge_dose_readout.py.

The positive-control direction p comes from the actor rows with the dead emit row dropped: on a
synthetic linear-policy agent whose emit row dominates, p is the thrust direction and the push
moves the env action. Magnitude 1.0 with --positive-control none reproduces
run_mechanism_readouts' gap_closed path to the bit.

The saved agents are small untrained trunks and the L3 surrogate a fixed damping law; the
worlds, rollouts and probes run for real at small n.
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from _git_checkout import assert_commit_recorded

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import itasorl.control_diagnostics as cd  # noqa: E402
import itasorl.experiment_b2 as b2  # noqa: E402
import itasorl.experiment_c as ec  # noqa: E402
import run_mechanism_readouts as rmr  # noqa: E402
import run_nudge_dose_readout as nd  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402

HID = 8


def _agent(seed: int = 0) -> RecurrentActorCritic:
    w = make_world(None, 0.0, 5)
    torch.manual_seed(seed)
    return RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=HID).train(False)


def test_linear_policy_moves_along_p_with_the_emit_row_excluded():
    ag = _agent()
    e = np.eye(HID)
    W = np.zeros((5, HID))
    W[0] = 3.0 * e[1]                     # thrust mean
    W[1] = 1.0 * e[2]                     # turn mean
    W[4] = 100.0 * e[0]                   # emit logit: dominant, and read by nothing
    with torch.no_grad():
        ag.actor.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
        ag.actor.bias.zero_()
    p, sv = nd.actor_direction(ag)
    assert abs(p @ e[1]) > 1 - 1e-9 and p @ e[1] > 0
    assert abs(p @ e[0]) < 1e-12           # the emit direction is not the control
    assert sv[0] == pytest.approx(3.0)
    # with the emit row kept the top direction would be the emit one
    assert abs(np.linalg.svd(W)[2][0] @ e[0]) > 1 - 1e-9
    shift = nd.env_action_shift(ag, np.zeros((5, HID)), p, 1.0)
    assert shift["columns"] == ["thrust", "turn", "eat", "drink"]
    thrust, turn, eat, drink = shift["mean_abs_shift"]
    assert thrust > 0.1 and turn < 1e-6 and eat < 1e-6 and drink < 1e-6
    assert nd.env_action_shift(ag, np.zeros((5, HID)), e[0], 1.0)["mean_abs_shift"] == [0.0, 0.0, 0.0, 0.0]


def _fake_bundle(path, device="cpu"):
    ag = _agent(zlib.crc32(os.path.basename(path).encode()))
    w = make_world(None, 0.0, 5)
    return ag, RunningNorm(w.obs_spec.size).freeze()


def test_magnitude_one_without_control_reproduces_the_mechanism_path(monkeypatch, tmp_path):
    monkeypatch.setattr(b2, "load_agent_bundle", _fake_bundle)
    monkeypatch.setattr(b2, "DRIFT_MODE", b2.DRIFT_MODE)
    monkeypatch.setattr(b2, "OBJECTIVE", b2.OBJECTIVE)
    monkeypatch.setattr(b2, "MORTAL", b2.MORTAL)
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    monkeypatch.setattr(cd, "sequence_gru_auroc", lambda *a, **k: {"auroc": float("nan")})
    monkeypatch.setattr(ec, "value_of_world_information", lambda *a, **k: {"value_of_information": float("nan")})
    common = {"run_dir": str(tmp_path), "drift": 0.45, "n_eps": 10, "steps": 8, "pool_n": 16, "pool_steps": 6,
              "ray_steps": 5, "objective": "survival", "mortal": True, "knobs": {}, "seed": 0}
    ref = rmr.run_seed({**common, "gru_epochs": 1,
                        "cells": {"0.00/0": {"xeval": {}}, "0.45/0": {"xeval": {}}}})
    mine = nd.run_seed({**common, "arms": ["survival", "untrained"], "magnitudes": [1.0],
                        "positive_control": "none"})
    compared = 0
    for arm, want in (("survival", ref["intervention"]), ("untrained", ref["untrained_arm"]["intervention"])):
        got = mine[arm]
        assert got["n_common"] == want["n_common"]
        if not want["informative"] and want.get("reason") == "empty pool":
            continue
        compared += 1
        m1 = got["magnitudes"]["1"]
        for k in ("score_real", "score_sham", "score_real_reverse", "score_sham_reverse", "gap"):
            assert json.dumps(m1[k]) == json.dumps(want[k]), (arm, k)
        assert m1["informative"] == want["informative"]
        assert got["u"] == want["u"] and got["v"] == want["v"]
        for k in ("score_real", "score_sham", "gap", "n_common"):
            assert json.dumps(got["integrity_1s"][k]) == json.dumps(want[k]), (arm, k)
    assert compared == 2
    assert "control" not in mine["survival"]

    # end to end with the control and two magnitudes, gated on a mechanism-format reference
    run = tmp_path / "run"
    (run / "cells").mkdir(parents=True)
    for d in ("0.00", "0.45"):
        (run / "cells" / f"cell_d{d}_s0.json").write_text(json.dumps(
            {"cell": {"drift": float(d), "seed": 0, "knobs": {}, "objective": "survival", "mortal": True}}))
    small = {k: common[k] for k in ("n_eps", "steps", "pool_n", "pool_steps")}
    ref1 = nd.run_seed({**common, **small, "run_dir": str(run), "arms": ["survival", "untrained"],
                        "magnitudes": [1.0], "positive_control": "none"})
    mech = {"per_seed": [{"seed": 0, "intervention": ref1["survival"]["integrity_1s"],
                          "untrained_arm": {"intervention": ref1["untrained"]["integrity_1s"]}}]}
    (tmp_path / "mech.json").write_text(json.dumps(mech))
    out = tmp_path / "x01.json"
    argv = ["--run-dir", str(run), "--out", str(out), "--seeds", "0", "--magnitudes", "1", "4",
            "--n-eps", "10", "--steps", "8", "--pool-n", "16", "--pool-steps", "6",
            "--integrity-ref", str(tmp_path / "mech.json")]
    assert nd.main(argv) == 0
    doc = json.loads(out.read_text())
    assert_commit_recorded(doc["provenance"])
    assert doc["integrity"]["gate_I"] == "PASS"
    blk = doc["per_seed"][0]["survival"]
    assert set(blk["n_common_per_magnitude"]) == {"1", "4"}
    assert blk["control"]["direction"] == "actor rows without the emit row"
    assert "label" in doc["summary"]["survival"] and "floor_passes_somewhere" in doc["summary"]["untrained"]
