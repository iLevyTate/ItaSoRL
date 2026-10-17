"""Closing item X07 (E10): scripts/run_pellet_dose_readout.py.

The density override reaches experiment_b2.SURVIVAL_FOOD and is recorded; the 24-pellet path
returns regenerate_state_dumps.run_task's target on a tmp agent; --n-pellets 0 exits non-zero
with a message. The agent is a small untrained trunk and the L3 surrogate a fixed damping law.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import itasorl.experiment_b2 as b2  # noqa: E402
import regenerate_state_dumps as rsd  # noqa: E402
import run_pellet_dose_readout as pd  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402


def _fake_bundle(path, device="cpu"):
    w = make_world(None, 0.0, 5)
    torch.manual_seed(zlib.crc32(Path(path).name.encode()))
    ag = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return ag, RunningNorm(w.obs_spec.size).freeze()


@pytest.fixture
def run_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(b2, "load_agent_bundle", _fake_bundle)
    for name in ("DRIFT_MODE", "OBJECTIVE", "MORTAL"):
        monkeypatch.setattr(b2, name, getattr(b2, name))
    monkeypatch.setattr(b2, "SURVIVAL_FOOD", dict(b2.SURVIVAL_FOOD))
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    monkeypatch.setenv("ITASORL_FOLDS", "explicit")
    run = tmp_path / "run"
    (run / "agents").mkdir(parents=True)
    (run / "cells").mkdir()
    for arm in ("survival", "untrained"):
        (run / "agents" / f"agent_d0.45_s0_{arm}.pt").write_bytes(b"")
    knobs = {"n_pellets": None, "reach": None, "basal_e": None, "l3_hidden": 8, "l3_seed": 0}
    cell = {"drift": 0.45, "seed": 0, "objective": "survival", "mortal": True, "knobs": knobs,
            "agents": {a: {"pool": {"target": 0.5}} for a in ("survival", "untrained")}}
    (run / "cells" / "cell_d0.45_s0.json").write_text(json.dumps({"cell": cell}))
    return run


def test_override_reaches_survival_food_and_is_recorded(run_dir, monkeypatch, tmp_path):
    seen = []

    def fake_pooled(agent, norm, P, drift, **kw):
        seen.append(b2.SURVIVAL_FOOD["n_pellets"])
        w = make_world(P, drift, 5)
        w.reset(b2._seeds(800_000))
        seen.append(len(w.pellets))
        return {"target": 0.5}

    monkeypatch.setattr(b2, "pooled_readout", fake_pooled)
    import itasorl.eval_protocols as ep
    real_record = ep.record_trajectories

    def small_record(agent, norm, P, drift, n_eps, *rest, **kw):     # the reference row, at small n
        seen.append(b2.SURVIVAL_FOOD["n_pellets"])
        return real_record(agent, norm, P, drift, 12, *rest, **kw)

    monkeypatch.setattr(ep, "record_trajectories", small_record)
    out = tmp_path / "x07.json"
    assert pd.main(["--run-dir", str(run_dir), "--n-pellets", "6", "24", "--arms", "survival", "untrained",
                    "--out", str(out)]) == 0
    assert seen.count(6) >= 2 * 2 + 2 and seen.count(24) >= 2 * 2 + 2   # flag value, pellet count, reference row
    assert b2.SURVIVAL_FOOD["n_pellets"] == 24                       # restored after every task
    doc = json.loads(out.read_text())
    assert doc["settings"]["n_pellets"] == [6, 24]
    arm_rows = [r for r in doc["rows"] if r["kind"] == "arm"]
    assert sorted({r["n_pellets"] for r in arm_rows}) == [6, 24]
    assert all(r["applied_n_pellets"] == r["n_pellets"] for r in doc["rows"])
    assert all(r["note"] == pd.NOTE for r in doc["rows"])
    assert {r["n_pellets"] for r in doc["rows"] if r["kind"] == "reference"} == {6, 24}
    assert doc["provenance"]["git_commit_at_promotion"]


def test_24_pellet_path_returns_run_task_target(run_dir, monkeypatch, tmp_path):
    real_collect = b2.collect_pool

    def small_collect(agent, norm, params, drift, n_eps, *rest, **kw):  # both paths at the same small n
        return real_collect(agent, norm, params, drift, min(n_eps, 30), *rest, **kw)

    monkeypatch.setattr(b2, "collect_pool", small_collect)
    task = next(t for t in rsd.tasks_for(str(run_dir)) if t["arm"] == "untrained")
    want = rsd.run_task({**task, "out_dir": str(tmp_path)})["target"]
    got = pd.run_density({"kind": "arm", "arm": "untrained", "seed": 0, "n_pellets": 24, "regen": task})
    assert got["target"] == want
    assert got["applied_n_pellets"] == 24


def test_zero_pellets_exits_nonzero_with_a_message(run_dir, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        pd.main(["--run-dir", str(run_dir), "--n-pellets", "0", "24", "--out", str(tmp_path / "o.json")])
    assert e.value.code != 0
    assert "at least one pellet" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()
