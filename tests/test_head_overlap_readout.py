"""Closing item X03 (E04): scripts/run_head_overlap_readout.py.

A synthetic actor whose row equals u reads the ratio 1 / (4/96); an actor orthogonal to u reads
0; the emit row is ignored. A small untrained trunk then runs the script end to end, gated on
its own cell targets.
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
import run_head_overlap_readout as ho  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402

HID = 96


def _agent(hidden: int = HID, seed: int = 0) -> RecurrentActorCritic:
    w = make_world(None, 0.0, 5)
    torch.manual_seed(seed)
    return RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=64, hidden=hidden).train(False)


def _set_actor(agent, W: np.ndarray) -> None:
    with torch.no_grad():
        agent.actor.weight.copy_(torch.as_tensor(W, dtype=torch.float32))


def _rows(u: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n unit rows orthogonal to u and to each other."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([u, rng.standard_normal((HID, n))]))
    return Q[:, 1:].T


def test_actor_row_equal_to_u_gives_ratio_24():
    rng = np.random.default_rng(1)
    u = rng.standard_normal(HID)
    u /= np.linalg.norm(u)
    ag = _agent()
    W = np.vstack([u, _rows(u, 3, 2), 5.0 * rng.standard_normal(HID)])    # thrust, turn, eat, drink, emit
    _set_actor(ag, W)
    ov = ho.head_overlaps(ag, u)["actor"]
    assert ov["rank"] == 4 and ov["expectation"] == pytest.approx(4 / 96)
    assert ov["share_u"] == pytest.approx(1.0, abs=1e-6)
    assert ov["ratio_u"] == pytest.approx(1 / (4 / 96), rel=1e-5)


def test_orthogonal_actor_gives_zero_and_the_emit_row_is_ignored():
    rng = np.random.default_rng(3)
    u = rng.standard_normal(HID)
    u /= np.linalg.norm(u)
    ag = _agent()
    W = np.vstack([_rows(u, 4, 4), 10.0 * u])                               # the emit row IS u
    _set_actor(ag, W)
    ov = ho.head_overlaps(ag, u)
    assert ov["actor"]["share_u"] == pytest.approx(0.0, abs=1e-6)
    assert ov["actor"]["ratio_u"] == pytest.approx(0.0, abs=1e-4)
    assert ov["critic"]["expectation"] == pytest.approx(1 / 96)
    assert ov["decoder_state_block"]["expectation"] == pytest.approx(64 / 96)
    # the action sensitivity along u is zero too: the emit column is not in the env action read
    X = torch.zeros((4, HID))
    r = ho.jvp_ratio(ho.action_fn(ag), X, u, ho._unit_dirs(HID, 20, 0))
    assert r["u"] == pytest.approx(0.0, abs=1e-6) and r["ratio"] == pytest.approx(0.0, abs=1e-4)


def _fake_load(agents_dir, drift, seed, arm, updates=None):
    w = make_world(None, 0.0, 5)
    torch.manual_seed(zlib.crc32(f"{drift:.2f}|{seed}|{arm}".encode()))
    ag = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return ag, RunningNorm(w.obs_spec.size).freeze()


def test_end_to_end_on_a_small_trunk(monkeypatch, tmp_path):
    monkeypatch.setattr(ho, "load", _fake_load)
    monkeypatch.setattr(ho, "setup_l3", lambda: None)
    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    run = tmp_path / "run"
    (run / "agents").mkdir(parents=True)
    (run / "cells").mkdir()
    task = {"agents_dir": str(run / "agents"), "seed": 0, "arm": "survival", "pool_n": 16, "pool_steps": 6,
            "ray_steps": 5, "n_random": 30}
    row = ho.run_cell(task)
    assert {"actor", "critic", "decoder_state_block"} <= set(row["overlap"])
    assert np.isfinite(row["action_sensitivity"]["ratio"]) and np.isfinite(row["recurrence_sensitivity"]["ratio"])
    (run / "cells" / "cell_d0.45_s0.json").write_text(json.dumps(
        {"cell": {"agents": {"survival": {"pool": {"target": row["pooled_target"]}}}}}))
    out = tmp_path / "x03.json"
    argv = ["--agents-dir", str(run / "agents"), "--arms", "survival", "--seeds", "0", "--pool-n", "16",
            "--pool-steps", "6", "--n-random", "30", "--out", str(out)]
    assert ho.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["provenance"]["git_commit_at_promotion"]
    assert doc["integrity"]["gate"] == "PASS"
    (run / "cells" / "cell_d0.45_s0.json").write_text(json.dumps(
        {"cell": {"agents": {"survival": {"pool": {"target": row["pooled_target"] + 0.01}}}}}))
    assert ho.main(argv[:-1] + [str(tmp_path / "none.json")]) == 1
    assert not (tmp_path / "none.json").exists()
