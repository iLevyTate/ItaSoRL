"""Closing item B2 (runbook 1.6, R13): scripts/run_walk_speed_readout.py.

Smoke on a tmp agent set (small untrained trunks, the L3 surrogate replaced by a fixed damping
law): the four walks for every arm, the speed calibration, the base refusal, and two consistency
checks that tie the new walks to the registered ones: the world-only speed walk keeps the same
episodes as collect_pool_scripted at the same thrust scale, and replaying a scripted action
stream through collect_pool_replayed reproduces collect_pool_scripted's states exactly."""

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
import run_walk_speed_readout as ws  # noqa: E402
from itasorl import post_review as pr  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.eval_protocols import POLICY_SEED_BASE, collect_pool_scripted  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402


def _fake_load(agents_dir, drift, seed, arm, updates=None):
    w = make_world(None, 0.0, 5)
    torch.manual_seed(zlib.crc32(f"{drift:.2f}|{seed}|{arm}|{updates}".encode()))
    ag = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return ag, RunningNorm(w.obs_spec.size).freeze()


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(ws, "load", _fake_load)
    monkeypatch.setattr(ws, "setup_l3", lambda: None)
    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    return monkeypatch


def test_bases_are_the_registered_ones_and_refused_on_collision(monkeypatch):
    assert ws.b2_bases(110) == (2_800_000, 2_850_000)
    with pytest.raises(SystemExit, match="footprint"):
        ws.b2_bases(111)
    monkeypatch.setattr(pr, "all_bases_in_use", lambda: pr.BASES_IN_USE + (("x", 2_800_050, 3),))
    with pytest.raises(SystemExit, match="collide"):
        ws.b2_bases(110)


def test_calibration_bisects_and_flags_an_unreachable_speed():
    c = ws.calibrate(lambda k: 0.1 * k, 0.123)
    assert c["matched"] and abs(c["speed"] - 0.123) <= ws.SPEED_TOL and abs(c["thrust_scale"] - 1.23) < 0.02
    far = ws.calibrate(lambda k: 0.1 * k, 0.5)
    assert not far["matched"] and far["thrust_scale"] == ws.SCALE_HI and "outside" in far["note"]


def test_speed_walk_and_replay_match_the_scripted_pool(fakes):
    ag, nm = _fake_load(None, ws.DRIFT, 0, "survival")
    P = ws.world_params()
    for drift, base in ((0.0, 2_800_000), (ws.DRIFT, 2_850_000)):
        for k in (1.0, 1.4):
            H, kept = collect_pool_scripted(ag, nm, P, drift, 6, ws.STEPS, "cpu", base, ws.RAY_STEPS, thrust_scale=k)
            kw, speed = ws.walk_speed(P, drift, base, 6, ws.scripted_actions(POLICY_SEED_BASE, k))
            assert kw.tolist() == kept.tolist() and np.all(np.isfinite(speed))
        # the scripted stream, recorded and replayed open loop, gives the scripted pool's states
        make = ws.scripted_actions(POLICY_SEED_BASE, 1.0)
        acts = np.stack([np.stack([f(t) for t in range(ws.STEPS)]) for f in (make(i) for i in range(6))])
        Hr, kr = ws.collect_pool_replayed(ag, nm, P, drift, acts, base)
        H, kept = collect_pool_scripted(ag, nm, P, drift, 6, ws.STEPS, "cpu", base, ws.RAY_STEPS)
        assert kr.tolist() == kept.tolist() and np.array_equal(Hr, H)
    sp_fast = ws.pooled_speed(P, (2_800_000, 2_850_000), (6, 6), ws.scripted_actions(POLICY_SEED_BASE, 1.5))
    sp_slow = ws.pooled_speed(P, (2_800_000, 2_850_000), (6, 6), ws.scripted_actions(POLICY_SEED_BASE, 0.6))
    assert sp_fast > sp_slow


def test_smoke_writes_every_walk_for_every_arm(fakes, tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    for arm in ws.ARMS:
        Path(ws.agent_path(str(agents), ws.DRIFT, 0, arm)).write_bytes(b"x")
    out = tmp_path / "b2" / "walk_speed.json"
    argv = ["--agents-dir", str(agents), "--out", str(out), "--seeds", "0", "--n-eps", "8"]
    assert ws.main([*argv, "--dry-run"]) == 0 and not out.exists()
    assert ws.main(argv) == 0
    doc = json.loads(out.read_text())
    prov = doc["provenance"]
    assert prov["git_commit_at_promotion"] and "git_commit" not in prov
    assert {"tracked_changes_at_generation", "stack", "argv", "agents_dir"} <= set(prov)
    assert doc["bases"]["B2-auth"] == 2_800_000 and doc["bases"]["B2-surr"] == 2_850_000
    row = doc["rows"][0]
    assert set(row["arms"]) == set(ws.ARMS)
    for arm in ws.ARMS:
        assert set(row["arms"][arm]) == set(ws.WALKS)
    assert {"survival_own", "scripted", "speed_matched", "replay"} <= set(row["speed"])
    assert row["calibration"]["target_speed"] == row["speed"]["survival_own"]
    assert set(doc["summary"]["contrasts"]["survival"]) == {f"{a} - {b}" for a, b in ws.CONTRASTS}
    assert (tmp_path / "b2" / "ckpt_walk_speed" / "s0.json").exists()
