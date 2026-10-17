"""Closing item X02 (E02): scripts/run_development_fixed_walk.py.

Smoke on a tmp snapshot set (small untrained trunks per budget, the L3 surrogate replaced by a
fixed damping law), gated on its own replay targets; and the observation assertion fires when a
recorded episode is shifted against the collected ones.
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
import run_development_fixed_walk as fw  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.eval_protocols import record_trajectories  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, collect_pool, make_world  # noqa: E402

SMALL = ["--seeds", "0", "--n-eps", "12", "--steps", "6"]


def _fake_load(agents_dir, drift, seed, arm, updates=None):
    """A small untrained trunk per (seed, arm, snapshot), the same one on every call."""
    w = make_world(None, 0.0, 5)
    torch.manual_seed(zlib.crc32(f"{drift:.2f}|{seed}|{arm}|{updates}".encode()))
    ag = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return ag, RunningNorm(w.obs_spec.size).freeze()


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(fw, "load", _fake_load)
    monkeypatch.setattr(fw, "setup_l3", lambda: None)
    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    return monkeypatch


def test_smoke_on_a_tmp_snapshot_set(fakes, tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    out = tmp_path / "x02"
    assert fw.main(["--run", f"wm={agents}", "--run", f"nowm={agents}", "--out-dir", str(out), *SMALL]) == 0
    doc = json.loads((out / "fixed_walk_wm.json").read_text())
    assert doc["provenance"]["git_commit_at_promotion"]
    assert doc["integrity"]["gate"].startswith("NOT CHECKED")
    row = doc["rows"][0]
    assert set(row["budgets"]) == {"0", "100", "200", "300", "450"}
    for b in row["budgets"].values():
        assert set(b) == {"replay_target", "resid_obs_fixed", "resid_trace_fixed"}
    assert "label" in doc["summary"]["readings"]["1-fixed (resid_obs 450 - 100)"]
    assert doc["auxiliary_fixed"] is not None
    # gate: the stored development format, built from these replay targets, passes; a moved value stops it
    ref = {"rows": [{"seed": 0}], "summary": {"aggregate": {
        f"{b} replay_target": {"per_seed": [row["budgets"][str(b)]["replay_target"]]} for b in (0, 100, 200, 300, 450)}}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    argv = ["--run", f"wm={agents}", "--out-dir", str(out), "--ckpt", str(out / "ckpt_wm"), *SMALL,
            "--integrity-ref", f"wm={tmp_path / 'ref.json'}"]
    assert fw.main(argv) == 0
    assert json.loads((out / "fixed_walk_wm.json").read_text())["integrity"]["gate"] == "PASS"
    ref["summary"]["aggregate"]["450 replay_target"]["per_seed"][0] += 1e-4
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (out / "fixed_walk_wm.json").unlink()
    assert fw.main(argv) == 1
    assert not (out / "fixed_walk_wm.json").exists()


def test_observation_assertion_fires_on_a_shifted_episode(fakes):
    ag, nm = _fake_load("x", 0.45, 0, "survival", 300)
    P = None
    O, A, K = record_trajectories(ag, nm, P, 0.0, 8, 6, "cpu", 800_000, 5)
    res = collect_pool(ag, nm, P, 0.0, 8, 6, "cpu", 800_000, 5, return_anchors=True, return_obs=True,
                       return_actions=True, return_index=True)
    Ot, At, It = res[7], res[8], res[9]
    fw.assert_same_walk(O, A, K, Ot, At, It, nm)                       # the same walk passes
    shifted = np.roll(O, 1, axis=0)                                     # episode i holds episode i-1's observations
    with pytest.raises(fw.WalkMismatch, match="observations"):
        fw.assert_same_walk(shifted, A, K, Ot, At, It, nm)
    with pytest.raises(fw.WalkMismatch):
        fw.assert_same_walk(O[1:], A[1:], K[1:], Ot, At, It, nm)       # a dropped episode
