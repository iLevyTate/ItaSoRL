"""Goal-and-stakes runs (docs/specs/2026-10-07-goal-and-stakes-design.md): the world's
mortality switch, the touch objective, the engagement rule per objective, and the two
runner flags' fingerprint behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def _world(drift=0.0):
    import itasorl.experiment_b2 as b2
    w = b2.make_world(P, drift, 5)
    w.reset(b2._seeds(123))
    return w


def test_world_is_mortal_by_default_and_immortal_when_switched():
    w = _world()
    assert w.mortal is True
    w.E = 0.0
    assert w._focal_dead() is True
    w.mortal = False
    assert w._focal_dead() is False
    w.Hyd = 0.0
    assert w._focal_dead() is False


def test_immortal_world_step_never_terminates_and_never_pays_the_death_penalty():
    w = _world()
    w.mortal = False
    w.E = 0.0
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    for _ in range(5):
        r = w.step(a)
        assert r.terminated is False
        assert r.reward > -0.5          # the -1 death transition penalty is never charged
    assert w.alive is True


class _R:
    def __init__(self, reward, ate):
        self.reward, self.info = reward, {"ate": ate, "intake": 0.1 if ate else 0.0}


def test_task_reward_follows_the_objective(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "OBJECTIVE", "survival")
    assert b2.task_reward(_R(-0.03, True)) == pytest.approx(-0.03)
    monkeypatch.setattr(b2, "OBJECTIVE", "touch")
    assert b2.task_reward(_R(-0.03, True)) == 1.0
    assert b2.task_reward(_R(0.5, False)) == 0.0


def test_make_world_applies_mortal_knob(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "MORTAL", False)
    assert _world().mortal is False
    monkeypatch.setattr(b2, "MORTAL", True)
    assert _world().mortal is True


def test_engagement_rule_survival_uses_absolute_margin(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "OBJECTIVE", "survival")
    better, life = b2.engagement_rule(trained_ret=0.0, rnd_ret=-0.3, scr_ret=-0.2,
                                      trained_len=70.0, rnd_len=68.0)
    assert better is True and life is True
    better, _ = b2.engagement_rule(trained_ret=-0.1, rnd_ret=-0.3, scr_ret=-0.2,
                                   trained_len=70.0, rnd_len=68.0)
    assert better is False                      # +0.10 < ENGAGE_MARGIN 0.15


def test_engagement_rule_touch_uses_ratio(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "OBJECTIVE", "touch")
    better, _ = b2.engagement_rule(trained_ret=3.0, rnd_ret=1.0, scr_ret=2.0,
                                   trained_len=80.0, rnd_len=80.0)
    assert better is True                       # 3.0 >= 1.5 * 2.0
    better, _ = b2.engagement_rule(trained_ret=2.9, rnd_ret=1.0, scr_ret=2.0,
                                   trained_len=80.0, rnd_len=80.0)
    assert better is False
    better, _ = b2.engagement_rule(trained_ret=0.0, rnd_ret=0.0, scr_ret=0.0,
                                   trained_len=80.0, rnd_len=80.0)
    assert better is False                      # zero touches never passes


def test_collector_true_return_is_touch_count_under_touch_objective(monkeypatch):
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "OBJECTIVE", "touch")
    monkeypatch.setattr(b2, "MORTAL", False)
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                     device="cpu", seed=0)
    batch = b2.collect_episodes_ac(agent, norm, P, 0.0, n_eps=3, max_steps=12, device="cpu",
                                   seed_base=5, ray_steps=5, deterministic=True)
    assert np.all(batch["lengths"] == 12)                      # immortal: no early end
    assert np.all(batch["ret"] >= 0.0)                         # touches are never negative
    assert np.all(batch["ret"] == np.round(batch["ret"]))      # integer counts
