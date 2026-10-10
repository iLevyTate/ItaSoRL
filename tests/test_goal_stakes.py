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
    def __init__(self, reward, consumed, ate=True):
        self.reward = reward
        self.info = {"ate": ate, "intake": 0.1 if ate else 0.0, "consumed": consumed}


def test_task_reward_follows_the_objective(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "OBJECTIVE", "survival")
    assert b2.task_reward(_R(-0.03, True)) == pytest.approx(-0.03)
    monkeypatch.setattr(b2, "OBJECTIVE", "touch")
    assert b2.task_reward(_R(-0.03, True)) == 1.0
    assert b2.task_reward(_R(0.5, False)) == 0.0          # eating without finishing: 0
    assert b2.task_reward(_R(0.5, False, ate=False)) == 0.0


def test_make_world_applies_mortal_knob(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "MORTAL", False)
    assert _world().mortal is False
    monkeypatch.setattr(b2, "MORTAL", True)
    assert _world().mortal is True
    monkeypatch.setattr(b2, "MORTAL", False)
    assert b2.make_world(P, 0.0, 5, mortal=True).mortal is True    # per-world override


def test_predictor_arm_trains_on_mortal_episodes_whatever_mortal_says(monkeypatch):
    """Integrity check 2 needs the objective-free predictor arm to equal C1's under
    --mortal off. Its scripted training episodes must stop at death as C1's did."""
    pytest.importorskip("torch")
    import torch
    import itasorl.experiment_b2 as b2
    kw = dict(n_eps=8, updates=2, hidden=16, embed=16, max_steps=80, ray_steps=5, seed=3,
              device="cpu")
    # The test bites: in this batch some scripted episodes die inside 80 steps, so an
    # immortal world would hand the predictor longer episodes than C1's predictor saw.
    probe = b2.make_world(P, 0.0, 5)
    agent = b2.RecurrentActorCritic(probe.obs_spec.size, probe.action_spec.size, 16, 16, True)
    _, _, _, mask = b2._collect_scripted(agent, b2.RunningNorm(probe.obs_spec.size),
                                         P, 0.0, 8, 80, "cpu", 200_000 + 3 * 9000, 5)
    assert int(mask.sum(dim=-1).min()) < 80
    monkeypatch.setattr(b2, "MORTAL", True)
    a_mortal, _ = b2.train_predictor_only(0.0, P, **kw)
    monkeypatch.setattr(b2, "MORTAL", False)
    a_off, _ = b2.train_predictor_only(0.0, P, **kw)
    for p, q in zip(a_mortal.state_dict().values(), a_off.state_dict().values()):
        assert torch.equal(p, q)


def test_engagement_rule_survival_uses_absolute_margin(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "OBJECTIVE", "survival")
    better, life = b2.engagement_rule(trained_ret=0.0, rnd_ret=-0.3, scr_ret=-0.2,
                                      trained_len=70.0, rnd_len=68.0)
    assert better is True and life is True
    better, _ = b2.engagement_rule(trained_ret=-0.1, rnd_ret=-0.3, scr_ret=-0.2,
                                   trained_len=70.0, rnd_len=68.0)
    assert better is False                      # +0.10 < ENGAGE_MARGIN 0.15
    better, _ = b2.engagement_rule(trained_ret=-0.2 + b2.ENGAGE_MARGIN, rnd_ret=-0.3,
                                   scr_ret=-0.2, trained_len=70.0, rnd_len=68.0)
    assert better is True                       # exact boundary: best + ENGAGE_MARGIN passes


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


def test_consumed_fires_once_when_a_parked_eater_finishes_a_pellet(monkeypatch):
    """A creature parked on a pellet with eat pressed depletes it by food_gain * dt per
    step (0.1 at the registered 2.0 and 0.05): info["consumed"] is False while it eats and
    True exactly once, on the step the amount reaches zero. The touch reward is that
    event; the survival reward is untouched by it."""
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "MORTAL", False)
    w = _world()
    eat = np.array([0.0, 0.0, 1.0, 0.0, 0.0], np.float32)
    steps_per_pellet = int(round(1.0 / (w.food_gain * w.params.dt)))     # 10 at 2.0 and 0.05
    consumed_steps = []
    for t in range(1, 3 * steps_per_pellet + 1):
        if not consumed_steps:
            # Park on the pellet (terrain gravity would slide the creature out of reach).
            w.pos = w.pellets[0].copy()
            w.vel[:] = 0.0
        r = w.step(eat)
        monkeypatch.setattr(b2, "OBJECTIVE", "touch")
        if r.info["consumed"]:
            consumed_steps.append(t)
            assert w.pellet_amt[0] == 1.0                   # respawned, full again
            assert b2.task_reward(r) == 1.0
            monkeypatch.setattr(b2, "OBJECTIVE", "survival")
            assert b2.task_reward(r) == r.reward
        else:
            assert b2.task_reward(r) == 0.0
            if not consumed_steps:
                assert r.info["ate"] is True              # eating, but not yet finished
    assert consumed_steps == [steps_per_pellet]           # exactly once, when the amount hit zero


def test_collector_true_return_is_consumption_count_under_touch_objective(monkeypatch):
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "OBJECTIVE", "touch")
    monkeypatch.setattr(b2, "MORTAL", False)
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                     device="cpu", seed=0)
    batch = b2.collect_episodes_ac(agent, norm, P, 0.0, n_eps=3, max_steps=12, device="cpu",
                                   seed_base=5, ray_steps=5, deterministic=True)
    assert np.all(batch["lengths"] == 12)                      # immortal: no early end
    assert np.all(batch["ret"] >= 0.0)                         # consumptions are never negative
    assert np.all(batch["ret"] == np.round(batch["ret"]))      # integer counts
    # Consuming a pellet needs ten consecutive eating steps, so a positive sum is not
    # guaranteed for an untrained agent; only non-negativity is.
    assert batch["ret"].sum() >= 0.0


import run_expB2  # noqa: E402

BASE = {"updates": 1, "n_eps": 1, "max_steps": 8, "hidden": 8, "ray_steps": 5,
        "shaping_coef": 1.0, "pool_n": 4, "pool_steps": 4, "mp_pairs": 2, "mp_prefix": 2,
        "mp_branch": 2, "basal_e": None, "n_pellets": None, "reach": None,
        "dump_states": None, "sysid_aux": False, "sysid_coef": 1.0, "drift_mode": "l3",
        "l3_hidden": 8, "l1_delta": 1 / 64, "sensor_sigma": 0.01, "l3_seed": 0,
        "world_model": True, "drifts": [0.0, 0.45], "device": "cpu", "out_dir": ".",
        "save_agents": False}


def test_fingerprint_unchanged_at_default_objective_and_mortal():
    a = run_expB2.config_fingerprint(BASE)
    assert run_expB2.config_fingerprint({**BASE, "objective": "survival", "mortal": True}) == a


def test_fingerprint_changes_with_touch_or_immortal():
    a = run_expB2.config_fingerprint(BASE)
    assert run_expB2.config_fingerprint({**BASE, "objective": "touch"}) != a
    assert run_expB2.config_fingerprint({**BASE, "mortal": False}) != a
    assert (run_expB2.config_fingerprint({**BASE, "objective": "touch", "mortal": False})
            != run_expB2.config_fingerprint({**BASE, "objective": "touch"}))


def test_cli_defaults_and_choices():
    a = run_expB2.cfg_from_argv(["--drift-mode", "l3"])
    assert a.objective == "survival" and a.mortal == "on"
    a = run_expB2.cfg_from_argv(["--drift-mode", "l3", "--objective", "touch", "--mortal", "off"])
    assert a.objective == "touch" and a.mortal == "off"
    with pytest.raises(SystemExit):
        run_expB2.cfg_from_argv(["--objective", "beacon"])


def test_c1_fingerprint_reproduces():
    """The recorded C1 config (artifacts/corrected_runs/corrected_l3_h8_wm/cells/
    cell_d0.00_s0.json) hashes as it did before the objective and mortal knobs."""
    c1 = {"updates": 300, "n_eps": 16, "max_steps": 80, "hidden": 96, "ray_steps": 5,
          "shaping_coef": 1.0, "pool_n": 110, "pool_steps": 24, "mp_pairs": 60, "mp_prefix": 20,
          "mp_branch": 24, "basal_e": None, "n_pellets": None, "reach": None, "dump_states": "x",
          "sysid_aux": False, "sysid_coef": 1.0, "drift_mode": "l3", "l3_hidden": 8,
          "l1_delta": 1 / 64, "sensor_sigma": 0.01, "l3_seed": 0, "world_model": True,
          "survival_updates": None, "gae_bootstrap": "successor", "budget_extend": 450,
          "budget_snapshots": [100, 200], "l3_family": "gmotion", "l3_family_param": None,
          "drifts": [0.0, 0.45], "device": "cpu", "out_dir": "x", "save_agents": True}
    assert run_expB2.config_fingerprint(c1) == "98dfbde61e361984"


def test_calibration_choice_rule():
    import calibrate_scarcity as cs
    def row(n, s0, s1, e0=0.0, e1=0.0):
        return {"n_pellets": n,
                "scripted": {"0.00": {"death_rate": s0, "early_death_rate": e0},
                             "0.45": {"death_rate": s1, "early_death_rate": e1}}}
    rows = [row(24, 0.05, 0.06), row(16, 0.30, 0.33), row(12, 0.45, 0.50), row(8, 0.55, 0.70)]
    assert cs.choose(rows)["n_pellets"] == 12          # first in band, within tolerance
    rows = [row(24, 0.05, 0.06), row(12, 0.45, 0.60), row(8, 0.52, 0.58)]
    assert cs.choose(rows)["n_pellets"] == 8           # 12 is out of tolerance (0.15 gap)
    rows = [row(24, 0.05, 0.06), row(12, 0.45, 0.47, e0=0.01)]
    assert cs.choose(rows) is None                     # an early death disqualifies


def test_death_stats_counts_deaths_and_early_deaths(monkeypatch):
    """A basal burn far above the starting energy starves the creature within a few steps:
    every episode dies, every death is early for a window that covers the death step, and
    none is early for a window that ends just before it. Drift 0 needs no surrogate."""
    import calibrate_scarcity as cs
    import itasorl.experiment_b2 as b2
    monkeypatch.setitem(b2.SURVIVAL_METAB, "basal_E", 400.0)
    r = cs.death_stats("scripted", 0.0, n_eps=2, max_steps=10, ray_steps=5, seed_base=3, window=3)
    assert r["death_rate"] == 1.0
    assert r["early_death_rate"] == 1.0
    assert r["mean_len"] <= 3
    death_step = int(round(r["mean_len"]))
    assert death_step >= 1
    r2 = cs.death_stats("scripted", 0.0, n_eps=2, max_steps=10, ray_steps=5, seed_base=3,
                        window=death_step - 1)
    assert r2["death_rate"] == 1.0
    assert r2["early_death_rate"] == 0.0
