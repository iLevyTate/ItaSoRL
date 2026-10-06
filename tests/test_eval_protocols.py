"""Revision step 6: evaluation protocols that hold the policy or the data fixed across arms."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.eval_protocols import (  # noqa: E402
    collect_pool_scripted,
    readout_from_states,
    record_trajectories,
    replay_states,
    train_predictor_on_logged,
)
from itasorl.experiment_b2 import RunningNorm, collect_pool, make_world  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent(seed=0):
    w = make_world(P, 0.0, RS)
    torch.manual_seed(seed)
    a = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return a, RunningNorm(w.obs_spec.size).freeze()


def test_scripted_protocol_gives_every_arm_the_same_trajectory():
    """Two different agents under the scripted protocol see identical observations, so a
    third agent replaying either one's inputs reproduces its states exactly."""
    a1, n1 = _agent(0)
    a2, n2 = _agent(1)
    H1, k1 = collect_pool_scripted(a1, n1, P, 0.0, 4, 6, "cpu", 800_000, RS)
    H2, k2 = collect_pool_scripted(a2, n2, P, 0.0, 4, 6, "cpu", 800_000, RS)
    assert np.array_equal(k1, k2) and len(k1) == 4
    assert not np.allclose(H1, H2)            # different trunks, same inputs


def test_replay_reproduces_the_closed_loop_states():
    """Replaying an agent's own recorded trajectory into itself reproduces the states it
    had while acting: replay is the open-loop version of the same computation."""
    a, n = _agent(0)
    obs, act, kept = record_trajectories(a, n, P, 0.45, 4, 6, "cpu", 800_000, RS)
    H_closed, _ = collect_pool(a, n, P, 0.45, 4, 6, "cpu", 800_000, RS)
    H_replay = replay_states(a, n, obs, act)
    np.testing.assert_allclose(H_replay, H_closed, atol=1e-5)


def test_readout_from_states_is_chance_on_identical_pools():
    H = np.random.default_rng(0).normal(size=(12, 5, 8)).astype(np.float32)
    r = readout_from_states(H, H.copy(), groups=np.r_[np.arange(12), np.arange(12)])
    assert r["target"] == pytest.approx(0.5, abs=1e-12)


def test_logged_training_batches_reproduce_the_trainer_and_feed_a_predictor():
    """Logging the survival trainer's batches changes nothing it computes, and a predictor
    trained on those batches takes one update per logged batch, deterministically."""
    from itasorl.experiment_b2 import train_actor_critic
    kw = dict(n_eps=3, hidden=8, embed=16, max_steps=10, ray_steps=RS, seed=2, device="cpu",
              shaping_coef=1.0)
    log: list = []
    a1, n1, _ = train_actor_critic(0.45, P, updates=4, log_batches=log, **kw)
    a2, n2, _ = train_actor_critic(0.45, P, updates=4, **kw)
    assert all(torch.equal(x, y) for x, y in zip(a1.state_dict().values(), a2.state_dict().values()))
    assert np.array_equal(n1.mean, n2.mean)
    assert len(log) == 4 and log[0]["obs_raw"].shape[:2] == log[0]["mask"].shape
    p1, m1 = train_predictor_on_logged(log, embed=16, hidden=8, seed=3)
    p2, _ = train_predictor_on_logged(log, embed=16, hidden=8, seed=3)
    assert all(torch.equal(x, y) for x, y in zip(p1.state_dict().values(), p2.state_dict().values()))
    valid = np.concatenate([b["obs_raw"][b["mask"] > 0.5] for b in log])
    assert m1.frozen and np.allclose(m1.mean, valid.mean(0), atol=1e-4)
