"""Closing hook B2: `thrust_scale` in eval_protocols.collect_pool_scripted.

The walk-speed readout needs a scripted walk whose speed matches the survival agent's. The
hook scales the scripted thrust and leaves the headings and draws alone. At the default 1.0
the pool must be the historical one byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.eval_protocols import POLICY_SEED_BASE, collect_pool_scripted  # noqa: E402
from itasorl.experiment_b import scripted_policy  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, _seeds, make_world  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent(seed=0):
    w = make_world(P, 0.0, RS)
    torch.manual_seed(seed)
    a = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return a, RunningNorm(w.obs_spec.size).freeze()


def _reference(agent, norm, params, drift_sigma, n_eps, steps, device, seed_base, ray_steps,
               policy_seed_base=POLICY_SEED_BASE):
    """collect_pool_scripted as it stood at main 1bbe7f5, before the thrust_scale hook."""
    Hs, kept = [], []
    for i in range(n_eps):
        w = make_world(params, drift_sigma, ray_steps)
        w.reset(_seeds(seed_base + i))
        rng = np.random.default_rng(policy_seed_base + i)
        h = agent.initial_state(1, device)
        prev = torch.zeros(1, agent.act_dim, device=device)
        obs = w.observe().astype(np.float64)
        row, died = [], False
        with torch.no_grad():
            for _ in range(steps):
                o = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
                h = agent.step_state(o, prev, h)
                row.append(h[0].cpu().numpy())
                a = scripted_policy(rng)
                r = w.step(a)
                prev = torch.as_tensor(a[None], dtype=torch.float32, device=device)
                obs = r.obs.astype(np.float64)
                if r.terminated:
                    died = True
                    break
        if not died and len(row) == steps:
            Hs.append(np.asarray(row, np.float32))
            kept.append(i)
    H = np.stack(Hs) if Hs else np.zeros((0, steps, agent.hidden), np.float32)
    return H, np.asarray(kept, dtype=int)


@pytest.mark.parametrize("drift", [0.0, 0.45])
def test_default_is_byte_identical_to_the_pre_hook_collector(drift):
    a, n = _agent(0)
    H_ref, k_ref = _reference(a, n, P, drift, 5, 8, "cpu", 800_000, RS)
    for kw in ({}, {"thrust_scale": 1.0}):
        H, k = collect_pool_scripted(a, n, P, drift, 5, 8, "cpu", 800_000, RS, **kw)
        assert H.dtype == H_ref.dtype and H.tobytes() == H_ref.tobytes()
        assert k.tobytes() == k_ref.tobytes()


def _mean_speed(thrust_scale, n_eps=4, steps=12):
    """Mean world speed of the scripted walk at a given thrust scale."""
    speeds = []
    for i in range(n_eps):
        w = make_world(P, 0.0, RS)
        w.reset(_seeds(800_000 + i))
        rng = np.random.default_rng(POLICY_SEED_BASE + i)
        for _ in range(steps):
            act = scripted_policy(rng)
            act[0] = np.float32(min(max(float(act[0]) * thrust_scale, 0.0), 1.0))
            w.step(act)
            speeds.append(float(np.linalg.norm(w.vel)))
    return float(np.mean(speeds))


def test_scaled_thrust_changes_the_states_and_slows_the_walk():
    a, n = _agent(0)
    H1, _ = collect_pool_scripted(a, n, P, 0.0, 4, 8, "cpu", 800_000, RS)
    Hh, _ = collect_pool_scripted(a, n, P, 0.0, 4, 8, "cpu", 800_000, RS, thrust_scale=0.5)
    assert H1.shape == Hh.shape and not np.array_equal(H1, Hh)
    assert _mean_speed(0.5) < _mean_speed(1.0)


def test_scaled_thrust_is_clipped_to_the_world_range():
    """A large scale saturates at the world's thrust clip, so the GRU never receives a
    thrust above 1."""
    a, n = _agent(0)
    H_big, _ = collect_pool_scripted(a, n, P, 0.0, 3, 6, "cpu", 800_000, RS, thrust_scale=1e4)
    H_sat, _ = collect_pool_scripted(a, n, P, 0.0, 3, 6, "cpu", 800_000, RS, thrust_scale=1e6)
    assert H_big.tobytes() == H_sat.tobytes()
