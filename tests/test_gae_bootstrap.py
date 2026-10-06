"""GAE truncation bootstrap (revision steps 2 and 3, docs/CORRECTIONS.md 2026-10-06).

The intended semantics: an episode alive at the rollout cutoff is a truncation of a
continuing task, so its final residual is r_T + gamma * V(successor) - V(h_T), where the
successor is the state after the final recorded action. An episode that died bootstraps
from 0. Padded slots past an episode's end never touch a valid advantage.

These tests replace `test_compute_gae_truncation_bootstraps_last_value`, which asserted the
historical pre-transition bootstrap and so protected the bug. Each one fails if the code
returns to bootstrapping from the current (pre-transition) value or from a padded slot."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import (  # noqa: E402
    RunningNorm,
    _seeds,
    collect_episodes_ac,
    compute_gae,
    make_world,
    truncation_bootstrap,
)
from itasorl.world import WorldParams  # noqa: E402

GAMMA, LAM = 0.99, 0.95
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _t(rows):
    return torch.tensor(rows, dtype=torch.float32)


def _reference_gae(rewards, values, bootstrap, terminated, gamma=GAMMA, lam=LAM):
    """Independent reference for ONE unpadded episode, written from the definition
    A_t = sum_k (gamma*lam)^k delta_{t+k} with delta_t = r_t + gamma V_{t+1} - V_t and
    V_{T} = bootstrap (or 0 if the episode terminated). No recursion shared with the code."""
    r = np.asarray(rewards, np.float64)
    v = np.asarray(values, np.float64)
    v_next = np.append(v[1:], 0.0 if terminated else bootstrap)
    delta = r + gamma * v_next - v
    n = len(r)
    return np.array([sum((gamma * lam) ** k * delta[t + k] for k in range(n - t))
                     for t in range(n)])


# --------------------------------------------------------------------------- compute_gae
def test_one_step_truncated_episode_uses_the_successor_value():
    """reward 1, V(current) 0.3, V(successor) 2, gamma 0.99 -> 1 + 0.99*2 - 0.3 = 2.68.
    The historical pre-transition bootstrap would give 1 + 0.99*0.3 - 0.3 = 0.997."""
    adv, ret = compute_gae(_t([[1.0]]), _t([[0.3]]), _t([[1.0]]), _t([0.0]), GAMMA, LAM,
                           bootstrap=_t([2.0]))
    assert adv[0, 0].item() == pytest.approx(2.68, abs=1e-6)
    assert ret[0, 0].item() == pytest.approx(2.98, abs=1e-6)       # adv + V(current)
    assert abs(adv[0, 0].item() - 0.997) > 1.0                     # not the old bootstrap


@pytest.mark.parametrize("successor", [2.0, -50.0, 1e6])
def test_identical_terminal_episode_ignores_the_successor_value(successor):
    """Same step, but the agent died on it: the residual is r - V = 0.7 whatever the
    successor value passed in."""
    adv, _ = compute_gae(_t([[1.0]]), _t([[0.3]]), _t([[1.0]]), _t([1.0]), GAMMA, LAM,
                         bootstrap=_t([successor]))
    assert adv[0, 0].item() == pytest.approx(0.7, abs=1e-6)


def test_mixed_lengths_and_endings_match_the_reference():
    """A batch of episodes of lengths 1, 3, 5 and 4, some truncated and some dead, padded to
    T = 5. Each row must equal the reference computed on that episode alone."""
    rng = np.random.default_rng(7)
    lengths = [1, 3, 5, 4]
    terminated = [0.0, 1.0, 0.0, 1.0]
    boots = [2.0, 3.5, -1.25, 0.8]
    T = max(lengths)
    R = rng.normal(size=(4, T)).astype(np.float32)
    V = rng.normal(size=(4, T)).astype(np.float32)
    M = np.zeros((4, T), np.float32)
    for i, n in enumerate(lengths):
        M[i, :n] = 1.0
    adv, ret = compute_gae(torch.from_numpy(R), torch.from_numpy(V), torch.from_numpy(M),
                           _t(terminated), GAMMA, LAM, bootstrap=_t(boots))
    for i, n in enumerate(lengths):
        ref = _reference_gae(R[i, :n], V[i, :n], boots[i], bool(terminated[i]))
        np.testing.assert_allclose(adv[i, :n].numpy(), ref, atol=1e-5)
        np.testing.assert_allclose(ret[i, :n].numpy(), ref + V[i, :n], atol=1e-5)
        assert np.all(adv[i, n:].numpy() == 0.0)


@pytest.mark.parametrize("pad", [7.0, 1e30, float("inf"), float("-inf"), float("nan")])
def test_extreme_padded_entries_do_not_touch_valid_advantages(pad):
    """Padded reward and value slots hold a deliberately extreme value. The valid steps must
    match the reference exactly, and a code path that bootstrapped from the padded value
    slot (7.0 here) instead of the successor value (2.0) must fail."""
    R = _t([[1.0, 0.5, pad, pad]])
    V = _t([[0.2, 0.3, pad, pad]])
    M = _t([[1.0, 1.0, 0.0, 0.0]])
    adv, ret = compute_gae(R, V, M, _t([0.0]), GAMMA, LAM, bootstrap=_t([2.0]))
    ref = _reference_gae([1.0, 0.5], [0.2, 0.3], 2.0, False)
    np.testing.assert_allclose(adv[0, :2].numpy(), ref, atol=1e-6)
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()
    assert np.all(adv[0, 2:].numpy() == 0.0)
    wrong = _reference_gae([1.0, 0.5], [0.2, 0.3], 7.0, False)
    assert abs(ref[-1] - wrong[-1]) > 1.0


def test_multistep_truncated_episode_matches_the_reference():
    """A 12-step truncated episode with random rewards and values against the reference."""
    rng = np.random.default_rng(11)
    R = rng.normal(size=12)
    V = rng.normal(size=12)
    adv, _ = compute_gae(_t([R]), _t([V]), torch.ones(1, 12), _t([0.0]), GAMMA, LAM,
                         bootstrap=_t([1.7]))
    np.testing.assert_allclose(adv[0].numpy(), _reference_gae(R, V, 1.7, False), atol=1e-5)
    pre = _reference_gae(R, V, V[-1], False)                     # the historical bootstrap
    assert not np.allclose(adv[0].numpy(), pre, atol=1e-3)


def test_bootstrap_is_a_required_argument():
    with pytest.raises(TypeError):
        compute_gae(_t([[1.0]]), _t([[0.3]]), _t([[1.0]]), _t([0.0]), GAMMA, LAM)


# ----------------------------------------------------- recurrent integration (collector)
def _agent_and_batch(n_eps=3, steps=5):
    od = make_world(P, 0.0, RS).obs_spec.size
    ad = make_world(P, 0.0, RS).action_spec.size
    torch.manual_seed(0)
    agent = RecurrentActorCritic(od, ad, embed=16, hidden=8).train(False)
    norm = RunningNorm(od)
    norm.update(np.random.default_rng(0).normal(size=(64, od)))
    norm.freeze()
    b = collect_episodes_ac(agent, norm, P, 0.45, n_eps, steps, "cpu", 4321, RS,
                            deterministic=True, update_norm=False)
    return agent, norm, b


def test_collector_keeps_the_successor_observation_and_final_action():
    """next_obs must be the (normalized) observation AFTER the final recorded action, and
    last_env_act that action: replaying the stored actions in a fresh world reproduces both."""
    agent, norm, b = _agent_and_batch()
    assert (b["lengths"] == 5).all() and torch.all(b["terminated"] == 0.0)
    for i in range(3):
        w = make_world(P, 0.45, RS)
        w.reset(_seeds(4321 + i))
        for t in range(5):
            r = w.step(b["env_act"][i, t].numpy().astype(np.float32))
        np.testing.assert_allclose(b["next_obs"][i].numpy(),
                                   norm(r.obs.astype(np.float64)).astype(np.float32), atol=1e-6)
        np.testing.assert_array_equal(b["last_env_act"][i].numpy(), b["env_act"][i, 4].numpy())


def test_successor_bootstrap_runs_one_gru_step_from_the_final_state():
    agent, _, b = _agent_and_batch()
    _, value, _, states = agent.score_actions(b["obs"], b["act_in"], b["raw"],
                                              agent.initial_state(3, "cpu"))
    boot = truncation_bootstrap(agent, b, states, value, "successor")
    with torch.no_grad():
        h_next = agent.step_state(b["next_obs"], b["last_env_act"], states[:, -1])
        manual = agent.critic(h_next).squeeze(-1)
    torch.testing.assert_close(boot, manual)
    assert not boot.requires_grad
    pre = truncation_bootstrap(agent, b, states, value, "pre_transition")
    torch.testing.assert_close(pre, value[:, -1].detach())
    assert not torch.allclose(boot, pre)


def test_changing_only_the_successor_observation_changes_the_bootstrap():
    """The integration check: with everything else fixed, a different successor
    observation must move the bootstrap and the final advantage of a truncated episode,
    and must not move them under the historical pre-transition rule."""
    agent, _, b = _agent_and_batch()
    _, value, _, states = agent.score_actions(b["obs"], b["act_in"], b["raw"],
                                              agent.initial_state(3, "cpu"))
    b2 = dict(b, next_obs=b["next_obs"] + 3.0)
    s1 = truncation_bootstrap(agent, b, states, value, "successor")
    s2 = truncation_bootstrap(agent, b2, states, value, "successor")
    assert torch.all((s1 - s2).abs() > 1e-6)
    a1, _ = compute_gae(b["reward"], value.detach(), b["mask"], b["terminated"], GAMMA, LAM,
                        bootstrap=s1)
    a2, _ = compute_gae(b["reward"], value.detach(), b["mask"], b["terminated"], GAMMA, LAM,
                        bootstrap=s2)
    assert torch.all((a1[:, -1] - a2[:, -1]).abs() > 1e-6)
    p1 = truncation_bootstrap(agent, b, states, value, "pre_transition")
    p2 = truncation_bootstrap(agent, b2, states, value, "pre_transition")
    torch.testing.assert_close(p1, p2)


def test_dead_episodes_carry_no_successor(monkeypatch):
    """A lethal metabolism kills every episode before the cutoff: next_obs and
    last_env_act stay zero and the bootstrap is ignored by compute_gae."""
    import itasorl.experiment_b2 as b2mod
    monkeypatch.setattr(b2mod, "SURVIVAL_METAB",
                        {"E0": 0.05, "basal_E": 4.0, "Hyd0": 8.0, "basal_Hyd": 0.005})
    agent, _, b = _agent_and_batch(steps=30)
    assert torch.all(b["terminated"] == 1.0)
    assert torch.all(b["next_obs"] == 0.0) and torch.all(b["last_env_act"] == 0.0)


def test_unknown_bootstrap_mode_is_rejected():
    agent, _, b = _agent_and_batch()
    _, value, _, states = agent.score_actions(b["obs"], b["act_in"], b["raw"],
                                              agent.initial_state(3, "cpu"))
    with pytest.raises(ValueError):
        truncation_bootstrap(agent, b, states, value, "last_value")
