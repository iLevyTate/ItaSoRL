"""Matched-input surprise (replayed prediction error) and the correlated-jitter family G_ar
(docs/specs/2026-10-15-surprise-dissociation-design.md, 2026-10-15-coherence-integration-sweep-design.md)."""

from __future__ import annotations

import numpy as np
import pytest

from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def _authentic(vel, a):
    return (1.0 - 1.5 * P.dt) * np.asarray(vel, float) + np.asarray(a, float) * P.dt


def test_replayed_prediction_error_equals_the_error_on_the_agents_own_rollout():
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    from itasorl import mechanism_readouts as mr
    from itasorl.eval_protocols import record_trajectories
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                     device="cpu", seed=0)
    own = mr.rollout_behavior(agent, norm, P, 0.0, n_eps=3, steps=10, seed_base=7, ray_steps=5)
    O, A, kept = record_trajectories(agent, norm, P, 0.0, 3, 10, "cpu", 7, 5)
    assert list(kept) == list(own["kept"])
    E = mr.replay_prediction_error(agent, norm, O, A)
    assert E.shape == own["E"].shape == (len(kept), 9)
    np.testing.assert_allclose(E, own["E"], rtol=1e-5, atol=1e-7)


def test_replayed_prediction_error_is_nan_without_a_decoder():
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    from itasorl import mechanism_readouts as mr
    from itasorl.eval_protocols import record_trajectories
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=False,
                                     device="cpu", seed=0)
    O, A, _ = record_trajectories(agent, norm, P, 0.0, 2, 6, "cpu", 7, 5)
    assert np.isnan(mr.replay_prediction_error(agent, norm, O, A)).all()


def test_g_ar_sigma_zero_is_the_authentic_law():
    from itasorl.surrogate_l3_families import make_g_ar
    g = make_g_ar(sigma=0.0, rho=0.9, params=P, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v, a = rng.normal(size=2), rng.normal(size=2)
        np.testing.assert_allclose(g(v, a, None), _authentic(v, a), atol=1e-12)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.95])
def test_g_ar_is_stationary_with_the_named_sd_and_lag1_autocorrelation(rho):
    from itasorl.surrogate_l3_families import make_g_ar
    g = make_g_ar(sigma=0.01, rho=rho, params=P, seed=3)
    z = np.zeros(2)
    eta = np.array([g(z, z, None) for _ in range(40_000)])
    assert abs(eta.std() - 0.01) < 0.0006
    r1 = float((eta[1:] * eta[:-1]).sum() / (eta ** 2).sum())
    assert abs(r1 - rho) < 0.03


def test_g_ar_reseed_restores_the_stream_and_starts_stationary():
    from itasorl.surrogate_l3_families import make_g_ar
    g = make_g_ar(sigma=0.01, rho=0.99, params=P, seed=0)
    z = np.zeros(2)
    g.reseed(11)
    a = [g(z, z, None) for _ in range(5)]
    g.reseed(11)
    b = [g(z, z, None) for _ in range(5)]
    np.testing.assert_array_equal(np.array(a), np.array(b))
    first = []
    for s in range(3000):
        g.reseed(s)
        first.append(g(z, z, None))
    assert abs(np.std(first) - 0.01) < 0.0008   # the first step is already stationary


def test_accumulated_offset_sd_matches_simulation():
    from itasorl.surrogate_l3_families import ar_accumulated_sd, make_g_ar
    lam = 1.0 - 1.5 * P.dt
    for rho in (0.0, 0.9):
        g = make_g_ar(sigma=0.01, rho=rho, params=P, seed=5)
        z = np.zeros(2)
        x = np.zeros(2)
        xs = []
        for t in range(60_000):
            x = lam * x + g(z, z, None)
            if t > 500:
                xs.append(x.copy())
        sim = float(np.std(np.array(xs)))
        assert abs(sim / ar_accumulated_sd(0.01, rho, lam) - 1.0) < 0.06


def test_sigma_for_matched_accumulation_inverts_the_formula():
    from itasorl.surrogate_l3_families import ar_accumulated_sd, sigma_for_accumulated_sd
    lam = 0.925
    target = ar_accumulated_sd(0.01, 0.9, lam)
    for rho in (0.0, 0.5, 0.99):
        s = sigma_for_accumulated_sd(target, rho, lam)
        assert abs(ar_accumulated_sd(s, rho, lam) - target) < 1e-12


def test_scripted_recorder_keeps_raw_observations_and_the_scripted_actions():
    from itasorl import mechanism_readouts as mr
    from itasorl.experiment_b import scripted_policy
    O, A, kept = mr.record_scripted_trajectories(P, 0.0, n_eps=3, steps=8, seed_base=7, ray_steps=5,
                                                 policy_seed_base=600_000)
    assert O.shape[0] == A.shape[0] == len(kept) and O.shape[1] == A.shape[1] == 8
    i = int(kept[0])
    rng = np.random.default_rng(600_000 + i)
    np.testing.assert_array_equal(A[0], np.stack([scripted_policy(rng) for _ in range(8)]))
    assert np.abs(O).max() > 1.5   # raw, not normalized
