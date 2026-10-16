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


def _scripts():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_arc_family_draws_each_episode_from_the_stationary_distribution():
    """Sweep C (spec 2026-10-15-coherence-finite-horizon-companion-design.md): every world
    creation reseeds the family from base + 1000 + k, so each episode starts from its own
    stationary draw and nothing runs on from one episode to the next."""
    _scripts()
    import run_texture_fresh_probe as rt
    from itasorl.surrogate_l3_families import make_g_ar
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    fam = make_g_ar(sigma=0.01, rho=0.99, params=P, seed=0)
    with rt.per_episode_draw(fam, base=2_550_000) as make_world:
        make_world(P, 0.45, 5)
        first = fam(np.zeros(2), np.zeros(2)).copy()
        make_world(P, 0.45, 5)
        second = fam(np.zeros(2), np.zeros(2)).copy()
    fresh = make_g_ar(sigma=0.01, rho=0.99, params=P, seed=2_550_000 + 1000 + 1)
    assert np.allclose(second, fresh(np.zeros(2), np.zeros(2)))
    assert not np.allclose(first, second)


def test_sweep_c_sigmas_match_the_shared_gap():
    _scripts()
    import run_ar_sweep_c as c
    pts = c.points()
    assert [p["rho"] for p in pts] == [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
    assert pts[3]["sigma"] == pytest.approx(0.01)
    assert pts[0]["sigma"] == pytest.approx(0.01 * 0.1150 / 0.0364, rel=1e-6)
    assert c.BASES == (2_500_000, 2_550_000)


def test_sweep_c_readings_are_the_three_of_the_spec():
    _scripts()
    import run_ar_sweep_c as c
    k = ("1 coherence raises at matched gap", "2 within the margin unit", "3 neither")
    assert [c.readings({"mean": 0.02, "t90": [0.005, 0.035]})[x] for x in k] == [True, True, False]
    assert [c.readings({"mean": 0.0, "t90": [-0.04, 0.04]})[x] for x in k] == [False, True, False]
    assert [c.readings({"mean": 0.03, "t90": [-0.02, 0.08]})[x] for x in k] == [False, False, True]


def test_sweep_c_resume_reads_the_point_file_it_wrote():
    """A sweep C point file keeps its inputs at the top level with a repo-relative agents label;
    the resume check read a provenance block the probe never writes, so it refused every point."""
    _scripts()
    import run_ar_sweep_c as c
    p = c.points()[2]
    doc = {"family": "arc", "rho": p["rho"], "param": p["sigma"], "seed_bases": list(c.BASES),
           "agents_dir": "fullruns/corrected_l3_h8_wm/agents"}
    assert c._point_matches(doc, p, "fullruns/corrected_l3_h8_wm/agents")
    assert c._point_matches(doc, p, "C:/elsewhere/ITASORL/fullruns/corrected_l3_h8_wm/agents")
    assert not c._point_matches({**doc, "rho": 0.95}, p, "fullruns/corrected_l3_h8_wm/agents")
    assert not c._point_matches({**doc, "agents_dir": "C:/x/fullruns/corrected_l3_h8_wm/agents"}, p,
                                "fullruns/corrected_l3_h8_wm/agents")
