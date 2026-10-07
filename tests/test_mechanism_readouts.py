"""Mechanism readouts of the goal-and-stakes spec (section 'Mechanism readouts'):
state nudge, probe direction, gap-closed score, behavior difference, surprise, adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from itasorl import mechanism_readouts as mr  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_probe_direction_points_toward_the_surrogate_pool():
    rng = np.random.default_rng(0)
    Ha = rng.normal(size=(40, 6, 5))
    Hs = rng.normal(size=(40, 6, 5)) + np.array([2.0, 0, 0, 0, 0])
    u, s = mr.probe_direction(Ha, Hs)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert abs(u[0]) > 0.9 and u[0] > 0               # the separating axis, surrogate-positive
    assert s > 0


def test_sham_direction_is_unit_and_orthogonal():
    u = np.array([1.0, 0, 0, 0])
    v = mr.sham_direction(u, seed=3)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert abs(v @ u) < 1e-9
    assert np.allclose(v, mr.sham_direction(u, seed=3))  # fixed RNG


def test_gap_closed_scores():
    b_auth = np.zeros((10, 5)); b_surr = np.ones((10, 5))
    scale = np.ones(5)
    assert mr.gap_closed(b_auth, b_surr, b_surr, scale)[0] == pytest.approx(1.0)
    assert mr.gap_closed(b_auth, b_surr, b_auth, scale)[0] == pytest.approx(0.0)
    half = np.full((10, 5), 0.5)
    assert mr.gap_closed(b_auth, b_surr, half, scale)[0] == pytest.approx(0.5)
    score, gap = mr.gap_closed(b_auth, b_auth, half, scale)
    assert np.isnan(score) and gap == 0.0


def test_surprise_summaries_and_correlation():
    E = np.tile(np.linspace(0.0, 1.0, 9), (4, 1))        # rising error, 4 episodes
    S = mr.surprise_summaries(E)
    assert S.shape == (4, 3)
    assert np.allclose(S[:, 0], 0.5) and np.allclose(S[:, 1], 1.0) and np.all(S[:, 2] > 0)
    H = np.zeros((4, 10, 3)); H[:, :, 1] = np.linspace(0.0, 1.0, 10)   # projection tracks error
    u = np.array([0.0, 1.0, 0.0])
    assert mr.direction_error_correlation(H, E, u) == pytest.approx(1.0)


def test_surprise_summaries_nan_row_does_not_raise():
    E = np.vstack([np.linspace(0.0, 1.0, 9), np.full(9, np.nan)])
    S = mr.surprise_summaries(E)
    assert np.all(np.isfinite(S[0])) and np.all(np.isnan(S[1]))


def test_adaptation_sign():
    ha = np.array([[1.0, 1.0]] * 5); hs = np.array([[0.5, 0.9]] * 5)   # gap narrows
    out = mr.adaptation(ha, hs)
    assert out["gap_first"] == pytest.approx(0.5)
    assert out["gap_second"] == pytest.approx(0.1)
    assert out["adaptation"] == pytest.approx(0.4)


def test_rollout_behavior_shapes_and_nudge_changes_states():
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                     device="cpu", seed=0)
    r0 = mr.rollout_behavior(agent, norm, P, 0.0, n_eps=3, steps=10, seed_base=7, ray_steps=5)
    assert r0["B"].shape == (3, 7) and r0["H"].shape == (3, 10, 8)
    assert r0["E"].shape == (3, 9) and r0["halves"].shape == (3, 2)
    u = np.zeros(8); u[0] = 1.0
    r1 = mr.rollout_behavior(agent, norm, P, 0.0, n_eps=3, steps=10, seed_base=7, ray_steps=5,
                             nudge=0.5 * u)
    assert not np.allclose(r0["H"], r1["H"])
    assert np.allclose(r0["H"][:, 0, 1:], r1["H"][:, 0, 1:])   # first step differs only on u


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
def test_nudge_changes_the_action_at_the_same_step():
    """The policy head reads the nudged state at step t (not one step late): a one-step
    rollout's action columns (abs_turn, thrust) move under a large nudge. (A one-step
    episode has an empty first half, so its `halves` entry is NaN by design.)"""
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                     device="cpu", seed=0)
    u = np.zeros(8); u[0] = 1.0
    kw = dict(n_eps=3, steps=1, seed_base=7, ray_steps=5)
    r0 = mr.rollout_behavior(agent, norm, P, 0.0, **kw)
    r1 = mr.rollout_behavior(agent, norm, P, 0.0, nudge=50.0 * u, **kw)
    assert r0["B"].shape == (3, 7)
    assert not np.allclose(r0["B"][:, 1:3], r1["B"][:, 1:3])


def test_scripted_streams_shapes():
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    _, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                 device="cpu", seed=0)
    O, A = mr.scripted_observation_streams(norm, P, 0.0, n_eps=4, steps=6, seed_base=11, ray_steps=5)
    assert O.shape[0] == A.shape[0] <= 4 and O.shape[1] == 6 and A.shape[2] == 5
