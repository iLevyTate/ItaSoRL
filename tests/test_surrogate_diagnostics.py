"""Revision step 10: the linear-fit control, held-out and rollout errors, and the
agent-accessible (observation-only) detector."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")

from itasorl.experiment_a_l3 import observation_law_detector  # noqa: E402
from itasorl.surrogate_l3 import surrogate_diagnostics, train_g_motion  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_linear_fit_recovers_the_uniform_drag_law_and_g_does_not():
    g = train_g_motion(hidden=4, n_eps=20, steps=10, epochs=30, params=P, ray_steps=3)
    d = surrogate_diagnostics(g, params=P, n_train_eps=20, train_steps=10, heldout_eps=6,
                              heldout_steps=12, horizons=(1, 10), ray_steps=3)
    assert d["uniform_drag"]
    assert d["linear_fit"]["rms_heldout"] < 1e-6
    assert d["linear_fit"]["coef_vel_x"][0] == pytest.approx(1 - 1.5 * P.dt, abs=1e-5)
    assert d["g_one_step"]["rms_heldout"] > 100 * d["linear_fit"]["rms_heldout"]
    assert d["rollout_rms_velocity_gap"]["10"] >= d["rollout_rms_velocity_gap"]["1"]


def test_observation_law_detector_is_chance_on_one_world_and_reads_g():
    import run_surrogate_diagnostics as rsd
    a = rsd.scripted_obs_pool(None, 800_000, 20, 8, ray_steps=3)
    b = rsd.scripted_obs_pool(None, 850_000, 20, 8, ray_steps=3)
    same = observation_law_detector(a, b, dt=P.dt)["auroc"]
    assert 0.2 < same < 0.8                                   # two authentic samples
    g = train_g_motion(hidden=4, n_eps=20, steps=10, epochs=30, params=P, ray_steps=3)
    s = rsd.scripted_obs_pool(g, 850_000, 20, 8, ray_steps=3)
    assert observation_law_detector(a, s, dt=P.dt)["auroc"] > 0.9
