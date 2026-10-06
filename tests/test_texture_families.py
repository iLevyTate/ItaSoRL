"""Revision step 9: the structured comparator family and the perturbation profile."""

from __future__ import annotations

import numpy as np

from itasorl.surrogate_l3_families import gate0_candidates, make_g_gn, make_g_qd, perturbation_profile
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_qd_is_the_authentic_law_at_eps_zero_and_slows_fast_motion():
    a = np.array([0.1, -0.2])
    v = np.array([0.6, 0.8])
    authentic = (1.0 - 1.5 * P.dt) * v + a * P.dt
    assert np.allclose(make_g_qd(eps=0.0, params=P)(v, a), authentic, atol=0, rtol=0)
    slowed = make_g_qd(eps=2.0, params=P)(v, a)
    assert np.allclose(authentic - slowed, 2.0 * 1.0 * v * P.dt)      # |v| = 1


def test_profile_separates_white_noise_from_a_structured_perturbation():
    gn = perturbation_profile(make_g_gn(sigma_v=0.01, params=P), params=P, n_eps=8, steps=20)
    qd = perturbation_profile(make_g_qd(eps=4.0, params=P), params=P, n_eps=8, steps=20)
    assert abs(gn["lag1_autocorr"]) < 0.2 and qd["lag1_autocorr"] > 0.8
    assert gn["linear_explained_share"] < 0.1 < qd["linear_explained_share"]


def test_qd_is_a_gate0_family():
    knobs = [k for k, _ in gate0_candidates("qd", params=P, sweep=[1.0, 0.5])]
    assert knobs == [("eps", 0.5), ("eps", 1.0)]


def test_comparator_training_reseeds_the_noise_per_cell():
    """A gn-trained cell must not depend on which worker ran it or in what order: the noise
    stream is reseeded from (drift, seed) when the family is installed."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import run_expB2

    import itasorl.experiment_b2 as b2
    saved = b2._L3_GMOTION
    try:
        v, a = np.array([0.3, 0.1]), np.array([0.2, 0.0])
        run_expB2.install_l3_family(b2, "gn", 0.01, 0.45, 3)
        first = [b2._L3_GMOTION(v, a) for _ in range(3)]
        run_expB2.install_l3_family(b2, "gn", 0.01, 0.45, 4)            # another cell in between
        run_expB2.install_l3_family(b2, "gn", 0.01, 0.45, 3)
        again = [b2._L3_GMOTION(v, a) for _ in range(3)]
        assert all(np.array_equal(x, y) for x, y in zip(first, again))
        run_expB2.install_l3_family(b2, "qd", 2.0, 0.45, 3)
        assert isinstance(b2._L3_GMOTION, type(make_g_qd(eps=2.0, params=P)))
    finally:
        b2._L3_GMOTION = saved
