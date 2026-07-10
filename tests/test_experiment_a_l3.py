"""Tests for the L3 dynamics-level oracle (Experiment A for the learned-dynamics rung).

The oracle re-applies the EXACT authentic velocity law to the logged motion transitions and
scores the residual against a measurement-noise floor: authentic residual ~ noise, L3
residual = G's dynamics error. Leakage (reward/length/metadata) must stay at chance because
the surrogate's observations come from the REAL sensor model on top of G's motion."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.experiment_a_l3 import (  # noqa: E402
    generate_l3_pairs,
    oracle_features_L3,
    run_experiment_a_l3,
)
from itasorl.surrogate_l3 import train_g_motion  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402


def _pairs(hidden=4, n_pairs=16):
    g = train_g_motion(hidden=hidden, n_eps=30, steps=25, epochs=60, ray_steps=4, seed=0)
    return generate_l3_pairs(g, n_pairs=n_pairs, prefix=6, branch=20, ray_steps=4, seed0=3000)


def test_l3_oracle_runs_and_leakage_clean():
    eps = _pairs(n_pairs=16)
    assert len(eps) == 32 and all(len(e["trans"]) > 0 for e in eps)   # one authentic + one L3 per pair
    res = run_experiment_a_l3(eps, sigma_meas=0.05, seed=0)
    assert 0.0 <= float(res["oracle_auroc"]) <= 1.0
    assert res["leakage_pass"], f"leakage not clean: {res['leakage']}"


def test_l3_residual_larger_for_surrogate():
    """The oracle's core invariant: at a tiny noise floor, the authentic branch residual is
    ~0 (it follows the law exactly) while the L3 branch residual is G's dynamics error."""
    eps = _pairs(n_pairs=12)
    rng = np.random.default_rng(0)
    dt = WorldParams().dt
    auth = [oracle_features_L3(e["trans"], 1e-4, rng, dt)[2] for e in eps if e["label"] == 0]
    l3 = [oracle_features_L3(e["trans"], 1e-4, rng, dt)[2] for e in eps if e["label"] == 1]
    assert np.mean(l3) > np.mean(auth), "L3 residual should exceed the ~zero authentic residual"
