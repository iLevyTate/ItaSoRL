"""Spec 2026-09-27-local-strengthening-probes: the in-configuration L2 oracle
and the nonlinear joint control, on ground truth.

The nonlinear-control tests use a VARIANCE-shift echo: a label that enters the
inputs only through a mean shift is absorbed by a linear regression on the
inputs whatever nonlinearity the state applies afterwards (measured while
writing these tests: tanh, square, abs, relu echoes all fell to chance under
the linear control). A label that changes the input variance and a state that
squares the input is the construction a linear control cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from itasorl.behavior_audit import sensory_residual_probe_auroc  # noqa: E402
from itasorl.experiment_a_l3 import generate_l2_pairs, run_experiment_a_l2  # noqa: E402
from itasorl.experiment_b import episode_features, probe_auroc  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_l2_pairs_are_bit_identical_at_zero_drift():
    eps = generate_l2_pairs(P, 0.0, drift_mode="ar1", n_pairs=4, prefix=3, branch=6)
    by_pair = {}
    for e in eps:
        by_pair.setdefault(e["pair"], {})[e["label"]] = e
    for pair in by_pair.values():
        a, b = pair[0]["trans"], pair[1]["trans"]
        assert len(a) == len(b) == 6
        for ta, tb in zip(a, b):
            assert np.array_equal(ta[3], tb[3])


def test_l2_oracle_detects_strong_drift_and_uses_authentic_drag():
    eps = generate_l2_pairs(P, 2.0, drift_mode="ar1", n_pairs=12, prefix=3, branch=12)
    r = run_experiment_a_l2(eps, sigma_meas=1e-4, params=P)
    assert r["drag_auth"] == 1.5
    assert r["oracle_auroc"] > 0.9
    assert r["leakage_pass"]


def test_l2_regime_keeps_reset_drawn_offset_across_restore():
    eps = generate_l2_pairs(P, 0.45, drift_mode="regime", n_pairs=3, prefix=2, branch=5)
    surr = [e for e in eps if e["label"] == 1]
    for e in surr:
        drags = {round(float(t[2]), 6) for t in e["trans"]}
        assert len(drags) == 1 and abs(next(iter(drags)) - 1.5) > 1e-6   # constant, perturbed


def test_l2_oracle_rejects_two_drag_worlds():
    eps = generate_l2_pairs(P, 0.0, n_pairs=2, prefix=2, branch=4)
    with pytest.raises(ValueError):
        run_experiment_a_l2(eps, sigma_meas=0.02, params=WorldParams(k_land=0.2, k_water=0.6))


def _variance_echo(k=50, T=12, C=20, hid=12, seed=0, scale=2.0, orthogonal=False):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(k, int), np.ones(k, int)])
    O = rng.normal(size=(2 * k, T, C))
    O[:, :, :4] *= (1.0 + scale * y)[:, None, None]        # the world changes input VARIANCE
    W = rng.normal(size=(C, hid)) / np.sqrt(C)
    H = np.abs(O @ W) + 0.1 * rng.normal(size=(2 * k, T, hid))   # nonlinear echo of the input
    if orthogonal:
        d = rng.normal(size=hid)
        H = H + 2.0 * y[:, None, None] * (d / np.linalg.norm(d))[None, None, :]
    return H, O, y


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_mlp_control_removes_a_nonlinear_echo_the_linear_control_leaves():
    H, O, y = _variance_echo()
    assert probe_auroc(episode_features(H), y) > 0.9
    lin = sensory_residual_probe_auroc(H, O, y)
    mlp = sensory_residual_probe_auroc(H, O, y, nonlinear=True)
    assert lin > 0.85                       # a linear control cannot absorb |W x| of a variance shift
    assert mlp < lin - 0.2 and mlp < 0.7    # the MLP control removes most of it


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_orthogonal_tag_survives_mlp_control_with_uninformative_inputs():
    H, O, y = _variance_echo(orthogonal=True, scale=0.0)
    assert sensory_residual_probe_auroc(H, O, y, nonlinear=True) > 0.85
