"""Control properties of the sensory-echo control (itasorl/behavior_audit.py,
spec docs/specs/2026-09-26-l3-sensory-echo-control-design.md), on synthetic
data where the ground truth is known:
  - a world signal that reaches the state ONLY as a linear echo of the
    observation trace is removed (no false positive);
  - a world direction orthogonal to the input pathway survives when the
    inputs are uninformative (no over-removal), and stays alive under the
    instantaneous basis even when the inputs carry the label;
  - the integrated (cummean) basis is the over-strict variant: it erodes the
    genuine tag once the inputs separate the worlds, which is why it is
    secondary;
  - the joint control also removes a behavior-mediated signal the sensory
    control alone leaves in place;
  - collect_pool(return_obs=True) appends the observation trace without
    changing any other output.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from itasorl.behavior_audit import sensory_residual_probe_auroc  # noqa: E402
from itasorl.experiment_b import episode_features, probe_auroc  # noqa: E402


def _data(k=50, T=12, C_obs=20, C_beh=4, hid=12, obs_shift=1.5, beh_shift=0.0,
          orthogonal=False, seed=0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(k, int), np.ones(k, int)])
    O = rng.normal(size=(2 * k, T, C_obs))
    O[:, :, 0] += obs_shift * y[:, None]                # the world shifts one obs channel
    B = rng.normal(size=(2 * k, T, C_beh))
    B[:, :, 0] += beh_shift * y[:, None]
    Wo = rng.normal(size=(C_obs, hid)) / np.sqrt(C_obs)
    Wb = rng.normal(size=(C_beh, hid)) / np.sqrt(C_beh)
    H = O @ Wo + B @ Wb + 0.3 * rng.normal(size=(2 * k, T, hid))
    if orthogonal:
        d = rng.normal(size=hid)
        span = np.concatenate([Wo, Wb], axis=0)            # rows span the input pathways
        d -= span.T @ np.linalg.lstsq(span.T, d, rcond=None)[0]
        H = H + 2.0 * y[:, None, None] * (d / np.linalg.norm(d))[None, None, :]
    return H, O, B, y


def test_observation_echo_is_removed():
    H, O, B, y = _data(obs_shift=1.5)
    assert probe_auroc(episode_features(H), y) > 0.9          # the echo decodes the world
    assert sensory_residual_probe_auroc(H, O, y) < 0.62      # the control removes it
    assert sensory_residual_probe_auroc(H, O, y, integrated=True) < 0.62


def test_orthogonal_direction_survives_uninformative_inputs():
    H, O, B, y = _data(obs_shift=0.0, orthogonal=True)
    assert sensory_residual_probe_auroc(H, O, y) > 0.85
    assert sensory_residual_probe_auroc(H, O, y, integrated=True) > 0.85


def test_orthogonal_direction_alive_under_instantaneous_basis_with_informative_inputs():
    H, O, B, y = _data(obs_shift=1.5, orthogonal=True)
    assert sensory_residual_probe_auroc(H, O, y) > 0.85


def test_integrated_basis_is_over_strict_with_informative_inputs():
    """Documents WHY the integrated variant is secondary: the genuine tag is
    eroded (still alive, but attenuated) once cummean(x) encodes the label."""
    H, O, B, y = _data(obs_shift=1.5, orthogonal=True)
    inst = sensory_residual_probe_auroc(H, O, y)
    integ = sensory_residual_probe_auroc(H, O, y, integrated=True)
    assert integ < inst and integ > 0.65


def test_joint_control_removes_behavior_mediation():
    H, O, B, y = _data(obs_shift=0.0, beh_shift=1.5)
    assert sensory_residual_probe_auroc(H, O, y) > 0.85      # sensory control alone leaves it
    assert sensory_residual_probe_auroc(H, O, y, Bt=B) < 0.62


def test_pure_noise_stays_near_chance():
    # k = 120 episodes per class: at k = 50 the out-of-fold AUROC of a
    # residualized null swings by more than 0.1 from sampling alone.
    H, O, B, y = _data(k=120, obs_shift=0.0)
    assert abs(sensory_residual_probe_auroc(H, O, y) - 0.5) < 0.12


def test_collect_pool_return_obs_is_additive():
    pytest.importorskip("torch")
    from itasorl.experiment_b2 import collect_pool, untrained_agent
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    agent, norm = untrained_agent(P, 0.0, 5, 16, 16, True, "cpu", seed=0)
    base = collect_pool(agent, norm, P, 0.0, 3, 6, "cpu", 100, 5, return_anchors=True)
    ext = collect_pool(agent, norm, P, 0.0, 3, 6, "cpu", 100, 5, return_anchors=True, return_obs=True)
    assert len(ext) == len(base) + 1
    for x, z in zip(base, ext[:-1]):
        assert np.array_equal(np.asarray(x), np.asarray(z))
    Ot = ext[-1]
    assert isinstance(Ot, np.ndarray)
    assert Ot.ndim == 3 and Ot.shape[0] == base[0].shape[0] and Ot.shape[1] == 6
    assert Ot.shape[2] == int(norm.mean.shape[0])
