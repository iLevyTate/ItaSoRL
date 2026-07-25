"""Tests for L1 H2 ablation surrogate families."""

import numpy as np

from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.surrogate_l1_families import L1Offset, LObsNoise
from itasorl.world import SeedBundle, WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def _base_world():
    w = PatchOfEarthV0(P, drift_sigma=0.0, drift_mode="ar1", sensor_sigma=0.0)
    w.ray_steps = 4
    w.reset(SeedBundle(world=0, weather=1, ecology=2))
    return w


def test_lobs_noise_sigma_zero_is_byte_identical():
    base = _base_world()
    wrap = LObsNoise(base, sigma=0.0, seed=7)
    a = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    assert np.array_equal(wrap._obs_transform(a), a)


def test_lobs_noise_fixed_seed_bit_identical():
    a = np.linspace(0.0, 1.0, 16)
    g1 = LObsNoise(_base_world(), sigma=0.05, seed=42)
    g2 = LObsNoise(_base_world(), sigma=0.05, seed=42)
    assert np.array_equal(g1._obs_transform(a.copy()), g2._obs_transform(a.copy()))


def test_lobs_noise_reseed_restores_stream():
    a = np.linspace(0.0, 1.0, 16)
    g = LObsNoise(_base_world(), sigma=0.05, seed=11)
    first = g._obs_transform(a.copy())
    g.reseed(11)
    second = g._obs_transform(a.copy())
    assert np.array_equal(first, second)


def test_l1_offset_zero_matches_plain_quantize():
    delta = 0.023
    a = np.array([0.011, 0.034, -0.05, 0.5], dtype=np.float64)
    plain = np.round(a / delta) * delta
    got = L1Offset(_base_world(), delta=delta, offset=0.0)._obs_transform(a)
    assert np.allclose(got, plain)


def test_l1_inline_quantize_in_patch_of_earth():
    w = PatchOfEarthV0(P, drift_sigma=0.023, drift_mode="l1",
                       l1_delta=0.023, sensor_sigma=0.0)
    w.ray_steps = 4
    w.reset(SeedBundle(world=3, weather=4, ecology=5))
    obs = w.observe()
    # Every continuous value lands on the grid (within float32 assemble noise).
    resid = obs - np.round(obs / 0.023) * 0.023
    assert float(np.max(np.abs(resid))) < 1e-5


def test_l1_delta_zero_skips_quantize():
    w = PatchOfEarthV0(P, drift_sigma=0.023, drift_mode="l1",
                       l1_delta=0.0, sensor_sigma=0.0)
    w.ray_steps = 4
    w.reset(SeedBundle(world=6, weather=7, ecology=8))
    # Must not raise (no division by zero) and must return finite obs.
    obs = w.observe()
    assert np.all(np.isfinite(obs))


def test_format_drift_roundtrips_subcent_delta():
    from itasorl.experiment_b2 import format_drift
    assert format_drift(0.45) == "0.45"
    assert format_drift(0.023) == "0.0230"
    assert abs(float(format_drift(0.023)) - 0.023) < 1e-9
