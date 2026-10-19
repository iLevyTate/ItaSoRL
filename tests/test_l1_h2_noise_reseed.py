"""Closing hook A7b: `--noise-reseed {per-pool,per-episode}` in scripts/run_l1_h2_ablations.py.

The recorded A2 noise knockout reseeds every surrogate world of a seed's pool to the same
Philox stream, so every episode sees the same noise draws. The default must stay exactly that;
per-episode must give each surrogate episode its own stream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from _git_checkout import assert_commit_recorded

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_l1_h2_ablations as rh  # noqa: E402
import itasorl.experiment_b2 as b2  # noqa: E402
from itasorl.patch_of_earth import PatchOfEarthV0  # noqa: E402
from itasorl.surrogate_l1_families import LObsNoise  # noqa: E402

HEADLINE, SENSOR, SIGMA, SEED = 0.023, 0.01, 0.05, 3


def _recorded_closure(old_make_world, seed):
    """The make_noise_world closure of _run_noise as it stood at main 1bbe7f5."""
    def make_noise_world(params, drift_sigma, ray_steps, food_override=None, _seed=seed):
        if drift_sigma == 0.0:
            return old_make_world(params, 0.0, ray_steps, food_override)
        w = PatchOfEarthV0(params or rh.P, drift_sigma=0.0, drift_mode="ar1",
                           l1_delta=HEADLINE, sensor_sigma=SENSOR)
        w.ray_steps = ray_steps
        for k, v in {**b2.SURVIVAL_METAB, **b2.SURVIVAL_FOOD, **(food_override or {})}.items():
            setattr(w, k, v)
        wrap = LObsNoise(w, sigma=SIGMA, seed=960_000 + _seed)
        wrap.reseed(970_000 + _seed)
        return wrap
    return make_noise_world


def _factory(reseed, old=None):
    return rh.noise_world_factory(old or b2.make_world, headline=HEADLINE, sensor_sigma=SENSOR,
                                  noise_sigma=SIGMA, seed=SEED, reseed=reseed)


def _trace(world, base, steps=4):
    obs = [np.asarray(world.reset(b2._seeds(base)).obs)]
    act = np.array([0.4, 0.2, 0.0, 0.0, 0.0], np.float32)
    for _ in range(steps):
        obs.append(np.asarray(world.step(act).obs))
    return np.stack(obs)


def _noise(world, n=6):
    return world._rng.normal(0.0, 1.0, size=n)


def test_default_is_per_pool():
    a = rh.cfg(["--agents-dir", "x", "--states-dir", "y", "--out-dir", "z"])
    assert a.noise_reseed == "per-pool"
    assert rh.NOISE_RESEED == ("per-pool", "per-episode")


def test_per_pool_factory_is_byte_identical_to_the_recorded_closure():
    new, old = _factory("per-pool"), _recorded_closure(b2.make_world, SEED)
    for k in range(2):
        wn, wo = new(rh.P, HEADLINE, 5), old(rh.P, HEADLINE, 5)
        assert wn._seed == wo._seed == 970_000 + SEED
        assert _trace(wn, 800_000 + k).tobytes() == _trace(wo, 800_000 + k).tobytes()


def test_per_pool_gives_every_episode_the_same_noise_and_per_episode_does_not():
    pool = _factory("per-pool")
    a, b = pool(rh.P, HEADLINE, 5), pool(rh.P, HEADLINE, 5)
    assert np.array_equal(_noise(a), _noise(b))
    ep = _factory("per-episode")
    worlds = [ep(rh.P, HEADLINE, 5) for _ in range(3)]
    draws = [_noise(w) for w in worlds]
    assert not np.array_equal(draws[0], draws[1]) and not np.array_equal(draws[1], draws[2])
    assert [w._seed for w in worlds] == [rh.noise_episode_seed(SEED, k) for k in range(3)]
    assert 970_000 + SEED not in {w._seed for w in worlds}
    assert rh.noise_episode_seed(SEED, 0) != rh.noise_episode_seed(SEED + 1, 0)
    again = _factory("per-episode")
    assert np.array_equal(_noise(again(rh.P, HEADLINE, 5)), draws[0])   # deterministic


def test_drift_zero_builds_the_authentic_world_and_bad_mode_is_refused():
    calls = []

    def old(params, drift_sigma, ray_steps, food_override=None):
        calls.append(drift_sigma)
        return "authentic"

    for mode in rh.NOISE_RESEED:
        assert _factory(mode, old)(rh.P, 0.0, 5) == "authentic"
    assert calls == [0.0, 0.0]
    with pytest.raises(ValueError, match="reseed"):
        _factory("per-step")


@pytest.mark.parametrize("mode", ["per-pool", "per-episode"])
def test_run_noise_installs_the_chosen_stream_and_labels_only_the_new_mode(monkeypatch, mode):
    seen = []
    monkeypatch.setattr(b2, "L1_DELTA", b2.L1_DELTA)   # _run_noise sets the module global
    old_make_world = b2.make_world
    monkeypatch.setattr(rh, "load_agent_bundle", lambda path, dev: (object(), object()))
    monkeypatch.setattr(rh, "collect_pool", lambda *a, **k: (np.zeros((6, 4, 2)), None))

    def fake_transfer(agent, norm, params, drift, Ha, Hs, **kw):
        seen.append([b2.make_world(params, drift, 5)._seed for _ in range(2)])
        return {"transfer_target": 0.5, "transfer_lo": 0.4, "transfer_hi": 0.6}

    monkeypatch.setattr(rh, "transfer_readout", fake_transfer)
    a = argparse.Namespace(quick=True, n_eps=110, steps=24, agents_dir="x", arms=["survival"],
                           sensor_sigma=SENSOR, noise_reseed=mode)
    rows, agg = rh._run_noise(a, "cpu", HEADLINE, ["agent_d0.0230_s3_survival.pt"], SIGMA)
    assert b2.make_world is old_make_world
    if mode == "per-pool":
        assert seen == [[970_000 + SEED] * 2] and "noise_reseed" not in agg
        assert list(agg) == ["noise_sigma", "quick", "n_eps", "steps", "gn_dropped_at_gate0",
                             "survival_mean", "survival_per_seed", "survival_n_ge_065",
                             "survival_tci90"]
    else:
        assert seen == [[rh.noise_episode_seed(SEED, 0), rh.noise_episode_seed(SEED, 1)]]
        assert agg["noise_reseed"] == "per-episode"
    assert set(rows[0]) == {"arm", "seed", "noise_sigma", "target", "target_lo", "target_hi"}


def test_provenance_json_records_the_readout_commit_key(tmp_path):
    a = rh.cfg(["--agents-dir", "fullruns/l1_heldout/agents", "--states-dir",
                "fullruns/l1_heldout/states", "--out-dir", str(tmp_path), "--channels", "noise",
                "--noise-reseed", "per-episode"])
    a.device_used = "cpu"
    path = rh.write_provenance(a, ["--noise-reseed", "per-episode"])
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert Path(path).name == "provenance.json"
    assert_commit_recorded(doc)
    assert "tracked_changes_at_generation" in doc and doc["stack"]["device"] == "cpu"
    assert doc["agents_dir"] == "fullruns/l1_heldout/agents"
    assert doc["noise_reseed"] == "per-episode" and doc["argv"] == ["--noise-reseed", "per-episode"]
