"""The 2026-10-15 readout pieces: salience, world-draw features, erasure, development."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from itasorl.world import WorldParams

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_write_output_records_no_machine_local_path(tmp_path):
    """The provenance block names the agents directory and the command line. Both must be
    repo-relative, as every promoter's record is (scripts/promote_bv3_gates.repo_rel), or
    tests/test_artifact_paths.py rejects the artifact once it is tracked. The development
    script passes its directories as `name=path` pairs, which must be rewritten too."""
    import _readout_common as rc
    agents = os.path.join(ROOT, "fullruns", "some_run", "agents")
    out = str(tmp_path / "o.json")
    rc.write_output(out, "scripts/run_x.py",
                    ["--agents-dir", agents, "--run", f"wm={agents}", "--out", out],
                    {"agents_dir": agents, "seeds": [0]}, "docs/specs/x.md", {"rows": []})
    with open(out, encoding="utf-8") as fh:
        prov = json.load(fh)["provenance"]
    assert prov["inputs"]["agents_dir"] == "fullruns/some_run/agents"
    assert prov["inputs"]["seeds"] == [0]
    assert ROOT.replace("\\", "/") not in prov["command"].replace("\\", "/")
    assert "--agents-dir fullruns/some_run/agents --run wm=fullruns/some_run/agents" in prov["command"]

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def _pools(sep: float, n: int = 60, T: int = 6, hid: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    Ha = rng.normal(size=(n, T, hid))
    Hs = rng.normal(size=(n, T, hid))
    Hs[:, :, 0] += sep
    return Ha, Hs


def test_salience_separates_a_strong_axis_and_not_identical_pools():
    from itasorl.novel_readouts import salience_measures
    strong = salience_measures(*_pools(4.0))
    assert strong["kmeans_agreement"] > 0.9
    assert strong["share_probe_dir"] > 0.1
    assert strong["cohen_d_oof"] > 2.0
    null = salience_measures(*_pools(0.0))
    assert null["kmeans_agreement"] < 0.7
    assert abs(null["cohen_d_oof"]) < 1.0


def test_erasure_pins_the_subspace_and_the_sham_is_orthogonal():
    from itasorl.novel_readouts import erase, probe_block_directions, projector, sham_projector
    Ha, Hs = _pools(1.0)
    D = probe_block_directions(Ha, Hs)
    assert D.shape == (2, 8)
    Pr = projector(D)
    np.testing.assert_allclose(Pr @ Pr, Pr, atol=1e-10)
    assert round(float(np.trace(Pr))) == 2
    m = 0.5 * (Ha.reshape(-1, 8).mean(0) + Hs.reshape(-1, 8).mean(0))
    E = erase(np.concatenate([Ha, Hs]), Pr, m)
    np.testing.assert_allclose(E @ Pr, np.broadcast_to(m @ Pr, E.shape), atol=1e-10)
    Pv = sham_projector(Pr, 2, seed=3)
    np.testing.assert_allclose(Pv @ Pr, 0.0, atol=1e-10)
    assert round(float(np.trace(Pv))) == 2


def test_ridge_r2_reads_a_linear_target_and_not_a_permuted_one():
    from itasorl.novel_readouts import ridge_r2_cv
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 10))
    t = X @ rng.normal(size=10) + 0.1 * rng.normal(size=400)
    assert ridge_r2_cv(X, t) > 0.9
    assert ridge_r2_cv(X, rng.permutation(t)) < 0.05


def test_residual_r2_removes_what_the_senses_carry():
    from itasorl.novel_readouts import residual_r2_cv
    rng = np.random.default_rng(2)
    O = rng.normal(size=(600, 12))
    t = O[:, 0] + 0.1 * rng.normal(size=600)
    echo = np.column_stack([O @ rng.normal(size=12) for _ in range(6)])
    assert residual_r2_cv(echo, O, t) < 0.05          # the state only echoes the senses
    extra = rng.normal(size=600)
    t2 = extra + 0.1 * rng.normal(size=600)
    kept = np.column_stack([echo, extra])
    assert residual_r2_cv(kept, O, t2) > 0.8           # the state keeps something the senses lack


def test_draw_features_read_the_world_without_consuming_its_random_streams():
    from itasorl.experiment_b2 import _seeds, make_world
    from itasorl.novel_readouts import INCIDENTAL, INSTRUMENTAL, draw_features, terrain_grid
    w1 = make_world(P, 0.0, 5)
    w1.reset(_seeds(2_600_000))
    f = draw_features(w1, 0.15)
    assert set(f) == set(INSTRUMENTAL + INCIDENTAL)
    assert 0.0 <= f["I3_water_near"] <= 1.0 and 0.0 <= f["N1_water_far"] <= 1.0
    assert f["I1_nearest_pellet"] <= np.sqrt(0.5) and f["I2_pellets_in_reach"] >= 0
    x, y, H = terrain_grid(w1)
    for i, j in ((0, 0), (17, 90), (127, 64)):
        assert H[i, j] == pytest.approx(w1._H(x[i, j], y[i, j]), abs=1e-12)
    w2 = make_world(P, 0.0, 5)
    w2.reset(_seeds(2_600_000))
    a = np.array([0.5, 0.2, 0.0, 0.0, 0.0], np.float32)
    for _ in range(5):
        o1, o2 = w1.step(a).obs, w2.step(a).obs
    np.testing.assert_array_equal(o1, o2)


def test_residualized_incidental_features_carry_no_linear_trace_of_the_near_field():
    from itasorl.novel_readouts import INCIDENTAL, INSTRUMENTAL, residualize_incidental
    rng = np.random.default_rng(4)
    F = {k: rng.normal(size=200) for k in INSTRUMENTAL}
    F.update({k: F["I1_nearest_pellet"] * 2 + rng.normal(size=200) for k in INCIDENTAL})
    R = residualize_incidental(F)
    for k in INCIDENTAL:
        assert abs(np.corrcoef(R[k], F["I1_nearest_pellet"])[0, 1]) < 1e-8


def test_rollout_with_a_zero_projector_equals_the_plain_rollout():
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    from itasorl import mechanism_readouts as mr
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                     device="cpu", seed=0)
    plain = mr.rollout_behavior(agent, norm, P, 0.0, n_eps=2, steps=8, seed_base=7, ray_steps=5)
    zero = mr.rollout_behavior(agent, norm, P, 0.0, n_eps=2, steps=8, seed_base=7, ray_steps=5,
                               erase=(np.zeros((8, 8)), np.zeros(8)))
    np.testing.assert_array_equal(plain["H"], zero["H"])
    np.testing.assert_array_equal(plain["ret"], zero["ret"])
    assert plain["ret"].shape == (len(plain["kept"]),)


def test_rollout_erasure_pins_the_states_inside_the_subspace():
    pytest.importorskip("torch")
    import itasorl.experiment_b2 as b2
    from itasorl import mechanism_readouts as mr
    from itasorl.novel_readouts import projector
    agent, norm = b2.untrained_agent(P, 0.0, 5, hidden=8, embed=8, world_model=True,
                                     device="cpu", seed=0)
    rng = np.random.default_rng(5)
    Pr = projector(rng.normal(size=(2, 8)))
    m = rng.normal(size=8)
    out = mr.rollout_behavior(agent, norm, P, 0.0, n_eps=2, steps=8, seed_base=7, ray_steps=5,
                              erase=(Pr, m))
    np.testing.assert_allclose(out["H"] @ Pr, np.broadcast_to(m @ Pr, out["H"].shape), atol=1e-5)


def test_cosine_and_paired_interval():
    from itasorl.novel_readouts import cosine, paired_t90
    assert cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    c = paired_t90([0.6, 0.7, 0.65, 0.62], [0.5, 0.5, 0.5, 0.5])
    assert c["n"] == 4 and c["t90"][0] < c["mean"] < c["t90"][1]


def test_ridge_r2_does_not_overfit_noise_in_the_readout_shape():
    """1100 episodes and 192 state features, the world-feature readout's shape: a target
    unrelated to the features must read R^2 near 0, the spec's sanity bound."""
    from itasorl.novel_readouts import residual_r2_cv, ridge_r2_cv
    rng = np.random.default_rng(6)
    X = rng.normal(size=(1100, 192))
    O = rng.normal(size=(1100, 120))
    t = rng.normal(size=1100)
    assert ridge_r2_cv(X, t) <= 0.02
    assert residual_r2_cv(X, O, t) <= 0.02


def test_arena_distance_does_not_wrap_through_the_walls():
    """patch_of_earth clips positions to [0, 1] and stops the normal velocity (line 207);
    the arena has walls, not a torus. Two points near opposite walls are 0.96 apart."""
    from itasorl.novel_readouts import arena_dist
    assert arena_dist(np.array([0.02, 0.5]), np.array([0.98, 0.5])) == pytest.approx(0.96)
    assert arena_dist(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.0, 0.0])).tolist() == \
        pytest.approx([0.0, 2 ** 0.5])


def test_torus_distance_is_gone():
    import itasorl.novel_readouts as nr
    assert not hasattr(nr, "_torus_dist")
