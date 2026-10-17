"""Common-garden channel tests (spec 2026-07-14): differing prefix world,
identical authentic tail; the probe reads tail-only states.

Load-bearing guarantees mirror test_experiment_b2.py:
  - L0 (drift off): auth- and surr-prefix tails are BIT-IDENTICAL, so the
    channel manufactures no signal;
  - drift on: tails diverge (prefix history is the only difference);
  - on synthetic tails, a PERSISTENT group signal scores high on both windows
    while a REACTIVE (early-only) signal collapses to chance on the late window."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, cg_probe, common_garden_rollout  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent_norm():
    from itasorl.patch_of_earth import PatchOfEarthV0
    w = PatchOfEarthV0(P)
    torch.manual_seed(0)
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


def test_cg_L0_tails_bit_identical():
    agent, norm = _agent_norm()
    auth, surr = common_garden_rollout(agent, norm, P, 0.0, n_pairs=3, prefix_steps=4,
                                       tail_steps=6, ray_steps=RS, device="cpu")
    assert len(auth) >= 1
    for a, s in zip(auth, surr):
        assert np.array_equal(a, s), "L0 common-garden tails diverged - channel is not confound-free"


def test_cg_drift_tails_diverge():
    agent, norm = _agent_norm()
    # this call is also the guard on the surrogate-RNG key filter inside
    # common_garden_rollout: without it, set_state KeyErrors here at drift>0
    auth, surr = common_garden_rollout(agent, norm, P, 0.5, n_pairs=3, prefix_steps=4,
                                       tail_steps=6, ray_steps=RS, device="cpu")
    assert any(not np.array_equal(a, s) for a, s in zip(auth, surr)), \
        "drift-on prefixes left identical tails - snapshot carry is broken"


def _synthetic_tails(rng, n, T, hid, offset_fn):
    """Tails where group 1 carries offset_fn(t) added to one hidden unit."""
    auth = [rng.normal(0, 1, (T, hid)).astype(np.float32) for _ in range(n)]
    surr = []
    for _ in range(n):
        H = rng.normal(0, 1, (T, hid)).astype(np.float32)
        H[:, 0] += np.array([offset_fn(t) for t in range(T)], np.float32)
        surr.append(H)
    return auth, surr


def test_cg_probe_pair_groups_keep_l0_floor_at_chance():
    """Regression: pair members must share ONE CV group. With per-episode groups,
    GroupKFold split twins across folds whenever the surviving pair count was not a
    multiple of n_splits (5), and the probe read the train twin's label off the
    near-identical test twin - bit-identical L0 tails scored AUROC 0.000 instead of
    0.5 (e.g. any heldout run where common_garden_rollout dropped a few pairs)."""
    rng = np.random.default_rng(0)
    T, hid = 12, 8
    for n_pairs in (23, 24, 25):  # 25 was accidentally safe; 23/24 hit the bug
        auth = [rng.normal(0, 1, (T, hid)).astype(np.float32) for _ in range(n_pairs)]
        surr = [a.copy() for a in auth]  # exact L0 twins: true AUROC is 0.5
        r = cg_probe(auth, surr, late_k=4, seed=0)
        assert abs(r["cg_tail_target"] - 0.5) < 0.05, \
            f"L0 floor off chance at n_pairs={n_pairs}: {r['cg_tail_target']}"
        assert abs(r["cg_latetail_target"] - 0.5) < 0.05, \
            f"L0 late-tail off chance at n_pairs={n_pairs}: {r['cg_latetail_target']}"


def test_probe_world_identity_pair_groups_keep_l0_at_chance():
    """Same regression for the SECONDARY matched-pair readout: identical twin pairs
    at a pair count that is not a multiple of 5 must probe at chance, not 0."""
    from itasorl.experiment_b2 import probe_world_identity
    rng = np.random.default_rng(1)
    n, T, hid = 23, 12, 8
    auth, surr = [], []
    for _ in range(n):
        H = rng.normal(0, 1, (T, hid)).astype(np.float32)
        spd = float(rng.uniform(0.1, 1.0))
        auth.append({"H": H, "label": 0, "speed": spd})
        surr.append({"H": H.copy(), "label": 1, "speed": spd})
    r = probe_world_identity(auth, surr, seed=0)
    assert abs(r["target"] - 0.5) < 0.05, f"L0 target off chance: {r['target']}"


def test_cg_probe_persistent_vs_reactive():
    rng = np.random.default_rng(0)
    T = 24
    pers_a, pers_s = _synthetic_tails(rng, 60, T, 8, lambda t: 3.0)          # persistent
    reac_a, reac_s = _synthetic_tails(rng, 60, T, 8, lambda t: 3.0 if t < 4 else 0.0)  # reactive
    pers = cg_probe(pers_a, pers_s, late_k=8, seed=0)
    reac = cg_probe(reac_a, reac_s, late_k=8, seed=0)
    assert pers["cg_tail_target"] > 0.9 and pers["cg_latetail_target"] > 0.9
    assert abs(reac["cg_latetail_target"] - 0.5) < 0.20, "late window must not see an early-only signal"


# --- closing hook A7a and G21: world-seed grouping, opt-in --------------------------------


def _tails(rng, n, T=10, hid=6):
    return ([rng.normal(0, 1, (T, hid)).astype(np.float32) for _ in range(n)],
            [rng.normal(0, 1, (T, hid)).astype(np.float32) + 0.3 for _ in range(n)])


def test_cg_groups_default_and_expansion():
    from itasorl.experiment_b2 import cg_groups
    assert np.array_equal(cg_groups(4), [0, 1, 2, 3, 0, 1, 2, 3])
    assert np.array_equal(cg_groups(3, [7, 7, 9]), [7, 7, 9, 7, 7, 9])
    assert np.array_equal(cg_groups(2, [5, 6, 5, 6]), [5, 6, 5, 6])
    with pytest.raises(ValueError, match="groups must have length"):
        cg_groups(3, [1, 2])


def test_cg_probe_default_equals_explicit_pair_groups():
    auth, surr = _tails(np.random.default_rng(3), 23)
    assert cg_probe(auth, surr) == cg_probe(auth, surr, groups=np.arange(23))
    assert cg_probe(auth, surr) == cg_probe(auth, surr, groups=None)


def test_world_seed_groups_keep_a_seed_in_one_fold():
    from itasorl import folds
    from itasorl.experiment_b2 import cg_groups
    from itasorl.experiment_c import group_by_world_seed
    widx = [[0, 1, 2, 4, 5, 6, 7], [0, 2, 3, 4, 6, 7], [1, 2, 3, 5, 6, 7]]  # deaths drop pairs
    g = group_by_world_seed(widx)
    assert g.tolist() == [p for ix in widx for p in ix]
    G = cg_groups(len(g), g)
    assert len(G) == 2 * len(g)
    for tr, te in folds.split(G):
        assert not set(G[te]) & set(G[tr]), "a world seed straddles a fold"
    assert group_by_world_seed([]).size == 0


def test_return_index_adds_kept_pairs_and_leaves_the_rollout_alone():
    agent, norm = _agent_norm()
    kw = dict(n_pairs=3, prefix_steps=4, tail_steps=6, ray_steps=RS, device="cpu")
    a0, s0 = common_garden_rollout(agent, norm, P, 0.5, **kw)
    a1, s1, ix = common_garden_rollout(agent, norm, P, 0.5, return_index=True, **kw)
    a2, s2, ast, sst, ix2 = common_garden_rollout(agent, norm, P, 0.5, return_index=True,
                                                  return_stats=True, **kw)
    assert len(ix) == len(a0) and ix == ix2 and ix == sorted(set(ix))
    assert all(0 <= p < 3 for p in ix) and len(ast) == len(sst) == len(a0)
    for x, y in ((a0, a1), (s0, s1), (a0, a2), (s0, s2)):
        assert all(np.array_equal(u, v) for u, v in zip(x, y))


def test_common_garden_panel_world_seed_groups_is_opt_in():
    from itasorl.experiment_c import common_garden_panel
    from itasorl.patch_of_earth import PatchOfEarthV0
    w = PatchOfEarthV0(P)
    pop = []
    for i in range(2):
        torch.manual_seed(500 + i)
        pop.append(RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=4, hidden=4,
                                        world_model=False))
    kw = dict(drift_sigma=0.02, n_pairs=6, prefix_steps=5, tail_steps=6, seed_base=930_000)
    base = common_garden_panel(pop, 0, **kw)
    off = common_garden_panel(pop, 0, world_seed_groups=False, **kw)
    on = common_garden_panel(pop, 0, world_seed_groups=True, **kw)
    assert "cv_groups" not in base
    assert list(off) == list(base)
    assert repr(off) == repr(base)
    assert on.pop("cv_groups") == "world_seed"
    assert list(on) == list(base) and on["cg_n_pairs"] == base["cg_n_pairs"]
    assert on["survival"] == base["survival"]


def _population_script():
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import validate_population_readout as vpr
    return vpr


def test_population_readout_flag_off_block_is_the_old_pooled_probe():
    vpr = _population_script()
    rng = np.random.default_rng(11)
    widx = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 3, 4, 5, 6, 8], [0, 2, 3, 4, 5, 7, 8]]
    auth, surr = _tails(rng, sum(len(ix) for ix in widx))
    off = vpr.pooled_probe_block(auth, surr)
    assert off == {"pooled_probe": float(cg_probe(auth, surr)["cg_tail_target"])}
    on = vpr.pooled_probe_block(auth, surr, widx)
    assert set(on) == {"pooled_probe", "pooled_probe_world_grouped",
                       "pooled_probe_world_grouped_ci", "n_world_groups"}
    assert on["pooled_probe"] == off["pooled_probe"]
    ref = cg_probe(auth, surr, groups=np.concatenate([np.asarray(ix) for ix in widx]))
    assert on["pooled_probe_world_grouped"] == float(ref["cg_tail_target"])
    assert on["pooled_probe_world_grouped_ci"] == [float(ref["cg_tail_lo"]), float(ref["cg_tail_hi"])]
    assert on["n_world_groups"] == 9


def test_population_readout_help_lists_the_world_seed_flag():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "validate_population_readout.py"),
                        "--help"], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert "--group-by-world-seed" in r.stdout
