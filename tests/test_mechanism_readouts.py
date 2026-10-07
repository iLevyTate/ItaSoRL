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


def test_probe_direction_degenerate_on_identical_pools():
    rng = np.random.default_rng(1)
    Ha = rng.normal(size=(30, 6, 5))
    with pytest.warns(UserWarning, match="degenerate probe direction"):
        u, s = mr.probe_direction(Ha, Ha.copy())
    assert np.isnan(s) and np.all(u == 0.0) and u.shape == (5,)


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
    score, gap = mr.gap_closed(b_auth, b_surr[:0], half, scale)       # empty pool
    assert np.isnan(score) and np.isnan(gap)
    assert np.all(np.isnan(mr.behavior_scale(b_auth[:0], b_surr[:0])))
    assert all(np.isnan(v) for v in mr.adaptation(np.zeros((0, 2)), np.ones((3, 2))).values())
    assert all(np.isnan(v) for v in mr.behavior_difference(b_auth[:0], b_surr, scale).values())


def test_surprise_summaries_and_correlation():
    E = np.tile(np.linspace(0.0, 1.0, 9), (4, 1))        # rising error, 4 episodes
    S = mr.surprise_summaries(E)
    assert S.shape == (4, 3)
    assert np.allclose(S[:, 0], 0.5) and np.allclose(S[:, 1], 1.0) and np.all(S[:, 2] > 0)
    H = np.zeros((4, 10, 3)); H[:, :, 1] = np.linspace(0.0, 1.0, 10)   # projection tracks error
    u = np.array([0.0, 1.0, 0.0])
    assert mr.direction_error_correlation(H, E, u) == pytest.approx(1.0)


def test_surprise_auroc_separable_null_and_nan():
    rng = np.random.default_rng(5)
    Ea = rng.uniform(0.0, 0.2, size=(100, 9))
    assert mr.surprise_auroc(Ea, Ea + 1.0) > 0.9                      # separable
    Eb = rng.uniform(0.0, 0.2, size=(100, 9))
    assert 0.2 < mr.surprise_auroc(Ea, Eb) < 0.8                      # same distribution
    assert np.isnan(mr.surprise_auroc(Ea, np.full((100, 9), np.nan)))  # no decoder


def test_surprise_summaries_nan_row_does_not_raise():
    E = np.vstack([np.linspace(0.0, 1.0, 9), np.full(9, np.nan)])
    S = mr.surprise_summaries(E)
    assert np.all(np.isfinite(S[0])) and np.all(np.isnan(S[1]))


def test_aggregate_behavior_uses_signed_turn_for_variability():
    turn = np.array([0.5, -0.5] * 4)                      # alternating left/right
    R = np.c_[np.ones(8), np.abs(turn), np.full(8, 0.3), np.zeros(8), np.zeros(8), turn]
    b = mr.aggregate_behavior(R)
    assert b.shape == (7,)
    names = dict(zip(mr.BEHAVIOR_NAMES, b))
    assert names["abs_turn"] == pytest.approx(0.5)
    assert names["std_turn"] == pytest.approx(0.5)
    assert names["std_thrust"] == pytest.approx(0.0)
    assert names["thrust"] == pytest.approx(0.3) and names["speed"] == pytest.approx(1.0)


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
    k = r0["B"].shape[0]
    assert r0["B"].shape[1] == 7 and 0 < k <= 3 and len(r0["kept"]) == k
    assert r0["H"].shape == (k, 10, 8) and r0["E"].shape == (k, 9) and r0["halves"].shape == (k, 2)
    assert r0["E"].dtype == np.float32
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


def test_common_rows_restricts_to_the_shared_survivors():
    from run_mechanism_readouts import common_rows
    r1 = {"B": np.arange(8.0).reshape(4, 2), "kept": np.array([0, 1, 2, 3])}
    r2 = {"B": np.arange(6.0).reshape(3, 2) + 100, "kept": np.array([1, 2, 5])}
    r3 = {"B": np.arange(4.0).reshape(2, 2) + 200, "kept": np.array([2, 1])}
    b1, b2, b3 = common_rows(r1, r2, r3)
    # every output is restricted to episodes {1, 2}, in the same (sorted) order
    np.testing.assert_array_equal(b1, [[2, 3], [4, 5]])
    np.testing.assert_array_equal(b2, [[100, 101], [102, 103]])
    np.testing.assert_array_equal(b3, [[202, 203], [200, 201]])
    assert common_rows(r1, {"B": np.zeros((0, 2)), "kept": np.array([], int)})[0].shape == (0, 2)


def test_driver_quick_on_a_smoke_run(tmp_path):
    """End to end on a quick-scale run directory produced by run_expB2 --quick."""
    pytest.importorskip("torch")
    import subprocess
    run_dir = tmp_path / "run"
    subprocess.run([sys.executable, "scripts/run_expB2.py", "--quick", "--drift-mode", "l3",
                    "--seeds", "0", "--drifts", "0.0", "0.45", "--workers", "1", "--device", "cpu",
                    "--save-agents", "--out-dir", str(run_dir)], check=True, cwd=ROOT)
    out = tmp_path / "mech.json"
    subprocess.run([sys.executable, "scripts/run_mechanism_readouts.py", "--run-dir", str(run_dir),
                    "--out", str(out), "--quick"], check=True, cwd=ROOT)
    import json
    d = json.loads(out.read_text())
    s0 = d["per_seed"][0]
    for key in ("intervention", "behavior", "surprise", "adaptation", "scripted_stream",
                "value_of_world_information"):
        assert key in s0, key
    assert set(s0["intervention"]) >= {"score_real", "score_sham", "gap", "score_real_reverse",
                                       "score_sham_reverse", "informative", "n_common_auth",
                                       "n_common_surr"}
    assert "n_seeds" in d["summary"]
    assert d["summary"]["n_seeds"] == len(d["per_seed"])


def _summary(real_sham=(0.2, 0.05, 0.35), n_inf=8, s_auc=0.7, s_corr=(0.3, 0.1, 0.5),
             gap1=(0.02, 0.01, 0.03), adapt=(0.01, 0.002, 0.02)):
    def pk(m, lo, hi):
        return {"n": n_inf, "mean": m, "ci90": [lo, hi]}
    return {"n_seeds": 10, "n_informative": n_inf,
            "intervention_real_minus_sham": pk(*real_sham),
            "surprise_auroc": pk(s_auc, s_auc - 0.05, s_auc + 0.05),
            "surprise_corr": pk(*s_corr),
            "adaptation_gap_first": pk(*gap1), "adaptation": pk(*adapt)}


def test_decide_intervention_rule():
    import decide_goal_stakes as dg
    assert dg.intervention_verdict(_summary())["pass"] is True
    assert dg.intervention_verdict(_summary(real_sham=(0.2, -0.01, 0.4)))["pass"] is False
    assert dg.intervention_verdict(_summary(real_sham=(0.08, 0.02, 0.14)))["pass"] is False
    assert dg.intervention_verdict(_summary(n_inf=6))["pass"] is False      # needs 7 seeds


def test_decide_surprise_rule():
    import decide_goal_stakes as dg
    assert dg.surprise_verdict(_summary())["wording"].startswith("the decoded direction tracks")
    v = dg.surprise_verdict(_summary(s_corr=(0.1, -0.1, 0.3)))
    assert v["wording"].startswith("the surrogate is surprising")
    v = dg.surprise_verdict(_summary(s_auc=0.55))
    assert v["wording"].startswith("prediction error does not")


def test_decide_adaptation_rule():
    import decide_goal_stakes as dg
    assert dg.adaptation_verdict(_summary())["wording"].startswith("foraging in the fake world recovers")
    assert dg.adaptation_verdict(_summary(gap1=(0.0, -0.01, 0.01)))["wording"].startswith("uninformative")
    assert dg.adaptation_verdict(_summary(adapt=(0.01, -0.002, 0.02)))["wording"].startswith("no within-lifetime")


def test_decide_stakes_rule():
    import decide_goal_stakes as dg
    means = {"S-immortal": 0.60, "C1": 0.733, "S-scarce": 0.80}
    contrast = {"mean": 0.20, "ci90": [0.08, 0.32]}
    assert dg.stakes_verdict(means, contrast)["pass"] is True
    assert dg.stakes_verdict({**means, "S-immortal": 0.75}, contrast)["pass"] is False  # order
    assert dg.stakes_verdict(means, {"mean": 0.04, "ci90": [0.01, 0.07]})["pass"] is False
