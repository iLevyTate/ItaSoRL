"""Post-review readouts (docs/specs/2026-10-11-post-review-readouts-design.md): the pure
pieces of itasorl/post_review.py and the resume and wiring logic of
scripts/run_post_review_readouts.py. No rollout, no world, no torch training: pools are
synthetic arrays, worlds are fakes, and the estimators the runner calls are monkeypatched."""

from __future__ import annotations

import ast
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from itasorl import folds  # noqa: E402
from itasorl import post_review as pr  # noqa: E402


def _pools(rng, n=30, T=4, hid=3, shift=1.5):
    Ha = rng.normal(size=(n, T, hid)).astype(np.float32)
    Hs = (rng.normal(size=(n, T, hid)) + np.array([shift] + [0.0] * (hid - 1))).astype(np.float32)
    return Ha, Hs


# ---------------------------------------------------------------------------
# r210: probe classes
# ---------------------------------------------------------------------------
def test_registered_permutations_are_the_pooled_readout_draws(monkeypatch):
    """pooled_readout's shuffled control, on synthetic pools, equals the linear class scored on
    registered_permutations: the nonlinear classes see the same eight label draws."""
    import itasorl.experiment_b2 as b2
    rng = np.random.default_rng(0)
    Ha, Hs = _pools(rng)

    def fake_collect_pool(agent, norm, params, drift, n_eps, steps, device, seed_base, ray_steps, **kw):
        H = Hs if seed_base == 850_000 else Ha
        k = len(H)
        r = np.random.default_rng(seed_base)
        return (H, r.random(k), r.random(k), r.random(k), r.random(k), r.random(k),
                r.random((k, steps, 7)).astype(np.float32))

    monkeypatch.setattr(b2, "collect_pool", fake_collect_pool)
    out, (A, S) = b2.pooled_readout(None, None, None, 0.45, n_eps=30, steps=4, device="cpu",
                                    seed=3, return_pools=True)
    fast = {**pr.PROBE_CLASSES,
            "mlp": lambda X, y, seed=0, **kw: pr.mlp_probe_auroc(X, y, seed=seed, max_iter=60)}
    cell = pr.r210_cell(A, S, pool=out, recorded={"target": out["target"], "shuffled": out["shuffled"]},
                        seed=3, rf_trees=10, scorers=fast)
    assert cell["reproduced"] and cell["shuffled_reproduced"] and cell["perm_check"]
    assert cell["classes"]["linear"]["target"] == out["target"]
    assert set(cell["classes"]) == {"linear", "rf", "mlp"}
    for c in ("rf", "mlp"):
        k = cell["classes"][c]
        assert len(k["shuffled_draws"]) == pr.N_PERM
        assert k["selectivity"] == pytest.approx(k["target"] - k["shuffled"])
        assert k["target"] > 0.8                       # the pools separate on unit 0


def test_unreproduced_cell_is_not_scored():
    rng = np.random.default_rng(1)
    Ha, Hs = _pools(rng)
    pool = {"target": 0.71, "shuffled": 0.5}
    cell = pr.r210_cell(Ha, Hs, pool=pool, recorded={"target": 0.7100000000000001, "shuffled": 0.5}, seed=0)
    assert cell["reproduced"] is False and "classes" not in cell
    assert cell["shuffled_reproduced"] is True


def test_probe_class_scorers_on_separable_and_shuffled_data():
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal(size=(40, 6)), rng.normal(size=(40, 6)) + 2.0]
    y = np.r_[np.zeros(40), np.ones(40)].astype(int)
    perms = pr.registered_permutations(y, seed=5, n_perm=3)
    assert all(np.array_equal(a, b) for a, b in zip(perms, pr.registered_permutations(y, 5, 3)))
    out = pr.score_probe_classes(X, y, seed=5, perms=perms, n_estimators=15)
    for c in ("rf", "mlp"):
        assert out[c]["target"] > 0.95
        assert abs(out[c]["shuffled"] - 0.5) < 0.2
    mlp = pr.mlp_probe_auroc(X, y, seed=0, max_iter=3)
    assert len(mlp["n_iter"]) == 5 and mlp["hit_max_iter"] == 5
    from itasorl.behavior_audit import behavior_only_auroc
    small = X[::2], y[::2]
    assert pr.rf_probe_auroc(*small, seed=1)["auroc"] == behavior_only_auroc(*small, nonlinear=True, seed=1)
    from itasorl.experiment_a import grouped_auroc
    assert pr.linear_probe_auroc(X, y)["auroc"] == grouped_auroc(X, y, np.arange(len(y)))


def _r210_seed(reproduced: dict, base=0.7):
    cells = {}
    for d, ok in reproduced.items():
        cells[d] = {}
        for i, arm in enumerate(pr.ARMS):
            t = base - 0.1 * i
            cells[d][arm] = {"reproduced": ok}
            if ok:
                cells[d][arm]["classes"] = {c: {"target": t + off, "shuffled": 0.5, "selectivity": t + off - 0.5}
                                            for c, off in (("linear", 0.0), ("rf", 0.02), ("mlp", -0.01))}
    return {"cells": cells, "reproduced": dict(reproduced)}


def test_r210_summary_excludes_a_seed_only_at_the_drift_it_failed():
    per_seed = {"0": _r210_seed({"0.00": True, "0.45": True}, 0.70),
                "1": _r210_seed({"0.00": True, "0.45": False}, 0.72),
                "2": _r210_seed({"0.00": True, "0.45": True}, 0.75)}
    s = pr.summarize_r210(per_seed)
    assert s["0.45"]["seeds_used"] == ["0", "2"] and s["0.45"]["seeds_excluded"] == ["1"]
    assert s["0.00"]["seeds_used"] == ["0", "1", "2"]
    rf = s["0.45"]["class_minus_linear"]["survival"]["rf"]
    assert rf["n"] == 2 and rf["mean"] == pytest.approx(0.02)
    m = s["0.45"]["margins"]["survival_minus_untrained"]["linear"]
    assert m["mean"] == pytest.approx(0.2)
    import run_post_review_readouts as rpr
    integ = rpr.integrity({"r210": per_seed}, ["0.00", "0.45"])
    assert integ["checked"] and integ["unreproduced"] == {"0.00": [], "0.45": ["1"]}


# ---------------------------------------------------------------------------
# r35: normalizer variants
# ---------------------------------------------------------------------------
def test_refit_normalizer_is_the_empirical_moments_with_running_norm_formula():
    from itasorl.experiment_b2 import RunningNorm
    rng = np.random.default_rng(3)
    a, b = rng.normal(2.0, 3.0, size=(5, 4, 3)), rng.normal(-1.0, 0.5, size=(7, 4, 3))
    nm = pr.refit_normalizer([a, b])
    rows = np.concatenate([a.reshape(-1, 3), b.reshape(-1, 3)])
    assert np.allclose(nm.mean, rows.mean(0)) and np.allclose(nm.var, rows.var(0))
    assert isinstance(nm, RunningNorm) and nm.frozen
    x = rng.normal(size=(2, 3))
    assert np.array_equal(nm(x), (x - nm.mean) / (np.sqrt(nm.var) + 1e-6))


def test_infold_normalizer_sees_training_episodes_only():
    rng = np.random.default_rng(4)
    n, T, O = 40, 3, 2
    obs = rng.normal(size=(n, T, O))
    obs[n // 2:, :, 0] += 1.0
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    seen = []

    def states_fn(norm):
        seen.append(norm.mean.copy())
        return norm(obs)                     # an identity "trunk" on the normalized stream

    auc = pr.infold_normalizer_auroc(obs, y, states_fn)
    splits = list(folds.split(np.arange(n)))
    assert len(seen) == len(splits)
    for m, (tr, _) in zip(seen, splits):
        assert np.allclose(m, obs[tr].reshape(-1, O).mean(0))
    assert 0.5 < auc <= 1.0


def _r35_seed(surv_own, surv_refit, unt):
    return {"arms": {"survival": {"own": surv_own, "survival_norm": surv_own, "refit": surv_refit,
                                  "refit_infold": surv_refit},
                     "predictor": {v: surv_own - 0.01 for v in pr.R35_VARIANTS},
                     "untrained": dict(zip(pr.R35_VARIANTS, unt))}}


def test_r35_margins_and_reading():
    rng = np.random.default_rng(5)
    trunk = {str(s): _r35_seed(0.73 + e, 0.72 + e, (0.63 + e, 0.64 + e, 0.63 + e, 0.63 + e))
             for s, e in enumerate(rng.normal(0, 0.01, 8))}
    s = pr.summarize_r35(trunk)
    m = s["margins"]["survival_minus_untrained"]
    assert m["own"]["mean"] == pytest.approx(0.10, abs=0.01)
    assert m["survival_norm"]["share_of_own_margin"] > 0.5
    assert s["reading"]["untrained"] == "TRUNK"
    norm = {str(s): _r35_seed(0.73 + e, 0.72 + e, (0.63, 0.73 + e, 0.72 + e, 0.72 + e))
            for s, e in enumerate(rng.normal(0, 0.01, 8))}
    assert pr.summarize_r35(norm)["reading"]["untrained"] == "NORMALIZER"
    above = {"t90": [0.05, 0.10], "share_of_own_margin": 0.8}
    m = {"survival_norm": above, "refit": above,
         "refit_infold": {"t90": [-0.01, 0.03], "share_of_own_margin": 0.1}}
    assert pr.r35_reading(m) == "SPLIT"                   # the in-fold refit governs
    assert pr.r35_reading({**m, "refit_infold": above}) == "TRUNK"


# ---------------------------------------------------------------------------
# r206: deviation integration
# ---------------------------------------------------------------------------
def test_law_deviation_is_zero_on_the_authentic_law_and_the_offset_otherwise():
    from itasorl.surrogate_l3 import GradedGMotion
    rng = np.random.default_rng(6)
    dt, drag = 0.05, 1.5
    authentic = GradedGMotion(None, 0.0, dt)              # exactly the authentic arithmetic
    log, shifted = [], []
    for _ in range(5):
        v, a = rng.normal(size=2), rng.normal(size=2)
        log.append((v, a, drag, authentic(v, a, drag)))
        shifted.append((v, a, drag, authentic(v, a, drag) + np.array([0.3, -0.4])))
    assert np.all(pr.law_deviation(log, dt) == 0.0)
    assert np.allclose(pr.law_deviation(shifted, dt), 0.5)


class _FakeWorld:
    """Records motion like PatchOfEarthV0: G = authentic law + a fixed offset."""

    def __init__(self, offset):
        self.params = types.SimpleNamespace(dt=0.05)
        self.offset, self._log_motion = np.asarray(offset, float), None

    def reset(self, seed):
        self.t, self.vel, self.die_at = 0, np.zeros(2), 2 if seed % 3 == 0 else None

    def observe(self):
        return np.array([float(self.t), 0.0])

    def step(self, a):
        acc, drag, dt = np.array([float(a[0]), 0.0]), 1.5, self.params.dt
        vnext = (1 - drag * dt) * self.vel + acc * dt + self.offset
        self._log_motion.append((self.vel.copy(), acc, drag, vnext))
        self.vel, self.t = vnext, self.t + 1
        return types.SimpleNamespace(obs=self.observe(), terminated=self.t == self.die_at)


def test_walk_deviation_integrates_full_length_episodes_only():
    seen = []

    def policy_factory(i):
        def act(obs):
            seen.append((i, obs[0]))
            return np.array([0.5, 0.0])
        return act

    w = pr.walk_deviation(policy_factory, None, 0.45, n_eps=6, steps=5, seed_base=0, ray_steps=5,
                          world_factory=lambda p, d, r: _FakeWorld([0.0, 0.2]), seeds_fn=lambda s: s)
    assert w["kept"].tolist() == [1, 2, 4, 5]            # worlds 0 and 3 die at step 2
    assert w["D"].shape == (4, 5) and np.allclose(w["D"], 0.2)
    assert [o for i, o in seen if i == 1] == [0.0, 1.0, 2.0, 3.0, 4.0]   # the policy reads observe()
    s = pr.deviation_summary(w)
    assert s["n_eps"] == 4 and s["sum_mean"] == pytest.approx(1.0) and s["step_q99"] == pytest.approx(0.2)
    assert pr.deviation_summary({"D": np.zeros((0, 5)), "speed": [], "acc": []})["n_eps"] == 0


def test_r206_reading_and_protocol_map():
    rng = np.random.default_rng(7)

    def seed(surv, other):
        keys = ("sum_mean", "step_mean", "step_q90", "speed_mean", "acc_mean")
        return {"walks": {w: {k: (surv if w == "survival_own" else other) + rng.normal(0, 0.01) for k in keys}
                          for w in pr.R206_WALKS}}
    hi = pr.summarize_r206({str(s): seed(1.0, 0.5) for s in range(6)},
                           {str(s): 0.7 + 0.01 * s for s in range(6)})
    assert hi["reading"] == "HIGHER" and hi["corr_with_recorded_target"]["n"] == 6
    assert pr.summarize_r206({str(s): seed(0.5, 0.5) for s in range(6)})["reading"] in ("NOT HIGHER", "MIXED")
    assert pr.R206_PROTOCOL_WALK["replay"] == {arm: "survival_own" for arm in pr.ARMS}


# ---------------------------------------------------------------------------
# r33: balanced estimator wiring and the two-run contrast
# ---------------------------------------------------------------------------
def test_r33_task_reads_every_agent_at_the_strongest_drift_on_both_bases(monkeypatch):
    import itasorl.experiment_b2 as b2
    import itasorl.l0_audit as l0
    import run_post_review_readouts as rpr
    calls, loads = [], []
    monkeypatch.setattr(b2, "_L3_GMOTION", object())     # no G fit in a test
    monkeypatch.setattr(b2, "DRIFT_MODE", b2.DRIFT_MODE)
    monkeypatch.setattr(b2, "OBJECTIVE", b2.OBJECTIVE)
    monkeypatch.setattr(b2, "MORTAL", b2.MORTAL)
    monkeypatch.setattr(b2, "load_agent_bundle", lambda path, device="cpu": (loads.append(path) or "A", "N"))

    def fake_paired(agent, norm, params, drift, **kw):
        calls.append((drift, kw["seed_base"], kw["seed"], kw["n_eps"], kw["steps"]))
        return {"target": 0.6 + 0.001 * len(calls), "n_pairs": kw["n_eps"], "dropped_pairs": 0,
                "cluster_ci95": [0.5, 0.7], "identical_first_state": True}

    monkeypatch.setattr(l0, "paired_pooled_readout", fake_paired)
    task = {"run_dir": "RUN", "seed": "3", "drifts": ["0.00", "0.45"], "dmax": "0.45", "knobs": {},
            "objective": "survival", "mortal": True, "settings": rpr.settings_for(False), "readout": "r33"}
    res = rpr.run_task(task)
    assert "error" not in res, res.get("error")
    cells = res["result"]["cells"]
    assert len(calls) == 12 and {c[0] for c in calls} == {0.45}
    assert {c[1] for c in calls} == {pr.BALANCED_FRESH_BASE, pr.REGISTERED_AUTH_BASE}
    assert {c[2:] for c in calls} == {(3, 110, 24)}
    assert sorted({(c["agent_drift"], c["arm"], c["base"]) for c in cells}) == sorted(
        (d, a, b) for d in (0.0, 0.45) for a in pr.ARMS for b in ("fresh", "registered"))
    assert [os.path.basename(p) for p in loads] == [
        f"agent_d{d}_s3_{a}.pt" for d in ("0.00", "0.45") for a in pr.ARMS]
    s = pr.summarize_r33({"3": res["result"]})
    assert set(s) == {"d0.00/fresh", "d0.00/registered", "d0.45/fresh", "d0.45/registered"}


def _patch_world(monkeypatch):
    """Skip the G fit and the agent files: a task then runs on whatever the patched
    rollouts return. Returns the list of agent paths the task loaded."""
    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b2 import RunningNorm
    loads = []
    for name in ("DRIFT_MODE", "OBJECTIVE", "MORTAL"):
        monkeypatch.setattr(b2, name, getattr(b2, name))
    monkeypatch.setattr(b2, "_L3_GMOTION", object())

    def load(path, device="cpu"):
        loads.append(os.path.basename(path))
        nm = RunningNorm(3)
        nm.mean, nm.var = np.full(3, 0.1 * len(loads)), np.full(3, 1.0 + len(loads))
        return f"agent{len(loads)}", nm.freeze()
    monkeypatch.setattr(b2, "load_agent_bundle", load)
    return loads


def _task(readout, quick=True):
    import run_post_review_readouts as rpr
    return {"run_dir": "RUN", "seed": "2", "drifts": ["0.00", "0.45"], "dmax": "0.45", "knobs": {},
            "objective": "survival", "mortal": True, "settings": rpr.settings_for(quick), "readout": readout}


def test_r210_task_bit_checks_every_cell_and_scores_only_reproduced_ones(monkeypatch):
    import itasorl.experiment_b2 as b2
    import run_post_review_readouts as rpr
    loads = _patch_world(monkeypatch)
    rng = np.random.default_rng(9)

    def fake_pooled(agent, norm, params, drift, **kw):
        Ha, Hs = _pools(rng, n=kw["n_eps"], T=kw["steps"])
        return {"target": 0.6 if drift == 0.45 else 0.5, "shuffled": 0.49}, (Ha, Hs)

    monkeypatch.setattr(b2, "pooled_readout", fake_pooled)
    monkeypatch.setitem(pr.PROBE_CLASSES, "mlp",
                        lambda X, y, seed=0, **kw: pr.mlp_probe_auroc(X, y, seed=seed, max_iter=20))
    task = _task("r210")
    rec = {"target": 0.6, "shuffled": 0.49}
    task["recorded"] = {"0.45": {"2": {a: rec for a in pr.ARMS}},
                        "0.00": {"2": {a: ({"target": 0.5, "shuffled": 0.49} if a == "predictor" else rec)
                                       for a in pr.ARMS}}}       # only the predictor reproduces at drift 0
    res = rpr.run_task(task)
    assert "error" not in res, res.get("error")
    r = res["result"]
    assert r["reproduced"] == {"0.00": False, "0.45": True}
    assert r["cells"]["0.00"]["predictor"]["reproduced"] and "classes" not in r["cells"]["0.00"]["survival"]
    assert set(r["cells"]["0.45"]["survival"]["classes"]) == {"linear", "rf", "mlp"}
    assert len(loads) == 6


def test_r35_task_replays_the_survival_streams_under_four_normalizers(monkeypatch):
    import itasorl.eval_protocols as ep
    import run_post_review_readouts as rpr
    _patch_world(monkeypatch)
    rng = np.random.default_rng(10)
    bases = []

    def fake_record(agent, norm, params, drift, n, steps, device, seed_base, ray_steps):
        bases.append((drift, seed_base))
        obs = rng.normal(size=(n, steps, 3)) + (1.0 if drift > 0 else 0.0)
        return obs.astype(np.float32), np.zeros((n, steps, 5), np.float32), np.arange(n)

    monkeypatch.setattr(ep, "record_trajectories", fake_record)
    monkeypatch.setattr(ep, "replay_states", lambda ag, norm, obs, acts, device="cpu": norm(obs))
    res = rpr.run_task(_task("r35"))
    assert "error" not in res, res.get("error")
    r = res["result"]
    assert bases == [(0.0, pr.FRESH_AUTH_BASE), (0.45, pr.FRESH_SURR_BASE)]
    assert set(r["arms"]) == set(pr.ARMS) and all(set(v) == set(pr.R35_VARIANTS) for v in r["arms"].values())
    assert r["arms"]["survival"]["own"] == r["arms"]["survival"]["survival_norm"] == r["reference"]
    assert r["arms"]["predictor"]["refit"] == r["arms"]["untrained"]["refit"]   # one shared input


def test_r206_task_walks_four_policies_on_the_fresh_surrogate_worlds(monkeypatch):
    import run_post_review_readouts as rpr
    _patch_world(monkeypatch)
    seen = []

    def fake_walk(policy_factory, params, drift, *, n_eps, steps, seed_base, ray_steps):
        seen.append((drift, seed_base, n_eps, steps))
        return {"D": np.full((n_eps, steps), 0.1), "speed": np.ones(n_eps), "acc": np.ones(n_eps),
                "kept": np.arange(n_eps)}

    monkeypatch.setattr(pr, "walk_deviation", fake_walk)
    res = rpr.run_task(_task("r206"))
    assert "error" not in res, res.get("error")
    assert set(res["result"]["walks"]) == set(pr.R206_WALKS)
    assert seen == [(0.45, pr.FRESH_SURR_BASE, 12, 8)] * 4
    assert res["result"]["walks"]["scripted"]["sum_mean"] == pytest.approx(0.8)


def test_r31_task_builds_the_17_8_sequences_and_every_decoder(monkeypatch):
    import itasorl.control_diagnostics as cd
    import itasorl.eval_protocols as ep
    import itasorl.experiment_b2 as b2
    import itasorl.mechanism_readouts as mr
    import run_post_review_readouts as rpr
    _patch_world(monkeypatch)
    rng = np.random.default_rng(11)
    shapes = []

    def fake_pool(agent, norm, params, drift, n, steps, device, base, rs, return_obs, return_actions):
        H = rng.normal(size=(n, steps, 4)) + drift
        return H, None, rng.normal(size=(n, steps, 3)), rng.random((n, steps, 5))

    def fake_gru(seq, y, *, embed, hidden, epochs, seed):
        shapes.append((seq.shape, len(y), hidden, epochs, seed))
        return {"auroc": 0.6, "final_train_loss": [0.0]}

    monkeypatch.setattr(b2, "collect_pool", fake_pool)
    monkeypatch.setattr(mr, "scripted_observation_streams", lambda nm, P, d, *, n_eps, steps, seed_base, ray_steps:
                        (rng.normal(size=(n_eps, steps, 3)), rng.random((n_eps, steps, 5))))
    monkeypatch.setattr(ep, "collect_pool_scripted", lambda ag, nm, P, d, n, steps, dev, base, rs:
                        (rng.normal(size=(n, steps, 4)) + d, np.arange(n)))
    monkeypatch.setattr(cd, "sequence_gru_auroc", fake_gru)
    monkeypatch.setattr(pr, "conv_sequence_auroc", lambda seq, y, *, epochs, seed: {"auroc": 0.55})
    res = rpr.run_task(_task("r31"))
    assert "error" not in res, res.get("error")
    r = res["result"]
    assert set(r) >= set(pr.R31_STREAMS)
    for st in pr.R31_STREAMS:
        assert list(r[st]["decoders"]["gru"]) == ["1x", "2x"] and r[st]["decoders"]["conv"]["auroc"] == 0.55
        assert 0.5 <= r[st]["state"] <= 1.0
    assert [s[0] for s in shapes] == [(24, 8, 8)] * 4      # 12 + 12 episodes, 8 steps, 3 + 5 channels
    assert [s[2] for s in shapes] == [96, 192, 96, 192] and {s[4] for s in shapes} == {2}


def _r33_doc(vals, bases=None, settings=None):
    per = {}
    for s, v in vals.items():
        per[s] = {"cells": [{"agent_drift": d, "arm": arm, "base": b, "target": v + (0 if arm == "survival" else -0.2)}
                            for d in (0.0, 0.45) for arm in pr.ARMS for b in ("fresh", "registered")]}
    return {"run_dir": "X", "seed_bases": bases or {"balanced": dict(pr.R33_BASES)},
            "settings": settings or {"n_eps": 110}, "per_seed": {"r33": per}}


def test_balanced_contrast_is_seed_paired_and_refuses_mismatched_estimators():
    a = _r33_doc({"0": 0.80, "1": 0.70, "2": 0.75, "3": 0.82})
    b = _r33_doc({"0": 0.60, "1": 0.62, "2": 0.66, "3": 0.61})
    c = pr.balanced_contrast(a, b)
    assert c["primary"] == "d0.45/fresh"
    p = c["contrasts"]["d0.45/fresh"]
    assert p["n"] == 4 and p["per_seed"] == pytest.approx([0.20, 0.08, 0.09, 0.21])
    assert p["mean"] == pytest.approx(0.145) and p["t90"][0] < p["mean"] < p["t90"][1]
    assert p["reading"] == "HOLDS"
    assert pr.contrast_reading({"t90": [0.01, 0.2]}) == "DIRECTION ONLY"
    assert pr.contrast_reading({"t90": [-0.01, 0.2]}) == "NOT RESOLVED"
    with pytest.raises(ValueError):
        pr.balanced_contrast(a, _r33_doc({"0": 0.6}, bases={"balanced": {"fresh": 1}}))


def test_contrast_entry_point_writes_the_paired_contrast(tmp_path):
    import run_post_review_readouts as rpr
    a, b = tmp_path / "c1.json", tmp_path / "c2.json"
    a.write_text(json.dumps(_r33_doc({"0": 0.80, "1": 0.70, "2": 0.75})), encoding="utf-8")
    b.write_text(json.dumps(_r33_doc({"0": 0.60, "1": 0.62, "2": 0.66})), encoding="utf-8")
    out = tmp_path / "contrast.json"
    assert rpr.main(["--contrast", str(a), str(b), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["primary"] == "d0.45/fresh" and doc["contrasts"]["d0.45/fresh"]["n"] == 3


def test_balanced_reference_check_matches_the_l0_audit_layout():
    per = _r33_doc({"0": 0.80, "1": 0.70})["per_seed"]["r33"]
    ref = {"paired": [{"seed": 0, "arm": "survival", "drift": 0.45, "seed_base": 800000, "target": 0.80},
                      {"seed": 1, "arm": "survival", "drift": 0.45, "seed_base": 800000, "target": 0.71}]}
    r = pr.balanced_reference_check(per, ref, drift=0.45)
    assert r["n_checked"] == 2 and r["n_bit_equal"] == 1 and r["max_abs_dev"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# r31: the decoder-capacity table
# ---------------------------------------------------------------------------
def test_capacity_table_scales_the_recurrent_width_from_the_17_8_decoder():
    t = pr.capacity_table()
    assert [(r["label"], r["embed"], r["hidden"]) for r in t] == [
        ("1x", 64, 96), ("2x", 64, 192), ("4x", 64, 384), ("8x", 64, 768)]
    seen = []

    def gru(seq, y, *, embed, hidden, epochs, seed):
        seen.append((embed, hidden, epochs, seed))
        return {"auroc": hidden / 1000.0, "final_train_loss": [0.0]}

    curve = pr.stream_decoder_curve(np.zeros((10, 4, 3)), np.r_[np.zeros(5), np.ones(5)].astype(int),
                                    seed=7, epochs=3, gru_fn=gru,
                                    conv_fn=lambda s, y, *, epochs, seed: {"auroc": 0.5, "epochs": epochs})
    assert seen == [(64, 96, 3, 7), (64, 192, 3, 7), (64, 384, 3, 7), (64, 768, 3, 7)]
    assert list(curve["gru"]) == ["1x", "2x", "4x", "8x"] and curve["conv"]["epochs"] == 3


def test_conv_reader_maps_a_sequence_batch_to_one_logit_each():
    import torch
    m = pr._conv_reader(4)
    assert m(torch.zeros(3, 7, 4)).shape == (3,)


def test_r31_readings():
    rng = np.random.default_rng(8)

    def seed(state, dec):
        return {st: {"state": state + rng.normal(0, 0.005),
                     "decoders": {"gru": {k: {"auroc": dec[k] + rng.normal(0, 0.005)} for k in ("1x", "2x", "4x", "8x")},
                                  "conv": {"auroc": dec["conv"] + rng.normal(0, 0.005)}}}
                for st in pr.R31_STREAMS}
    flat = {"1x": 0.72, "2x": 0.72, "4x": 0.71, "8x": 0.71, "conv": 0.70}
    rows = {str(i): seed(0.73, flat) for i in range(8)}
    rows["8"] = {st: {"n_auth": 3, "n_surr": 2, "error": "too few survivors"} for st in pr.R31_STREAMS}
    s = pr.summarize_r31(rows)
    assert s["survival_own"]["state"]["n"] == 8           # the seed without decoders is skipped
    assert s["survival_own"]["capacity_reading"] == "NOT CAPACITY-LIMITED"
    assert s["survival_own"]["state_reading"] == "EQUIVALENT"
    wide = {"1x": 0.60, "2x": 0.66, "4x": 0.70, "8x": 0.74, "conv": 0.62}
    s = pr.summarize_r31({str(i): seed(0.62, wide) for i in range(8)})
    assert s["scripted"]["capacity_reading"] == "CAPACITY-LIMITED"
    assert s["scripted"]["best_decoder"] == "8x" and s["scripted"]["state_reading"] == "DECODER ABOVE"


# ---------------------------------------------------------------------------
# Partial-file resume
# ---------------------------------------------------------------------------
def test_partial_store_resumes_drops_a_cut_line_and_refuses_another_header(tmp_path):
    path = tmp_path / "x.partial.jsonl"
    st = pr.PartialStore(str(path), {"fingerprints": {"d0.45_s0": "abc"}, "settings": {"n": (1, 2)}})
    assert st.load() == {}
    st.append("r33", 0, result={"v": np.float32(0.5)})
    st.append("r35", 1, error="boom")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"readout": "r31", "seed": "0", "res')        # a kill mid-write
    done = pr.PartialStore(str(path), {"fingerprints": {"d0.45_s0": "abc"}, "settings": {"n": [1, 2]}}).load()
    assert done == {("r33", "0"): {"v": 0.5}}
    assert path.read_text(encoding="utf-8").endswith("\n")
    st.append("r35", 1, result={"v": 1})
    assert set(st.load()) == {("r33", "0"), ("r35", "1")}
    st.append("r35", 1, error="later failure")
    assert set(st.load()) == {("r33", "0")}
    with pytest.raises(ValueError, match="different run"):
        pr.PartialStore(str(path), {"fingerprints": {"d0.45_s0": "zzz"}, "settings": {"n": [1, 2]}}).load()


def test_plan_tasks_skips_done_and_puts_the_decoders_first():
    import run_post_review_readouts as rpr
    todo = rpr.plan_tasks(["r206", "r31", "r33"], [0, 1], {("r31", "0"): {}})
    assert todo == [("r31", "1"), ("r33", "0"), ("r33", "1"), ("r206", "0"), ("r206", "1")]


def _fake_run(tmp_path, seeds=(0, 1), knobs=None):
    run = tmp_path / "RUN"
    (run / "cells").mkdir(parents=True)
    (run / "agents").mkdir()
    for d in ("0.00", "0.45"):
        for s in seeds:
            cell = {"drift": float(d), "seed": s, "objective": "survival", "mortal": True,
                    "gae_bootstrap": "successor",
                    "knobs": knobs or {"drift_mode": "l3", "l3_hidden": 8, "l3_seed": 0, "l3_family": "gmotion"},
                    "agents": {a: {"pool": {"target": 0.6, "shuffled": 0.5}} for a in pr.ARMS}}
            (run / "cells" / f"cell_d{d}_s{s}.json").write_text(
                json.dumps({"fingerprint": "fp", "cell": cell}), encoding="utf-8")
            for a in pr.ARMS:
                (run / "agents" / f"agent_d{d}_s{s}_{a}.pt").write_bytes(b"")
    return run


def test_main_resumes_from_the_partial_file(tmp_path, monkeypatch):
    import run_post_review_readouts as rpr
    run = _fake_run(tmp_path)
    out = tmp_path / "out" / "post_review.json"
    monkeypatch.setenv("ITASORL_FOLDS", "explicit")
    ran = []

    def fake_task(fail_seed):
        def run_task(task):
            ran.append((task["readout"], task["seed"]))
            assert task["recorded"]["0.45"][task["seed"]]["survival"]["target"] == 0.6
            if task["seed"] == fail_seed:
                return {"readout": "r33", "seed": task["seed"], "error": "Traceback: boom", "seconds": 0.0}
            cells = [{"agent_drift": float(d), "arm": a, "base": b, "target": 0.7 if a == "survival" else 0.55}
                     for d in task["drifts"] for a in pr.ARMS for b in ("fresh", "registered")]
            return {"readout": "r33", "seed": task["seed"], "result": {"cells": cells, "seconds": 0.0}}
        return run_task

    argv = ["--run-dir", str(run), "--out", str(out), "--readouts", "r33", "--workers", "1"]
    monkeypatch.setattr(rpr, "run_task", fake_task("1"))
    assert rpr.main(argv) == 1
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["complete"] is False and doc["errors"][0]["seed"] == "1"
    assert list(doc["per_seed"]["r33"]) == ["0"]
    ran.clear()
    monkeypatch.setattr(rpr, "run_task", fake_task(None))
    assert rpr.main(argv) == 0
    assert ran == [("r33", "1")]                         # seed 0 came from the partial file
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["complete"] is True and list(doc["per_seed"]["r33"]) == ["0", "1"]
    assert doc["cell_fingerprints"]["d0.45_s1"] == "fp" and doc["fold_scheme"] == "explicit"
    assert doc["summary"]["r33"]["d0.45/fresh"]["arms"]["survival"]["mean"] == pytest.approx(0.7)
    assert set(doc["stack"]) >= {"python", "torch", "numpy", "sklearn"}
    assert doc["integrity"]["checked"] is False          # r210 was not among the readouts
    src = (ROOT / "scripts" / "run_post_review_readouts.py").read_text(encoding="utf-8")
    assert '"git_commit": git_head()' in src             # the readout-script provenance convention
    with pytest.raises(SystemExit, match="different run"):          # another setting, same file
        rpr.main(argv + ["--quick"])


def test_run_config_refuses_runs_outside_the_design(tmp_path):
    import run_post_review_readouts as rpr
    run = _fake_run(tmp_path, knobs={"drift_mode": "l1", "l3_family": None})
    with pytest.raises(SystemExit, match="l3"):
        rpr.run_config(rpr.load_cells(str(run)))


# ---------------------------------------------------------------------------
# Seed bases
# ---------------------------------------------------------------------------
def _literal(node):
    """Plain literal of a constant, tuple or dict node (the shapes these constants use)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    raise AssertionError(f"unexpected node {ast.dump(node)}")


def _script_constant(script: str, name: str):
    tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return _literal(node.value)
    raise AssertionError(f"{name} not found in scripts/{script}")


def test_post_review_bases_miss_every_base_in_use_and_each_other():
    assert pr.base_collisions() == []
    assert all(b >= 2_000_000 for _, b, _ in pr.NEW_BASES)
    assert pr.footprints_overlap(0, 10, 17_001, 10)                   # weather stream of one,
    assert not pr.footprints_overlap(800_000, 110, 850_000, 110)      # world stream of the other


def test_the_in_use_list_carries_every_base_the_runners_define():
    from itasorl.eval_protocols import POLICY_SEED_BASE
    from itasorl.l0_audit import AUDIT_BASES, STANDARD_BASES
    listed = {b for _, b, _ in pr.BASES_IN_USE}
    wanted = set(STANDARD_BASES) | {b for pair in AUDIT_BASES for b in pair}
    for script, name in (("run_l3_h2_ablations.py", "SEED_BASES"), ("run_texture_fresh_probe.py", "BASES"),
                         ("run_l3_crossrecipe.py", "SEED_BASES")):
        wanted |= {b for pair in _script_constant(script, name).values() for b in pair}
    wanted.add(_script_constant("run_mechanism_readouts.py", "WORLD_BASE"))
    assert wanted <= listed, sorted(wanted - listed)
    assert pr.SCRIPTED_POLICY_BASE == POLICY_SEED_BASE
    import run_post_review_readouts as rpr
    sb = rpr.seed_bases()
    assert (sb["fresh_authentic"], sb["fresh_surrogate"]) == (pr.FRESH_AUTH_BASE, pr.FRESH_SURR_BASE)
    assert sb["balanced"] == {"fresh": pr.BALANCED_FRESH_BASE, "registered": 800_000}


# ---------------------------------------------------------------------------
# r205: which half of [mean h, h_T] carries the signal
# ---------------------------------------------------------------------------
def test_feature_halves_are_the_two_blocks_of_episode_features():
    from itasorl.experiment_b import episode_features
    rng = np.random.default_rng(12)
    H = rng.normal(size=(7, 5, 3))
    f = pr.feature_halves(H)
    assert np.array_equal(f["mean"], H.mean(1)) and np.array_equal(f["final"], H[:, -1])
    assert np.array_equal(f["concat"], episode_features(H))


def _mean_only_pools(rng, n=30, T=6, hid=3, shift=1.5):
    """The surrogate pool is shifted on unit 0 at every step but the last, so the time average
    separates the pools and the final state does not."""
    Ha = rng.normal(size=(n, T, hid)).astype(np.float32)
    Hs = rng.normal(size=(n, T, hid)).astype(np.float32)
    Hs[:, :-1, 0] += shift
    return Ha, Hs


def test_r205_cell_scores_each_half_on_the_bit_checked_pools(monkeypatch):
    import itasorl.experiment_b2 as b2
    rng = np.random.default_rng(13)
    Ha, Hs = _mean_only_pools(rng)

    def fake_collect_pool(agent, norm, params, drift, n_eps, steps, device, seed_base, ray_steps, **kw):
        H = Hs if seed_base == 850_000 else Ha
        k = len(H)
        r = np.random.default_rng(seed_base)
        return (H, r.random(k), r.random(k), r.random(k), r.random(k), r.random(k),
                r.random((k, steps, 7)).astype(np.float32))

    monkeypatch.setattr(b2, "collect_pool", fake_collect_pool)
    out, (A, S) = b2.pooled_readout(None, None, None, 0.45, n_eps=30, steps=6, device="cpu",
                                    seed=4, return_pools=True)
    cell = pr.r205_cell(A, S, pool=out, recorded={"target": out["target"], "shuffled": out["shuffled"]}, seed=4)
    assert cell["reproduced"] and cell["shuffled_reproduced"] and cell["perm_check"]
    sets = cell["sets"]
    assert sets["concat"] == {"target": out["target"], "shuffled": out["shuffled"],
                              "selectivity": out["target"] - out["shuffled"]}
    assert sets["mean"]["target"] > 0.9 and sets["final"]["target"] < 0.7
    assert len(sets["mean"]["shuffled_draws"]) == len(sets["final"]["shuffled_draws"]) == pr.N_PERM
    assert sets["final"]["selectivity"] == pytest.approx(sets["final"]["target"] - sets["final"]["shuffled"])
    bad = pr.r205_cell(A, S, pool=out, recorded={"target": out["target"] + 1e-9, "shuffled": out["shuffled"]},
                       seed=4)
    assert bad["reproduced"] is False and "sets" not in bad


def _r205_seed(ok45: bool, e: float) -> dict:
    vals = {"survival": (0.72, 0.60, 0.73), "predictor": (0.55, 0.60, 0.61), "untrained": (0.52, 0.51, 0.53)}
    cells, rep = {}, {}
    for d, ok in (("0.00", True), ("0.45", ok45)):
        cells[d] = {}
        for arm, v in vals.items():
            cells[d][arm] = {"reproduced": ok}
            if ok:
                cells[d][arm]["sets"] = {name: {"target": x + e, "shuffled": 0.5, "selectivity": x + e - 0.5}
                                         for name, x in zip(pr.R205_SETS, v)}
        rep[d] = ok
    return {"cells": cells, "reproduced": rep}


def test_r205_summary_readings_counts_and_exclusion():
    rng = np.random.default_rng(14)
    per = {str(s): _r205_seed(s != 3, e) for s, e in enumerate(rng.normal(0, 0.05, 6))}
    s = pr.summarize_r205(per)["0.45"]
    assert s["seeds_excluded"] == ["3"] and s["seeds_used"] == ["0", "1", "2", "4", "5"]
    assert s["reading"] == {"survival": "MEAN CARRIES", "predictor": "FINAL CARRIES", "untrained": "BOTH"}
    used = s["seeds_used"]
    want = sum(per[k]["cells"]["0.45"]["survival"]["sets"]["concat"]["target"] >= 0.65 for k in used)
    assert s["arms"]["survival"]["concat"]["target"]["n_at_or_above_bar"] == want
    assert s["differences"]["survival"]["mean_minus_final"]["mean"] == pytest.approx(0.12)
    assert s["differences"]["survival"]["concat_minus_mean"]["n"] == 5
    assert pr.summarize_r205(per)["0.00"]["seeds_excluded"] == []
    nan = {"mean": float("nan")}
    assert pr.r205_reading({"concat_minus_mean": nan, "concat_minus_final": {"mean": 0.1}}) == "NA"


def test_r205_task_bit_checks_and_scores_the_halves(monkeypatch):
    import itasorl.experiment_b2 as b2
    import run_post_review_readouts as rpr
    loads = _patch_world(monkeypatch)
    rng = np.random.default_rng(15)

    def fake_pooled(agent, norm, params, drift, **kw):
        Ha, Hs = _mean_only_pools(rng, n=kw["n_eps"], T=kw["steps"])
        return {"target": 0.6 if drift == 0.45 else 0.5, "shuffled": 0.49}, (Ha, Hs)

    monkeypatch.setattr(b2, "pooled_readout", fake_pooled)
    task = _task("r205")
    rec = {"target": 0.6, "shuffled": 0.49}
    task["recorded"] = {"0.45": {"2": {a: rec for a in pr.ARMS}},
                        "0.00": {"2": {a: ({"target": 0.5, "shuffled": 0.49} if a == "untrained" else rec)
                                       for a in pr.ARMS}}}       # only the untrained arm reproduces at drift 0
    res = rpr.run_task(task)
    assert "error" not in res, res.get("error")
    r = res["result"]
    assert r["reproduced"] == {"0.00": False, "0.45": True} and len(loads) == 6
    assert set(r["cells"]["0.45"]["survival"]["sets"]) == set(pr.R205_SETS)
    assert "sets" in r["cells"]["0.00"]["untrained"] and "sets" not in r["cells"]["0.00"]["survival"]
    integ = rpr.integrity({"r205": {"2": r}}, ["0.00", "0.45"])
    assert integ["checked"] and integ["checked_by"] == "r205" and integ["unreproduced"]["0.00"] == ["2"]
    assert rpr.integrity({"r210": {"2": r}, "r205": {}}, ["0.00", "0.45"])["checked_by"] == "r210"


def test_r205_is_selectable_and_scheduled_after_r210():
    import run_post_review_readouts as rpr
    assert "r205" in pr.READOUTS and "r205" in rpr.TASKS
    assert rpr.plan_tasks(["r205", "r210", "r31"], [0], {}) == [("r31", "0"), ("r210", "0"), ("r205", "0")]
