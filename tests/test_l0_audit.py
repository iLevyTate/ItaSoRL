"""Revision step 5: versioned folds, pair-preserving readouts, and the L0 audit tools."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl import folds  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, collect_pool, make_world  # noqa: E402
from itasorl.l0_audit import paired_pooled_readout, pre_intervention_probe  # noqa: E402
from itasorl.stats import auroc_ci, cluster_auroc_ci  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _agent_norm():
    w = make_world(P, 0.0, RS)
    torch.manual_seed(0)
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


def test_serialized_partitions_match_the_versioned_generator():
    with open(os.path.join(ROOT, "artifacts", "folds", "explicit_v1.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored["version"] == folds.EXPLICIT_VERSION
    assert stored == json.loads(json.dumps(folds.standard_partitions()))


def test_pairs_stay_in_one_fold_and_the_digest_identifies_the_partition():
    g = np.r_[np.arange(60), np.arange(60)]
    y = np.r_[np.zeros(60), np.ones(60)]
    rec = folds.partition_record(g, y, scheme="explicit")
    assert rec["groups_intact"] and rec["version"] == "explicit-v1"
    other = folds.partition_record(g, y, n_splits=4, scheme="explicit")
    assert other["sha256"] != rec["sha256"]
    assert folds.partition_record(g, y, scheme="explicit")["sha256"] == rec["sha256"]


def test_cluster_bootstrap_resamples_whole_pairs():
    rng = np.random.default_rng(0)
    n = 40
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    s = np.r_[rng.normal(0, 1, n), rng.normal(1, 1, n)]
    lo, hi = cluster_auroc_ci(y, s, np.r_[np.arange(n), np.arange(n)], seed=0)
    assert 0.5 < lo < hi < 1.0
    lo1, hi1 = cluster_auroc_ci(y, s, np.arange(2 * n), seed=0)
    alo, ahi = auroc_ci(y, s, seed=0)
    assert abs(lo1 - alo) < 0.05 and abs(hi1 - ahi) < 0.05     # singletons ~ the plain bootstrap


def test_collect_pool_reports_the_kept_episode_indices(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "SURVIVAL_METAB", {"E0": 0.2, "basal_E": 0.4, "Hyd0": 8.0, "basal_Hyd": 0.005})
    agent, norm = _agent_norm()
    H, spd, idx = collect_pool(agent, norm, P, 0.45, 10, 12, "cpu", 999, RS, return_index=True)
    assert len(idx) == H.shape[0] and np.all(np.diff(idx) > 0) and idx.max() < 10


def test_paired_readout_is_exactly_chance_at_L0():
    """Same seeds, no surrogate: the twins are identical, so the target is 0.5 by
    construction and the first states match. A balanced design adds no signal of its own."""
    agent, norm = _agent_norm()
    r = paired_pooled_readout(agent, norm, P, 0.0, n_eps=12, steps=5, ray_steps=RS)
    assert r["n_pairs"] == 12 and r["identical_first_state"] is True
    assert r["target"] == pytest.approx(0.5, abs=1e-12)
    assert r["partition"]["groups_intact"]


def test_paired_readout_twins_share_the_initial_state_under_drift(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "DRIFT_MODE", "regime")
    agent, norm = _agent_norm()
    r = paired_pooled_readout(agent, norm, P, 0.5, n_eps=12, steps=5, ray_steps=RS)
    assert r["identical_first_state"] is True and np.isfinite(r["target"])


def test_pre_intervention_probe_on_one_sample_is_chance():
    r = pre_intervention_probe(P, bases=(5000, 5000), n=20, ray_steps=RS)
    assert r["target"] == pytest.approx(0.5, abs=1e-12)
    r2 = pre_intervention_probe(P, bases=(5000, 7000), n=20, ray_steps=RS)
    assert np.isfinite(r2["target"]) and r2["partition"]["n_samples"] == 40
