"""Budget snapshots (revision steps 4 and 11). The headline survival arm of an extended run
must be exactly the agent a protocol-budget run returns, so a budget curve can share one
training run without changing the principal comparison."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import run_expB2  # noqa: E402

from itasorl.experiment_b2 import train_actor_critic  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
KW = dict(n_eps=3, hidden=8, embed=16, max_steps=12, ray_steps=4, seed=1, device="cpu",
          shaping_coef=1.0)


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_snapshot_equals_a_run_of_that_budget():
    stats: dict = {}
    full, full_norm, hist, snaps = train_actor_critic(0.45, P, updates=5, snapshot_at=(3,),
                                                      stats=stats, **KW)
    short, short_norm, short_hist = train_actor_critic(0.45, P, updates=3, **KW)
    snap, snap_norm = snaps[3]
    assert _same(snap, short)
    assert np.array_equal(snap_norm.mean, short_norm.mean)
    assert np.array_equal(snap_norm.var, short_norm.var) and snap_norm.count == short_norm.count
    assert snap_norm.frozen
    assert hist[:3] == short_hist
    assert not _same(full, short)                      # training really continued
    assert len(stats["env_steps"]) == 5 and stats["env_steps"] == sorted(stats["env_steps"])


def test_snapshot_must_lie_inside_the_run():
    with pytest.raises(ValueError):
        train_actor_critic(0.45, P, updates=3, snapshot_at=(3,), **KW)


def test_budget_plan_keeps_the_protocol_budget_as_headline():
    assert run_expB2.budget_plan({"updates": 300}) == (300, 300, [])
    assert run_expB2.budget_plan({"updates": 300, "budget_extend": 450,
                                  "budget_snapshots": [100, 200]}) == (300, 450, [100, 200, 450])
    assert run_expB2.budget_plan({"updates": 300, "survival_updates": 450,
                                  "budget_snapshots": [300]}) == (450, 450, [300])
    with pytest.raises(SystemExit):
        run_expB2.budget_plan({"updates": 300, "budget_snapshots": [500]})


def test_default_budget_flags_leave_the_fingerprint_unchanged():
    base = {"updates": 300, "hidden": 96, "drift_mode": "l3"}
    fp = run_expB2.config_fingerprint(base)
    assert run_expB2.config_fingerprint({**base, "budget_extend": None, "budget_snapshots": []}) == fp
    assert run_expB2.config_fingerprint({**base, "budget_extend": 450}) != fp
    assert run_expB2.config_fingerprint({**base, "budget_snapshots": [100]}) != fp
