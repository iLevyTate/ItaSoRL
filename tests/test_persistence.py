"""Revision step 7: the controlled persistence conditions."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402
from itasorl.persistence import collect_persistence, persistence_readout  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent():
    w = make_world(P, 0.0, RS)
    torch.manual_seed(0)
    a = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return a, RunningNorm(w.obs_spec.size).freeze()


def test_at_L0_every_condition_is_exactly_chance():
    """Both prefixes authentic: the two branches are identical, so no condition can decode
    a prefix label. The design adds no signal of its own."""
    a, n = _agent()
    r = persistence_readout(a, n, P, 0.0, n_pairs=10, prefix_steps=4, tail_steps=5, ray_steps=RS)
    for key in ("replay:prefix", "common_state:prefix", "reset_hidden:prefix",
                "factorial:hidden", "factorial:physical"):
        assert r[key]["n_pairs"] == 10, key
        assert r[key]["window"] == pytest.approx(0.5, abs=1e-12), key
        assert all(x == pytest.approx(0.5, abs=1e-12) for x in r[key]["auc_by_t"]), key


def test_replay_members_read_identical_inputs_and_factorial_labels_are_balanced(monkeypatch):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "DRIFT_MODE", "regime")
    a, n = _agent()
    d = collect_persistence(a, n, P, 0.5, n_pairs=6, prefix_steps=4, tail_steps=5, ray_steps=RS)
    rep = d["replay"]
    assert len(rep["groups"]) == 2 * len(np.unique(rep["groups"]))
    assert not np.allclose(rep["H"][0::2], rep["H"][1::2])     # different prefix states differ
    # the prefix difference fades under identical input: the last step is closer than the first
    gap = np.abs(rep["H"][0::2] - rep["H"][1::2]).mean(axis=(0, 2))
    assert gap[-1] <= gap[0] + 1e-6
    fac = d["factorial"]
    yh, yp = fac["labels"]["hidden"], fac["labels"]["physical"]
    assert len(yh) == 4 * len(np.unique(fac["groups"]))
    assert abs(np.corrcoef(yh, yp)[0, 1]) < 1e-12              # the two labels are orthogonal
    assert set(d["reset_hidden"]["labels"]["prefix"]) == {0, 1}
