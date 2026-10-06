"""Revision step 12: the per-individual evolutionary readout and the value of world information."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402
from itasorl.experiment_c import (individual_probe_panel, lineage_summary,  # noqa: E402
                                  value_of_world_information)
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_individual_panel_scores_each_policy_and_reports_the_pooled_value():
    w = make_world(P, 0.0, 4)
    pop = []
    for s in range(3):
        torch.manual_seed(s)
        pop.append(RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=8, hidden=8))
    r = individual_probe_panel(pop, drift_sigma=0.0, n_pairs=10, prefix_steps=4, tail_steps=5,
                               seed_base=930_000, params=P, ray_steps=4,
                               norm=RunningNorm(w.obs_spec.size).freeze())
    assert r["n_individuals"] == 3 and len(r["per_individual"]) == 3
    assert all(x == pytest.approx(0.5, abs=1e-12) for x in r["per_individual"])   # L0: identical branches
    assert r["pooled_probe_same_tails"] == pytest.approx(0.5, abs=1e-12)


def test_lineage_summary_uses_lineages_as_units():
    panels = [{"mean": m, "share_at_or_above_bar": sh, "pooled_probe_same_tails": pp}
              for m, sh, pp in ((0.7, 0.8, 0.5), (0.6, 0.4, 0.5), (0.65, 0.5, 0.52))]
    s = lineage_summary(panels)
    assert s["n_lineages"] == 3 and s["mean_individual_auroc"] == pytest.approx(0.65)
    assert s["pooled_minus_individual"] == pytest.approx((0.5 - 0.7 + 0.5 - 0.6 + 0.52 - 0.65) / 3)


def test_value_of_world_information():
    # authentic-trained policy is best in A, surrogate-trained best in S: information helps
    v = value_of_world_information({"0.00": 1.0, "0.45": 0.0}, {"0.00": 0.0, "0.45": 1.0})
    assert v["value_of_information"] == pytest.approx(0.5)
    # one policy dominates in both worlds: knowing the world buys nothing
    v = value_of_world_information({"0.00": 1.0, "0.45": 1.0}, {"0.00": 0.0, "0.45": 0.5})
    assert v["value_of_information"] == pytest.approx(-0.25)
    assert np.isfinite(v["matched_return"])
