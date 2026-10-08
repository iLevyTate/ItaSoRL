"""Drift-0.45 world-sample sensitivity (spec docs/specs/2026-10-08-drift-045-...-design.md).

The decision rule and the prohibition on folding are frozen in that spec before the run, so
these tests pin them rather than the other way round.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl import l0_audit  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402
from itasorl.l0_audit import (  # noqa: E402
    AUDIT_BASES, l0_world_samples, world_sample_scan, world_sample_summary,
)
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4
BAR = 0.65


def _agent_norm():
    w = make_world(P, 0.0, RS)
    torch.manual_seed(0)
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size,
                                 embed=16, hidden=8).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


# --------------------------------------------------------------- the frozen decision rule

def test_secure_when_every_draw_clears_the_bar_and_the_registered_draw_is_not_favourable():
    independent = [0.70, 0.72, 0.68, 0.75, 0.80, 0.66, 0.69, 0.74]
    s = world_sample_summary(independent, 0.67, bar=BAR)
    assert s["verdict"] == "SECURE"
    assert s["registered_rank"] >= 5


def test_draw_dependent_when_any_single_draw_falls_below_the_bar():
    independent = [0.70, 0.72, 0.68, 0.75, 0.80, 0.62, 0.69, 0.74]
    s = world_sample_summary(independent, 0.67, bar=BAR)
    assert s["verdict"] == "DRAW-DEPENDENT"
    assert s["min"] == pytest.approx(0.62)


def test_indeterminate_when_all_clear_but_the_registered_draw_sits_above_the_median():
    independent = [0.70, 0.72, 0.68, 0.75, 0.66, 0.67, 0.69, 0.74]
    s = world_sample_summary(independent, 0.90, bar=BAR)
    assert s["verdict"] == "INDETERMINATE"
    assert s["registered_rank"] == 1


def test_the_three_verdicts_are_exhaustive_and_disjoint_over_a_grid():
    seen = set()
    rng = np.random.default_rng(0)
    for _ in range(300):
        independent = list(rng.uniform(0.55, 0.85, size=8))
        reg = float(rng.uniform(0.55, 0.85))
        v = world_sample_summary(independent, reg, bar=BAR)["verdict"]
        assert v in {"SECURE", "DRAW-DEPENDENT", "INDETERMINATE"}
        seen.add(v)
    assert seen == {"SECURE", "DRAW-DEPENDENT", "INDETERMINATE"}


def test_a_low_draw_is_never_rescued_by_folding_it_about_chance():
    """Folding is conservative at drift 0 and anti-conservative here; the spec forbids it.

    0.30 folds to |0.30 - 0.5| + 0.5 = 0.70, which would clear the bar. It must not.
    """
    independent = [0.70, 0.72, 0.68, 0.75, 0.80, 0.30, 0.69, 0.74]
    s = world_sample_summary(independent, 0.67, bar=BAR)
    assert s["verdict"] == "DRAW-DEPENDENT"
    assert s["min"] == pytest.approx(0.30)
    assert s["n_at_or_above_bar"] == 7


def test_the_registered_rank_counts_all_nine_draws():
    independent = [0.60, 0.61, 0.62, 0.63, 0.64, 0.66, 0.67, 0.68]
    assert world_sample_summary(independent, 0.99, bar=BAR)["registered_rank"] == 1
    assert world_sample_summary(independent, 0.01, bar=BAR)["registered_rank"] == 9
    assert world_sample_summary(independent, 0.655, bar=BAR)["n_draws"] == 8


def test_the_spread_is_the_sample_sd_of_the_independent_draws_only():
    independent = [0.70, 0.72, 0.68, 0.75, 0.80, 0.66, 0.69, 0.74]
    s = world_sample_summary(independent, 0.99, bar=BAR)
    assert s["between_draw_sd"] == pytest.approx(float(np.std(independent, ddof=1)))
    lo, hi = s["t90_over_draws"]
    assert lo < float(np.mean(independent)) < hi


# --------------------------------------------------------------- the scan, and the drift

def test_the_scan_passes_the_drift_to_the_pooled_readout_by_keyword(monkeypatch):
    """The drift-plumbing guard: this fails if the scan hard-codes 0.0 as l0_world_samples does."""
    seen = []

    def recorder(agent, norm, params, *args, **kwargs):
        seen.append((args, kwargs.get("drift_sigma")))
        return {"target": 0.5, "n": 4}

    monkeypatch.setattr(l0_audit, "pooled_readout", recorder)
    agent, norm = _agent_norm()
    world_sample_scan(agent, norm, P, 0.45, bases=AUDIT_BASES[:2], n_eps=2, steps=2,
                      ray_steps=RS)
    assert len(seen) == 2
    for positional, drift_kw in seen:
        assert positional == (), "drift must be passed by keyword, not positionally"
        assert drift_kw == 0.45


def test_the_scan_reports_the_bases_it_scored_and_omits_the_first_state_probe(monkeypatch):
    monkeypatch.setattr(l0_audit, "pooled_readout",
                        lambda *a, **k: {"target": 0.6, "n": 4})
    agent, norm = _agent_norm()
    rows = world_sample_scan(agent, norm, P, 0.45, bases=AUDIT_BASES[:3], n_eps=2, steps=2,
                             ray_steps=RS)
    assert [r["bases"] for r in rows] == [list(b) for b in AUDIT_BASES[:3]]
    assert all("first_state_target" not in r for r in rows)


def test_l0_world_samples_still_scores_drift_zero_with_its_recorded_key_order(monkeypatch):
    """The refactor guard: the committed drift-0 artifact must stay regenerable."""
    calls = []
    monkeypatch.setattr(l0_audit, "pooled_readout",
                        lambda *a, **k: calls.append(k.get("drift_sigma")) or
                        {"target": 0.52, "n": 4})
    monkeypatch.setattr(l0_audit, "_first_state_target", lambda *a, **k: 0.47)
    agent, norm = _agent_norm()
    rows = l0_world_samples(agent, norm, P, bases=AUDIT_BASES[:2], n_eps=2, steps=2,
                            ray_steps=RS)
    assert calls == [0.0, 0.0], "l0_world_samples must stay a drift-zero readout"
    assert list(rows[0]) == ["bases", "target", "n", "first_state_target"]


@pytest.mark.slow
def test_the_scan_runs_end_to_end_on_a_tiny_agent():
    agent, norm = _agent_norm()
    rows = world_sample_scan(agent, norm, P, 0.45, bases=AUDIT_BASES[:1], n_eps=6, steps=4,
                             ray_steps=RS)
    assert len(rows) == 1
    assert 0.0 <= rows[0]["target"] <= 1.0 and rows[0]["n"] == 12
