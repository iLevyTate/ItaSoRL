"""itasorl/draw_crossed_ext.py and scripts/run_draw_crossed_extension.py: the reading net of
its control, its label, the three sets of draws and the frozen constants of
docs/specs/2026-11-16-draw-crossed-extension-design.md, on synthetic values only (plus the
committed summary of the first readout, from which the number of draws was fixed)."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from itasorl import draw_crossed as dc  # noqa: E402
from itasorl import draw_crossed_ext as ext  # noqa: E402

REG = [800000, 850000]


def _cells(bases, level=0.75, control=0.52, n_seeds=4, seed=3, with_registered=True):
    """Synthetic C1 cells in the artifact's layout, own walks only."""
    rng = np.random.default_rng(seed)
    cells = []
    for drift in (0.0, 0.45):
        for s in range(n_seeds):
            draws = []
            for k, b in enumerate(([REG] if with_registered else []) + [list(x) for x in bases]):
                readings = {}
                for i, fs in enumerate(ext.FEATURE_SETS):
                    base = control if drift == 0.0 else level - 0.08 * i
                    v = base + 0.01 * s + 0.004 * k + rng.normal(0, 0.004)
                    readings[fs] = {"c1": v, "nested": v - 0.005, "n_auth": 110, "n_surr": 110,
                                    "chosen_C": [1.0] * 5, "convergence_warnings": 0}
                draws.append({"bases": b, "readings": readings})
            cells.append({"run": "C1", "drift": drift, "seed": s, "agent_sha256": {}, "draws": draws})
    return cells


@pytest.mark.parametrize("lo, hi, want", [
    (0.15, 0.30, "at least the bar's distance from chance"),
    (0.20, 0.30, "at least the bar's distance from chance"),
    (0.149, 0.30, "above its control"),
    (0.001, 0.02, "above its control"),
    (0.0, 0.30, "not shown above its control"),
    (-0.02, 0.01, "not shown above its control"),
])
def test_label_net(lo, hi, want):
    assert ext.label_net({"computable": True, "lo": lo, "hi": hi}) == want


def test_label_net_without_an_interval():
    assert ext.label_net({"computable": False, "lo": None, "hi": None}) == dc.NOT_COMPUTABLE
    assert ext.NET_FLOOR == pytest.approx(dc.BAR - 0.5) == pytest.approx(0.15)


def test_net_of_control_is_the_paired_difference_of_the_two_drifts():
    bases = [(6800000 + 100000 * k, 6850000 + 100000 * k) for k in range(5)]
    cells = _cells(bases)
    s = ext.summarize(cells, [list(b) for b in bases], REG, ok=True)
    for probe in ext.PROBES:
        for arm in ext.ARMS:
            fs = f"own/{arm}"
            hi = np.asarray(dc.matrix(cells, "C1", 0.45, fs, probe, bases))
            lo = np.asarray(dc.matrix(cells, "C1", 0.0, fs, probe, bases))
            want = dc.crossed_interval(hi - lo)
            got = s[probe]["net_of_control"][ext.net_name(arm)]
            assert got["mean"] == pytest.approx(want["mean"])
            assert got["lo"] == pytest.approx(want["lo"]) and got["hi"] == pytest.approx(want["hi"])
            assert got["mean"] == pytest.approx(
                s[probe]["readings"]["0.45"][fs]["mean"] - s[probe]["readings"]["0.00"][fs]["mean"])
            # only the survival arm's net reading carries a label
            assert (got["label"] is not None) == (arm == "survival")


def test_labels_are_the_first_readouts_and_baselines_carry_none_at_the_surrogate_drift():
    bases = [(6800000 + 100000 * k, 6850000 + 100000 * k) for k in range(5)]
    s = ext.summarize(_cells(bases), [list(b) for b in bases], REG, ok=True)["nested"]
    assert s["readings"]["0.45"]["own/survival"]["label"] == dc.label_level(s["readings"]["0.45"]["own/survival"])
    assert s["readings"]["0.45"]["own/predictor"]["label"] is None
    assert s["readings"]["0.45"]["own/untrained"]["label"] is None
    for fs in ext.FEATURE_SETS:
        assert s["readings"]["0.00"][fs]["label"] == dc.label_band(s["readings"]["0.00"][fs])
    assert [name for name, _p, _m in ext.LEADS] == ["survival minus predictor, own walks",
                                                   "survival minus untrained, own walks"]
    for name, _p, _m in ext.LEADS:
        assert s["leads"][name]["label"] == dc.label_margin(s["leads"][name])
    assert s["registered_draw"]["0.45"]["own/survival"]["per_seed"]


def test_a_failed_integrity_gate_voids_every_label_and_unlabelled_sets_carry_none():
    bases = [(6800000 + 100000 * k, 6850000 + 100000 * k) for k in range(4)]
    cells = _cells(bases)
    void = ext.summarize(cells, [list(b) for b in bases], REG, ok=False)["nested"]
    assert void["readings"]["0.00"]["own/survival"]["label"] == dc.VOID
    assert void["readings"]["0.45"]["own/survival"]["label"] == dc.VOID
    assert void["net_of_control"][ext.net_name("survival")]["label"] == dc.VOID
    assert void["readings"]["0.45"]["own/predictor"]["label"] is None
    plain = ext.summarize(cells, [list(b) for b in bases], REG, ok=True, labels=False)["nested"]
    assert all(r["label"] is None for d in plain["readings"].values() for r in d.values())
    assert all(r["label"] is None for r in plain["leads"].values())
    assert all(r["label"] is None for r in plain["net_of_control"].values())


def test_the_three_sets_of_draws():
    first_bases = [(6000000 + 100000 * k, 6050000 + 100000 * k) for k in range(3)]
    further_bases = [(6000000 + 100000 * k, 6050000 + 100000 * k) for k in range(3, 9)]
    first = _cells(first_bases, seed=11)
    cells = _cells(further_bases, seed=12)
    out = ext.summarize_all(cells, first, [list(b) for b in further_bases],
                            [list(b) for b in first_bases], REG, ok=True)
    assert set(out) == {"further", "first_eight", "all_fresh"}
    net = ext.net_name("survival")
    # the further draws carry every label; the other two sets carry none
    assert out["further"]["nested"]["net_of_control"][net]["label"] is not None
    assert set(out["first_eight"]["nested"]) == {"net_of_control"}
    assert out["first_eight"]["nested"]["net_of_control"][net]["label"] is None
    assert out["all_fresh"]["nested"]["readings"]["0.00"]["own/survival"]["label"] is None
    assert "registered_draw" not in out["all_fresh"]["nested"]
    # sizes of the three matrices
    assert out["further"]["nested"]["readings"]["0.00"]["own/survival"]["n_draws"] == 6
    assert out["first_eight"]["nested"]["net_of_control"][net]["n_draws"] == 3
    pooled = out["all_fresh"]["nested"]["readings"]["0.00"]["own/survival"]
    assert pooled["n_draws"] == 9 and pooled["n_seeds"] == 4
    # the pooled mean is the mean over all nine draws, in either order of the two sets
    both = np.concatenate([np.asarray(dc.matrix(first, "C1", 0.0, "own/survival", "nested", first_bases)),
                           np.asarray(dc.matrix(cells, "C1", 0.0, "own/survival", "nested", further_bases))],
                          axis=1)
    assert pooled["mean"] == pytest.approx(both.mean())


def test_first_cells_keeps_the_own_walks_of_c1_and_rescales_nothing():
    art = {"cells": [
        {"run": "C1", "drift": 0.0, "seed": 0, "draws": [
            {"bases": [6000000, 6050000],
             "readings": {fs: {"c1": 0.5, "nested": 0.51} for fs in dc.FEATURE_SETS}}]},
        {"run": "C2", "drift": 0.0, "seed": 0, "draws": []}]}
    got = ext.first_cells(art)
    assert len(got) == 1 and got[0]["run"] == "C1"
    assert set(got[0]["draws"][0]["readings"]) == set(ext.FEATURE_SETS)
    assert got[0]["draws"][0]["readings"]["own/survival"] == {"c1": 0.5, "nested": 0.51}


def test_run_script_constants_match_the_frozen_design():
    import run_draw_crossed_extension as r
    import run_draw_crossed_readout as first
    assert r.REGISTERED_BASES == (800_000, 850_000)
    assert r.FIRST_BASES == first.FRESH_BASES
    assert r.FURTHER_BASES == tuple((6_000_000 + 100_000 * k, 6_050_000 + 100_000 * k)
                                    for k in range(8, 64))
    assert len(r.FURTHER_BASES) == 56
    used = {b for pair in r.FURTHER_BASES for b in pair}
    assert len(used) == 112 and min(used) == 6_800_000
    assert not used & {b for pair in r.FIRST_BASES for b in pair}
    assert not used & set(r.REGISTERED_BASES)
    assert (r.N_EPS, r.STEPS, r.RAY_STEPS) == (110, 24, 5)
    assert r.SEEDS == tuple(range(10)) and r.DRIFTS == (0.0, 0.45)
    assert r.ARMS == ("survival", "predictor", "untrained")
    assert ext.FEATURE_SETS == ("own/survival", "own/predictor", "own/untrained")
    assert ext.PROBES[0] == "nested"                                # the primary probe
    assert r.SPEC == "docs/specs/2026-11-16-draw-crossed-extension-design.md"
    assert os.path.exists(os.path.join(ROOT, r.SPEC))


def test_fifty_six_is_the_number_the_first_readout_fixes():
    """The design fixes the number of further draws from FINDINGS 25 alone: the smallest number
    for which the draw component of the standard error at drift zero is no larger than the seed
    component of ten seeds."""
    with open(os.path.join(ROOT, "artifacts", "draw_crossed", "corrected_c1_c2.json"),
              encoding="utf-8") as fh:
        first = json.load(fh)
    r = first["summary"]["C1"]["0.00"]["nested"]["readings"]["own/survival"]
    sd_draw, sd_seed, n_seeds = round(r["sd_draw_means"], 4), round(r["sd_seed_means"], 4), r["n_seeds"]
    assert (sd_draw, sd_seed, n_seeds) == (0.0524, 0.0222, 10)
    seed_part = sd_seed / math.sqrt(n_seeds)
    smallest = next(d for d in range(2, 500) if sd_draw / math.sqrt(d) <= seed_part)
    assert smallest == 56
