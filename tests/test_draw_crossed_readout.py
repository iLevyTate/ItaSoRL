"""itasorl/draw_crossed.py: the crossed interval, the pre-stated labels and the integrity gate
of docs/specs/2026-11-12-draw-crossed-readout-design.md, on synthetic values only."""

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


def _additive(seed_eff, draw_eff, mean=0.7, noise=None):
    a = np.asarray(seed_eff, float)[:, None]
    b = np.asarray(draw_eff, float)[None, :]
    M = mean + a + b
    return M if noise is None else M + noise


def test_crossed_interval_matches_the_closed_form():
    rng = np.random.default_rng(0)
    S, D = 10, 8
    M = _additive(rng.normal(0, 0.05, S), rng.normal(0, 0.04, D),
                  noise=rng.normal(0, 0.03, (S, D)))
    ci = dc.crossed_interval(M)
    g = M.mean()
    a, b = M.mean(1), M.mean(0)
    ms_s = D * ((a - g) ** 2).sum() / (S - 1)
    ms_d = S * ((b - g) ** 2).sum() / (D - 1)
    r = M - a[:, None] - b[None, :] + g
    ms_e = (r ** 2).sum() / ((S - 1) * (D - 1))
    se = math.sqrt((ms_s + ms_d - ms_e) / (S * D))
    assert ci["computable"]
    assert ci["mean"] == pytest.approx(g)
    assert ci["se"] == pytest.approx(se)
    assert ci["lo"] < g < ci["hi"]
    assert ci["hi"] - g == pytest.approx(g - ci["lo"])
    assert ci["n_seeds"] == S and ci["n_draws"] == D
    assert ci["sd_draw_means"] == pytest.approx(b.std(ddof=1))
    assert ci["sd_seed_means"] == pytest.approx(a.std(ddof=1))


def test_crossed_interval_is_wider_than_the_seed_only_interval_when_draws_vary():
    """The point of the design: a draw component that every seed shares widens the interval,
    while the interval over seed means on one draw does not see it."""
    from scipy import stats
    rng = np.random.default_rng(1)
    S, D = 10, 8
    M = _additive(rng.normal(0, 0.01, S), rng.normal(0, 0.05, D),
                  noise=rng.normal(0, 0.005, (S, D)))
    ci = dc.crossed_interval(M)
    one_draw = M[:, 0]
    half_seed = stats.t.ppf(0.95, S - 1) * one_draw.std(ddof=1) / math.sqrt(S)
    assert ci["computable"]
    assert (ci["hi"] - ci["lo"]) / 2 > 2 * half_seed


def test_no_interval_is_returned_when_it_cannot_be_computed():
    # A pure seed-by-draw interaction: every seed mean and draw mean is equal, so the
    # variance estimate MS_seed + MS_draw - MS_resid is negative.
    checker = 0.7 + 0.01 * np.array([[(-1) ** (s + k) for k in range(8)] for s in range(10)])
    assert not dc.crossed_interval(checker)["computable"]
    bad = np.full((10, 8), 0.7)
    bad[3, 2] = float("nan")
    ci = dc.crossed_interval(bad)
    assert not ci["computable"] and ci["lo"] is None and ci["hi"] is None
    assert not dc.crossed_interval([[0.7, 0.8]])["computable"]      # one seed
    for label in (dc.label_level, dc.label_band, dc.label_margin):
        assert label(ci) == dc.NOT_COMPUTABLE                       # never a favorable label


@pytest.mark.parametrize("lo,hi,want", [
    (0.66, 0.80, "above the bar"), (0.65, 0.80, "above the bar"),
    (0.50, 0.649, "below the bar"), (0.60, 0.70, "straddles the bar"),
    (0.649, 0.65, "straddles the bar"),
])
def test_label_level(lo, hi, want):
    assert dc.label_level({"computable": True, "lo": lo, "hi": hi}) == want


@pytest.mark.parametrize("lo,hi,want", [
    (0.46, 0.54, "inside the band"), (0.45, 0.55, "inside the band"),
    (0.56, 0.60, "outside the band"), (0.40, 0.449, "outside the band"),
    (0.44, 0.50, "straddles the band edge"), (0.50, 0.56, "straddles the band edge"),
])
def test_label_band(lo, hi, want):
    assert dc.label_band({"computable": True, "lo": lo, "hi": hi}) == want


@pytest.mark.parametrize("lo,hi,want", [
    (-0.02, 0.03, "within the margin"), (0.05, 0.20, "above the margin"),
    (0.06, 0.20, "above the margin"), (-0.20, 0.04, "below the margin"),
    (-0.20, -0.06, "below the margin"), (0.02, 0.09, "straddles the margin"),
    (-0.05, 0.03, "below the margin"), (-0.01, 0.05, "straddles the margin"),
])
def test_label_margin(lo, hi, want):
    assert dc.label_margin({"computable": True, "lo": lo, "hi": hi}) == want


def _cells(runs=("C1", "C2"), value=None):
    """Synthetic cells in the artifact's layout: 3 seeds, the registered pair and 3 fresh pairs."""
    reg, fresh = [800000, 850000], [[6000000, 6050000], [6100000, 6150000], [6200000, 6250000]]
    rng = np.random.default_rng(7)
    cells = []
    for run in runs:
        for drift in (0.0, 0.45):
            for seed in range(3):
                draws = []
                for k, bases in enumerate([reg] + fresh):
                    readings = {}
                    for i, fs in enumerate(dc.FEATURE_SETS):
                        base = (0.5 if drift == 0.0 else 0.75 - 0.03 * i) if value is None else value
                        v = base + 0.01 * seed + 0.02 * k + rng.normal(0, 0.005)
                        readings[fs] = {"c1": v, "nested": v - 0.01, "n_auth": 110, "n_surr": 110,
                                        "chosen_C": [1.0] * 5, "convergence_warnings": 0}
                    draws.append({"bases": bases, "readings": readings})
                cells.append({"run": run, "drift": drift, "seed": seed, "agent_sha256": {},
                              "draws": draws})
    return cells, reg, fresh


def _reference(cells, reg, shift=0.0):
    ref = {}
    for c in cells:
        dkey = f"{c['drift']:.2f}"
        d = next(d for d in c["draws"] if d["bases"] == reg)
        for arm in ("survival", "predictor", "untrained"):
            lst = ref.setdefault(c["run"], {}).setdefault(dkey, {}).setdefault(arm, [None] * 3)
            lst[c["seed"]] = d["readings"][f"own/{arm}"]["c1"] + shift
    return ref


def test_matrix_orders_seeds_and_draws_and_marks_missing_cells():
    cells, _reg, fresh = _cells(runs=("C1",))
    M = dc.matrix(cells, "C1", 0.45, "own/survival", "c1", fresh)
    assert np.asarray(M).shape == (3, 3)
    by_seed = {c["seed"]: c for c in cells if c["drift"] == 0.45}
    assert M[2][1] == by_seed[2]["draws"][2]["readings"]["own/survival"]["c1"]
    missing = dc.matrix(cells, "C1", 0.45, "own/survival", "c1", fresh + [[1, 2]])
    assert all(math.isnan(row[-1]) for row in missing)


def test_integrity_gate_passes_only_when_every_agent_reproduces():
    cells, reg, _fresh = _cells()
    gate = dc.integrity_gate(cells, _reference(cells, reg), reg)
    assert gate["C1"]["pass"] and gate["C2"]["pass"]
    assert gate["C1"]["n_checked"] == gate["C1"]["n_expected"] == 18
    off = dc.integrity_gate(cells, _reference(cells, reg, shift=0.02), reg)
    assert not off["C1"]["pass"] and off["C1"]["worst_abs_dev"] == pytest.approx(0.02)
    # A cell without its registered draw fails the gate: a missing check is never a pass.
    cut = json.loads(json.dumps(cells))
    cut[0]["draws"] = [d for d in cut[0]["draws"] if d["bases"] != reg]
    partial = dc.integrity_gate(cut, _reference(cells, reg), reg)
    assert not partial[cut[0]["run"]]["pass"] and partial[cut[0]["run"]]["worst_abs_dev"] is None


def test_summary_labels_and_void_runs():
    cells, reg, fresh = _cells()
    gate = dc.integrity_gate(cells, _reference(cells, reg), reg)
    s = dc.summarize(cells, ("C1", "C2"), (0.0, 0.45), fresh, reg, gate)
    blk = s["C1"]["0.45"]["nested"]
    assert blk["readings"]["own/survival"]["label"] in (
        "above the bar", "below the bar", "straddles the bar")
    assert blk["readings"]["own/predictor"]["label"] is None       # baselines carry no label
    assert s["C1"]["0.00"]["nested"]["readings"]["own/untrained"]["label"] in (
        "inside the band", "outside the band", "straddles the band edge")
    assert set(blk["contrasts"]) == {name for name, _, _ in dc.CONTRASTS}
    assert all(c["label"] is None for c in s["C1"]["0.00"]["nested"]["contrasts"].values())
    # The registered draw is reported beside the fresh draws and enters no crossed interval.
    assert blk["readings"]["own/survival"]["n_draws"] == len(fresh)
    assert len(blk["registered_draw"]["own/survival"]["per_seed"]) == 3

    bad = dc.integrity_gate(cells, _reference(cells, reg, shift=0.5), reg)
    v = dc.summarize(cells, ("C1", "C2"), (0.0, 0.45), fresh, reg, bad)
    assert v["C1"]["0.45"]["nested"]["readings"]["own/survival"]["label"] == dc.VOID
    assert v["C1"]["0.00"]["c1"]["readings"]["own/survival"]["label"] == dc.VOID
    between = dc.summarize_between_runs(cells, "C1", "C2", (0.0, 0.45), fresh, bad)
    assert between["0.45"]["nested"]["label"] == dc.VOID
    assert between["0.00"]["nested"]["label"] is None


def test_run_script_constants_match_the_frozen_design():
    import run_draw_crossed_readout as r
    assert r.REGISTERED_BASES == (800_000, 850_000)
    assert r.FRESH_BASES == tuple((6_000_000 + 100_000 * k, 6_050_000 + 100_000 * k)
                                  for k in range(8))
    assert (r.N_EPS, r.STEPS, r.RAY_STEPS) == (110, 24, 5)
    assert r.SEEDS == tuple(range(10)) and r.DRIFTS == (0.0, 0.45)
    assert (dc.BAR, dc.MARGIN, dc.BAND, dc.LEVEL) == (0.65, 0.05, (0.45, 0.55), 0.90)
    assert dc.PROBES[0] == "nested"                                # the primary probe
    assert r._clean({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}


def test_fmt_ci_quotes_as_the_record_does():
    ci = {"mean": 0.7761, "lo": 0.7364, "hi": 0.8158, "computable": True}
    assert dc.fmt_ci(ci) == "0.776 [0.736, 0.816]"
    d = {"mean": -0.0002, "lo": -0.0181, "hi": 0.0171, "computable": True}
    assert dc.fmt_ci(d, signed=True) == "+0.000 [-0.018, +0.017]"
    assert dc.fmt_ci({"mean": -0.003, "lo": None, "hi": None, "computable": False},
                     signed=True) == "-0.003 [not computable]"
    assert dc.fmt_ci({"mean": None, "computable": False}) == "not computable"
