"""Closing item X09 (E19): scripts/build_bayes_factors.py.

The point-null JZS BF01 matches the closed-form one-dimensional integral of Rouder et al.
(2009, eq. 1); the interval-null BF01 matches a Monte Carlo draw from the Cauchy prior; and a
change to any per-seed value of any input array moves the printed BF of its row.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_bayes_factors as bbf  # noqa: E402

KNOWN = np.array([0.572727, 0.550413, 0.56405, 0.573554, 0.538017, 0.53843, 0.594215, 0.560331,
                  0.548, 0.551])


def rouder_bf01(t: float, n: int, r: float) -> float:
    """Rouder, Speckman, Sun, Morey and Iverson (2009), eq. 1, with prior scale r."""
    from scipy.integrate import quad
    v = n - 1
    num = (1 + t * t / v) ** (-(v + 1) / 2)

    def f(g):
        a = 1 + n * g * r * r
        return a ** -0.5 * (1 + t * t / (a * v)) ** (-(v + 1) / 2) * (2 * math.pi) ** -0.5 \
            * g ** -1.5 * math.exp(-1 / (2 * g))

    den, _ = quad(f, 0, np.inf, limit=400, epsrel=1e-11)
    return num / den


@pytest.mark.parametrize("r", bbf.SCALES)
@pytest.mark.parametrize("mu0", [0.5, 0.55])
def test_point_null_matches_the_closed_form(r, mu0):
    x = KNOWN
    t = (x.mean() - mu0) / (x.std(ddof=1) / math.sqrt(x.size))
    assert bbf.point_bf01(x, mu0, r) == pytest.approx(rouder_bf01(t, x.size, r), rel=1e-6)


@pytest.mark.parametrize("x,mu0", [(KNOWN, 0.5), (KNOWN - 0.04, 0.5),
                                   (0.03 + 0.06 * np.random.default_rng(1).standard_normal(8), 0.0)])
def test_interval_null_matches_monte_carlo(x, mu0):
    from scipy.stats import nct
    r = bbf.LABEL_SCALE
    got = bbf.interval_bf01(x, mu0, bbf.HALF, r)
    rng = np.random.default_rng(7)
    delta = r * np.tan(np.pi * (rng.random(400_000) - 0.5))       # Cauchy(0, r) draws
    w = nct.pdf(got["t"], got["n"] - 1, delta * math.sqrt(got["n"]))
    inside = np.abs(delta) <= got["c_delta"]
    mc = (w[inside].mean()) / (w[~inside].mean())
    assert got["bf01"] == pytest.approx(mc, rel=0.03)
    assert got["prior_mass_in"] == pytest.approx(inside.mean(), abs=0.003)


def _docs() -> dict:
    rng = np.random.default_rng(3)
    c1 = (0.56 + 0.015 * rng.standard_normal(10)).tolist()
    rows = 0.5 + 0.15 * rng.standard_normal((10, 9))
    pairs = [float(np.mean(rows[:, p])) for p in range(1, 9)]
    bal = {rid: (0.55 + 0.02 * rng.standard_normal(10)).tolist() for rid in ("B1", "B2")}
    fresh_s, fresh_u = 0.6 + 0.03 * rng.standard_normal(10), 0.52 + 0.03 * rng.standard_normal(10)
    m0 = [float(a - b) for a, b in zip(fresh_s, fresh_u)]
    m99 = (np.asarray(m0) + 0.01 + 0.03 * rng.standard_normal(10)).tolist()
    fm99 = (np.asarray(m0) - 0.01 + 0.03 * rng.standard_normal(10)).tolist()
    l1 = {a: (0.52 + 0.03 * rng.standard_normal(10)).tolist() for a in ("survival", "untrained", "predictor")}
    return {
        "c1": {"runs": {"C1": {"drift_zero": {"survival": {"per_seed": c1, "mean": float(np.mean(c1))}}}}},
        "l0": {"l0_summary": {"independent_pair_means": pairs},
               "l0": [{"seed": s, "rows": [{"target": float(v)} for v in rows[s]]} for s in range(10)]},
        "balanced": {"runs": {rid: {"pool_target": {"0.00": {"survival": {
            "per_seed": v, "mean": float(np.mean(v))}}}} for rid, v in bal.items()}},
        "sweep_c": {"points": [{"rho": 0.0, "M": {"per_seed": m0}}, {"rho": 0.99, "M": {"per_seed": m99}}],
                    "contrast": {"C: M(0.99) - M(0)": {"mean": float(np.mean(np.subtract(m99, m0)))}}},
        "sweep_c_rho0": {"cells": [c for s in range(10) for c in (
            {"seed": s, "arm": "survival", "fresh": float(fresh_s[s])},
            {"seed": s, "arm": "untrained", "fresh": float(fresh_u[s])})]},
        "sweep_c_followup": {"points": [{"rho": 0.99, "M": {"per_seed": fm99}}],
                             "contrast": {"follow-up": {"mean": float(np.mean(np.subtract(fm99, m0)))}}},
        "l1": {"arms": {"0.023": {a: {"pool_target": {"per_seed": v}} for a, v in l1.items()}}},
    }


# (path to a per-seed array inside the docs, the reading prefixes whose printed BF it feeds)
ARRAYS = [
    (("c1", "runs", "C1", "drift_zero", "survival", "per_seed"), ("1 ",)),
    (("l0", "l0_summary", "independent_pair_means"), ("2 eight independent L0 pairs (signed)",
                                                      "2 eight independent L0 pairs (folded")),
    (("balanced", "runs", "B1", "pool_target", "0.00", "survival", "per_seed"), ("3 B1",)),
    (("balanced", "runs", "B2", "pool_target", "0.00", "survival", "per_seed"), ("3 B2",)),
    (("sweep_c", "points", 1, "M", "per_seed"), ("4 sweep C stored",)),
    (("sweep_c", "points", 0, "M", "per_seed"), ("4 sweep C stored", "4 sweep C follow-up")),
    (("sweep_c_followup", "points", 0, "M", "per_seed"), ("4 sweep C follow-up",)),
    (("l1", "arms", "0.023", "survival", "pool_target", "per_seed"), ("5 corrected L1 survival minus",)),
    (("l1", "arms", "0.023", "untrained", "pool_target", "per_seed"), ("5 corrected L1 survival minus untrained",)),
    (("l1", "arms", "0.023", "predictor", "pool_target", "per_seed"), ("5 corrected L1 survival minus predictor",)),
]


def _printed(rows: list) -> dict:
    return {r["reading"]: bbf.fmt_bf(r["bf01_interval"]["0.707"]) for r in rows}


def _get(doc, path):
    for k in path:
        doc = doc[k]
    return doc


def test_any_per_seed_change_moves_the_printed_bf(monkeypatch):
    monkeypatch.setattr(bbf, "point_bf01", lambda *a, **k: float("nan"))   # not under test; speed
    monkeypatch.setattr(bbf, "SCALES", (bbf.LABEL_SCALE,))
    cache: dict = {}
    real = bbf.interval_bf01

    def memo(x, mu0, half, r):     # unchanged arrays are not recomputed
        key = (tuple(np.asarray(x, float).tolist()), mu0, half, r)
        if key not in cache:
            cache[key] = real(x, mu0, half, r)
        return cache[key]

    monkeypatch.setattr(bbf, "interval_bf01", memo)
    docs = _docs()
    rows, fails = bbf.readings(docs)
    assert fails == []
    assert len(rows) == 9
    base = _printed(rows)
    for path, readers in ARRAYS:
        n = len(_get(docs, path))
        for i in range(n):
            d = copy.deepcopy(docs)
            _get(d, path)[i] += 0.003
            moved = _printed(bbf.readings(d)[0])
            for name in base:
                if any(name.startswith(p) for p in readers):
                    assert moved[name] != base[name], (path, i, name)


def test_gate_refuses_a_file_that_differs_from_its_committed_version(tmp_path):
    import json
    for key, rel in bbf.INPUTS.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_docs()[key]))
    same = lambda rel: (tmp_path / rel).read_bytes()  # noqa: E731
    assert bbf.load_inputs(str(tmp_path), same)[1] == []
    stale = lambda rel: same(rel) + (b" " if rel == bbf.INPUTS["l1"] else b"")  # noqa: E731
    fails = bbf.load_inputs(str(tmp_path), stale)[1]
    assert fails == [f"{bbf.INPUTS['l1']}: differs from the committed version"]
