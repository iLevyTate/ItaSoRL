"""Smoke coverage for the builder and readout scripts nothing imported (audit finding F48).

Eight scripts had no test importing them at all, so a syntax error, a bad import or a renamed
flag would only surface in a multi-hour run or in CI's --check steps. These tests import each
one, pin the constants a run depends on, and exercise the pure helpers on synthetic input.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytest.importorskip("torch")      # the readout scripts import itasorl, which needs torch

import build_budget_curve as bbc  # noqa: E402
import run_l0_audit as rla  # noqa: E402
import run_texture_fresh_probe as rtfp  # noqa: E402
import validate_population_readout as vpr  # noqa: E402


# --------------------------------------------------------------- build_budget_curve

def _cells(tmp_path, series):
    """A minimal run directory: one cell per seed, each carrying a budget curve."""
    d = tmp_path / "cells"
    d.mkdir()
    for seed, by_u in series.items():
        cell = {"cell": {"drift": 0.45, "seed": seed, "budget_curve": {
            str(u): {"pool": {"target": t}, "eng": {"trained_return": r}, "env_steps": s}
            for u, (t, r, s) in by_u.items()}}}
        (d / f"cell_d0.45_s{seed}.json").write_text(json.dumps(cell), encoding="utf-8")
    # A drift-0 cell must be ignored: the curve is read at the strongest drift only.
    (d / "cell_d0.00_s0.json").write_text(
        json.dumps({"cell": {"drift": 0.0, "seed": 0, "budget_curve": {
            "300": {"pool": {"target": 0.0}, "eng": {"trained_return": 0.0}, "env_steps": 0.0}}}}),
        encoding="utf-8")
    return str(tmp_path)


def test_load_curve_reads_the_strongest_drift_and_orders_by_updates(tmp_path):
    run = _cells(tmp_path, {0: {300: (0.70, 1.0, 10.0), 450: (0.80, 2.0, 20.0)},
                            1: {300: (0.60, 1.5, 11.0), 450: (0.90, 2.5, 21.0)}})
    dmax, by_u = bbc.load_curve(run)
    assert dmax == 0.45
    assert list(by_u) == [300, 450]
    assert sorted(by_u[300]["target"]) == [0.60, 0.70]


def test_summarize_means_and_intervals_come_from_the_per_seed_values(tmp_path):
    run = _cells(tmp_path, {0: {300: (0.70, 1.0, 10.0)}, 1: {300: (0.60, 1.5, 11.0)}})
    _, by_u = bbc.load_curve(run)
    rows = bbc.summarize(by_u)
    assert len(rows) == 1
    row = rows[0]
    assert row["updates"] == 300 and row["seeds"] == [0, 1]
    assert row["target"]["mean"] == pytest.approx(0.65)
    assert row["target"]["per_seed"] == [0.70, 0.60]          # reordered by seed
    lo, hi = row["target"]["t90"]
    assert lo < 0.65 < hi


def test_summarize_reports_no_interval_for_a_single_seed(tmp_path):
    run = _cells(tmp_path, {0: {300: (0.70, 1.0, 10.0)}})
    _, by_u = bbc.load_curve(run)
    assert bbc.summarize(by_u)[0]["target"]["t90"] is None


def test_budget_figure_draws_what_summarize_produces(tmp_path):
    """figure() consumes summarize() output directly, so build its input that way: a renamed
    or dropped key in either function breaks this test rather than a multi-hour run."""
    pytest.importorskip("matplotlib")
    run = _cells(tmp_path, {0: {300: (0.70, 1.0, 10.0), 450: (0.80, 2.0, 20.0)},
                            1: {300: (0.60, 1.5, 11.0), 450: (0.90, 2.5, 21.0)}})
    _, by_u = bbc.load_curve(run)
    rows = bbc.summarize(by_u)
    out = tmp_path / "figures" / "curve.png"          # a directory it must create
    bbc.figure({"decoder on": rows, "decoder off": rows}, str(out))
    assert out.exists() and out.stat().st_size > 1000
    assert len(bbc.SERIES_COLORS) >= 2                # one color per series


# --------------------------------------------------------------- readout scripts

AGENT_SCRIPTS = [("run_l0_audit", rla), ("run_texture_fresh_probe", rtfp)]


@pytest.mark.parametrize("name,mod", AGENT_SCRIPTS)
def test_agent_filename_pattern_parses_a_real_saved_agent_name(name, mod):
    m = mod.AGENT_RE.search("agent_d0.45_s7_survival.pt")
    assert m and m.group(1) == "0.45" and m.group(2) == "7" and m.group(3) == "survival"
    assert mod.AGENT_RE.search("agent_d0.45_s7_decoder.pt") is None   # unknown arm
    assert mod.AGENT_RE.search("agent_d0.45_s7_survival.json") is None


def test_texture_comparator_seed_bases_are_disjoint_ranges():
    gn, qd = rtfp.BASES["gn"], rtfp.BASES["qd"]
    assert set(rtfp.BASES) == {"gn", "qd"}
    for lo, hi in (gn, qd):
        assert lo < hi
    assert max(gn) < min(qd), "the qd pools must not reuse a gn world seed"


def test_l0_audit_world_sample_pairs_are_distinct_from_the_standard_pair():
    assert rla.STANDARD_BASES not in rla.AUDIT_BASES
    assert len(set(rla.AUDIT_BASES)) == len(rla.AUDIT_BASES)
    assert rla.P.k_land == 1.5 and rla.P.gravity == 0.4      # world P, as every readout uses


@pytest.mark.parametrize("name,mod", [("build_budget_curve", bbc), ("run_l0_audit", rla),
                                      ("run_texture_fresh_probe", rtfp),
                                      ("validate_population_readout", vpr)])
def test_each_script_exposes_a_main(name, mod):
    assert callable(mod.main)


@pytest.mark.parametrize("script", ["build_budget_curve.py", "run_l0_audit.py",
                                    "run_texture_fresh_probe.py",
                                    "validate_population_readout.py"])
def test_each_script_prints_its_usage_and_exits_zero(script):
    """--help goes through argparse, so a renamed or duplicated flag fails here."""
    import subprocess
    r = subprocess.run([sys.executable, os.path.join("scripts", script), "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    assert "usage" in r.stdout.lower()
