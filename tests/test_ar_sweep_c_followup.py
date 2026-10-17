"""scripts/run_ar_sweep_c_followup.py (A6c): the closed-form 24-step gap, the sigma matching,
the seed-paired contrast, the recorded off-target values and the resume check. Nothing here
reruns an agent: the stored point files are reused, and the summary is rebuilt from them."""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_ar_sweep_c_followup as fu  # noqa: E402
from run_ar_sweep import SIGMA_A, contrast, margin_by_seed  # noqa: E402

FU = os.path.join(ROOT, "artifacts", "texture", "ar_sweep_c_followup")
SC = os.path.join(ROOT, "artifacts", "texture", "ar_sweep_c")


def _load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return json.load(fh)


def test_lam_is_the_audit_constant():
    """The audit rebuilds the gap with lam = 0.925 (its _gap24 default); the runner reads it from
    the world parameters. They must agree."""
    assert fu.lam() == pytest.approx(0.925, abs=1e-15)
    with open(os.path.join(ROOT, "scripts", "audit_stats_recheck.py"), encoding="utf-8") as fh:
        assert "def _gap24(sig, rho, lam=0.925, h=24):" in fh.read()


@pytest.mark.parametrize("rho", [0.0, 0.9, 0.99])
def test_the_closed_form_gap_matches_a_monte_carlo_of_g_ar(rho):
    """Open loop from a stationary start: G_ar minus the authentic law after 24 steps, RMS over
    both components and 3000 independently seeded processes."""
    from itasorl.surrogate_l3_families import make_g_ar
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    sigma, n = 0.01, 3000
    sq = np.empty(n)
    for k in range(n):
        g = make_g_ar(sigma=sigma, rho=rho, params=P, seed=10_000 + k)
        auth = make_g_ar(sigma=0.0, rho=0.0, params=P)
        v = w = np.zeros(2)
        for _ in range(fu.H):
            v, w = g(v, np.zeros(2)), auth(w, np.zeros(2))
        sq[k] = float(np.sum((v - w) ** 2))
    mc = float(np.sqrt(sq.mean()))
    assert mc == pytest.approx(fu.exact_gap(sigma, rho), rel=0.05)


def test_the_follow_up_sigmas_match_the_target_gap_exactly():
    s = _load(FU, "summary.json")
    target = fu.exact_gap(SIGMA_A, 0.9)
    assert s["gap_target_exact"] == pytest.approx(target, rel=1e-12)
    for p in s["points"]:
        assert p["rho"] in fu.RERUN
        assert p["sigma"] == pytest.approx(target / fu.exact_gap(1.0, p["rho"]), rel=1e-12)
        assert fu.exact_gap(p["sigma"], p["rho"]) == pytest.approx(target, rel=1e-12)
        assert p["exact_gap"] == pytest.approx(target, rel=1e-12)


def test_the_recorded_off_target_values_recompute():
    s = _load(FU, "summary.json")
    stored = _load(SC, "summary.json")["points"]
    target = fu.exact_gap(SIGMA_A, 0.9)
    assert [r["rho"] for r in s["stored_points_exact_gaps"]] == [p["rho"] for p in stored]
    for r, p in zip(s["stored_points_exact_gaps"], stored):
        assert r["sigma"] == p["sigma"]
        assert r["exact_gap"] == pytest.approx(fu.exact_gap(p["sigma"], p["rho"]), rel=1e-12)
        assert r["off_target"] == pytest.approx(fu.exact_gap(p["sigma"], p["rho"]) / target - 1.0, abs=1e-12)
        assert r["within_tolerance"] is (abs(r["off_target"]) <= fu.GAP_TOL)
    # the four lower-rho points sit within 1.1 percent; only 0.99 is outside the 3 percent tolerance
    off = {r["rho"]: r["off_target"] for r in s["stored_points_exact_gaps"]}
    assert max(abs(off[r]) for r in (0.0, 0.5, 0.8)) < 0.0115 and off[0.9] == 0.0
    assert [r for r, v in off.items() if abs(v) > fu.GAP_TOL] == [0.99]


def _close(a, b, path="$"):
    """Equal up to float rounding: the t quantile moves in the last digits between scipy
    releases (1.15 here, 1.18 on CI), so a recomputed interval matches its stored copy to
    about 1e-13, not bit for bit. Every non-float value must still match exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), (path, sorted(set(a) ^ set(b)))
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12, nan_ok=True), (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_contrast_pairs_by_seed_not_by_order():


    hi = {3: 0.30, 1: 0.10, 2: 0.25, 9: 0.5}
    lo = {1: 0.05, 2: 0.20, 3: 0.10, 7: 0.0}
    c = contrast(hi, lo)
    d = np.array([0.10 - 0.05, 0.25 - 0.20, 0.30 - 0.10])
    assert c["n"] == 3 and c["mean"] == pytest.approx(d.mean())
    from itasorl.stats import t_ci90
    assert c["t90"] == pytest.approx(list(t_ci90(d)))


def test_the_stored_contrast_is_the_seed_paired_one():
    m99 = margin_by_seed(_load(FU, "C_rho0.99.json"))
    m0 = margin_by_seed(_load(SC, "C_rho0.00.json"))
    c = contrast(m99, m0)
    st = _load(FU, "summary.json")["contrast"]["follow-up: M(0.99, matched) - M(0, stored)"]
    _close(c, st)
    assert c["n"] == 10


def _run(tmp_path, monkeypatch, edit=None):
    out = tmp_path / "fu"
    out.mkdir()
    for f in ("C_rho0.95.json", "C_rho0.99.json"):
        shutil.copy(os.path.join(FU, f), out / f)
    if edit:
        edit(out)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["run_ar_sweep_c_followup.py", "--out-dir", str(out)])
    real_run = fu.subprocess.run

    def no_rerun(cmd, *a, **k):
        if any("run_texture_fresh_probe" in str(x) for x in cmd):
            pytest.fail("no point may be rerun")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(fu.subprocess, "run", no_rerun)
    return out, fu.main()


def test_resume_from_the_stored_points_rebuilds_the_stored_summary(tmp_path, monkeypatch, capsys):
    out, rc = _run(tmp_path, monkeypatch)
    capsys.readouterr()
    assert rc == 0
    new = _load(str(out), "summary.json")
    old = _load(FU, "summary.json")
    new.pop("provenance")
    old.pop("provenance")
    _close(new, old)


def test_resume_refuses_a_point_file_of_another_family(tmp_path, monkeypatch):
    def gn(out):
        p = out / "C_rho0.95.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["family"] = "gn"
        p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different inputs"):
        _run(tmp_path, monkeypatch, gn)


def test_a_contrast_short_of_ten_seeds_stops_before_the_readings(tmp_path, monkeypatch):
    def drop(out):
        p = out / "C_rho0.99.json"
        doc = json.loads(p.read_text(encoding="utf-8"))
        gone = next(c for c in doc["cells"] if c["arm"] == "survival")
        doc["cells"] = [c for c in doc["cells"] if c is not gone]
        p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pairs 9 seeds"):
        _run(tmp_path, monkeypatch, drop)
