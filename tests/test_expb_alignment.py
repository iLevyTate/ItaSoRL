"""Closing hook A7d and G35: the Experiment B collector's observation-action alignment and the
--align / --out-dir wiring of the five scripts/run_expB_*.py scripts.

CORRECTIONS 2026-10-16 measured, on 60 scripted episodes of 30 steps at seed0 10000, that the
heading change from O[t] to O[t+1] correlates 1.000 with the turn in A[t+1] and 0.015 with the
turn in A[t] under the stored (legacy) pairing. The tests below pin that measurement, check
that the corrected pairing moves the correlation onto A[t], that the legacy default is the
pre-hook collector byte for byte, and that an --out-dir run leaves the record figures alone.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from _git_checkout import assert_commit_recorded

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from itasorl.experiment_b import collect_episodes, scripted_policy  # noqa: E402
from itasorl.patch_of_earth import PatchOfEarthV0  # noqa: E402
from itasorl.world import DEFAULT_OBS_SPEC, SeedBundle, WorldParams  # noqa: E402

pytest.importorskip("torch")
pytest.importorskip("sklearn")


def _collect_reference(label, n, steps=30, drift_sigma=0.0, ray_steps=6, seed0=0, params=None):
    """experiment_b.collect_episodes as it stood at main 1bbe7f5, before the align hook."""
    params = params or WorldParams()
    eps = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        w = PatchOfEarthV0(params, drift_sigma=drift_sigma)
        w.ray_steps = ray_steps
        w.reset(SeedBundle(world=seed0 + i, weather=seed0 + 7000 + i, ecology=seed0 + 13000 + i))
        O, A, spd = [], [], []
        w.observe()
        for _ in range(steps):
            a = scripted_policy(rng)
            r = w.step(a)
            O.append(r.obs.copy()); A.append(a); spd.append(float(np.linalg.norm(w.vel)))
        eps.append({"obs": np.asarray(O, np.float32), "act": np.asarray(A, np.float32),
                    "label": label, "speed": float(np.mean(spd))})
    return eps


def _same_bytes(e1, e2):
    assert len(e1) == len(e2)
    for a, b in zip(e1, e2):
        assert a.keys() == b.keys()
        assert a["obs"].dtype == b["obs"].dtype and a["obs"].tobytes() == b["obs"].tobytes()
        assert a["act"].tobytes() == b["act"].tobytes()
        assert a["label"] == b["label"] and a["speed"] == b["speed"]


def _heading_index():
    off = 0
    for c in DEFAULT_OBS_SPEC.channels:
        if c.name == "intero":
            return off + 2, off + 3  # intero: velocity(2) then sin, cos of the heading
        off += c.size
    raise AssertionError("no intero channel")


def _turn_corr(eps, lag):
    """Correlation of the heading change O[t] -> O[t+1] with the turn in A[t + lag]."""
    si, ci = _heading_index()
    dh, tu = [], []
    for e in eps:
        h = np.arctan2(e["obs"][:, si], e["obs"][:, ci])
        d = np.angle(np.exp(1j * (h[1:] - h[:-1])))
        dh.append(d)
        tu.append(e["act"][lag:lag + len(d), 1])
    return float(np.corrcoef(np.concatenate(dh), np.concatenate(tu))[0, 1])


@pytest.mark.parametrize("drift", [0.0, 0.2])
def test_legacy_default_is_byte_identical_to_the_pre_hook_collector(drift):
    params = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    ref = _collect_reference(1, 3, 12, drift, 5, seed0=30200, params=params)
    _same_bytes(collect_episodes(1, 3, 12, drift, 5, seed0=30200, params=params), ref)
    _same_bytes(collect_episodes(1, 3, 12, drift, 5, seed0=30200, params=params,
                                 align="legacy"), ref)


def test_aligned_pairing_shifts_observations_only():
    leg = collect_episodes(0, 3, 10, seed0=10000)
    ali = collect_episodes(0, 3, 10, seed0=10000, align="obs_then_action")
    for i, (l, a) in enumerate(zip(leg, ali)):
        assert a["obs"][1:].tobytes() == l["obs"][:-1].tobytes()
        w = PatchOfEarthV0(WorldParams())
        w.ray_steps = 6
        w.reset(SeedBundle(world=10000 + i, weather=17000 + i, ecology=23000 + i))
        assert a["obs"][0].tobytes() == np.asarray(w.observe(), np.float32).tobytes()
        assert a["act"].tobytes() == l["act"].tobytes() and a["speed"] == l["speed"]


def test_unknown_alignment_is_refused():
    with pytest.raises(ValueError, match="align"):
        collect_episodes(0, 1, 2, align="action_then_obs")


def test_turn_correlation_matches_the_correction_record():
    """The CORRECTIONS 2026-10-16 measurement: 60 episodes, 30 steps, seed0 10000."""
    leg = collect_episodes(0, 60, 30, seed0=10000)
    ali = collect_episodes(0, 60, 30, seed0=10000, align="obs_then_action")
    assert _turn_corr(leg, 1) > 0.999
    assert abs(_turn_corr(leg, 0) - 0.015) < 0.005
    assert _turn_corr(ali, 0) > 0.99
    assert abs(_turn_corr(ali, 1)) < 0.05


# --- the five scripts -------------------------------------------------------------------

SMOKE = ["--pool", "12", "--sub", "10", "--epochs", "1", "--seeds", "0"]
SCRIPTS = ("expB_full", "expB_kstep", "expB_nonlinear", "expB_gap", "expB_surprise")
FIGURES = {"expB_full": "expB_incidental.png", "expB_kstep": "expB_kstep.png",
           "expB_surprise": "expB_channels.png"}


def _record_figures():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "docs" / "figures").glob("expB_*.png"))}


def _script(name):
    return importlib.import_module(f"run_{name}")


def test_parsers_default_to_the_registered_constants():
    import expb_common
    for name in SCRIPTS:
        m = _script(name)
        a = expb_common.parser("x", pool=m.POOL, sub=m.SUB, epochs=m.EPOCHS,
                               seeds=m.SEEDS).parse_args([])
        assert (a.align, a.out_dir, a.pool, a.sub, a.epochs, a.seeds) == (
            "legacy", None, 150, 110, m.EPOCHS, [0, 1, 2])
    assert expb_common.figure_path(None, "expB_kstep.png") == "docs/figures/expB_kstep.png"


def test_surprise_refuses_to_run_without_the_out_dir_full_summary(tmp_path):
    with pytest.raises(SystemExit, match="expB_full.json is missing"):
        _script("expB_surprise").main(["--out-dir", str(tmp_path), *SMOKE])
    (tmp_path / "expB_full.json").write_text(json.dumps(
        {"align": "legacy", "drift_sweep": [{"drift": d, "target_mean": 0.5}
                                             for d in (0.0, 0.2, 0.45)]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="obs_then_action"):
        _script("expB_surprise").main(["--out-dir", str(tmp_path), "--align",
                                       "obs_then_action", *SMOKE])


def test_out_dir_runs_write_json_and_leave_record_figures_alone(tmp_path):
    before = _record_figures()
    assert set(FIGURES.values()) <= set(before)
    for name in SCRIPTS:  # expB_full first: surprise reads its summary
        assert _script(name).main(["--out-dir", str(tmp_path), "--align", "obs_then_action",
                                   *SMOKE]) == 0
    assert _record_figures() == before
    for name in SCRIPTS:
        doc = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        assert doc["align"] == "obs_then_action" and doc["script"] == f"scripts/run_{name}.py"
        assert doc["config"]["pool"] == 12 and doc["config"]["seeds"] == [0]
        prov = doc["provenance"]
        assert_commit_recorded(prov, "git_commit")
        assert "tracked_changes_at_generation" in prov and prov["stack"]["device"] == "cpu"
        assert "scipy" in prov["stack"] and "--out-dir" in prov["argv"]
        if name in FIGURES:
            assert (tmp_path / FIGURES[name]).is_file()
    full = json.loads((tmp_path / "expB_full.json").read_text(encoding="utf-8"))
    sur = json.loads((tmp_path / "expB_surprise.json").read_text(encoding="utf-8"))
    assert [r["drift"] for r in full["drift_sweep"]] == [0.0, 0.2, 0.45]
    assert [r["recurrent_state_target_mean"] for r in sur["drift_sweep"]] == [
        r["target_mean"] for r in full["drift_sweep"]]
    assert all(len(r["per_seed"]["target"]) == 1 for r in full["drift_sweep"])
