"""Closing hook G03: `--integrity-seed-base` in scripts/audit_sensory_echo.py.

The fresh-base repeat of the balanced sensory-echo control rolls each cell twice: at the run's
own balanced base (910000), whose target must reproduce the recorded `balanced.target`, and at a
fresh read base, which every reading comes from. Four cases:

1. Without the flag the cells and the aggregate are the pre-hook script's output byte for byte
   (the aggregate's new provenance key aside).
2. With the flag the integrity target comes from the integrity rollout and the readings from
   the read rollout, and the aggregate names both bases.
3. A pair dropped at the fresh base still withholds the zone.
4. A read base of 2200000 (the post-review balanced pairs) is refused; 2250000 is accepted.

The rollouts and probes are replaced by cheap deterministic fakes; the script's own control
flow, integrity gate and aggregation run for real.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_sensory_echo as ase  # noqa: E402
import itasorl.experiment_b2 as b2  # noqa: E402

PRE_HOOK = "1bbe7f5"
SEEDS = range(10, 20)
ARMS = ("survival", "predictor", "untrained")
N_EPS, STEPS, HID = 110, 4, 3
INTEGRITY_BASE, FRESH_BASE, TAKEN_BASE = 910_000, 2_250_000, 2_200_000


def _pre_hook_module(tmp_path, name="audit_sensory_echo"):
    """The script as it stood at main 1bbe7f5, loaded from git; skip outside a full clone."""
    try:
        src = subprocess.run(["git", "show", f"{PRE_HOOK}:scripts/{name}.py"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"git history at {PRE_HOOK} is not available")
    path = tmp_path / f"{name}_pre_hook.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(f"{name}_pre_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_pool(calls, drop_base=None):
    def pool(agent, norm, params, drift, n_eps, steps, device, seed_base, ray, **kw):
        calls.append((agent, seed_base, float(drift)))
        key = zlib.crc32(f"{agent}|{seed_base}|{drift}".encode())
        rng = np.random.default_rng(key)
        kept = np.arange(n_eps)
        if drop_base is not None and seed_base == drop_base and drift != 0.0:
            kept = np.delete(kept, 3)
        k = len(kept)
        H = rng.normal(size=(k, steps, HID)).astype(np.float32) + (0.3 if drift else 0.0)
        a0 = zlib.crc32(f"{agent}|{seed_base}".encode()) % 97
        H[:, 0] = np.sin(kept[:, None] + a0 + np.arange(HID)[None, :])   # shared by twins
        bt = rng.normal(size=(k, steps, 7)).astype(np.float32)
        O = rng.normal(size=(k, steps, 2)).astype(np.float32)
        z = np.zeros(k)
        return H, z, z, z, z, z, bt, O, kept
    return pool


def _fake_auroc(X, y, groups=None):
    X = np.asarray(X, float).reshape(len(y), -1)
    return float(0.5 + 0.4 * np.tanh(X[y == 1].mean() - X[y == 0].mean()))


def _fake_trace(H, Bt, y, groups=None):
    return _fake_auroc(H, y) - 0.02


def _fake_resid(H, Ot, y, groups=None, **kw):
    return _fake_auroc(H, y) - 0.01 * (1 + len(kw))


def _patch(monkeypatch, mod, calls, drop_base=None):
    monkeypatch.setattr(mod, "collect_pool", _fake_pool(calls, drop_base))
    monkeypatch.setattr(mod, "load_agent_bundle", lambda path, dev: (Path(path).name, None))
    monkeypatch.setattr(mod, "grouped_auroc", _fake_auroc)
    monkeypatch.setattr(mod, "trace_residual_probe_auroc", _fake_trace)
    monkeypatch.setattr(mod, "sensory_residual_probe_auroc", _fake_resid)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Agent stubs for the frozen balanced population and cells whose recorded balanced
    target is the fake integrity rollout's own, so the integrity gate passes."""
    for attr in ("DRIFT_MODE", "OBJECTIVE", "MORTAL"):
        monkeypatch.setattr(b2, attr, getattr(b2, attr))
    monkeypatch.setattr(b2, "setup_l3_surrogate", lambda **kw: None)
    agents, cells = tmp_path / "agents", tmp_path / "cells"
    agents.mkdir()
    cells.mkdir()
    _patch(monkeypatch, ase, [])
    for s in SEEDS:
        rec = {"cell": {"agents": {}}}
        for arm in ARMS:
            name = f"agent_d0.45_s{s}_{arm}.pt"
            (agents / name).write_bytes(b"")
            icell = ase.integrity_rollout(name, None, 0.45, n_eps=N_EPS, steps=STEPS, device="cpu",
                                          integrity_seed_base=INTEGRITY_BASE)
            rec["cell"]["agents"][arm] = {"balanced": {"target": icell["target"], "n_pairs": N_EPS}}
        (cells / f"cell_d0.45_s{s}.json").write_text(json.dumps(rec), encoding="utf-8")
    return tmp_path


def _main(mod, monkeypatch, run_dir, out, *extra):
    argv = ["audit_sensory_echo.py", "--agents-dir", str(run_dir / "agents"),
            "--cells-dir", str(run_dir / "cells"), "--out-dir", str(out), "--drifts", "0.45",
            "--device", "cpu", "--n-eps", str(N_EPS), "--steps", str(STEPS), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    agg = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    cells = json.loads((out / "cells.json").read_text(encoding="utf-8"))
    return agg, cells


def test_no_flag_output_is_byte_identical_to_the_pre_hook_script(run_dir, monkeypatch, tmp_path):
    old = _pre_hook_module(tmp_path)
    _patch(monkeypatch, old, [])
    _patch(monkeypatch, ase, [])
    base = ("--balanced-seed-base", str(INTEGRITY_BASE))
    _main(old, monkeypatch, run_dir, tmp_path / "old", *base)
    agg, _ = _main(ase, monkeypatch, run_dir, tmp_path / "new", *base)
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    assert (new_dir / "cells.json").read_bytes() == (old_dir / "cells.json").read_bytes()
    prov = agg.pop("provenance")
    assert json.dumps(agg, indent=1) == (old_dir / "aggregate.json").read_text(encoding="utf-8")
    assert "integrity_seed_base" not in agg and "read_seed_base" not in agg
    assert "git_commit_at_promotion" in prov and "git_commit" not in prov
    assert agg["integrity"]["adjudication_blockers"] == []


def test_flag_reads_integrity_and_metrics_from_their_own_rollouts(run_dir, monkeypatch, tmp_path):
    calls = []
    _patch(monkeypatch, ase, calls)
    agg, cells = _main(ase, monkeypatch, run_dir, tmp_path / "fresh",
                       "--balanced-seed-base", str(FRESH_BASE),
                       "--integrity-seed-base", str(INTEGRITY_BASE))
    assert {b for _, b, _ in calls} == {INTEGRITY_BASE, FRESH_BASE}
    assert agg["integrity_seed_base"] == INTEGRITY_BASE and agg["read_seed_base"] == FRESH_BASE
    assert agg["balanced_seed_base"] == FRESH_BASE
    assert agg["integrity"]["adjudication_blockers"] == [] and agg["integrity"]["all_match"] is True
    assert len(cells) == len(SEEDS) * len(ARMS)
    for c in cells:
        name = f"agent_d0.45_s{c['seed']}_{c['agent']}.pt"
        want_i = ase.integrity_rollout(name, None, 0.45, n_eps=N_EPS, steps=STEPS, device="cpu",
                                       integrity_seed_base=INTEGRITY_BASE)["target"]
        want_r = ase.integrity_rollout(name, None, 0.45, n_eps=N_EPS, steps=STEPS, device="cpu",
                                       integrity_seed_base=FRESH_BASE)["target"]
        assert want_i != want_r
        assert c["integrity_target"] == want_i == c["recorded_target"]
        assert c["target"] == want_r
        assert c["integrity_match"] is True and c["integrity_n_pairs"] == N_EPS
        assert c["n_pairs"] == N_EPS and c["dropped_pairs"] == 0


def test_fresh_base_pair_drop_withholds_the_zone(run_dir, monkeypatch, tmp_path):
    _patch(monkeypatch, ase, [], drop_base=FRESH_BASE)
    agg, cells = _main(ase, monkeypatch, run_dir, tmp_path / "drop",
                       "--balanced-seed-base", str(FRESH_BASE),
                       "--integrity-seed-base", str(INTEGRITY_BASE))
    assert all(c["integrity_match"] is True for c in cells)      # the integrity rollout is clean
    assert all(c["dropped_pairs"] == 1 for c in cells)
    blockers = agg["integrity"]["adjudication_blockers"]
    assert any(b.startswith("n_pairs is not 110") for b in blockers)
    assert any(b.startswith("pair bookkeeping failed") for b in blockers)
    zone = agg["decision_balanced"]["zone"]
    assert zone.startswith("WITHHELD") and agg["decision_balanced"]["pass_bar"] is False


def test_integrity_gap_is_read_from_the_integrity_target():
    """A clean read target does not hide a bad integrity target, and a broken integrity
    rollout withholds the zone."""
    cell = {"drift": "0.45", "seed": 10, "agent": "survival", "target": 0.70, "resid_obs": 0.6,
            "integrity_match": True, "n_pairs": 110, "dropped_pairs": 0,
            "identical_first_state": True, "recorded_target": 0.70, "integrity_target": 0.71,
            "integrity_n_pairs": 110, "integrity_dropped_pairs": 0,
            "integrity_identical_first_state": True}
    keys = {("0.45", 10, "survival")}
    got = ase.adjudication_blockers([cell], quick=False, mismatches=[], balanced=True,
                                    expected_keys=keys)
    assert any(b.startswith("target gap above tolerance") for b in got)
    cell.update(integrity_target=0.70, integrity_dropped_pairs=2)
    got = ase.adjudication_blockers([cell], quick=False, mismatches=[], balanced=True,
                                    expected_keys=keys)
    assert got == ["integrity pair bookkeeping failed at 0.45 s10 survival"]


def test_read_base_of_the_post_review_pairs_is_refused(run_dir, monkeypatch, tmp_path):
    hits = ase.read_base_collisions(TAKEN_BASE, N_EPS)
    assert ("G03 read", "post-review balanced pairs") in hits
    assert ase.read_base_collisions(FRESH_BASE, N_EPS) == []
    calls = []
    _patch(monkeypatch, ase, calls)
    monkeypatch.setattr(sys, "argv", [
        "audit_sensory_echo.py", "--agents-dir", str(run_dir / "agents"),
        "--cells-dir", str(run_dir / "cells"), "--out-dir", str(tmp_path / "taken"),
        "--balanced-seed-base", str(TAKEN_BASE), "--integrity-seed-base", str(INTEGRITY_BASE),
        "--drifts", "0.45", "--device", "cpu"])
    with pytest.raises(SystemExit, match="shares world draws"):
        ase.main()
    assert calls == [] and not (tmp_path / "taken").exists()
    # 2250000 is accepted (and runs: the flag test above reads from it)
    from itasorl import post_review
    assert post_review.registered_later("G03-fresh")[1] == FRESH_BASE


def test_flag_needs_balanced_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "audit_sensory_echo.py", "--agents-dir", str(tmp_path), "--states-dir", str(tmp_path),
        "--out-dir", str(tmp_path / "o"), "--integrity-seed-base", str(INTEGRITY_BASE)])
    with pytest.raises(SystemExit, match="needs --balanced-seed-base"):
        ase.main()
