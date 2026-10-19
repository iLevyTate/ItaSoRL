"""Closing item B6 (runbook 1.6, R14): scripts/run_development_heldout_transfer.py.

Smoke on a tmp snapshot set (small untrained trunks per budget, the L3 surrogate replaced by a
fixed damping law): the in-sample integrity gate against stored development outputs for both
runs, the held-out rows at the registered B6 bases with the b = 300 self-transfer row, the world
overlap guard, the base refusal, and the fail-closed path (a stored value off by 1e-6 writes
nothing)."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from _git_checkout import assert_commit_recorded

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import itasorl.experiment_b2 as b2  # noqa: E402
import run_development_heldout_transfer as ht  # noqa: E402
from itasorl import post_review as pr  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402

N = ["--seeds", "0", "--n-eps", "12"]


def _fake_load(agents_dir, drift, seed, arm, updates=None):
    w = make_world(None, 0.0, 5)
    torch.manual_seed(zlib.crc32(f"{drift:.2f}|{seed}|{arm}|{updates}".encode()))
    ag = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return ag, RunningNorm(w.obs_spec.size).freeze()


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(ht, "load", _fake_load)
    monkeypatch.setattr(ht, "setup_l3", lambda: None)
    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    return monkeypatch


def _agents(tmp_path) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    for b in ht.BUDGETS:
        Path(ht.agent_path(str(d), ht.DRIFT, 0, "survival", b)).write_bytes(b"x")
    return d


def _refs(tmp_path, values: dict) -> Path:
    dev = tmp_path / "development"
    dev.mkdir(exist_ok=True)
    for run, name in ht.RUN_FILES.items():
        rows = [{"seed": 0, "budgets": {str(b): {"transfer_from_300": values[b]} for b in ht.BUDGETS}}]
        (dev / f"{name}.json").write_text(json.dumps({"run": run, "rows": rows}))
    return dev / "corrected_l3_h8_wm.json"


def test_bases_are_the_registered_ones_and_refused_on_collision(monkeypatch):
    assert ht.fresh_bases(110) == (2_700_000, 2_750_000)
    with pytest.raises(SystemExit, match="footprint"):
        ht.fresh_bases(111)
    monkeypatch.setattr(pr, "all_bases_in_use", lambda: pr.BASES_IN_USE + (("x", 2_750_005, 3),))
    with pytest.raises(SystemExit, match="collide"):
        ht.fresh_bases(110)


def test_smoke_integrity_and_heldout_rows(fakes, tmp_path):
    agents = _agents(tmp_path)
    fa, fs = ht.fresh_bases(12)
    row = ht.run_task({"run": "wm", "agents_dir": str(agents), "seed": 0, "n_eps": 12,
                       "fresh_auth": fa, "fresh_surr": fs})
    insample = {b: row["budgets"][str(b)]["insample_transfer_from_300"] for b in ht.BUDGETS}
    ref = _refs(tmp_path, insample)
    out = tmp_path / "b6"
    argv = ["--run", f"wm={agents}", "--run", f"nowm={agents}", "--integrity-ref", str(ref),
            "--out-dir", str(out), *N]
    assert ht.main([*argv, "--dry-run"]) == 0 and not out.exists()
    assert ht.main([*argv, "--workers", "1"]) == 0
    doc = json.loads((out / "heldout_transfer.json").read_text())
    prov = doc["provenance"]
    assert_commit_recorded(prov)
    assert {"tracked_changes_at_generation", "stack", "argv", "agents_dir"} <= set(prov)
    assert doc["bases"]["fresh"] == [2_700_000, 2_750_000]
    assert all(v["pass"] and v["n_checked"] == 4 for v in doc["integrity"].values())
    for run in ("wm", "nowm"):
        r = doc["runs"][run]["rows"][0]["budgets"]
        assert set(r) == {"100", "200", "300", "450"}           # 300 is the self-transfer row
        for v in r.values():
            assert np.isfinite(v["heldout_transfer_from_300"]) and np.isfinite(v["fresh_cv_target"])
        assert set(doc["runs"][run]["summary"]["300"]) >= {"heldout_transfer_from_300", "heldout_minus_insample"}
    # a stored value 1e-6 away withholds everything
    (out / "heldout_transfer.json").unlink()
    insample[450] += 1e-6
    _refs(tmp_path, insample)
    assert ht.main(argv) == 1
    assert not (out / "heldout_transfer.json").exists()


def test_the_overlap_guard_fires_on_shared_worlds(fakes):
    ag, nm = _fake_load(None, ht.DRIFT, 0, "survival", 300)
    P = ht.world_params()
    Xr, yr, wr = ht.pools(ag, nm, P, 6, ht.AUTH_BASE, ht.SURR_BASE)
    Xf, yf, wf = ht.pools(ag, nm, P, 6, 2_700_000, 2_750_000)
    assert not set(wr) & set(wf)
    assert set(wr) == {ht.AUTH_BASE + i for i in range(6)} | {ht.SURR_BASE + i for i in range(6)}
    with pytest.raises(ValueError, match="share"):
        b2.transfer_probe(Xr, yr, Xr, yr, train_worlds=wr, test_worlds=wr)


def test_missing_reference_fails_closed(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        ht.load_references(str(tmp_path / "none.json"), ["wm"])
    (tmp_path / "corrected_l3_h8_wm.json").write_text(json.dumps({"run": "wm", "rows": []}))
    with pytest.raises(SystemExit, match="nowm"):
        ht.load_references(str(tmp_path / "corrected_l3_h8_wm.json"), ["wm", "nowm"])
