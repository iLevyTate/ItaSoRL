"""Closing item X10 (E20 with E21 folded in): full-rank erasure on scripts/run_erasure_readout.py.

On a synthetic state cloud whose world signal spans three dimensions (with noise correlated
inside that subspace, so the rank-two probe block does not span it), the in-fold stop curve stays
near 0.5 where the all-episode curve, whose erasure saw the test folds, dips below it, and k from
the in-fold curve covers the world subspace; the orthogonalized extension reads the rank cleanly where a projector of the
stacked rows does not; and the rank-two default path writes what it wrote at main 1bbe7f5.
The flags then run end to end on small untrained trunks.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import itasorl.experiment_b2 as b2  # noqa: E402
import run_erasure_readout as rer  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402
from itasorl.novel_readouts import probe_block_directions, projector  # noqa: E402

PRE_HOOK = "1bbe7f5"
HID, T = 8, 4


def _cloud(seed: int, n: int = 500):
    M = np.eye(HID)
    M[:3, :3] = [[1.0, 0.9, 0.0], [0.0, 1.0, 0.9], [0.9, 0.0, 1.0]]
    rng = np.random.default_rng(seed)
    Ha = rng.standard_normal((n, T, HID)) @ M
    Hs = rng.standard_normal((n, T, HID)) @ M
    Hs[:, :-1, 0] += 0.8
    Hs[:, -1, 1] += 0.8
    Hs[:, :, 2] += 0.8
    return Ha, Hs


def test_infold_stop_curve_does_not_leak_and_k_covers_the_world_subspace():
    """The all-episode curve fits its erasure on the test folds too and dips below chance; the
    in-fold curve (growth and probe on training rows only) stays near 0.5, and its k is at
    least the world subspace's rank. Ranks stop at 6 of 8 units: a full-rank erasure leaves
    only rounding noise for the scaler to blow up."""
    Ha, Hs = _cloud(1)
    n = len(Ha)
    groups = np.r_[np.arange(n), np.arange(n)]
    infold = rer.infold_curve(Ha, Hs, groups, 6)
    m = 0.5 * (Ha.reshape(-1, HID).mean(0) + Hs.reshape(-1, HID).mean(0))
    B0 = rer.basis_of(projector(probe_block_directions(Ha, Hs), rank=2))
    B, alldata, _ = rer.grow_erasure(Ha, Hs, m, B0, 1, 6, None, groups)
    assert [c["rank"] for c in infold] == [c["rank"] for c in alldata] == [2, 4, 6]
    assert infold[0]["refit_auroc"] > 0.6                               # rank two leaves the world readable
    lo_in = min(c["refit_auroc"] for c in infold)
    lo_all = min(c["refit_auroc"] for c in alldata)
    assert lo_in >= 0.45
    assert lo_all < lo_in
    assert all(c["n_scored"] == 2 * n for c in infold)
    k, reason = rer.choose_k(infold, 0.55)
    assert k >= 3 and reason == "in-fold refit at or under the stop value"
    assert rer.choose_k(infold, 0.3) == (6, "rank cap reached (in-fold refit never at or under the stop value)")
    assert np.allclose(B @ B.T, np.eye(len(B)), atol=1e-10)
    P = B[:4].T @ B[:4]
    assert np.trace(P[:3, :3]) > 2.5                                     # the world subspace is inside


def test_orthogonalized_extension_reads_the_rank_cleanly():
    rng = np.random.default_rng(0)
    B, _ = np.linalg.qr(rng.standard_normal((HID, 3)))
    B = B.T                                                              # 3 orthonormal rows
    # refit directions that lie almost entirely inside the erased subspace
    D = np.vstack([B[0] + 1e-9 * rng.standard_normal(HID), 2.0 * B[1] - B[2] + 1e-9 * rng.standard_normal(HID)])
    grown, added = rer.extend_basis(B, D, room=2)
    assert added == 0 and len(grown) == 3
    stacked = projector(np.vstack([B, D]))                               # does not raise ...
    assert round(float(np.trace(stacked))) == 5                          # ... and reads rank 5
    # a direction with a real component outside is added, orthogonal to the subspace
    e = rng.standard_normal(HID)
    grown, added = rer.extend_basis(B, np.vstack([e]), room=2)
    assert added == 1 and np.allclose(grown @ grown.T, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# the script, on small untrained trunks
# ---------------------------------------------------------------------------

SMALL = ["--seeds", "0", "--arms", "survival", "--n-eps", "10", "--steps", "6", "--pool-n", "24"]


def _pre_hook_module(tmp_path):
    try:
        src = subprocess.run(["git", "show", f"{PRE_HOOK}:scripts/run_erasure_readout.py"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"git history at {PRE_HOOK} is not available")
    path = tmp_path / "run_erasure_readout_pre_hook.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("run_erasure_readout_pre_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_load(agents_dir, drift, seed, arm, updates=None):
    w = make_world(None, 0.0, 5)
    torch.manual_seed(zlib.crc32(f"{drift:.2f}|{seed}|{arm}".encode()))
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=HID).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    return monkeypatch


def _run(mod, mp, tmp_path, name, *extra) -> Path:
    mp.setattr(mod, "load", _fake_load)
    mp.setattr(mod, "setup_l3", lambda: None)
    out = tmp_path / f"{name}.json"
    mp.setattr(sys, "argv", ["run_erasure_readout.py", "--agents-dir", str(tmp_path / "agents"), "--out", str(out),
                             "--ckpt", str(tmp_path / f"{name}.ckpt"), *SMALL, *extra])
    assert mod.main() == 0
    return out


def _strip(doc: dict) -> dict:
    doc = {k: v for k, v in doc.items() if k != "provenance"}
    doc["rows"] = [{k: v for k, v in r.items() if k != "seconds"} for r in doc["rows"]]
    return doc


def test_rank_two_default_path_is_byte_identical(fakes, tmp_path):
    pre = _pre_hook_module(tmp_path)
    old = json.loads(_run(pre, fakes, tmp_path, "pre").read_text())
    new = json.loads(_run(rer, fakes, tmp_path, "new").read_text())
    assert json.dumps(_strip(new)) == json.dumps(_strip(old))
    assert set(new["provenance"]) == set(old["provenance"])
    assert "fullrank" not in new and all("fullrank" not in r for r in new["rows"])


def test_flags_run_end_to_end_and_gate_on_the_rank_two_values(fakes, tmp_path):
    ref = _run(rer, fakes, tmp_path, "ref")
    out = _run(rer, fakes, tmp_path, "full", "--max-rank", "4", "--refit-stop", "0.55", "--swap",
               "--erasure-ref", str(ref))
    doc = json.loads(out.read_text())
    assert doc["provenance"]["git_commit_at_promotion"]
    assert "tracked_changes_at_generation" in doc["provenance"]
    assert doc["fullrank"]["integrity"]["gate_I"] == "PASS"
    row = doc["rows"][0]
    full = row["fullrank"]
    assert 2 <= full["k"] <= 4 and full["refit_curve_infold"][0]["rank"] == 2
    assert full["refit_curve_alldata"]["curve"][0]["rank"] == 2 and "biased low" in full["refit_curve_alldata"]["note"]
    assert full["stop_reason"]
    assert full["swap"]["label"].startswith("class-mean counterfactual")
    assert {k: row[k] for k in rer.INTEGRITY_KEYS if k in row} == \
        {k: json.loads(ref.read_text())["rows"][0][k] for k in rer.INTEGRITY_KEYS if k in row}
    # a stored value off by more than 0.01 stops the readout and writes nothing
    bad = json.loads(ref.read_text())
    bad["rows"][0]["frozen_probe_unerased"] = float(bad["rows"][0].get("frozen_probe_unerased", 0.5)) + 0.02
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    fakes.setattr(sys, "argv", ["run_erasure_readout.py", "--agents-dir", str(tmp_path / "agents"),
                                "--out", str(tmp_path / "none.json"), "--ckpt", str(tmp_path / "full.ckpt"),
                                *SMALL, "--max-rank", "4", "--refit-stop", "0.55", "--swap",
                                "--erasure-ref", str(tmp_path / "bad.json")])
    assert rer.main() == 1
    assert not os.path.exists(tmp_path / "none.json")
