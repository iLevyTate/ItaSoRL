"""Closing item X04 (E05): `--snapshots` on scripts/run_surprise_readout.py.

Without the flag the script writes what it wrote at main 1bbe7f5 (provenance and wall-clock
fields aside). With it, the snapshot arms resolve to the `_u100`, `_u200` and `_u450` agent
files, are read on the same recorded streams, and the snapshot block is written only when the
b = 300 and b = 0 rows reproduce the stored readout.

The saved agents are replaced by small untrained trunks and the L3 surrogate by a fixed damping
law; the worlds, the recording, the replay and the probes run for real at small n.
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
import run_surprise_readout as rsr  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402

PRE_HOOK = "1bbe7f5"
SMALL = ["--drift", "0.45", "--seeds", "0", "--n-eps", "10", "--steps", "6"]


def _pre_hook_module(tmp_path):
    try:
        src = subprocess.run(["git", "show", f"{PRE_HOOK}:scripts/run_surprise_readout.py"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"git history at {PRE_HOOK} is not available")
    path = tmp_path / "run_surprise_readout_pre_hook.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("run_surprise_readout_pre_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LOADED: list = []


def _fake_bundle(path, device="cpu"):
    """A small untrained trunk per agent file name, the same one on every call."""
    LOADED.append(os.path.basename(path))
    w = make_world(None, 0.0, 5)
    torch.manual_seed(zlib.crc32(os.path.basename(path).encode()))
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(b2, "load_agent_bundle", _fake_bundle)
    monkeypatch.setattr(b2, "DRIFT_MODE", b2.DRIFT_MODE)           # restored after the test
    monkeypatch.setattr(b2, "_L3_GMOTION", lambda vel, a, drag: 0.95 * np.asarray(vel, float))
    LOADED.clear()
    return monkeypatch


def _run(mod, monkeypatch, tmp_path, name, *extra) -> Path:
    out = tmp_path / f"{name}.json"
    monkeypatch.setattr(sys, "argv", ["run_surprise_readout.py", "--agents-dir", str(tmp_path / "agents"),
                                      "--out", str(out), *SMALL, *extra])
    assert mod.main() == 0
    return out


def _strip(doc: dict) -> dict:
    doc = {k: v for k, v in doc.items() if k != "provenance"}
    doc["rows"] = [{k: v for k, v in r.items() if k != "seconds"} for r in doc["rows"]]
    return doc


def test_flag_off_output_is_byte_identical(fakes, tmp_path):
    pre = _pre_hook_module(tmp_path)
    old = json.loads(_run(pre, fakes, tmp_path, "pre").read_text())
    new = json.loads(_run(rsr, fakes, tmp_path, "new").read_text())
    assert json.dumps(_strip(new), sort_keys=False) == json.dumps(_strip(old), sort_keys=False)
    assert set(new["provenance"]) == set(old["provenance"])
    assert "snapshots" not in new


def test_snapshot_arms_resolve_to_the_u_files(fakes, tmp_path):
    paths = rsr.snapshot_paths(str(tmp_path / "agents"), 0.45, 3, [100, 200, 450])
    assert {k: os.path.basename(v) for k, v in paths.items()} == {
        "survival_u100": "agent_d0.45_s3_survival_u100.pt",
        "survival_u200": "agent_d0.45_s3_survival_u200.pt",
        "survival_u450": "agent_d0.45_s3_survival_u450.pt"}
    # end to end: the flag reads those files on the same streams and passes its gate
    (tmp_path / "agents").mkdir()
    for s in (0,):
        for p in rsr.snapshot_paths(str(tmp_path / "agents"), 0.45, s, [100, 200, 450]).values():
            Path(p).write_bytes(b"")
    ref = _run(rsr, fakes, tmp_path, "ref")
    dev = {"rows": [{"seed": 0}],
           "summary": {"aggregate": {f"{b} replay_target": {"per_seed": [0.5 + b / 1e4]}
                                     for b in (0, 100, 200, 300, 450)}}}
    (tmp_path / "dev.json").write_text(json.dumps(dev))
    LOADED.clear()
    out = _run(rsr, fakes, tmp_path, "snap", "--snapshots", "100", "200", "450",
               "--snapshot-ref", str(ref), "--development-ref", str(tmp_path / "dev.json"))
    assert {f"agent_d0.45_s{s}_survival_u{u}.pt" for s in (0,) for u in (100, 200, 450)} <= set(LOADED)
    doc = json.loads(out.read_text())
    blk = doc["snapshots"]
    assert blk["gate"] == "PASS"
    assert blk["budgets"] == [0, 100, 200, 300, 450]
    assert blk["per_budget"]["100"]["arm"] == "survival_u100"
    assert doc["provenance"]["git_commit_at_promotion"]
    assert "tracked_changes_at_generation" in doc["provenance"]
    assert _strip(doc)["summary"] == _strip(json.loads(ref.read_text()))["summary"]
    # a stored row that does not match stops the readout and writes nothing
    bad = json.loads(ref.read_text())
    bad["rows"][0]["own"]["survival"]["S"] += 1e-3
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    fakes.setattr(sys, "argv", ["run_surprise_readout.py", "--agents-dir", str(tmp_path / "agents"),
                                "--out", str(tmp_path / "none.json"), *SMALL, "--snapshots", "100", "200", "450",
                                "--snapshot-ref", str(tmp_path / "bad.json"),
                                "--development-ref", str(tmp_path / "dev.json")])
    assert rsr.main() == 1
    assert not (tmp_path / "none.json").exists()
