"""Closing hook G22 (B5): `--also-ungrouped-surprise` in scripts/run_mechanism_readouts.py.

B5 rescores the surprise AUROC with twin-pair groups and ties the rerun to the stored values,
which used one CV group per episode. The flag writes that ungrouped estimate as
`surprise.auroc_ungrouped` from the same rollouts. Without the flag the output is the pre-hook
script's byte for byte (the new provenance key and the wall-clock fields aside); with it, the
only change is the added key in every arm's surprise block.

Runs end to end at --quick scale on a tiny run directory of untrained world-model agents, with
a cheaply trained L3 velocity net installed in place of the registered one.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import itasorl.experiment_b2 as b2  # noqa: E402
import run_mechanism_readouts as rmr  # noqa: E402
from itasorl import mechanism_readouts as mr  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.surrogate_l3 import train_g_motion  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

PRE_HOOK = "1bbe7f5"
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
ARMS = ("survival", "untrained", "predictor")


def _pre_hook_module(tmp_path, name="run_mechanism_readouts"):
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


@pytest.fixture(scope="module")
def g_small():
    return train_g_motion(hidden=4, n_eps=6, steps=8, epochs=3, params=P)


@pytest.fixture
def run_dir(tmp_path, monkeypatch, g_small):
    for attr in ("DRIFT_MODE", "OBJECTIVE", "MORTAL"):
        monkeypatch.setattr(b2, attr, getattr(b2, attr))
    monkeypatch.setattr(b2, "_L3_GMOTION", g_small)
    d = tmp_path / "run"
    probe = b2.make_world(P, 0.0, 5)
    for i, arm in enumerate(ARMS):
        torch.manual_seed(i)
        agent = RecurrentActorCritic(probe.obs_spec.size, probe.action_spec.size, 16, 8, True)
        norm = b2.RunningNorm(probe.obs_spec.size).freeze()
        b2.save_agent_bundle(str(d / "agents" / f"agent_d0.45_s0_{arm}.pt"), agent.train(False), norm)
    (d / "cells").mkdir()
    for drift, xe in (("0.00", {"0.00": 1.0, "0.45": 0.4}), ("0.45", {"0.00": 0.7, "0.45": 0.9})):
        cell = {"cell": {"drift": float(drift), "seed": 0, "xeval": xe, "objective": "survival",
                         "mortal": True, "knobs": {}}}
        (d / "cells" / f"cell_d{drift}_s0.json").write_text(json.dumps(cell), encoding="utf-8")
    return d


def _run(mod, monkeypatch, run_dir, out, *extra):
    monkeypatch.setattr(sys, "argv", ["run_mechanism_readouts.py", "--run-dir", str(run_dir),
                                      "--out", str(out), "--quick", *extra])
    assert mod.main() == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _strip(doc: dict) -> dict:
    doc = {k: v for k, v in copy.deepcopy(doc).items() if k != "provenance"}
    for r in doc["per_seed"]:
        r.pop("seconds")
    return doc


def _blocks(row: dict):
    yield row
    for g in ("untrained", "predictor"):
        yield row[f"{g}_arm"]


def test_default_is_byte_identical_and_the_flag_only_adds_the_key(run_dir, monkeypatch, tmp_path):
    old = _pre_hook_module(tmp_path)
    ref = _run(old, monkeypatch, run_dir, tmp_path / "old.json")
    new = _run(rmr, monkeypatch, run_dir, tmp_path / "new.json")
    assert json.dumps(_strip(new), indent=1) == json.dumps(_strip(ref), indent=1)
    assert all("auroc_ungrouped" not in b["surprise"] for r in new["per_seed"] for b in _blocks(r))
    flagged = _run(rmr, monkeypatch, run_dir, tmp_path / "flag.json", "--also-ungrouped-surprise")
    for r in flagged["per_seed"]:
        for b in _blocks(r):
            v = b["surprise"].pop("auroc_ungrouped")
            assert v is None or isinstance(v, float)
            assert math.isnan(v) or 0.0 <= v <= 1.0
    assert json.dumps(_strip(flagged), indent=1) == json.dumps(_strip(ref), indent=1)
    assert "git_commit_at_promotion" in new["provenance"]
    assert new["provenance"]["agents_dir"].endswith("run/agents")


def test_ungrouped_surprise_is_the_one_group_per_episode_estimate():
    rng = np.random.default_rng(4)
    Ea = rng.uniform(0.0, 1.0, size=(44, 12))
    Es = Ea + 1e-3 * rng.normal(size=Ea.shape)
    assert rmr.ungrouped_surprise(Ea, Es) == mr.surprise_auroc(Ea, Es)
    assert rmr.ungrouped_surprise(Ea, Es) != mr.surprise_auroc(
        Ea, Es, groups=np.r_[np.arange(44), np.arange(44)])
