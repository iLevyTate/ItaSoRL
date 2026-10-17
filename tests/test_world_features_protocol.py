"""Closing hook G07 and H24: `--protocol {replay,own}` in scripts/run_world_features_readout.py.

replay (the default, the registered protocol) must write what the pre-hook script wrote, the
provenance block's added keys and the timing fields aside. own (the spec's secondary
protocol, post hoc) reads every arm on its own episodes, so each arm has its own episode
count, and the output carries every recorded episode's feature values (the spec's Output
clause at docs/specs/2026-10-15-world-draw-features-design.md:95).

The saved agents are replaced by small untrained trunks; the worlds, features, decoders and
the script's checkpoint and summary plumbing run for real at small n.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_world_features_readout as rwf  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, make_world  # noqa: E402
from itasorl.novel_readouts import INCIDENTAL, INSTRUMENTAL  # noqa: E402

PRE_HOOK = "1bbe7f5"
SMALL = ["--seeds", "0", "1", "--n-eps", "40", "--n-calib", "12", "--steps", "8"]


def _pre_hook_module(tmp_path, name="run_world_features_readout"):
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


def _fake_load(agents_dir, drift, seed, arm, updates=None):
    """A small untrained trunk per (drift, seed, arm), the same one on every call."""
    w = make_world(rwf.world_params(), 0.0, 5)
    torch.manual_seed(zlib.crc32(f"{drift:.2f}|{seed}|{arm}".encode()))
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


def _patch(monkeypatch, mod):
    monkeypatch.setattr(mod, "load", _fake_load)
    monkeypatch.setattr(mod, "setup_l3", lambda: None)


def _run(mod, monkeypatch, tmp_path, name, *extra):
    out = tmp_path / f"{name}.json"
    monkeypatch.setattr(sys, "argv", ["run_world_features_readout.py", "--agents-dir",
                                      str(tmp_path / "agents"), "--out", str(out), *SMALL, *extra])
    assert mod.main() == 0
    return out


def _strip(doc: dict) -> dict:
    """Drop the provenance block and the wall-clock fields, which differ run to run."""
    doc = {k: v for k, v in doc.items() if k != "provenance"}
    doc["rows"] = [{k: v for k, v in r.items() if k != "seconds"} for r in doc["rows"]]
    return doc


def test_replay_default_is_unchanged_against_the_pre_hook_script(monkeypatch, tmp_path):
    old = _pre_hook_module(tmp_path)
    _patch(monkeypatch, old)
    _patch(monkeypatch, rwf)
    old_doc = json.loads(_run(old, monkeypatch, tmp_path, "old").read_text(encoding="utf-8"))
    new_doc = json.loads(_run(rwf, monkeypatch, tmp_path, "new").read_text(encoding="utf-8"))
    assert json.dumps(_strip(new_doc), indent=1) == json.dumps(_strip(old_doc), indent=1)
    assert "protocol" not in new_doc and "protocol" not in new_doc["provenance"]["inputs"]
    # every provenance key the pre-hook script wrote keeps its value, the stack included
    op, np_ = old_doc["provenance"], new_doc["provenance"]
    for k, v in op.items():
        if k == "stack":
            assert all(np_["stack"][s] == x for s, x in v.items())
        elif k == "command":     # the two runs differ only in the output file name
            assert np_[k].replace("new.json", "o.json") == v.replace("old.json", "o.json")
        else:
            assert np_[k] == v, k
    assert "git_commit_at_promotion" in np_ and "tracked_changes_at_generation" in np_
    assert np_["agents_dir"] == new_doc["provenance"]["inputs"]["agents_dir"]
    assert not any(r.get("protocol") for r in new_doc["rows"])


def test_own_protocol_gives_per_arm_n_and_per_episode_features(monkeypatch, tmp_path):
    _patch(monkeypatch, rwf)
    doc = json.loads(_run(rwf, monkeypatch, tmp_path, "own", "--protocol", "own")
                     .read_text(encoding="utf-8"))
    assert doc["protocol"].startswith("own") and doc["provenance"]["inputs"]["protocol"] == "own"
    arms = [f"{g}_d{d:.2f}" for g, d in rwf.ARMS]
    feats = INSTRUMENTAL + INCIDENTAL
    for row in doc["rows"]:
        assert row["protocol"] == "own" and sorted(row["per_arm"]) == sorted(arms)
        for arm in arms:
            pa = row["per_arm"][arm]
            assert isinstance(pa["n"], int) and 0 < pa["n"] <= pa["n_recorded"] <= 40
            assert len(pa["episode_index"]) == len(pa["episode_ok"]) == pa["n_recorded"]
            assert sum(pa["episode_ok"]) == pa["n"]
            assert sorted(pa["episode_features"]) == sorted(feats)
            assert all(len(v) == pa["n_recorded"] for v in pa["episode_features"].values())
            assert sorted(pa["echo_r2"]) == sorted(feats)
            assert sorted(row["arms"][arm]) == sorted(feats)
        head = row["per_arm"]["survival_d0.45"]
        assert row["n"] == head["n"] and row["echo_r2"] == head["echo_r2"]
        assert all("r2_permuted" in row["arms"]["survival_d0.45"][k] for k in feats)
        # each arm's senses are its own episodes: the echo differs between arms
        assert row["per_arm"]["predictor_d0.45"]["echo_r2"] != head["echo_r2"]
    assert doc["readings"]["label"] in ("follows usefulness", "keeps everything",
                                        "keeps nothing beyond the senses", "mixed")


def test_own_checkpoints_never_mix_with_replay(monkeypatch, tmp_path):
    """The own tasks carry the protocol and live in their own checkpoint dir; the replay task
    dict (and its fingerprint) is the pre-hook one."""
    _patch(monkeypatch, rwf)
    _run(rwf, monkeypatch, tmp_path, "both")
    ck = tmp_path / "both.json.ckpt"
    replay = json.loads((ck / "read" / "s0.json").read_text(encoding="utf-8"))
    assert "protocol" not in replay["_task"]
    _run(rwf, monkeypatch, tmp_path, "both", "--protocol", "own")
    own = json.loads((ck / "read_own" / "s0.json").read_text(encoding="utf-8"))
    assert own["_task"]["protocol"] == "own"
