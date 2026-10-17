"""Closing hook A7e: `--jitter-key {pool,agent}` in scripts/run_texture_fresh_probe.py.

Sweep C (family arc) reseeds the jitter family at every surrogate world. The recorded run keys
those seeds by the pool only (2551000 + k for the k-th world, whatever the agent seed), so all
agents share one realization. The default must stay that sequence; `agent` gives each agent
seed its own block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_texture_fresh_probe as rt  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
BASE = rt.BASES["arc"][1]


class _Fam:
    def __init__(self):
        self.seeds = []

    def reseed(self, seed):
        self.seeds.append(seed)


def _draws(n, **kw):
    fam = _Fam()
    with rt.per_episode_draw(fam, BASE, **kw) as make_world:
        for _ in range(n):
            make_world(P, 0.0, 4)        # authentic worlds never reseed
            make_world(P, 0.45, 4)
    return fam.seeds


@pytest.mark.parametrize("agent_seed", [0, 3, 9])
def test_default_sequence_is_2551000_plus_k_for_any_seed(agent_seed):
    want = [2_551_000 + k for k in range(4)]
    assert _draws(4, agent_seed=agent_seed) == want
    assert _draws(4, key="pool", agent_seed=agent_seed) == want
    assert _draws(4) == want


def test_agent_key_gives_each_seed_its_own_disjoint_block():
    blocks = {s: _draws(5, key="agent", agent_seed=s) for s in range(10)}
    assert blocks[0] == [2_552_000 + k for k in range(5)]
    assert blocks[9] == [2_561_000 + k for k in range(5)]
    seen = [x for b in blocks.values() for x in b]
    assert len(seen) == len(set(seen))
    assert not set(seen) & set(range(2_551_000, 2_552_000))     # the recorded pool block
    assert rt.jitter_seed(BASE, rt.AGENT_STRIDE - 1, "agent", 0) < rt.jitter_seed(BASE, 0, "agent", 1)


def test_bad_key_and_block_overrun_are_refused():
    with pytest.raises(ValueError, match="jitter key"):
        rt.jitter_seed(BASE, 0, "episode")
    with pytest.raises(ValueError, match="overruns"):
        rt.jitter_seed(BASE, rt.AGENT_STRIDE, "agent", 0)
    with pytest.raises(ValueError, match="jitter key"):
        with rt.per_episode_draw(_Fam(), BASE, key="episode"):
            pass


def test_agent_key_is_refused_outside_family_arc():
    with pytest.raises(SystemExit, match="arc only"):
        rt.main(["--agents-dir", "x", "--family", "ar", "--param", "0.01", "--rho", "0.5",
                 "--out", "y", "--jitter-key", "agent"])


@pytest.mark.parametrize("key", ["pool", "agent"])
def test_output_keys_default_unchanged_and_agent_labelled(tmp_path, monkeypatch, key):
    agents = tmp_path / "agents"
    agents.mkdir()
    for s in (0, 1):
        (agents / f"agent_d0.45_s{s}_survival.pt").write_bytes(b"")
    tasks = []

    def fake_run_one(task):
        tasks.append(task)
        return {"drift": task["drift"], "seed": task["seed"], "arm": task["arm"],
                "family": task["family"], "param": task["param"], "rho": task["rho"],
                "transfer": 0.5, "fresh": 0.5 + task["seed"] / 10, "fresh_paired": 0.5,
                "n_comparator": [3, 3], "seconds": 0.0}

    monkeypatch.setattr(rt, "run_one", fake_run_one)
    out = tmp_path / "C.json"
    argv = ["--agents-dir", str(agents), "--family", "arc", "--param", "0.0076", "--rho", "0.99",
            "--out", str(out), "--jitter-key", key]
    assert rt.main(argv) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    recorded = json.loads((ROOT / "artifacts/texture/ar_sweep_c/C_rho0.99.json").read_text(
        encoding="utf-8"))
    prov = {"git_commit_at_promotion", "tracked_changes_at_generation", "stack", "argv"}
    extra = {"jitter_key"} if key == "agent" else set()
    assert set(doc) == set(recorded) | prov | extra
    assert list(doc)[:len(recorded)] == list(recorded)
    assert {t["jitter_key"] for t in tasks} == {key}
    assert doc.get("jitter_key", "pool") == key
    assert doc["stack"]["device"] == "cpu" and doc["argv"][-2:] == ["--jitter-key", key]
