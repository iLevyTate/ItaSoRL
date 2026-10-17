"""Closing item H31: the quadratic-drag oracle reading at eps = 6.0 runs through the existing
scripts/run_expA_l3.py CLI, whose --json payload now carries the 1.4 provenance block.

The calibration is agent free and is the run, so the commit key is git_commit. The parser
must accept `--family qd --qd-epss 4.0 6.0 8.0` with no argparse change, and the payload's
numbers must be the pre-hook script's. The surrogate candidates, the oracle and the organism
floor are replaced by deterministic fakes; the script's selection and writing run for real.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import itasorl.experiment_b2 as b2  # noqa: E402
import itasorl.surrogate_l3_families as fam  # noqa: E402
import run_expA_l3 as l3  # noqa: E402

PRE_HOOK = "1bbe7f5"
QD_ARGS = ["--family", "qd", "--qd-epss", "4.0", "6.0", "8.0"]
PROV_KEYS = {"generated_by", "git_commit", "tracked_changes_at_generation", "stack", "argv"}


def _pre_hook_module(tmp_path, name="run_expA_l3"):
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


def test_parser_accepts_the_qd_eps_sweep(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_expA_l3.py", *QD_ARGS, "--json", "x.json"])
    a = l3.cfg()
    assert a.family == "qd" and a.qd_epss == [4.0, 6.0, 8.0] and a.json == "x.json"


def _patch(monkeypatch, mod, seen):
    def candidates(family, params=None, sweep=None):
        seen.append((family, list(sweep)))
        return (((("eps", e), f"g{e}") for e in sweep))

    monkeypatch.setattr(fam, "gate0_candidates", candidates)
    monkeypatch.setattr(b2, "DRIFT_MODE", b2.DRIFT_MODE)
    monkeypatch.setattr(b2, "_L3_GMOTION", b2._L3_GMOTION)
    monkeypatch.setattr(mod, "generate_l3_pairs", lambda g, **kw: g)
    monkeypatch.setattr(mod, "run_experiment_a_l3", lambda eps, **kw: {
        "oracle_auroc": {"g4.0": 0.97, "g6.0": 0.91, "g8.0": 0.83}[eps],
        "leakage_pass": True, "reward_leak": 0.52})
    monkeypatch.setattr(mod, "untrained_agent", lambda *a, seed=0, **kw: (seed, None))
    monkeypatch.setattr(mod, "pooled_readout", lambda agent, *a, **kw: {
        "target": 0.48 + 0.01 * agent, "pool_reward_leak": 0.5})


def _run(mod, monkeypatch, out):
    monkeypatch.setattr(sys, "argv", ["run_expA_l3.py", *QD_ARGS, "--device", "cpu",
                                      "--json", str(out)])
    code = mod.main()
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_payload_carries_provenance_and_keeps_its_numbers(monkeypatch, tmp_path):
    old = _pre_hook_module(tmp_path)
    seen = []
    _patch(monkeypatch, old, seen)
    code_old, ref = _run(old, monkeypatch, tmp_path / "old.json")
    _patch(monkeypatch, l3, seen)
    code_new, doc = _run(l3, monkeypatch, tmp_path / "new.json")
    assert seen == [("qd", [4.0, 6.0, 8.0])] * 2
    assert code_new == code_old == 0
    prov = doc.pop("provenance")
    assert json.dumps(doc, indent=2) == (tmp_path / "old.json").read_text(encoding="utf-8")
    assert [r["eps"] for r in doc["rows"]] == [4.0, 6.0, 8.0]
    assert doc["selected"] == {"family": "qd", "eps": 6.0}
    assert PROV_KEYS <= set(prov) and "git_commit_at_promotion" not in prov
    assert prov["generated_by"] == "scripts/run_expA_l3.py"
    assert prov["stack"]["device"] == "cpu"
    assert prov["argv"][:6] == QD_ARGS
