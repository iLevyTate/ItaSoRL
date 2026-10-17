"""Closing hook G57: `--tolerance ABS` in scripts/regenerate_state_dumps.py.

A dump rebuilt on another stack can miss the recorded target in the last digits. The default
stays the bit-exact check; with the flag a cell passes when |rebuilt - recorded| <= ABS, the
report says so per cell and records the mode and the stack, and the exit code follows the
tolerance. The pooled readout is monkeypatched to return the recorded target plus an offset.
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
import regenerate_state_dumps as rsd  # noqa: E402

PRE_HOOK = "1bbe7f5"
RECORDED = 0.6123456789


def _pre_hook_module(tmp_path, name="regenerate_state_dumps"):
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


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_FOLDS", "explicit")
    for attr in ("DRIFT_MODE", "OBJECTIVE", "MORTAL"):
        monkeypatch.setattr(b2, attr, getattr(b2, attr))
    monkeypatch.setattr(b2, "setup_l3_surrogate", lambda **kw: None)
    monkeypatch.setattr(b2, "load_agent_bundle", lambda path, device="cpu": (None, None))
    d = tmp_path / "run"
    (d / "agents").mkdir(parents=True)
    (d / "cells").mkdir()
    cell = {"cell": {"agents": {}, "objective": "survival", "mortal": True, "knobs": {}}}
    for arm in ("survival", "untrained"):
        (d / "agents" / f"agent_d0.45_s0_{arm}.pt").write_bytes(b"")
        cell["cell"]["agents"][arm] = {"pool": {"target": RECORDED}}
    (d / "cells" / "cell_d0.45_s0.json").write_text(json.dumps(cell), encoding="utf-8")
    return d


def _offset(monkeypatch, delta):
    monkeypatch.setattr(b2, "pooled_readout", lambda *a, **kw: {"target": RECORDED + delta})


def _run(mod, monkeypatch, run_dir, tmp_path, name, *extra):
    report = tmp_path / f"{name}.json"
    monkeypatch.setattr(sys, "argv", ["regenerate_state_dumps.py", "--run-dir", str(run_dir),
                                      "--out-dir", str(tmp_path / "states"),
                                      "--report", str(report), *extra])
    code = mod.main()
    return code, json.loads(report.read_text(encoding="utf-8"))


def test_offset_of_1e4_fails_by_default_and_passes_at_1e3(run_dir, monkeypatch, tmp_path):
    _offset(monkeypatch, 1e-4)
    code, rep = _run(rsd, monkeypatch, run_dir, tmp_path, "exact")
    assert code == 1 and rep["all_reproduced"] is False and "mode" not in rep
    code, rep = _run(rsd, monkeypatch, run_dir, tmp_path, "tol", "--tolerance", "1e-3")
    assert code == 0
    assert rep["mode"] == "cross_stack_replay" and rep["tolerance"] == 1e-3
    assert rep["all_reproduced"] is False and rep["all_within_tolerance"] is True
    assert rep["within_tolerance"] == rep["n"] == 2
    assert {"python", "numpy", "torch", "sklearn", "scipy"} <= set(rep["stack"])
    for c in rep["cells"]:
        assert c["reproduced_exact"] is False and c["within_tolerance"] is True
        assert c["abs_diff"] == pytest.approx(1e-4, rel=1e-6)
    code, rep = _run(rsd, monkeypatch, run_dir, tmp_path, "tight", "--tolerance", "1e-5")
    assert code == 1 and rep["all_within_tolerance"] is False and rep["within_tolerance"] == 0


def test_a_non_finite_rebuild_never_passes(run_dir, monkeypatch, tmp_path):
    _offset(monkeypatch, float("nan"))
    code, rep = _run(rsd, monkeypatch, run_dir, tmp_path, "nan", "--tolerance", "1.0")
    assert code == 1 and all(c["within_tolerance"] is False for c in rep["cells"])


def test_an_exact_rebuild_passes_both_ways(run_dir, monkeypatch, tmp_path):
    _offset(monkeypatch, 0.0)
    code, rep = _run(rsd, monkeypatch, run_dir, tmp_path, "exact")
    assert code == 0 and rep["all_reproduced"] is True
    code, rep = _run(rsd, monkeypatch, run_dir, tmp_path, "tol", "--tolerance", "0")
    assert code == 0 and all(c["reproduced_exact"] and c["within_tolerance"] for c in rep["cells"])


@pytest.mark.parametrize("delta", [0.0, 1e-4])
def test_default_report_is_the_pre_hook_report_plus_provenance(run_dir, monkeypatch, tmp_path, delta):
    old = _pre_hook_module(tmp_path)
    _offset(monkeypatch, delta)
    code_old, _ = _run(old, monkeypatch, run_dir, tmp_path, "old")
    code_new, rep = _run(rsd, monkeypatch, run_dir, tmp_path, "new")
    assert code_new == code_old == (0 if delta == 0.0 else 1)
    prov = rep.pop("provenance")
    rep["cells"] = [{k: v for k, v in c.items() if k != "seconds"} for c in rep["cells"]]
    ref = json.loads((tmp_path / "old.json").read_text(encoding="utf-8"))
    ref["cells"] = [{k: v for k, v in c.items() if k != "seconds"} for c in ref["cells"]]
    assert json.dumps(rep, indent=1) == json.dumps(ref, indent=1)
    assert "git_commit_at_promotion" in prov and prov["agents_dir"].endswith("run/agents")


def test_tolerance_must_be_finite_and_non_negative(run_dir, monkeypatch, tmp_path):
    for bad in ("-1e-3", "nan"):
        monkeypatch.setattr(sys, "argv", ["regenerate_state_dumps.py", "--run-dir", str(run_dir),
                                          "--out-dir", str(tmp_path / "s"), "--tolerance", bad])
        with pytest.raises(SystemExit):
            rsd.main()
