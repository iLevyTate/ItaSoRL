"""The retrain route (`reproduce.py retrain`, scripts/revision/run_corrected_confirmation.sh and
the policy-controlled readout it prints).

A failed corrected-run retrain used to delete the committed C1 and C2 cells and still exit 0;
on Windows `bash` resolved to the WSL launcher, so it always failed; `retrain C1` also
retrained C2; the C2 follow-up omitted --no-world-model and wrote an empty readout with exit
0; and every printed follow-up wrote over a committed artifact or a reviewed figure. These
tests pin each fix.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import reproduce  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "revision", "run_corrected_confirmation.sh")
OUTPUT_FLAGS = ("--out", "--out-dir", "--figure", "--report", "--dump-states", "--json")
FORBIDDEN = ("artifacts/", "docs/figures/")


# ------------------------------------------------------------------ the bash script
_STUB_PYTHON = """#!/bin/bash
# A python stand-in: the preflight passes; run_expB2.py writes N_CELLS cells and the results
# file unless STUB_FAIL=train; the behavior audit writes its JSON unless STUB_FAIL=audit.
case "$1" in -c) exit 0 ;; esac
arg() { local want="$1"; shift; local prev=""; for a in "$@"; do
  [ "$prev" = "$want" ] && { echo "$a"; return; }; prev="$a"; done; }
if [ "$1" = scripts/run_expB2.py ]; then
  [ "${STUB_FAIL:-}" = train ] && exit 1
  out="$(arg --out-dir "$@")"
  mkdir -p "$out/cells"
  i=0
  for d in 0.00 0.45; do for s in 0 1 2 3 4 5 6 7 8 9; do
    i=$((i + 1)); [ "$i" -gt "${N_CELLS:-20}" ] && break
    echo '{"cell": {"new": true}}' > "$out/cells/cell_d${d}_s${s}.json"
  done; done
  echo '{}' > "$out/expB2_results.json"
  exit 0
fi
if [ "$1" = scripts/audit_behavior_mediation.py ]; then
  [ "${STUB_FAIL:-}" = audit ] && exit 1
  echo '{"stub": true}' > "$(arg --json "$@")"
  exit 0
fi
exit 3
"""


@pytest.fixture
def stub_repo(tmp_path):
    """A folder laid out like the repository, holding only the confirmation script, and a bin
    folder whose `python` is the stub above. Skipped where no usable bash exists."""
    bash = reproduce.git_bash()
    if bash is None:
        pytest.skip("no usable bash here")
    repo = tmp_path / "repo"
    (repo / "scripts" / "revision").mkdir(parents=True)
    shutil.copyfile(SCRIPT, repo / "scripts" / "revision" / "run_corrected_confirmation.sh")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    (stub_bin / "python").write_bytes(_STUB_PYTHON.encode("ascii"))
    os.chmod(stub_bin / "python", 0o755)
    promoted = repo / "dest" / "corrected_runs" / "corrected_l3_h8_wm" / "cells"
    promoted.mkdir(parents=True)
    (promoted / "cell_d0.45_s0.json").write_text('{"cell": {"recorded": true}}', encoding="utf-8")

    def run(*args, **env):
        e = dict(os.environ, PATH=str(stub_bin) + os.pathsep + os.environ.get("PATH", ""))
        e.pop("STUB_FAIL", None)
        e.update(env)
        return subprocess.run([bash, "scripts/revision/run_corrected_confirmation.sh", *args],
                              cwd=repo, env=e, capture_output=True, text=True, timeout=300)

    return repo, promoted, run


def _snapshot(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


@pytest.mark.parametrize("env", [{"STUB_FAIL": "train"}, {"STUB_FAIL": "audit"},
                                 {"N_CELLS": "19"}])
def test_a_failed_run_exits_non_zero_and_leaves_the_promoted_cells(stub_repo, env):
    repo, promoted, run = stub_repo
    before = _snapshot(promoted)
    res = run("dest", "1", "wm", **env)
    assert res.returncode != 0, res.stdout + res.stderr
    assert _snapshot(promoted) == before
    leftovers = [p.name for p in promoted.parent.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_a_complete_run_is_promoted_through_a_temporary_folder(stub_repo):
    repo, promoted, run = stub_repo
    res = run("dest", "1", "wm")
    assert res.returncode == 0, res.stdout + res.stderr
    run_dir = promoted.parent
    assert sorted(p.name for p in run_dir.iterdir()) == ["behavior_audit.json", "cells",
                                                         "expB2_results.json"]
    assert len(list(promoted.glob("cell_d*_s*.json"))) == 20
    assert "recorded" not in (promoted / "cell_d0.45_s0.json").read_text(encoding="utf-8")
    assert [p.name for p in run_dir.parent.iterdir()] == ["corrected_l3_h8_wm"]   # wm only
    assert (repo / "dest" / "fullruns" / "corrected_l3_h8_wm" / "cells").is_dir()


def test_the_script_refuses_a_bad_mode_and_the_repository_root(stub_repo):
    repo, promoted, run = stub_repo
    before = _snapshot(promoted)
    assert run("dest", "1", "bogus").returncode == 2
    assert run(".", "1", "wm").returncode == 2
    assert run("artifacts", "1", "wm").returncode == 2
    assert not (repo / "fullruns").exists()
    assert _snapshot(promoted) == before


# ------------------------------------------------------------------ reproduce.py retrain
def _printed(capsys, run, dest="dist/retrain_x"):
    assert reproduce.retrain(run, execute=False, dest=dest) == 0
    return capsys.readouterr().out


def test_c2_follow_ups_carry_no_world_model_and_c1_retrains_c1_only(capsys):
    c2 = _printed(capsys, "C2")
    c1 = _printed(capsys, "C1")
    assert "--no-world-model" in c2
    assert "--no-world-model" not in c1
    assert "run_policy_controlled_readouts.py --run-dir dist/retrain_x/fullruns/corrected_l3_h8_nowm " \
           "--no-world-model" in c2
    assert c1.splitlines()[0] == "bash scripts/revision/run_corrected_confirmation.sh dist/retrain_x 4 wm"
    assert c2.splitlines()[0] == "bash scripts/revision/run_corrected_confirmation.sh dist/retrain_x 4 nowm"


def _outputs(line: str):
    """(flag, value) for every output argument of a printed command."""
    toks = shlex.split(line.split("   #")[0])
    return [(t, toks[i + 1]) for i, t in enumerate(toks[:-1]) if t in OUTPUT_FLAGS]


@pytest.mark.parametrize("run", sorted(reproduce.RETRAIN))
def test_no_printed_command_writes_into_the_committed_record(run):
    dest = reproduce.retrain_dest()
    recipe, follow = reproduce.retrain_lines(run, dest)
    for line in recipe + follow:
        assert "{" not in line and "}" not in line, line            # every field filled in
        if line.startswith("#"):
            continue
        for flag, value in _outputs(line):
            v = value.replace("\\", "/").lstrip("./")
            assert not v.startswith(FORBIDDEN), (run, flag, value)
    for line in follow:
        for flag, value in _outputs(line):
            if flag in ("--out", "--figure"):
                assert value.startswith(f"{dest}/readouts/"), (run, line)


def test_the_follow_ups_keep_the_committed_relative_names():
    _, follow = reproduce.retrain_lines("C1", "D")
    outs = {v for line in follow if not line.startswith("#") for _, v in _outputs(line)}
    for rel in ("l0_audit/corrected_l3_h8_wm.json", "policy_controls/corrected_l3_h8_wm.json",
                "persistence/corrected_l3_h8_wm.json", "control_diagnostics/corrected_l3_h8_wm.json",
                "texture/corrected_l3_h8_wm_gn.json", "texture/corrected_l3_h8_wm_qd.json",
                "population_readout/corrected_l3_h8_wm.json", "budget_curve.json",
                "budget_curve.png", "cross_replay/corrected_c1_c2.json"):
        assert f"D/readouts/{rel}" in outs, rel
    assert "python scripts/build_corrected_verdicts.py --check" in follow


def test_execute_refuses_the_record_and_runs_an_argv_list(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(reproduce.subprocess, "call", lambda argv, **kw: calls.append(argv) or 0)
    for bad in (".", "artifacts/retrain", "docs/figures"):
        assert reproduce.retrain("B1-BALANCED", True, dest=bad) == 1
    assert calls == []
    dest = str(tmp_path / "a b")                        # a space stays inside one argument
    assert reproduce.retrain("B1-BALANCED", True, dest=dest) == 0
    (argv,) = calls                                     # the note line is printed, not run
    assert argv[0] == reproduce.PY and argv[1] == "scripts/run_expB2.py"
    assert argv[argv.index("--out-dir") + 1] == f"{dest}/fullruns/balanced_confirmation_wm"
    assert argv[argv.index("--device") + 1] == "cuda"


def test_execute_resolves_bash_and_preflights_torch(monkeypatch):
    calls = []
    monkeypatch.setattr(reproduce.subprocess, "call", lambda argv, **kw: calls.append(argv) or 0)
    monkeypatch.setattr(reproduce, "git_bash", lambda: None)
    assert reproduce.retrain("C1", True, dest="dist/x") == 1 and calls == []
    monkeypatch.setattr(reproduce, "git_bash", lambda: "/stub/bash")
    monkeypatch.setattr(reproduce, "bash_preflight", lambda b: "no torch")
    assert reproduce.retrain("C1", True, dest="dist/x") == 1 and calls == []
    monkeypatch.setattr(reproduce, "bash_preflight", lambda b: None)
    assert reproduce.retrain("C2", True, dest="dist/x") == 0
    assert calls == [["/stub/bash", "scripts/revision/run_corrected_confirmation.sh", "dist/x",
                      "4", "nowm"]]


@pytest.mark.skipif(os.name != "nt", reason="the WSL and Store launchers exist only on Windows")
def test_git_bash_rejects_the_wsl_and_store_launchers(monkeypatch, tmp_path):
    for launcher in (r"C:\Windows\System32\bash.exe",
                     str(tmp_path / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "bash.exe")):
        monkeypatch.setattr(reproduce.shutil, "which", lambda name, p=launcher: p)
        monkeypatch.setattr(reproduce.os.path, "isfile", lambda p: True)
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "none"))
        assert reproduce.git_bash() == str(tmp_path / "none" / "Git" / "bin" / "bash.exe")
        monkeypatch.undo()
    git = tmp_path / "Git" / "bin" / "bash.exe"
    monkeypatch.setattr(reproduce.shutil, "which", lambda name: r"C:\Windows\System32\bash.exe")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert reproduce.git_bash() is None                 # neither found: refuse
    git.parent.mkdir(parents=True)
    git.write_bytes(b"")
    assert reproduce.git_bash() == str(git)


# ------------------------------------------------------------------ policy-controlled readout
def _policy_run_dir(tmp_path):
    run = tmp_path / "run"
    (run / "cells").mkdir(parents=True)
    for s in (0, 1):
        (run / "cells" / f"cell_d0.45_s{s}.json").write_text(
            json.dumps({"cell": {"gae_bootstrap": "successor"}}), encoding="utf-8")
    (run / "expB2_results.json").write_text(json.dumps({"0.0": {}, "0.45": {}}), encoding="utf-8")
    return run


@pytest.mark.parametrize("ok", [False, True])
def test_policy_controlled_main_fails_when_every_seed_is_excluded(tmp_path, monkeypatch, ok):
    import run_policy_controlled_readouts as pcr

    def fake_seed(task):
        if not ok:
            return {"seed": task["seed"], "drift": task["drift"], "retrain_identical": False,
                    "targets": {}, "error": "retrained survival agent differs from the saved one"}
        return {"seed": task["seed"], "drift": task["drift"], "retrain_identical": True,
                "targets": {g: {p: {"target": 0.6 + 0.01 * task["seed"]} for p in pcr.PROTOCOLS}
                            for g in pcr.ARMS}}

    run = _policy_run_dir(tmp_path)
    out = tmp_path / "readouts" / "pc.json"
    monkeypatch.setattr(pcr, "run_seed", fake_seed)
    monkeypatch.setattr(sys, "argv", ["x", "--run-dir", str(run), "--out", str(out)])
    assert pcr.main() == (0 if ok else 1)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["seeds_excluded"]) == (0 if ok else 2)
    assert bool(doc["aggregate"]) is ok
