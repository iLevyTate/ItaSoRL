"""Closing item G46 (runbook 1.4): every runner the closing pass touches records provenance.

The block comes from scripts/agent_free_common.provenance: the generator, a commit, the tracked
changes at generation and the stack, plus argv (and agents_dir for a readout of saved agents).
The commit key is fixed per output kind ([R2-4c]): an agent-free computation IS the run and
records `git_commit`; a readout of saved agents or stored dumps records
`git_commit_at_promotion`, because scripts/build_results_manifest.py reads `git_commit` as the
commit a run executed at. The block is a new key only, so every number a runner writes must be
what the pre-hook script (main 1bbe7f5) wrote.

Three checks:
1. Every provenance() call in each runner uses the commit key of its output kind.
2. End to end, with the agents and rollouts replaced by deterministic fakes, each runner below
   writes the provenance keys and the pre-hook numbers. Runners whose end-to-end check lives in
   their hook's own test file are listed in COVERED_ELSEWHERE, and that test must exist.
3. The provenance helpers of the runners wired by the other closing item (run_expB_* through
   expb_common, run_l1_h2_ablations, run_texture_fresh_probe, validate_population_readout)
   write the right keys.
Later closing items add their new scripts to RUNNERS.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

PRE_HOOK = "1bbe7f5"
AT_RUN, AT_PROMOTION = "git_commit", "git_commit_at_promotion"

#: script -> the commit key of its output kind
RUNNERS = {
    # closing item agent D (this file's own runners)
    "scripts/audit_sensory_echo.py": AT_PROMOTION,
    "scripts/run_world_features_readout.py": AT_PROMOTION,
    "scripts/run_mechanism_readouts.py": AT_PROMOTION,
    "scripts/regenerate_state_dumps.py": AT_PROMOTION,
    "scripts/reverse_direction.py": AT_PROMOTION,
    "scripts/run_world_sample_sensitivity.py": AT_PROMOTION,
    "scripts/run_expA_l1.py": AT_RUN,
    "scripts/run_expA_l3.py": AT_RUN,
    # wired by the other 1.4 item
    "scripts/expb_common.py": AT_RUN,          # every run_expB_* writes through it
    "scripts/run_l1_h2_ablations.py": AT_PROMOTION,
    "scripts/run_texture_fresh_probe.py": AT_PROMOTION,
    "scripts/validate_population_readout.py": AT_PROMOTION,
    # the 1.6 new scripts (closing items B2, B6, H53, G01, G04): readouts of saved agents or stored dumps
    "scripts/run_walk_speed_readout.py": AT_PROMOTION,
    "scripts/run_development_heldout_transfer.py": AT_PROMOTION,
    "scripts/run_l1_nonlinear_probe.py": AT_PROMOTION,
    "scripts/run_probe_penalty_sensitivity.py": AT_PROMOTION,
    "scripts/run_step_resolved_readout.py": AT_PROMOTION,
    # the 1.6b extension readouts (Stage 5X, X01 to X10); X08 and X09 are agent-free
    "scripts/run_nudge_dose_readout.py": AT_PROMOTION,
    "scripts/run_development_fixed_walk.py": AT_PROMOTION,
    "scripts/run_head_overlap_readout.py": AT_PROMOTION,
    # scripts/run_surprise_readout.py is not listed: its default output keeps the older
    # optional_readouts.provenance block, and the 1.4 block (git_commit_at_promotion) is written
    # under --snapshots only, through an alias; tests/test_surprise_snapshots.py checks it.
    "scripts/run_mdl_probe.py": AT_PROMOTION,
    "scripts/run_pellet_dose_readout.py": AT_PROMOTION,
    "scripts/read_budget_curve_dynamics.py": AT_RUN,
    "scripts/build_bayes_factors.py": AT_RUN,
    "scripts/run_erasure_readout.py": AT_PROMOTION,        # written in full-rank mode only
}

#: runner -> (test file, test) holding its end-to-end provenance and byte-identity check
COVERED_ELSEWHERE = {
    "scripts/audit_sensory_echo.py": ("tests/test_sensory_echo_integrity_base.py",
                                      "test_no_flag_output_is_byte_identical_to_the_pre_hook_script"),
    "scripts/run_world_features_readout.py": ("tests/test_world_features_protocol.py",
                                              "test_replay_default_is_unchanged_against_the_pre_hook_script"),
    "scripts/run_mechanism_readouts.py": ("tests/test_mechanism_ungrouped_surprise.py",
                                          "test_default_is_byte_identical_and_the_flag_only_adds_the_key"),
    "scripts/regenerate_state_dumps.py": ("tests/test_regenerate_dumps_tolerance.py",
                                          "test_default_report_is_the_pre_hook_report_plus_provenance"),
    "scripts/run_expA_l3.py": ("tests/test_expa_l3_qd_eps.py",
                               "test_payload_carries_provenance_and_keeps_its_numbers"),
    # new scripts: no pre-hook version exists, so their smoke tests check the written block
    "scripts/run_walk_speed_readout.py": ("tests/test_walk_speed_readout.py",
                                          "test_smoke_writes_every_walk_for_every_arm"),
    "scripts/run_development_heldout_transfer.py": ("tests/test_development_heldout_transfer.py",
                                                    "test_smoke_integrity_and_heldout_rows"),
    "scripts/run_l1_nonlinear_probe.py": ("tests/test_l1_nonlinear_probe.py",
                                          "test_smoke_scores_every_class_after_the_gate"),
    "scripts/run_probe_penalty_sensitivity.py": ("tests/test_probe_penalty_sensitivity.py",
                                                 "test_cell_target_dir_and_regenerated_dir"),
    "scripts/run_step_resolved_readout.py": ("tests/test_step_resolved_readout.py", "test_smoke_and_void"),
}


def _base_keys(commit_key: str) -> set:
    return {commit_key, "tracked_changes_at_generation", "stack", "argv"}


def _check_block(prov: dict, commit_key: str, *, agents: bool, generated_by: str | None = None):
    other = AT_PROMOTION if commit_key == AT_RUN else AT_RUN
    assert _base_keys(commit_key) <= set(prov), sorted(prov)
    assert other not in prov
    assert isinstance(prov["tracked_changes_at_generation"], (list, type(None)))
    assert {"python", "numpy", "scipy", "sklearn", "torch"} <= set(prov["stack"])
    assert isinstance(prov["argv"], list)
    if agents:
        assert "agents_dir" in prov and "\\" not in prov["agents_dir"]
    if generated_by is not None:
        assert prov["generated_by"] == generated_by


def _pre_hook_module(tmp_path, script: str):
    """The script as it stood at main 1bbe7f5, loaded from git; skip outside a full clone."""
    try:
        src = subprocess.run(["git", "show", f"{PRE_HOOK}:{script}"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"git history at {PRE_HOOK} is not available")
    name = Path(script).stem + "_pre_hook"
    path = tmp_path / f"{name}.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. commit key per output kind, at every provenance() call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script,commit_key", sorted(RUNNERS.items()))
def test_every_provenance_call_uses_the_commit_key_of_its_output_kind(script, commit_key):
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and ast.unparse(c.func).split(".")[-1] == "provenance"
             and any(k.arg == "commit_key" for k in c.keywords)]
    assert calls, f"{script} records no provenance() block with a commit key"
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert isinstance(kw["commit_key"], ast.Constant) and kw["commit_key"].value == commit_key


@pytest.mark.parametrize("script", sorted(COVERED_ELSEWHERE))
def test_end_to_end_check_lives_in_the_hook_test(script):
    path, name = COVERED_ELSEWHERE[script]
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    assert name in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


# ---------------------------------------------------------------------------
# 2. end to end, against the pre-hook script
# ---------------------------------------------------------------------------
def _agents(tmp_path, drift: str, seeds, arms) -> Path:
    d = tmp_path / "agents"
    d.mkdir(exist_ok=True)
    for s in seeds:
        for arm in arms:
            (d / f"agent_d{drift}_s{s}_{arm}.pt").write_bytes(f"{drift}{s}{arm}".encode())
    return d


def _fake_target(agent, drift, seed, base=0) -> float:
    arm_shift = {"survival": 0.12, "predictor": 0.05, "untrained": 0.0}[agent]
    return round(0.5 + arm_shift + 0.01 * seed + 0.1 * float(drift) + 1e-7 * (base % 997), 9)


def _reference(tmp_path, cells) -> Path:
    from itasorl import folds
    scheme = folds.current_scheme()
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"cells": [{"drift": str(d), "agent": arm, "seed": s,
                                          scheme: {"target": t}} for d, arm, s, t in cells]}),
                   encoding="utf-8")
    return ref


def _strip(doc: dict, *timing) -> dict:
    return {k: v for k, v in doc.items() if k != "provenance" and k not in timing}


def _common(monkeypatch, mod):
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "DRIFT_MODE", b2.DRIFT_MODE)
    monkeypatch.setattr(b2, "setup_l3_surrogate", lambda **kw: None)
    monkeypatch.setattr(mod, "load_agent_bundle",
                        lambda path, dev="cpu": (Path(path).stem.split("_")[-1], None))


def _run(mod, monkeypatch, argv, out: Path) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", argv)
    code = mod.main()
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_reverse_direction(monkeypatch, tmp_path):
    script = "scripts/reverse_direction.py"
    old, new = _pre_hook_module(tmp_path, script), importlib.import_module("reverse_direction")
    arms = ("survival", "predictor", "untrained")
    agents = _agents(tmp_path, "0.00", (0, 1, 2), arms)
    ref = _reference(tmp_path, [(0.0, a, s, _fake_target(a, 0.0, s)) for a in arms for s in (0, 1, 2)])
    for mod in (old, new):
        _common(monkeypatch, mod)
        monkeypatch.setattr(mod, "pooled_readout", lambda agent, norm, params, drift_sigma, seed, **kw:
                            {"target": _fake_target(agent, drift_sigma, seed)})
        monkeypatch.setattr(mod, "paired_pooled_readout", lambda agent, norm, params, drift, seed, **kw:
                            {"target": _fake_target(agent, drift, seed) - 0.01})
    docs = []
    for mod, name in ((old, "old"), (new, "new")):
        out = tmp_path / f"{name}.json"
        docs.append(_run(mod, monkeypatch, ["reverse_direction.py", "--agents-dir", str(agents),
                                            "--out", str(out), "--integrity-ref", str(ref)], out))
    (code_old, ref_doc), (code_new, doc) = docs
    assert code_old == code_new == 0 and doc["integrity"]["pass"] is True
    _check_block(doc["provenance"], RUNNERS[script], agents=True, generated_by=script)
    assert doc["git_commit"] == ref_doc["git_commit"]           # the historical key stays
    assert _strip(doc, "wall_seconds") == _strip(ref_doc, "wall_seconds")


def test_run_world_sample_sensitivity(monkeypatch, tmp_path):
    script = "scripts/run_world_sample_sensitivity.py"
    old = _pre_hook_module(tmp_path, script)
    new = importlib.import_module("run_world_sample_sensitivity")
    agents = _agents(tmp_path, "0.45", (0, 1, 2), ("survival",))
    ref = _reference(tmp_path, [(0.45, "survival", s, _fake_target("survival", 0.45, s, 800_000))
                                for s in (0, 1, 2)])
    for mod in (old, new):
        _common(monkeypatch, mod)
        monkeypatch.setattr(mod, "world_sample_scan", lambda agent, norm, params, drift, bases, seed, **kw:
                            [{"target": _fake_target(agent, drift, seed, b[0]), "bases": list(b)}
                             for b in bases])
    docs = []
    for mod, name in ((old, "old"), (new, "new")):
        out = tmp_path / f"{name}.json"
        docs.append(_run(mod, monkeypatch, ["run_world_sample_sensitivity.py", "--agents-dir",
                                            str(agents), "--out", str(out), "--integrity-ref",
                                            str(ref)], out))
    (code_old, ref_doc), (code_new, doc) = docs
    assert code_old == code_new == 0 and doc["integrity"]["pass"] is True
    _check_block(doc["provenance"], RUNNERS[script], agents=True, generated_by=script)
    assert _strip(doc, "wall_seconds") == _strip(ref_doc, "wall_seconds")


def test_run_expA_l1(monkeypatch, tmp_path):
    script = "scripts/run_expA_l1.py"
    old, new = _pre_hook_module(tmp_path, script), importlib.import_module("run_expA_l1")
    small = ["--n-pairs", "10", "--steps", "6", "--deltas", "0.05", "0.02"]
    for mod, name in ((old, "old"), (new, "new")):
        monkeypatch.setattr(sys, "argv", ["run_expA_l1.py", "--out", str(tmp_path / f"{name}.json"),
                                          *small])
        mod.main()
    doc = json.loads((tmp_path / "new.json").read_text(encoding="utf-8"))
    _check_block(doc.pop("provenance"), RUNNERS[script], agents=False, generated_by=script)
    assert json.dumps(doc, indent=2) == (tmp_path / "old.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. the other 1.4 item's provenance helpers
# ---------------------------------------------------------------------------
def test_expb_common_writes_the_run_commit(tmp_path):
    expb_common = importlib.import_module("expb_common")
    path = expb_common.write_out_json(str(tmp_path), "expB_full", "legacy", {"n": 1},
                                      {"auroc": 0.5}, argv=["--out-dir", str(tmp_path)])
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_block(doc["provenance"], AT_RUN, agents=False, generated_by="scripts/run_expB_full.py")
    assert doc["auroc"] == 0.5


def test_l1_h2_ablations_writes_provenance_json(tmp_path):
    rh = importlib.import_module("run_l1_h2_ablations")
    a = Namespace(out_dir=str(tmp_path), device_used="cpu", agents_dir=str(tmp_path / "agents"),
                  states_dir=str(tmp_path / "states"), noise_json=None, channels=["noise"],
                  arms=["survival"], noise_reseed="per-pool")
    path = rh.write_provenance(a, argv=["--out-dir", str(tmp_path)])
    prov = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_block(prov, AT_PROMOTION, agents=True, generated_by="scripts/run_l1_h2_ablations.py")


def test_texture_fresh_probe_block(tmp_path):
    tf = importlib.import_module("run_texture_fresh_probe")
    _check_block(tf.provenance_block(str(tmp_path / "o.json"), argv=["--quick"]), AT_PROMOTION,
                 agents=False)


def test_validate_population_readout_block(tmp_path):
    vp = importlib.import_module("validate_population_readout")
    prov = vp.provenance_block(str(tmp_path / "o.json"), str(tmp_path / "run"), argv=["--quick"])
    _check_block(prov, AT_PROMOTION, agents=True)
