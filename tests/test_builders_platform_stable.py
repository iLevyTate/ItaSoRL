"""The generated artifacts must be byte-stable across platforms (floats rounded before dumping),
the results manifest must honor .gitignore when it lists artifacts, provenance must name every
commit that lives off main, and the readout scripts must stamp the commit they ran at
(revision audit items 5 and 7)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_contrast_intervals as bci  # noqa: E402
import build_gate_table as bgt  # noqa: E402
import build_results_manifest as brm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _floats(x):
    if isinstance(x, float):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _floats(v)
    elif isinstance(x, list):
        for v in x:
            yield from _floats(v)


def test_gate_table_json_is_rounded_to_six_decimals():
    doc = json.loads(bgt.dump_json(bgt.build()))
    assert all(round(f, 6) == f for f in _floats(doc))


def test_contrast_intervals_json_is_rounded_to_six_decimals():
    doc = json.loads(bci.dump_json(bci.build()))
    assert all(round(f, 6) == f for f in _floats(doc))


def test_committed_gate_table_and_contrast_intervals_are_current():
    assert bgt.main(["--check"]) == 0
    assert bci.main(["--check"]) == 0


def test_manifest_file_list_skips_gitignored_files_but_sees_untracked_ones():
    ignored = os.path.join(ROOT, "artifacts", "clip_audit", "_probe_ignored.json")
    unowned = os.path.join(ROOT, "artifacts", "expB2", "_probe_unowned.json")
    os.makedirs(os.path.dirname(ignored), exist_ok=True)
    try:
        for p in (ignored, unowned):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("{}\n")
        rel = brm._rel_files()
        assert "clip_audit/_probe_ignored.json" not in rel
        assert "expB2/_probe_unowned.json" in rel
    finally:
        for p in (ignored, unowned):
            if os.path.exists(p):
                os.remove(p)
        try:
            os.rmdir(os.path.dirname(ignored))
        except OSError:
            pass


def test_manifest_file_list_falls_back_to_the_filesystem_without_git(monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(brm.subprocess, "run", no_git)
    rel = brm._rel_files()
    assert "gate_table.json" in rel


def test_provenance_lists_every_commit_that_lives_off_main():
    off = brm.build()["commits_off_main"]
    assert isinstance(off, list) and all({"branch", "commits", "note"} <= set(g) for g in off)
    listed = {c for g in off for c in g["commits"]}
    assert {"4d57253", "f676b95", "34e2c0d", "7870bba", "2606e7f", "06ecc10"} <= listed
    md = brm.render_md(brm.build())
    for c in ("f676b95", "06ecc10", "claude/affectionate-carson-azhxt2"):
        assert c in md


# The scripts that run a readout on saved agents. The two deterministic builders
# (build_corrected_verdicts.py, build_budget_curve.py) are deliberately NOT stamped: their
# output is compared byte for byte against the committed artifact by --check, and a commit
# field would make that comparison depend on where it runs.
READOUT_SCRIPTS = ["run_l0_audit.py", "run_policy_controlled_readouts.py", "run_persistence_readout.py",
                   "run_control_diagnostics.py", "run_texture_fresh_probe.py",
                   "validate_population_readout.py", "run_cross_replay.py"]


def test_readout_scripts_record_the_commit_they_ran_at():
    for name in READOUT_SCRIPTS:
        src = open(os.path.join(ROOT, "scripts", name), encoding="utf-8").read()
        assert '"git_commit": git_head()' in src, name
    for name in ("build_corrected_verdicts.py", "build_budget_curve.py"):
        src = open(os.path.join(ROOT, "scripts", name), encoding="utf-8").read()
        assert "git_head()" not in src, name


def test_git_head_helper_returns_the_short_hash():
    from itasorl.results_io import git_head
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    assert git_head() == head
