"""Committed files record repo-relative paths, never a machine-local user profile path.

Provenance fields such as source_run, states_dir and bundle once stored absolute Windows
paths that carried the owner's account name and resolved on no other machine. The promoters
now write `fullruns/<run>/...` through promote_bv3_gates.repo_rel, and this scan keeps a
profile path from coming back. The pattern is assembled from pieces so this file does not
match itself.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_bv3_gates as pb  # noqa: E402

_PROFILE = "Use" + "rs"
USER_PATH = re.compile(r"(?:[A-Za-z]:[/\\]+|/(?:c|mnt/c)/)" + _PROFILE + r"[/\\]+[^/\\\s\"'<>]+")

# Not scanned: files other work owns or that are generated from the ones below.
SKIP = {"artifacts/corrected_verdicts.json", "artifacts/surrogate_diagnostics.json",
        "artifacts/results_manifest.json", "docs/RESULTS_MANIFEST.md"}
SKIP_PREFIXES = ("artifacts/goal_stakes/",)
# A dated audit record that quotes such a path while describing it; it stays as written.
ALLOW = {"docs/REVISION_AUDIT_2026-10-07.md"}
# source_run is a field the results manifest copies from each artifact, so rewriting it here
# would leave the manifest stale. These four change together with a manifest rebuild.
PENDING_MANIFEST_FOLLOWUP = {
    "artifacts/expB2/sensory_echo_l3_h7.json",
    "artifacts/expB2/sensory_echo_l3_h8.json",
    "artifacts/expB2/sensory_echo_l3_h8_mlp.json",
    "artifacts/expC/emergence_pilot_per_individual_summary.json",
}


def _tracked_files() -> list[str] | None:
    try:
        res = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [n for n in res.stdout.decode("utf-8", errors="replace").split("\0") if n]


def _offenders() -> set[str]:
    files = _tracked_files()
    if files is None:
        pytest.skip("needs a git checkout (an extracted supplement is scrubbed at build time)")
    out = set()
    for rel in files:
        if rel in SKIP or rel in ALLOW or rel.startswith(SKIP_PREFIXES):
            continue
        path = ROOT / rel
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue                                   # binary
        if USER_PATH.search(data.decode("utf-8", errors="replace")):
            out.add(rel)
    return out


def test_no_tracked_file_records_a_user_profile_path():
    found = _offenders()
    new = sorted(found - PENDING_MANIFEST_FOLLOWUP)
    assert not new, f"absolute user-profile paths in {new}; record them with repo_rel"
    # Strict: once a pending file is fixed (with the manifest rebuild), drop it from the list.
    assert found == PENDING_MANIFEST_FOLLOWUP, sorted(PENDING_MANIFEST_FOLLOWUP - found)


def test_the_pattern_catches_the_forms_it_must():
    for bad in ("C:/" + _PROFILE + "/someone/x", "c:\\\\" + _PROFILE + "\\\\someone\\\\x",
                "/c/" + _PROFILE + "/someone/x", "/mnt/c/" + _PROFILE + "/someone/x"):
        assert USER_PATH.search(bad), bad
    for ok in ("fullruns/l3_h8_heldout/states", "<local-path>/x", "/home/user/ItaSoRL/fullruns/r"):
        assert not USER_PATH.search(ok), ok


@pytest.mark.parametrize("given,want", [
    ("fullruns/07062026", "fullruns/07062026"),
    ("C:/x/y/fullruns/07062026", "fullruns/07062026"),
    ("C:\\" + _PROFILE + "\\someone\\ITASORL\\fullruns\\07062026\\states", "fullruns/07062026/states"),
    ("/home/user/ItaSoRL/fullruns/corrected_l3_h8_wm/agents", "fullruns/corrected_l3_h8_wm/agents"),
    ("D:\\bundles\\run7", "D:/bundles/run7"),
    ("artifacts/corrected_runs/corrected_l3_h8_wm", "artifacts/corrected_runs/corrected_l3_h8_wm"),
    ("", ""),
])
def test_repo_rel(given, want):
    assert pb.repo_rel(given) == want


def test_repo_rel_makes_a_path_inside_the_checkout_relative():
    assert pb.repo_rel(str(ROOT / "artifacts" / "expB2" / "x.json")) == "artifacts/expB2/x.json"


def test_rel_flags_rewrites_only_absolute_path_arguments():
    flags = ["--dump-states", "C:\\" + _PROFILE + "\\someone\\ITASORL\\fullruns\\07062026\\states",
             "--seeds", "0", "--drift-mode", "regime"]
    assert pb.rel_flags(flags) == ["--dump-states", "fullruns/07062026/states",
                                   "--seeds", "0", "--drift-mode", "regime"]
    assert pb.rel_flags(None) is None


@pytest.mark.parametrize("script", [
    "promote_cg_rescore", "promote_mp_rescore", "promote_sensory_echo", "promote_expC_summary",
    "promote_heldout_artifact", "promote_reviewer_gaps_runs", "promote_texture_trained",
])
def test_promoters_record_paths_through_repo_rel(script):
    src = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert "repo_rel(" in src
    # the old rule only swapped separators and kept the machine-local prefix
    assert not re.search(r"\"(source_run|states_dir|bundle|source_rescore|compare_run)\": [^,\n]*\.replace\(", src)
