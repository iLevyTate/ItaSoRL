"""The results manifest (revision step 1) must cover every committed artifact exactly once,
and must mark every survival-trained run with the trainer that produced it."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_results_manifest as brm  # noqa: E402


def test_every_artifact_has_exactly_one_owner():
    for rel in brm._rel_files():
        assert len(brm._owner(rel)) == 1, rel


def test_manifest_check_passes_on_the_committed_tree():
    assert brm.check(brm.build()) == []


def test_survival_runs_name_their_trainer():
    for r in brm.RUNS:
        if r["experiment"] in ("A", "B", "C"):
            assert r["survival_trainer"] == "none", r["id"]
        elif r["readout_only_on"]:
            assert r["survival_trainer"].startswith("inherited: "), r["id"]
            assert r["affected_by_gae_correction"], r["id"]
        else:
            assert r["survival_trainer"] in ("pre_transition_value", "successor_value"), r["id"]
            assert r["affected_by_gae_correction"], r["id"]


def test_check_flags_an_unowned_artifact(monkeypatch):
    monkeypatch.setattr(brm, "_rel_files", lambda: ["expB2/not_in_any_run.json"])
    errs = brm.check(brm.build())
    assert any("belongs to no run" in e for e in errs)
