"""Shared test helper: is the repository root a git checkout?

Provenance blocks record the commit a readout ran at (`git_commit_at_promotion`, or `git_commit`
for the runners). Inside an extracted supplement there is no git, so the field is written as None
by design (runbook Stage 11, G50). Tests check the key name everywhere and its value only here."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def in_git_checkout() -> bool:
    """True when ROOT is the top of a git checkout (an extracted archive placed inside some
    other repository is still an extracted archive)."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(top) and os.path.normcase(os.path.realpath(top)) == \
        os.path.normcase(os.path.realpath(ROOT))


def assert_commit_recorded(prov: dict, key: str = "git_commit_at_promotion") -> None:
    """The provenance block names the commit under `key` (and not under the other spelling);
    the value must be non-empty in a git checkout and may be None in an extracted archive."""
    other = "git_commit" if key == "git_commit_at_promotion" else "git_commit_at_promotion"
    assert key in prov and other not in prov, sorted(prov)
    if in_git_checkout():
        assert prov[key], f"{key} is empty in a git checkout"
