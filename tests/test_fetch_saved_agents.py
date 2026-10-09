"""scripts/fetch_saved_agents.py resolves the gitignored saved agents for a readout-only rerun.

The agents live outside the repository (a Drive folder, or an archive from the Zenodo
record), so the Colab cell needs one function that finds them wherever they are, extracts
an archive if given one, and refuses files whose sha256 does not match the committed
artifact. These tests build small fake agent files and archives; nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import fetch_saved_agents as fsa  # noqa: E402

NAMES = [f"agent_d0.45_s{i}_survival.pt" for i in range(3)]


def _write_agents(d, names=NAMES, salt=b""):
    d.mkdir(parents=True, exist_ok=True)
    shas = {}
    for n in names:
        p = d / n
        p.write_bytes(b"weights:" + n.encode() + salt)
        shas[n] = hashlib.sha256(p.read_bytes()).hexdigest()
    return shas


def _artifact(tmp_path, shas):
    """A minimal copy of the sensitivity artifact: cells[].seed and agent_sha256."""
    cells = [{"seed": int(n.split("_s")[1].split("_")[0]), "arm": "survival",
              "agent_sha256": sha} for n, sha in shas.items()]
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"drift": 0.45, "cells": cells}), encoding="utf-8")
    return p


def test_finds_the_folder_that_holds_the_agents_even_when_nested(tmp_path):
    shas = _write_agents(tmp_path / "deep" / "fullruns" / "l3_h8_heldout" / "agents")
    found = fsa.find_agents_dir(tmp_path, drift=0.45, arm="survival")
    assert found == tmp_path / "deep" / "fullruns" / "l3_h8_heldout" / "agents"
    assert fsa.find_agents_dir(tmp_path / "empty", drift=0.45, arm="survival") is None
    del shas


def test_extracts_a_zip_or_a_tarball_before_searching(tmp_path):
    src = tmp_path / "src"
    _write_agents(src / "agents")
    z = tmp_path / "agents.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for n in NAMES:
            zf.write(src / "agents" / n, f"l3_h8_heldout/agents/{n}")
    t = tmp_path / "agents.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        tf.add(src / "agents", arcname="agents")
    for archive in (z, t):
        out = fsa.extract_archive(archive, tmp_path / ("x_" + archive.name))
        found = fsa.find_agents_dir(out, drift=0.45, arm="survival")
        assert found is not None and sorted(p.name for p in found.iterdir()) == NAMES


def test_verifies_every_agent_against_the_committed_sha256(tmp_path):
    d = tmp_path / "agents"
    shas = _write_agents(d)
    ref = _artifact(tmp_path, shas)
    report = fsa.verify_agents(d, ref, drift=0.45, arm="survival")
    assert report["ok"] and report["n_checked"] == 3 and report["mismatched"] == []

    (d / NAMES[1]).write_bytes(b"tampered")
    report = fsa.verify_agents(d, ref, drift=0.45, arm="survival")
    assert not report["ok"] and report["mismatched"] == [NAMES[1]]


def test_a_missing_seed_is_reported_not_silently_skipped(tmp_path):
    d = tmp_path / "agents"
    shas = _write_agents(d)
    ref = _artifact(tmp_path, shas)
    (d / NAMES[2]).unlink()
    report = fsa.verify_agents(d, ref, drift=0.45, arm="survival")
    assert not report["ok"] and report["missing"] == [NAMES[2]]


def test_resolve_prefers_an_existing_dir_then_a_drive_folder_then_the_url(tmp_path, monkeypatch):
    local = tmp_path / "local"
    _write_agents(local)
    assert fsa.resolve(local_dir=local, drive_dir=None, url=None, work=tmp_path / "w") == local

    drive = tmp_path / "drive" / "agents"
    _write_agents(drive)
    assert fsa.resolve(local_dir=tmp_path / "nope", drive_dir=tmp_path / "drive", url=None,
                       work=tmp_path / "w") == drive

    calls = []

    def fake_download(url, dest):
        calls.append(url)
        with zipfile.ZipFile(dest, "w") as zf:
            for n in NAMES:
                zf.writestr(f"agents/{n}", b"weights:" + n.encode())
        return dest

    monkeypatch.setattr(fsa, "download", fake_download)
    got = fsa.resolve(local_dir=tmp_path / "nope", drive_dir=tmp_path / "nodrive",
                      url="https://zenodo.example/agents.zip", work=tmp_path / "w2")
    assert calls == ["https://zenodo.example/agents.zip"]
    assert got is not None and sorted(p.name for p in got.iterdir()) == NAMES


def test_resolve_with_nothing_available_says_what_to_do(tmp_path):
    with pytest.raises(SystemExit) as e:
        fsa.resolve(local_dir=tmp_path / "a", drive_dir=tmp_path / "b", url="", work=tmp_path / "w")
    assert "Zenodo" in str(e.value) and "Drive" in str(e.value)


def test_the_folder_with_the_most_agents_wins_over_a_stray_copy_nearer_the_root(tmp_path):
    # A loose copy of one file beside the archive root must not shadow the full folder.
    (tmp_path / NAMES[0]).write_bytes(b"stray")
    _write_agents(tmp_path / "pack" / "agents")
    found = fsa.find_agents_dir(tmp_path, drift=0.45, arm="survival")
    assert found == tmp_path / "pack" / "agents"
