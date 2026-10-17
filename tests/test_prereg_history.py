"""PREREG_HISTORY/: the full text history of every preregistration and spec in the supplement.

PROVENANCE.md lists when each frozen document was committed, but not what changed, so a
reader could not check a pre-registration claim from the archive. prereg_history() adds one
patch file per document from `git log -p --follow --cc`, and the supplement passes it through
the same scrub, pseudonym and number checks as every other text file. Every fixture below that
looks like a name, an address, a path or a commit ID is assembled at runtime, so the scrubber
does not rewrite this file inside the archive.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import reproduce  # noqa: E402

NAME = "Quill" + "worthy"                       # an author name the history must not keep
MAIL = "q.writer" + "@" + "example" + ".org"
DOC = "docs/PREREGISTRATION_X.md"
SUBJECT = "subject" + "marker"                # in every commit message
BRANCH = "qx" + "-topic"


def _git(d, *args, when=None):
    env = dict(os.environ)
    if when:
        env.update(GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    env.update(GIT_AUTHOR_NAME=NAME, GIT_AUTHOR_EMAIL=MAIL, GIT_COMMITTER_NAME=NAME,
               GIT_COMMITTER_EMAIL=MAIL)
    res = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
                          *args], cwd=d, env=env, check=True, capture_output=True, text=True)
    return res.stdout.strip()


@pytest.fixture
def history(tmp_path):
    """A small repository: the document is added under another path, moved, edited on a side
    branch and on main (one edit names the author and a local path, later removed), merged,
    and committed with a +02:00 offset throughout."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    d = tmp_path / "repo"
    d.mkdir()
    n = [0]

    def commit(msg, files):
        n[0] += 1
        for path, text in files.items():
            p = d / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode("utf-8"))
            _git(d, "add", path)
        _git(d, "commit", "-q", "-m", msg + " " + SUBJECT, when=f"2026-01-01T02:{n[0]:02d}:00+02:00")

    profile = "C:/" + "Us" + "ers/" + "qw" + "ill/notes"
    _git(d, "init", "-q")
    _git(d, "symbolic-ref", "HEAD", "refs/heads/main")
    (d / "CITATION.cff").write_text(f'authors:\n  - family-names: "{NAME}"\n', encoding="utf-8")
    commit("add", {"PREREGISTRATION_X.md": "# Plan\n\nbar 0.65\n", "CITATION.cff":
                   (d / "CITATION.cff").read_text(encoding="utf-8")})
    (d / "docs").mkdir()
    _git(d, "mv", "PREREGISTRATION_X.md", DOC)
    commit("move", {})
    commit("note", {DOC: f"# Plan\n\nbar 0.65\nwritten by {NAME} in {profile}\n"})
    _git(d, "checkout", "-q", "-b", BRANCH)
    commit("topic", {DOC: f"# Plan\n\nbar 0.65\nwritten by {NAME} in {profile}\nside 7\n"})
    _git(d, "checkout", "-q", "main")
    commit("main", {DOC: "# Plan\n\nbar 0.70\nwritten in a shared folder\n"})
    n[0] += 1
    try:
        _git(d, "merge", "-q", "--no-ff", "--no-edit", BRANCH, when=f"2026-01-01T02:{n[0]:02d}:00+02:00")
    except subprocess.CalledProcessError:
        (d / DOC).write_bytes(b"# Plan\n\nbar 0.70\nwritten in a shared folder\nside 7\n")
        _git(d, "add", DOC)
        _git(d, "commit", "-q", "--no-edit", when=f"2026-01-01T02:{n[0]:02d}:00+02:00")
    return d, profile


def _commits(d):
    full = _git(d, "log", "--all", "--format=%H").split()
    blobs = _git(d, "log", "--all", "-p", "--raw", "--no-abbrev", "--format=")
    blob_ids = set(re.findall(r"\b[0-9a-f]{40}\b", blobs)) - {"0" * 40}
    return full, blob_ids


def test_one_header_per_commit_in_utc_without_offsets_index_lines_or_prefixes(history, monkeypatch):
    d, profile = history
    monkeypatch.setattr(reproduce, "ROOT", str(d))
    patch = reproduce.prereg_history(DOC)
    rows, why = reproduce.provenance_rows([DOC])
    assert patch and why is None
    (_, commits, _), = rows
    headers = [ln for ln in patch.splitlines() if ln.startswith("commit ")]
    for when, h, note in commits:
        if note != "merge":
            assert f"commit {h} {when} UTC" in headers, (h, when)
    # oldest first, and every time is the UTC reading of a +02:00 commit
    times = [ln.split()[2] + " " + ln.split()[3] for ln in headers]
    assert times == sorted(times) and times[0] == "2026-01-01 00:01"
    lines = patch.splitlines()
    assert not any(ln.startswith("index ") for ln in lines)
    assert not any(re.search(r"[+-]\d{4}\s*$", ln) for ln in headers)          # no offsets
    assert "+0200" not in patch and "+02:00" not in patch
    for ln in lines:
        if ln.startswith(("diff --git ", "--- ", "+++ ")):
            assert " a/" not in ln and " b/" not in ln, ln
    assert MAIL not in patch and SUBJECT not in patch                          # no log fields
    full, blobs = _commits(d)
    for c in full:
        assert c not in patch
    for b in blobs:
        assert b[:7] not in patch, b
    assert NAME in patch and profile in patch           # the text itself; the scrub runs later


def test_no_history_outside_a_checkout_or_in_a_shallow_clone(history, monkeypatch, tmp_path):
    d, _ = history
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", (d.as_uri()), str(shallow)],
                   check=True, capture_output=True)
    monkeypatch.setattr(reproduce, "ROOT", str(shallow))
    assert reproduce.prereg_history(DOC) is None
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path / "nowhere"))
    assert reproduce.prereg_history(DOC) is None


def test_the_supplement_ships_each_history_scrubbed_and_pseudonymized(history, monkeypatch, tmp_path):
    d, profile = history
    monkeypatch.setattr(reproduce, "ROOT", str(d))
    monkeypatch.setattr(reproduce, "INCLUDE_DIRS", [])
    monkeypatch.setattr(reproduce, "DOC_DIRS", ["docs"])
    monkeypatch.setattr(reproduce, "INCLUDE_FILES", ["CITATION.cff"])
    monkeypatch.setattr(reproduce, "RUN_DIRS", [])
    monkeypatch.setattr(reproduce, "_pr_heads", lambda: "")
    out = tmp_path / "out" / "supp.zip"
    rc = reproduce.supplement(str(out), anon_name="blind" + "pkg", key_path=str(tmp_path / "k.txt"),
                              paper=str(tmp_path / "no-paper"))
    assert rc == 0 and out.exists()
    arc = f"{reproduce.PREREG_HISTORY_DIR}/{DOC}.patch"
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert arc in names
        patch = z.read(arc).decode("utf-8")
        prov = z.read(reproduce.PROVENANCE_NAME).decode("utf-8")
        readme = z.read(reproduce.README_NAME).decode("utf-8")
        sums = z.read("SHA256SUMS").decode("utf-8")
    assert f"  {arc}" in sums
    assert f"`{reproduce.PREREG_HISTORY_DIR}/`" in readme and "(1 files)" in readme
    # the scrub ran: no author name, address or local path, in the removed lines either
    assert NAME not in patch and MAIL not in patch and profile not in patch
    assert "<author>" in patch and "<local-path>" in patch
    # the pseudonyms ran: no commit of the repository, full or short, survives
    full, blobs = _commits(d)
    for c in full:
        assert c[:7] not in patch and c[:7] not in prov
    for b in blobs:
        assert b[:7] not in patch
    # every non-merge commit PROVENANCE.md lists has its header, under the same pseudonym
    rows = re.findall(r"^\| \d+ \| (\S+ \S+) \| ([0-9a-f]+) \| ([^|]*)\|$", prov, flags=re.M)
    assert rows
    for when, h, note in rows:
        if note.strip() != "merge":
            assert f"commit {h} {when} UTC" in patch, (h, when)
    # and the leak scan of the archive, which the build already ran, still passes
    rules = reproduce.scrub_rules([])
    assert reproduce.leak_scan(str(out), rules, None, titles=[], branches=[]) == []


def test_the_readme_names_the_histories_only_when_they_ship():
    assert "PREREG_HISTORY/" in reproduce._readme([], "blind" + "pkg", histories=4)
    assert "PREREG_HISTORY" not in reproduce._readme([], "blind" + "pkg")
