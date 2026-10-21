"""Packaging fixes of the 2026-10-16 critical review: `reproduce.py tables` reports stale shipped
pages in its exit code, the review profile of the supplement stays under the venue limit and
says what it leaves out, commit IDs ship as same-length pseudonyms, and the manuscript check
compares the counts the prose states about the rule ledger and the gate table."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_paper_tables as bpt  # noqa: E402
import reproduce  # noqa: E402


# ------------------------------------------------------------------ reproduce.py tables
def _fake_run(stale: bool, calls: list):
    """A stale page fails each builder's --check and, as in the real audit, which runs those
    checks itself, the audit of the pages as shipped. A rewrite makes the pages current, so the
    audit after a rewrite passes."""
    rewritten = []

    def run(cmd, **kw):
        calls.append(cmd)
        if "build_" in cmd[1]:
            if "--check" in cmd:
                return 1 if stale and not rewritten else 0
            rewritten.append(cmd)
            return 0
        return 1 if stale and not rewritten and "audit_stats_recheck" in cmd[1] else 0
    return run


def test_tables_exits_nonzero_when_the_shipped_pages_were_stale(monkeypatch):
    """The rewrite used to restore a stale page and the audit then passed on the rewritten
    copy, so the documented step exited 0 on tampered tables."""
    calls: list = []
    monkeypatch.setattr(reproduce, "_run", _fake_run(True, calls))
    assert reproduce.tables([]) == reproduce.STALE_EXIT != 0
    assert any("--check" not in c and "build_" in c[1] for c in calls)     # still rewrites
    calls.clear()
    monkeypatch.setattr(reproduce, "_run", _fake_run(False, calls))
    assert reproduce.tables([]) == 0


def test_tables_check_rewrites_nothing_and_runs_the_audit(monkeypatch):
    """The audit fails on a stale page too, and --check still exits 3, not the audit's 1."""
    calls: list = []
    monkeypatch.setattr(reproduce, "_run", _fake_run(True, calls))
    assert reproduce.tables([], check_only=True) == reproduce.STALE_EXIT
    builders = [c for c in calls if "build_" in c[1]]
    assert builders and all("--check" in c for c in builders)
    assert any("audit_stats_recheck" in c[1] for c in calls)
    calls.clear()
    monkeypatch.setattr(reproduce, "_run", _fake_run(False, calls))
    assert reproduce.main(["tables", "--check"]) == 0
    assert all("--check" in c for c in calls if "build_" in c[1])


def test_a_failing_audit_keeps_its_own_exit_code(monkeypatch):
    monkeypatch.setattr(reproduce, "_run",
                        lambda cmd, **kw: 1 if "audit_stats_recheck" in cmd[1] or "--check" in cmd else 0)
    assert reproduce.tables([]) == 1
    # With current pages, a failing audit is reported as itself under --check as well.
    monkeypatch.setattr(reproduce, "_run", lambda cmd, **kw: 1 if "audit_stats_recheck" in cmd[1] else 0)
    assert reproduce.tables([], check_only=True) == 1


# ------------------------------------------------------------------ commit pseudonyms
# Synthetic commit IDs, assembled at runtime so no literal here is a hex token.
def _ids(n: int) -> list[str]:
    import hashlib
    return sorted(hashlib.sha1(f"commit {i}".encode()).hexdigest() for i in range(n))


def test_commit_pseudonyms_keep_length_prefixes_and_order():
    commits = _ids(50)
    h = reproduce.CommitPseudonyms(commits, b"k" * 32)
    a, b = commits[3], commits[40]
    fp = "f" + "0" * 15                 # a fingerprint-like token that is no commit's prefix
    assert not h.resolves(fp)
    text = f"run {a} and {a[:7]} at {b[:10]}; fingerprint {fp}; sum {'a' * 64}"
    out = h.sub(text)
    pa, pa7, pb10 = out.split()[1], out.split()[3], out.split()[5].rstrip(";")
    assert len(pa) == 40 and len(pa7) == 7 and len(pb10) == 10
    assert pa.startswith(pa7) and pa != a and pa7 != a[:7]
    assert fp in out and "a" * 64 in out
    assert h.search(out) is None and h.search(text) is not None
    # Pseudonyms sort as the commits do, so a sorted list stays sorted in the archive.
    full = [h.full[c] for c in commits]
    assert full == sorted(full) and len({p[:7] for p in full}) == len(full)
    assert h.count() == 2 and h.mapping[pa7] == a[:7]
    # The same key gives the same pseudonyms; another key gives others.
    assert reproduce.CommitPseudonyms(commits, b"k" * 32).sub(text) == out
    assert reproduce.CommitPseudonyms(commits, b"j" * 32).sub(text) != out


def test_commit_pseudonyms_do_nothing_without_a_history():
    h = reproduce.CommitPseudonyms([], b"")
    t = "commit " + "ab" * 20
    assert h.sub(t) == t and h.search(t) is None


def test_the_commit_key_is_made_once_and_reused(tmp_path):
    p = str(tmp_path / "k" / "key.txt")
    k = reproduce.commit_key(p)
    assert len(k) == 32 and reproduce.commit_key(p) == k


# ------------------------------------------------------------------ the review profile
def _in_checkout() -> bool:
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(top) and os.path.normcase(os.path.realpath(top)) == \
        os.path.normcase(os.path.realpath(ROOT))


AGENT = b"weights \x00\x01"


def _paper(tmp_path) -> str:
    """A stand-in manuscript source, so the review build's title check has a title to read."""
    d = tmp_path / "paper"
    d.mkdir(exist_ok=True)
    (d / "main.tex").write_text("\\title{Quiet Words In A Row: Another Line Of Words}\n",
                                encoding="utf-8")
    return str(d)


def _review_build(tmp_path, monkeypatch, limit: int, agent: bytes = AGENT, paper: str | None = "make",
                  files=("pyproject.toml", "pytest.ini")):
    runs = tmp_path / "runs"
    for b in ("review_a/agents", "review_b/agents", "full_only"):
        (runs / b).mkdir(parents=True)
        (runs / b / "agent_s0.pt").write_bytes(agent)
    commits = reproduce.commit_ids()
    sha = hashlib.sha256(agent).hexdigest()
    note = (f'{{"git_commit": "{commits[-1]}", "short": "{commits[-1][:7]}", '
            f'"upper": "{commits[-1][:7].upper()}", "agent_sha256": "{sha}", '
            f'"mean": -9.765625e-05}}\n' if commits else "{}\n")
    (runs / "review_a" / "agents" / "note.json").write_text(note, encoding="utf-8")
    old = reproduce.project_name()
    monkeypatch.setattr(reproduce, "RUN_DIRS", ["fullruns/review_a/agents", "fullruns/review_b/agents",
                                                "fullruns/full_only"])
    monkeypatch.setattr(reproduce, "REVIEW_RUN_DIRS", ["fullruns/review_a/agents",
                                                       "fullruns/review_b/agents"])
    monkeypatch.setattr(reproduce, "INCLUDE_DIRS", [old])
    monkeypatch.setattr(reproduce, "INCLUDE_FILES", list(files))
    monkeypatch.setattr(reproduce, "REVIEW_LIMIT", limit)
    out = tmp_path / "review.zip"
    rc = reproduce.supplement(str(out), runs_root=str(runs), anon_name="blind" + "pkg",
                              profile="review", key_path=str(tmp_path / "key" / "k.txt"),
                              paper=_paper(tmp_path) if paper == "make" else paper)
    return rc, out, commits


def _skip_or_fail_on_gap(gap: str) -> None:
    """Skip when the review build cannot see the whole history. In CI the checkout is a full
    clone (ci.yml sets fetch-depth 0), so a shallow or partial clone there is a broken CI
    setup and fails instead of hiding the leak guards behind a skip. A network failure or a
    missing branch tip still skips: history_gap needs the commits on old remote branches,
    which a later branch prune can remove."""
    if os.environ.get("CI") == "true" and ("shallow clone" in gap or "partial clone" in gap):
        pytest.fail(f"CI checkout lacks the history the review build needs: {gap}")
    pytest.skip(f"the review build needs a full checkout: {gap}")


def _review_needs_history(rc, out, commits) -> None:
    """In a shallow clone or outside a checkout the review build refuses to start."""
    gap = reproduce.history_gap(commits)
    if gap:
        assert rc == 1 and not out.exists()
        _skip_or_fail_on_gap(gap)


def test_the_review_profile_ships_only_its_bundles_and_says_what_it_leaves_out(tmp_path, monkeypatch):
    rc, out, commits = _review_build(tmp_path, monkeypatch, reproduce.REVIEW_LIMIT)
    _review_needs_history(rc, out, commits)
    assert rc == 0
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "fullruns/review_a/agents/agent_s0.pt" in names
        assert "fullruns/review_b/agents/agent_s0.pt" in names
        assert not any(n.startswith("fullruns/full_only") for n in names)
        readme = z.read("README_SUPPLEMENT.md").decode("utf-8")
        assert "This is the review build" in readme
        assert "leaves out these run bundles" in readme and "`fullruns/full_only`" in readme
        assert "not included" not in readme
        note = z.read("fullruns/review_a/agents/note.json").decode("utf-8")
        sums = z.read("SHA256SUMS").decode("utf-8")
        table = json.loads(z.read(reproduce.DIGEST_TABLE_NAME).decode("utf-8")) \
            if reproduce.DIGEST_TABLE_NAME in names else None
    if commits and _in_checkout():
        h = reproduce.CommitPseudonyms(commits, b"")
        assert commits[-1] not in note and commits[-1][:7] not in note
        assert commits[-1][:7].upper() not in note           # c2: the uppercase spelling too
        assert h.search(note) is None and "Commit IDs (" in readme
        rec = json.loads(note)
        assert rec["upper"] == rec["short"].upper() and rec["mean"] == -9.765625e-05
        # c42: the recorded digest is a pseudonym; SHA256SUMS keeps the real one; inside the
        # extracted archive archive_digest() maps the real digest to the recorded pseudonym.
        sha = hashlib.sha256(AGENT).hexdigest()
        assert sha not in note and len(rec["agent_sha256"]) == 64
        assert f"{sha}  fullruns/review_a/agents/agent_s0.pt" in sums
        assert "SHA-256 digests (" in readme and table is not None
        ex = tmp_path / "extracted"
        with zipfile.ZipFile(out) as z:
            z.extractall(ex)
        assert reproduce.archive_digest(sha) == sha                        # in the checkout
        with monkeypatch.context() as mp:
            mp.setattr(reproduce, "ROOT", str(ex))
            assert reproduce.archive_digest(sha) == rec["agent_sha256"]
            assert reproduce.archive_digest_text(f"x {sha} y") == f"x {rec['agent_sha256']} y"
            assert reproduce.archive_digest("c" * 64) == "c" * 64         # not in the table
        # The map sits beside the archive, not beside the key, and can rebuild the pseudonyms.
        cmap = tmp_path / "review.commit-map.json"
        assert cmap.exists() and not os.path.exists(tmp_path / "key" / "review.commit-map.json")
        m = json.loads(cmap.read_text(encoding="utf-8"))
        assert m["archive"] == "review.zip" and m["commits"] == commits
        assert m["archive_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        key = reproduce.commit_key(str(tmp_path / "key" / "k.txt"))
        again = reproduce.CommitPseudonyms(m["commits"], key)
        assert all(again.full[again.resolves(c)[0]].startswith(p.lower())
                   for p, c in m["pseudonyms"].items())


def test_the_review_profile_fails_and_removes_an_archive_over_the_limit(tmp_path, monkeypatch):
    # An earlier archive and its map at the same path: the failed rebuild removes both, so no
    # map is left naming an archive that is gone.
    (tmp_path / "review.zip").write_bytes(b"an earlier archive")
    (tmp_path / "review.commit-map.json").write_text('{"archive": "review.zip"}\n', encoding="utf-8")
    rc, out, commits = _review_build(tmp_path, monkeypatch, 1000)
    gap = reproduce.history_gap(commits)
    if gap:
        # Refused before writing anything: the earlier pair stays as it was.
        assert rc == 1 and out.read_bytes() == b"an earlier archive"
        assert (tmp_path / "review.commit-map.json").exists()
        _skip_or_fail_on_gap(gap)
    assert rc == 1 and not out.exists()
    assert not (tmp_path / "review.commit-map.json").exists()


def test_the_commit_map_is_named_after_its_archive():
    assert reproduce.commit_map_path(os.path.join("dist", "supplement-review.zip")) == \
        os.path.join("dist", "supplement-review.commit-map.json")
    assert reproduce.commit_map_path("a.zip") != reproduce.commit_map_path("b.zip")


def test_an_anonymized_build_refuses_a_partial_history(tmp_path, monkeypatch):
    """Without the whole object store the commit scan is blind to the missing commits, so the
    build writes nothing: no archive, no key, no map."""
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))      # no README_SUPPLEMENT.md here
    assert "not the top of a git checkout" in reproduce.history_gap([])
    commits = _ids(3)
    monkeypatch.setattr(reproduce, "_git", _fake_git(shallow=True))
    assert "shallow clone" in reproduce.history_gap(commits)
    monkeypatch.setattr(reproduce, "_git", _fake_git())
    monkeypatch.setattr(reproduce, "_ls_remote", lambda r: f"{commits[0]}\trefs/heads/main\n")
    assert reproduce.history_gap(commits) is None
    # An extracted archive carries pseudonyms only, so it may rebuild itself without history.
    (tmp_path / reproduce.README_NAME).write_text("# Anonymized supplement\n", encoding="utf-8")
    assert reproduce.history_gap([]) is None
    (tmp_path / reproduce.README_NAME).unlink()

    monkeypatch.setattr(reproduce, "commit_ids", lambda: [])
    monkeypatch.setattr(reproduce, "project_name", lambda: None)
    monkeypatch.setattr(reproduce, "RUN_DIRS", [])
    key = tmp_path / "key" / "k.txt"
    for profile, flag in (("review", None), ("full", True)):
        out = tmp_path / f"{profile}.zip"
        assert reproduce.supplement(str(out), runs_root=str(tmp_path / "runs"), anon_name="blind" + "pkg",
                                    profile=profile, key_path=str(key), require_history=flag,
                                    paper=_paper(tmp_path)) == 1
        assert not out.exists() and not key.exists()
        assert not os.path.exists(reproduce.commit_map_path(str(out)))
    # The command line always asks for the history.
    seen = {}
    monkeypatch.setattr(reproduce, "supplement", lambda out, **kw: seen.update(kw) or 0)
    reproduce.main(["supplement"])
    assert seen["require_history"] is True


def _fake_git(shallow=False, partial=False, refspec="+refs/heads/*:refs/remotes/origin/*"):
    """A stand-in for reproduce._git: a non-shallow checkout with one remote, origin."""
    def git(args):
        if "--is-shallow-repository" in args:
            return "true\n" if shallow else "false\n"
        if args[:2] == ["config", "--get"] and args[2] == "extensions.partialClone":
            return "origin\n" if partial else None
        if args == ["remote"]:
            return "origin\n"
        if args[:2] == ["config", "--get-all"]:
            return refspec + "\n"
        return None
    return git


# ------------------------------------------------------------------ second review, 2026-10-16
def test_c3_a_partial_single_branch_or_stale_store_is_refused(tmp_path, monkeypatch):
    """A non-shallow clone can still lack commits the archive names: a single-branch clone, a
    partial clone, a store missing a branch tip of the remote, or one in which a commit field
    names a commit it does not hold. Each is refused, and so is a remote whose branches cannot
    be listed, since a branch tip the store lacks would then go unseen."""
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    commits = _ids(5)
    monkeypatch.setattr(reproduce, "_ls_remote", lambda r: f"{commits[1]}\trefs/heads/main\n")
    monkeypatch.setattr(reproduce, "_git", _fake_git(partial=True))
    assert "partial clone" in reproduce.history_gap(commits)
    monkeypatch.setattr(reproduce, "_git", _fake_git(refspec="+refs/heads/main:refs/remotes/origin/main"))
    assert "single-branch" in reproduce.history_gap(commits)
    monkeypatch.setattr(reproduce, "_git", _fake_git())
    assert reproduce.history_gap(commits, {commits[2][:7]}) is None
    other = hashlib.sha1(b"elsewhere").hexdigest()     # a commit this store does not hold
    assert other not in commits
    monkeypatch.setattr(reproduce, "_ls_remote", lambda r: f"{commits[1]}\trefs/heads/main\n"
                                                           f"{other}\trefs/heads/run/x\n")
    assert "branch tip" in reproduce.history_gap(commits)
    monkeypatch.setattr(reproduce, "_ls_remote", lambda r: None)
    assert "could not be listed" in reproduce.history_gap(commits)
    monkeypatch.setattr(reproduce, "_ls_remote", lambda r: f"{commits[1]}\trefs/heads/main\n")
    assert "commit fields" in reproduce.history_gap(commits, {other[:7]})


def test_c3_real_clones_single_branch_and_stale_are_refused(tmp_path, monkeypatch):
    """The same checks on real repositories: a single-branch clone of a two-branch origin, and a
    full clone that has not fetched a branch pushed after it was made."""
    def git(cwd, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        "-c", "init.defaultBranch=main", *args], cwd=cwd, check=True,
                       capture_output=True)
    try:
        origin = tmp_path / "origin"
        origin.mkdir()
        git(origin, "init")
        (origin / "a.txt").write_text("a\n", encoding="utf-8")
        git(origin, "add", "a.txt")
        git(origin, "commit", "-m", "a")
        git(origin, "checkout", "-b", "run/x")
        (origin / "b.txt").write_text("b\n", encoding="utf-8")
        git(origin, "add", "b.txt")
        git(origin, "commit", "-m", "b")
        git(origin, "checkout", "main")
        url = origin.as_uri()
        git(tmp_path, "clone", "--single-branch", "--branch", "main", url, "single")
        git(tmp_path, "clone", url, "full")
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git is not usable here: {e}")
    for name, want in (("single", "single-branch"), ("full", None)):
        monkeypatch.setattr(reproduce, "ROOT", str(tmp_path / name))
        commits = reproduce.commit_ids()
        assert commits, name
        gap = reproduce.history_gap(commits)
        assert (gap is None) if want is None else (want in gap), (name, gap)
    # A branch pushed to the origin after the clone: its tip is not in the clone's store.
    git(origin, "checkout", "-b", "run/y")
    (origin / "c.txt").write_text("c\n", encoding="utf-8")
    git(origin, "add", "c.txt")
    git(origin, "commit", "-m", "c")
    assert "branch tip" in reproduce.history_gap(reproduce.commit_ids())
    git(tmp_path / "full", "fetch", "origin")
    assert reproduce.history_gap(reproduce.commit_ids()) is None


def test_c3_commit_fields_are_pseudonymized_even_when_they_do_not_resolve():
    commits = _ids(20)
    stray = "0f" * 20                          # a commit ID this store does not hold
    assert not reproduce._resolves(commits, stray)
    text = (f'{{"git_commit": "{stray}", "commits": ["{stray[:7]}"], "padding_fix": "{commits[4][:7]}"}}'
            f' and commit `{stray[:9]}`; fingerprint "{stray[:16]}"'
            f', "fingerprint_committed": "{stray[:16]}", "git_sha": "{commits[5][:7]}"')
    named = reproduce.commit_field_tokens(text)
    assert {stray, stray[:7], stray[:9], commits[4][:7]} <= named and stray[:16] not in named
    assert commits[5][:7] not in named          # no "commit" or "head" in the key
    forced = sorted(t for t in named if not reproduce._resolves(commits, t))
    h = reproduce.CommitPseudonyms(commits, b"k" * 32, forced=forced)
    out = h.sub(text)
    tokens = set(re.findall(r"[0-9a-f]+", out))
    assert not {stray, stray[:7], stray[:9]} & tokens
    assert stray[:16] in tokens                # a 16-character fingerprint is no commit field
    assert h.search(text) is not None and h.search(out) is None
    assert set(h.forced_mapping.values()) == {stray, stray[:7], stray[:9]}


def test_c3_a_build_that_finds_a_resolvable_commit_leaves_no_archive_and_no_map(tmp_path, monkeypatch):
    commits = reproduce.commit_ids()
    if not commits:
        pytest.skip("needs a git checkout")
    rc, out, commits = _review_build(tmp_path, monkeypatch, reproduce.REVIEW_LIMIT,
                                     agent=b"weights " + commits[-1][:9].upper().encode() + b"\x00")
    _review_needs_history(rc, out, commits)
    assert rc == 1 and not out.exists()
    assert not (tmp_path / "review.commit-map.json").exists()


def test_c2_uppercase_commit_ids_are_replaced_in_their_case_and_found_in_bytes():
    commits = _ids(30)
    c = commits[7]
    text = f"REPRO-{c[:7].upper()} at {c[:7]}, full {c.upper()}"
    h = reproduce.CommitPseudonyms(commits, b"k" * 32)
    out = h.sub(text)
    p7 = h.full[c][:7]
    assert out == f"REPRO-{p7.upper()} at {p7}, full {h.full[c].upper()}"
    assert h.search(out) is None and h.search(text).group(0) == c[:7].upper()
    assert h.mapping[p7.upper()] == c[:7].upper() and h.count() == 1
    assert reproduce._binary_hits(b"\x00\x01" + c[:8].upper().encode() + b"\xff", [h])
    assert not reproduce._binary_hits(b"\x00\x01" + p7.upper().encode() + b"\xff", [h])


def test_c6_hex_runs_inside_numbers_are_left_alone():
    """A commit whose prefix reads like the decimals of a number: the number is not rewritten,
    while the same token standing alone, and an all-digit commit prefix, still are."""
    frac = "765625e"
    commits = sorted(_ids(10) + [frac + "0" * 33, "2718281" + "1" * 33, "1765625e" + "2" * 32])
    h = reproduce.CommitPseudonyms(commits, b"k" * 32)
    # The commit field is assembled at runtime: this file ships, and a literal commit field
    # naming no real commit would stop the archive build (history_gap).
    field = '"git_' + 'commit": "' + "2718281" + '"'
    text = ('{"mean": -9.765625e-05, "x": 1.2718281, "y": 2718281.5, "z": 1765625e-05, '
            + field + ', "note": "765625e"}')
    out = h.sub(text)
    for num in ("-9.765625e-05", "1.2718281", "2718281.5", "1765625e-05"):
        assert num in out, num
    assert field not in out and '"note": "765625e"' not in out
    assert reproduce.number_guard("a.json", text, out) is None
    assert json.loads(out)["mean"] == -9.765625e-05


def test_c6_the_number_rules_cover_digit_runs_only():
    """The number contexts excuse a run of digits (with at most an exponent "e"), never a token
    with another hex letter: a hash-date name or a version suffix is still replaced, and the
    leak scan, which uses the same pattern, still finds it."""
    tokens = lambda s: [m.group(0) for m in reproduce._HEX_TOKEN.finditer(s)]   # noqa: E731
    assert tokens("deadbee-2026") == ["deadbee"]
    assert tokens("v1.4dead1f3c") == ["4dead1f3c"]
    assert tokens("abcdef1.5") == ["abcdef1"]
    for num in ("-9.765625e-05", "1765625e-05", "2718281.5", "0.2718281", "1.234567e10"):
        assert tokens(num) == [], num
    commits = sorted(_ids(10) + ["deadbee" + "0" * 33, "4dead1f3c" + "1" * 31])
    h = reproduce.CommitPseudonyms(commits, b"k" * 32)
    for text in ("run deadbee-2026", "v1.4dead1f3c"):
        assert h.search(text) is not None, text
        out = h.sub(text)
        assert out != text and h.search(out) is None, text


def test_c6_the_number_guard_catches_a_changed_number_or_broken_json():
    g = reproduce.number_guard
    assert g("a.json", '{"a": 1}', '{"a": 1}') is None
    assert "decimal" in g("a.md", "x 9.765625e-05", "x 9.abcdef0e-05")
    assert "parses" in g("a.json", '{"a": 1234567}', '{"a": 12345f7}')
    assert "number" in g("a.json", '{"a": 1234567}', '{"a": 7654321}')
    assert "number" in g("a.jsonl", '{"a": 1}\n{"b": 2}\n', '{"a": 1}\n{"b": 3}\n')
    assert g("a.json", "not json abcdef1", "not json 1234567") is None    # not JSON before
    assert g("a.json", '{"k": "abcdef1"}', '{"k": "1234567"}') is None     # strings may change
    # An all-digit token is rewritten only where something marks it as a commit.
    for before in ("not json 1234567", '{"k": "1234567"}', "x 1234567 y"):
        after = before.replace("1234567", "abcdef1")
        assert "integer" in g("a.md", before, after), before
        assert g("a.md", before, after, known={"1234567"}) is None, before
    # The commit contexts are assembled at runtime: this file ships, and a literal one naming no
    # real commit would stop the archive build (history_gap).
    word = "com" + "mit"
    for before in ("at `1234567`.", f"{word} 1234567", f"git_{word} 1234567", f"{word} (`1234567`,",
                   '{"git_' + word + '": "1234567"}'):
        assert g("a.md", before, before.replace("1234567", "abcdef1")) is None, before
    # A branch head written after the branch name in a code span (the 2026-10-20 spec).
    assert g("a.md", "ITASORL `main` 1234567 plus", "ITASORL `main` abcdef1 plus") is None
    assert "integer" in g("a.md", "the `seed` 1234567 base", "the `seed` abcdef1 base")
    # A commit of the manuscript's own repository is not one of this repository's commits.
    assert reproduce.commit_field_tokens(f"up to paper {word} abcdef1 and {word} 1234abc") == {"1234abc"}
    assert reproduce.paper_commit_tokens(f"up to paper {word} abcdef1 and {word} 1234abc") == {"abcdef1"}
    assert "length" in g("a.md", "x 1234567", "x 1234567 y")


def test_c6_a_build_whose_pseudonyms_change_a_number_leaves_no_archive(tmp_path, monkeypatch):
    if not reproduce.commit_ids():
        pytest.skip("needs a git checkout")
    monkeypatch.setattr(reproduce, "number_guard",
                        lambda name, b, a, **kw: "a JSON number changed" if name.endswith("note.json")
                        else None)
    rc, out, commits = _review_build(tmp_path, monkeypatch, reproduce.REVIEW_LIMIT)
    _review_needs_history(rc, out, commits)
    assert rc == 1 and not out.exists()
    assert not (tmp_path / "review.commit-map.json").exists()


def test_c5_a_review_build_refuses_without_the_manuscript_title(tmp_path, monkeypatch, capsys):
    """The title pattern used to drop out silently when docs/paper was absent. A review build now
    refuses to start, writing nothing; an extracted archive, which has no manuscript to check
    against, only warns."""
    rc, out, commits = _review_build(tmp_path, monkeypatch, reproduce.REVIEW_LIMIT,
                                     paper=str(tmp_path / "no-such-paper"))
    said = capsys.readouterr().out
    if os.path.exists(os.path.join(ROOT, reproduce.README_NAME)):
        assert "WARNING: the manuscript title could not be read" in said
        return
    assert rc == 1 and not out.exists() and not (tmp_path / "key" / "k.txt").exists()
    assert "no archive written: the manuscript title could not be read" in said


def test_c5_a_full_build_without_the_title_says_so_in_its_readme():
    readme = reproduce._readme([], "blind" + "pkg", title_checked=False)
    assert "could not be read where this archive was built" in readme
    assert "could not be read" not in reproduce._readme([], "blind" + "pkg")


def test_c5_the_title_and_subtitle_are_read_from_the_main_checkout_of_a_worktree(tmp_path, monkeypatch):
    main = tmp_path / "main"
    (main / "docs" / "paper").mkdir(parents=True)
    (main / "docs" / "paper" / "main.tex").write_text(
        "% \\title{Not This}\n\\title{Quiet Words In A Row: Seven \\\\ Long Words"
        "\\thanks{A note {with} braces.}}\\hypersetup{pdftitle={Quiet Words In A Row: Seven Long "
        "Words}}\n", encoding="utf-8")
    wt = tmp_path / "main" / ".worktrees" / "wt"
    wt.mkdir(parents=True)
    monkeypatch.setattr(reproduce, "ROOT", str(wt))
    monkeypatch.setattr(reproduce, "_git", lambda args: str(main / ".git") + "\n"
                        if "--git-common-dir" in args else None)
    assert reproduce.paper_source() == str(main / "docs" / "paper" / "main.tex")
    assert reproduce.paper_titles() == ["Quiet Words In A Row", "Seven Long Words"]
    pats = reproduce._leak_patterns(None)
    assert any(p.search("seven long\nwords") for p in pats)
    # Without pdftitle, the \title line itself, its \thanks and line break dropped.
    (main / "docs" / "paper" / "main.tex").write_text(
        "\\title{Quiet Words In A Row: Seven \\\\ Long Words\\thanks{A note {with} braces.}}\n",
        encoding="utf-8")
    assert reproduce.paper_titles() == ["Quiet Words In A Row", "Seven Long Words"]
    # An explicit path wins, and a missing one gives no title rather than another file's.
    assert reproduce.paper_titles(str(tmp_path / "nowhere")) == []


def test_c42_digests_get_same_length_pseudonyms_and_a_lookup_table(tmp_path, monkeypatch):
    commits = _ids(10)
    d1 = hashlib.sha256(b"one").hexdigest()
    d2 = hashlib.sha256(b"two").hexdigest()
    pad = "0" * 64
    text = f"a {d1} b {d1[:16]} c {d2.upper()} d {pad} e {d1[:8]}"
    digests = reproduce.digest_tokens(text)
    assert digests == {d1, d2}                 # padding is no digest
    h = reproduce.CommitPseudonyms(commits, b"k" * 32, digests=digests)
    out = h.sub(text)
    p1, p2 = h.dfull[d1], h.dfull[d2]
    assert out == f"a {p1} b {p1[:16]} c {p2.upper()} d {pad} e {d1[:8]}"
    assert h.digest_count() == 2 and len(p1) == 64 and p1 != d1
    assert reproduce.CommitPseudonyms(commits, b"k" * 32, digests=digests).sub(text) == out
    table = h.digest_table()
    assert table == {reproduce.digest_lookup_key(d1): p1, reproduce.digest_lookup_key(d2): p2}
    assert d1 not in json.dumps(table) and d2 not in json.dumps(table)
    # The scan for real digests: any case, names and bytes, nested zips, never SHA256SUMS.
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ok.json", out)
        z.writestr("bad.bin", b"\x00" + d2.upper().encode() + b"\x00")
        z.writestr("SHA256SUMS", f"{d1}  ok.json\n")
    assert reproduce.digest_scan(str(path), {d1, d2}) == ["bad.bin"]
    # Inside an extracted archive a computed digest maps to the recorded pseudonym, and the
    # reviewed-image check still accepts the image whose listed digest is a pseudonym.
    (tmp_path / reproduce.README_NAME).write_text("x\n", encoding="utf-8")
    (tmp_path / reproduce.DIGEST_TABLE_NAME).write_text(json.dumps({"pseudonyms": table}),
                                                        encoding="utf-8")
    monkeypatch.setattr(reproduce, "ROOT", str(tmp_path))
    assert reproduce.archive_digest(d1) == p1 and reproduce.archive_digest(d2.upper()) == p2.upper()
    assert reproduce.image_reviewed(b"one", {"fig.png": p1})
    assert not reproduce.image_reviewed(b"one", {"fig.png": d1})


def test_c42_the_review_archive_carries_no_tracked_digest(tmp_path, monkeypatch):
    """Every 64-character digest a tracked file holds is a search key for the public repository;
    in the review archive only SHA256SUMS may hold one."""
    if not reproduce.tracked_digests():
        pytest.skip("needs a git checkout with digests in tracked files")
    files = ["pyproject.toml", "docs/REPRODUCE.md", "artifacts/surrogates/index.json"]
    rc, out, commits = _review_build(tmp_path, monkeypatch, reproduce.REVIEW_LIMIT, files=files)
    _review_needs_history(rc, out, commits)
    assert rc == 0
    tracked = reproduce.tracked_digests()
    assert reproduce.digest_scan(str(out), tracked) == []
    with zipfile.ZipFile(out) as z:
        index = json.loads(z.read("artifacts/surrogates/index.json").decode("utf-8"))
        assert index["files"] and all(len(m["sha256"]) == 64 and m["sha256"] not in tracked
                                      for m in index["files"].values())


# ------------------------------------------------------------------ third review, 2026-10-16
# Branch names here are assembled at runtime: this file ships, and a literal agent-session
# branch name would be replaced in the archived copy, changing what the test asserts.
SESSION = "cla" + "ude/"


def test_c13_branch_names_become_pseudonyms_in_text_and_file_names():
    names = ["run/" + "zz-alpha", "feat/" + "zz-beta-1.2.0", "zz-plain-runner"]
    rule = reproduce.branch_rules(names)
    text = (f"on branch {names[0]}, origin/{names[1]}. Spec 2026-07-03-{names[2]}-design.md; "
            f"session {SESSION}quiet-otter-x1. and {SESSION}quiet-otter-x1 again; "
            f"~/.{SESSION}projects stays, xrun/zz-alpha stays, {names[0]}x stays")
    out = reproduce.scrub(text, rule)
    by_name = {v: k for k, v in rule[0][1].used.items()}
    p0, p1, p2 = (by_name[n] for n in names)
    ps = by_name[SESSION + "quiet-otter-x1"]
    assert re.fullmatch(r"anon-branch-\d+", ps) and len({p0, p1, p2, ps}) == 4
    assert out == (f"on branch {p0}, origin/{p1}. Spec 2026-07-03-{p2}-design.md; "
                   f"session {ps}. and {ps} again; "
                   f"~/.{SESSION}projects stays, xrun/zz-alpha stays, {names[0]}x stays")
    # The known names take their sorted index, so every build of one checkout agrees.
    assert reproduce.branch_rules(names)[0][1].index == {n: i + 1 for i, n in enumerate(sorted(names))}
    # Generic names stay, and both listings parse.
    listing = ("refs/heads/main\nrefs/heads/" + names[0] + "\nrefs/remotes/origin/HEAD\n"
               "refs/remotes/origin/" + names[1] + "\n" + "0" * 40 + "\trefs/heads/" + names[2] + "\n")
    assert reproduce._branch_listing(listing) == {"main", "HEAD", *names}


def test_c13_branch_names_come_from_local_refs_and_the_remote_listing(monkeypatch):
    local = "refs/heads/main\nrefs/heads/run/" + "zz-a\nrefs/remotes/origin/feat/" + "zz-b\n"
    monkeypatch.setattr(reproduce, "_git", lambda args: local if args[0] == "for-each-ref"
                        else "origin\n" if args == ["remote"] else None)
    monkeypatch.setattr(reproduce, "_ls_remote", lambda r: "0" * 40 + "\trefs/heads/fix/" + "zz-c\n")
    monkeypatch.setattr(reproduce, "_pr_heads", lambda: "docs/" + "zz-squashed\nmain\n")
    assert reproduce.branch_names() == ["feat/zz-b", "run/zz-a"]
    assert reproduce.branch_names(remote=True) == ["docs/zz-squashed", "feat/zz-b", "fix/zz-c",
                                                   "run/zz-a"]
    monkeypatch.setattr(reproduce, "_pr_heads", lambda: None)
    assert reproduce.branch_names(remote=True) == ["feat/zz-b", "fix/zz-c", "run/zz-a"]


def test_c13_a_deleted_branch_is_named_by_the_merge_subject_its_pull_request_left(monkeypatch):
    """A branch deleted after its pull request merged is in no ref, but the merge commit's
    subject still names it, and its pull request is still public: the name counts as known."""
    gone = "feat/" + "zz-deleted-head"
    subjects = ("Merge pull request #7 from someone/" + gone + "\n"
                "Merge branch '" + "zz-local-only' into run/" + "zz-target\n"
                "Merge remote-tracking branch 'origin/fix/" + "zz-tracked' into main\n"
                "Merge branch 'main' of https://example.org/x/y\n"
                "Merge branches 'zz-" + "one' and 'zz-" + "two' into main\n"
                "Revert \"Merge pull request #8 from someone/zz-" + "not-a-merge\"\n"
                "Ordinary subject naming nothing\n")
    assert reproduce._merge_listing(subjects) == {gone, "zz-local-only", "run/zz-target",
                                                  "fix/zz-tracked", "main", "zz-one", "zz-two"}
    monkeypatch.setattr(reproduce, "_git", lambda args: subjects if args[0] == "log"
                        else "refs/heads/main\n" if args[0] == "for-each-ref" else None)
    names = reproduce.branch_names()
    assert gone in names and "main" not in names
    # It is replaced in the archive, and the leak scan fails on any spelling left.
    assert gone not in reproduce.scrub("built on " + gone + ".", reproduce.branch_rules(names))
    assert reproduce._hits("built on " + gone.upper(), [reproduce._branch_pattern(names, re.I)])


def test_c13_the_leak_scan_fails_on_a_surviving_branch_name(tmp_path):
    """Any spelling the replacement missed, in any case, in text, bytes or a member name, and any
    agent-session branch name, fails the build."""
    name = "run/" + "zz-alpha"
    for member, data in (("a.md", "see " + name.upper()), ("b.bin", b"\x00" + name.encode() + b"\x00"),
                         ("docs/x-" + name.replace("/", "_") + ".md", "ok"),
                         ("c.md", "left " + SESSION + "other-x9")):
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("fine.md", "anon-branch-1 and ~/." + SESSION + "projects")
            z.writestr(member, data)
        want = [] if member.startswith("docs/x-") else [member]
        assert reproduce.leak_scan(str(path), [], None, titles=[], branches=[name]) == want, member


def test_c13_c0_the_review_archive_carries_no_branch_name_and_ships_docs_optional(tmp_path, monkeypatch):
    """The results manifest names the branches that hold off-main commits; in the review archive
    each reads as its pseudonym, and no branch name the build knows survives anywhere: those of
    the refs and the remotes, and those of deleted branches that merge subjects and the pull
    request listing still carry (the results manifest records one as a runner branch). The
    archive also carries docs/optional, which the manuscript cites (O2), with its provenance
    commits pseudonymized."""
    names = reproduce.branch_names(remote=True)
    if not names:
        pytest.skip("needs a git checkout with branches")
    # Only two of the record's files ship here, so an all-digit commit ID that the full archive
    # marks as a commit in another file reads as a bare number; the number guard has its own tests.
    monkeypatch.setattr(reproduce, "number_guard", lambda *a, **k: None)
    rc, out, commits = _review_build(tmp_path, monkeypatch, reproduce.REVIEW_LIMIT,
                                     files=("pyproject.toml", "docs/RESULTS_MANIFEST.md",
                                            "artifacts/results_manifest.json"))
    _review_needs_history(rc, out, commits)
    assert rc == 0
    with zipfile.ZipFile(out) as z:
        listed = z.namelist()
        assert {"docs/optional/README.md", "docs/optional/o2_long_history_ridge.json"} <= set(listed)
        blob = "\n".join(listed + [z.read(n).decode("latin-1") for n in listed]).lower()
        manifest = z.read("docs/RESULTS_MANIFEST.md").decode("utf-8")
        readme = z.read(reproduce.README_NAME).decode("utf-8")
        o2 = json.loads(z.read("docs/optional/o2_long_history_ridge.json").decode("utf-8"))
    assert not [n for n in names if n.lower() in blob]
    assert SESSION not in blob.replace("." + SESSION, "")
    assert re.search(r"`anon-branch-\d+` \(head `", manifest) and "Branch names (" in readme
    if commits and _in_checkout():
        h = reproduce.CommitPseudonyms(commits, b"")
        assert h.search(o2["provenance"]["git_head"]) is None


def test_c12_a_directory_artifact_without_cells_reads_its_files_commits(tmp_path):
    """A sweep directory has no cells/; its point files record git_commit and its summary a
    top-level git_head. The manifest reads both, so the row's commit is "recorded"."""
    import build_results_manifest as brm
    word = "com" + "mit"                       # assembled: a literal commit field would ship
    c1, c2 = _ids(2)
    (tmp_path / "A_rho0.50.json").write_text(json.dumps({"git_" + word: c1[:7], "cells": []}),
                                             encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({"git_head": c2[:7], "points": []}),
                                           encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x\n", encoding="utf-8")
    rec = brm._dir_recorded("texture/sweep", str(tmp_path))
    assert rec == {"texture/sweep": {}, "texture/sweep/A_rho0.50.json": {"git_" + word: c1[:7]},
                   "texture/sweep/summary.json": {"git_head": c2[:7]}}
    assert brm._run_commits(rec) == {c1[:7], c2[:7]}
    assert brm.commit_source(f"{c1[:7]} (points); {c2[:7]} (summary)", brm._run_commits(rec)) == "recorded"
    assert brm.commit_source(f"{c2[:7]}; code uncommitted at run time",
                             brm._run_commits(rec)).startswith("curated: the code was uncommitted")


def test_the_review_limit_is_100_mb_and_the_profiles_differ():
    assert reproduce.REVIEW_LIMIT == 100_000_000
    assert reproduce.profile_run_dirs("review") == ["fullruns/l3_h8_heldout/agents",
                                                    "fullruns/l3_h8_gseed2_gpu/agents",
                                                    "fullruns/corrected_l3_h8_wm/agents",
                                                    "fullruns/corrected_l3_h8_nowm/agents",
                                                    "fullruns/corrected_l3_h8_wm/cells",
                                                    "fullruns/corrected_l3_h8_nowm/cells"]
    assert reproduce.profile_run_dirs("full") == reproduce.RUN_DIRS
    assert set(reproduce.RUN_DIRS) - set(reproduce.REVIEW_RUN_DIRS) <= set(reproduce.REVIEW_LEFT_OUT)
    with pytest.raises(SystemExit):
        reproduce.profile_run_dirs("tiny")


# ------------------------------------------------------------------ manuscript counts
def test_spelled_counts_parse():
    assert bpt._count("49") == 49 and bpt._count("eleven") == 11
    assert bpt._count("twenty-seven") == 27 and bpt._count("Forty") == 40
    assert bpt._count("the") is None and bpt._count("twenty-ten") is None


def _ledger_rows():
    ledger = bpt.load_ledger()
    if ledger is None:
        pytest.skip("no committed rule ledger here")
    return bpt.evaluate_ledger(ledger)[0]


def test_a_stale_rule_ledger_tally_is_flagged_and_a_current_one_is_not():
    rows = _ledger_rows()
    t = bpt._tally(rows)
    good = (f"The record adjudicates {len(rows)} rule-and-run pairs at the 0.65 bar: the threshold "
            f"condition holds on {t['threshold met']} and fails on {t['threshold not met']}, "
            f"{t['uninformative']} are uninformative and {t['no verdict']} give no verdict. Next.")
    assert bpt._count_issues("m.tex", [good], rows) == []
    stale = good.replace(f"{len(rows)} rule", f"{len(rows) - 1} rule").replace(
        f"fails on {t['threshold not met']}", f"fails on {t['threshold not met'] - 1}")
    issues = bpt._count_issues("m.tex", [stale], rows)
    assert len(issues) == 2 and all("rule-ledger tally" in i for i in issues)
    # Without a ledger the tally is not checked.
    assert bpt._count_issues("m.tex", [stale], None) == []


def test_a_stale_gate_table_count_is_flagged():
    rows = bpt._load("gate_table.json")["rows"]
    n_sets = bpt._l0_sets(rows)[1]
    good = f"of the {len(rows)} rows in that table, which hold {n_sets} distinct sets of per-seed values"
    assert bpt._count_issues("m.tex", [good], None) == []
    stale = f"The {len(rows) - 1} rows hold {n_sets} distinct sets of per-seed values"
    issues = bpt._count_issues("m.tex", [stale], None)
    assert len(issues) == 1 and "gate table" in issues[0]


def test_check_manuscript_reads_the_counts(tmp_path, capsys):
    """check_manuscript, not only _count_issues, flags a stale count: the same manuscript with
    one count changed reports more items. The generated tables are made in memory from the
    committed artifacts, so docs/paper_tables/ need not exist."""
    rows = bpt._load("gate_table.json")["rows"]
    n_sets = bpt._l0_sets(rows)[1]
    head = "\\input{paper_tables/gate_table}\n"

    def items(sentence: str) -> int:
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        (d / "main.tex").write_text(head + sentence + "\n", encoding="utf-8")
        bpt.check_manuscript(str(d))
        out = capsys.readouterr().out
        return int(out.rsplit(".tex files, ", 1)[1].split(" item")[0])

    good = f"Of the {len(rows)} rows in that table, which hold {n_sets} distinct sets of per-seed values."
    stale = f"Of the {len(rows) - 1} rows in that table, which hold {n_sets} distinct sets of per-seed values."
    assert items(stale) == items(good) + 1
    ledger = bpt.load_ledger()
    if ledger is not None:
        lrows = bpt.evaluate_ledger(ledger)[0]
        t = bpt._tally(lrows)
        tally = (f"The record adjudicates {len(lrows)} rule-and-run pairs: the threshold condition "
                 f"holds on {t['threshold met']} and fails on {t['threshold not met']}.")
        assert items(tally.replace(f"{len(lrows)} rule", f"{len(lrows) + 1} rule")) == items(tally) + 1


def test_a_forced_commit_and_its_short_prefix_get_one_pseudonym():
    """In a shallow clone a commit-field ID resolves to nothing and is pseudonymized by force.
    Code or prose that quotes its first seven characters must get the same pseudonym's prefix,
    and the leak scan must see the bare prefix (CI, 2026-10-16: the audit compared a forced
    artifact field with an unforced constant inside the archive)."""
    full = "06ecc10" + "a" * 33
    h = reproduce.CommitPseudonyms([], b"k" * 32, forced=[full])
    field = h.sub('{"git_commit_at_run": "%s"}' % full)
    code = h.sub('x.startswith("06ecc10")')
    p = h.forced[full]
    assert p in field and full not in field
    assert code == 'x.startswith("%s")' % p[:7]
    assert h.search('startswith("06ecc10")') is not None
    assert h.search('startswith("%s")' % p[:7]) is None


def test_forced_spellings_of_one_commit_share_a_prefix_consistent_pseudonym():
    """A 7-character commit_at_run and a 40-character git_commit_at_run of the same unresolved
    commit are both forced; the short pseudonym must be the long one's prefix."""
    full = "06ecc10" + "b" * 33
    h = reproduce.CommitPseudonyms([], b"k" * 32, forced=["06ecc10", full])
    assert h.forced["06ecc10"] == h.forced[full][:7]
    long_sub = h.sub('"git_commit_at_run": "%s"' % full)
    short_sub = h.sub('commit_at_run="06ecc10"')
    assert h.forced[full] in long_sub and h.forced[full][:7] in short_sub


def test_forced_pseudonyms_sort_as_the_ids_do():
    """The results manifest lists a run's commits sorted; forced pseudonyms must keep that order
    or the manifest's --check fails inside an archive built from a shallow clone."""
    ids = ["55fadf4", "3bb741f", "a8fe1bf", "dfd14b2", "8e85b03" + "c" * 33, "dde1e91"]
    h = reproduce.CommitPseudonyms([], b"k" * 32, forced=ids)
    order = sorted(t.lower() for t in ids)
    assert [h.forced[t] for t in order] == sorted(h.forced[t] for t in order)
    assert len({h.forced[t][:7] for t in order}) == len(order)
