"""Packaging fixes of the 2026-10-16 critical review: `reproduce.py tables` reports stale shipped
pages in its exit code, the review profile of the supplement stays under the venue limit and
says what it leaves out, commit IDs ship as same-length pseudonyms, and the manuscript check
compares the counts the prose states about the rule ledger and the gate table."""

from __future__ import annotations

import hashlib
import json
import os
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


def _review_build(tmp_path, monkeypatch, limit: int):
    runs = tmp_path / "runs"
    for b in ("review_a/agents", "review_b/agents", "full_only"):
        (runs / b).mkdir(parents=True)
        (runs / b / "agent_s0.pt").write_bytes(b"weights \x00\x01")
    commits = reproduce.commit_ids()
    note = f'{{"git_commit": "{commits[-1]}", "short": "{commits[-1][:7]}"}}\n' if commits else "{}\n"
    (runs / "review_a" / "agents" / "note.json").write_text(note, encoding="utf-8")
    old = reproduce.project_name()
    monkeypatch.setattr(reproduce, "RUN_DIRS", ["fullruns/review_a/agents", "fullruns/review_b/agents",
                                                "fullruns/full_only"])
    monkeypatch.setattr(reproduce, "REVIEW_RUN_DIRS", ["fullruns/review_a/agents",
                                                       "fullruns/review_b/agents"])
    monkeypatch.setattr(reproduce, "INCLUDE_DIRS", [old])
    monkeypatch.setattr(reproduce, "INCLUDE_FILES", ["pyproject.toml", "pytest.ini"])
    monkeypatch.setattr(reproduce, "REVIEW_LIMIT", limit)
    out = tmp_path / "review.zip"
    rc = reproduce.supplement(str(out), runs_root=str(runs), anon_name="blind" + "pkg",
                              profile="review", key_path=str(tmp_path / "key" / "k.txt"))
    return rc, out, commits


def _review_needs_history(rc, out, commits) -> None:
    """In a shallow clone (CI) or outside a checkout the review build refuses to start."""
    gap = reproduce.history_gap(commits)
    if gap:
        assert rc == 1 and not out.exists()
        pytest.skip(f"the review build needs a full checkout: {gap}")


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
    if commits and _in_checkout():
        h = reproduce.CommitPseudonyms(commits, b"")
        assert commits[-1] not in note and commits[-1][:7] not in note
        assert h.search(note) is None and "Commit IDs (" in readme
        # The map sits beside the archive, not beside the key, and can rebuild the pseudonyms.
        cmap = tmp_path / "review.commit-map.json"
        assert cmap.exists() and not os.path.exists(tmp_path / "key" / "review.commit-map.json")
        m = json.loads(cmap.read_text(encoding="utf-8"))
        assert m["archive"] == "review.zip" and m["commits"] == commits
        assert m["archive_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        key = reproduce.commit_key(str(tmp_path / "key" / "k.txt"))
        again = reproduce.CommitPseudonyms(m["commits"], key)
        assert all(again.full[again.resolves(c)[0]].startswith(p) for p, c in m["pseudonyms"].items())


def test_the_review_profile_fails_and_removes_an_archive_over_the_limit(tmp_path, monkeypatch):
    rc, out, commits = _review_build(tmp_path, monkeypatch, 1000)
    _review_needs_history(rc, out, commits)
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
    monkeypatch.setattr(reproduce, "_git", lambda args: "true\n" if "--is-shallow-repository" in args else "")
    assert "shallow clone" in reproduce.history_gap(_ids(3))
    monkeypatch.setattr(reproduce, "_git", lambda args: "false\n")
    assert reproduce.history_gap(_ids(3)) is None
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
                                    profile=profile, key_path=str(key), require_history=flag) == 1
        assert not out.exists() and not key.exists()
        assert not os.path.exists(reproduce.commit_map_path(str(out)))
    # The command line always asks for the history.
    seen = {}
    monkeypatch.setattr(reproduce, "supplement", lambda out, **kw: seen.update(kw) or 0)
    reproduce.main(["supplement"])
    assert seen["require_history"] is True


def test_the_review_limit_is_100_mb_and_the_profiles_differ():
    assert reproduce.REVIEW_LIMIT == 100_000_000
    assert reproduce.profile_run_dirs("review") == ["fullruns/l3_h8_heldout/agents",
                                                    "fullruns/l3_h8_gseed2_gpu/agents"]
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
