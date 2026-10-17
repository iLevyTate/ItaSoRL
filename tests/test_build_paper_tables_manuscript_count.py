"""The manuscript quotes the verification script's check count, its pin count and their ratio
(A5, G19). build_paper_tables.check_manuscript() compares each with CHECK_FLOOR and PIN_COUNT of
scripts/audit_stats_recheck.py, so a count changed in one place is flagged in the other. Every
fixture is a temporary .tex file; the manuscript itself is never read."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import audit_stats_recheck as asr  # noqa: E402
import build_paper_tables as bpt  # noqa: E402

_WORDS = {40: "forty", 41: "forty-one", 42: "forty-two", 43: "forty-three", 44: "forty-four",
          45: "forty-five", 46: "forty-six", 47: "forty-seven", 48: "forty-eight", 49: "forty-nine",
          50: "fifty", 51: "fifty-one", 52: "fifty-two"}


def _sentences(n: int, p: int, k: str) -> str:
    return (f"A verification script of {n} checks runs in continuous integration on every change.\n"
            "A verification script runs in continuous integration on every change and prints its\n"
            f"count ({n} at the submission commit, {p} of them comparing a stored summary with a\n"
            f"constant in the script, about one check in {k}; a stored L0 gate must equal its\n"
            "recomputation from the per-seed values).\n")


def _k() -> str:
    k = round(asr.CHECK_FLOOR / asr.PIN_COUNT)
    return _WORDS.get(k, str(k))


def _write(tmp_path, text: str) -> str:
    (tmp_path / "main.tex").write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_the_constants_are_the_ones_the_checker_reads():
    assert bpt.audit_counts() == (asr.CHECK_FLOOR, asr.PIN_COUNT)
    assert 0 < asr.PIN_COUNT < asr.CHECK_FLOOR


def test_sentences_that_match_the_constants_raise_nothing():
    text = _sentences(asr.CHECK_FLOOR, asr.PIN_COUNT, _k())
    assert bpt._audit_count_issues("main.tex", text) == []


@pytest.mark.parametrize("field", ["n", "p", "k"])
def test_each_stale_count_is_flagged(field):
    n, p, k = asr.CHECK_FLOOR, asr.PIN_COUNT, _k()
    if field == "n":
        n += 1
    elif field == "p":
        p -= 1
    else:
        k = "fifty" if k != "fifty" else "forty"
    issues = bpt._audit_count_issues("main.tex", _sentences(n, p, k))
    want = {"n": "check count", "p": "pin count", "k": "one check in K ratio"}[field]
    assert issues and all(want in i for i in issues), issues
    # the stale check count appears in two sentences and both are flagged
    assert len(issues) == (2 if field == "n" else 1), issues


def test_the_ratio_is_rounded_and_digits_are_read_too():
    assert bpt._audit_count_issues("m.tex", _sentences(3187, 76, "forty-two"), n=3187, p=76) == []
    assert bpt._audit_count_issues("m.tex", _sentences(3187, 76, "42"), n=3187, p=76) == []
    assert bpt._audit_count_issues("m.tex", _sentences(3179, 64, "fifty"), n=3179, p=64) == []
    assert bpt._audit_count_issues("m.tex", _sentences(3179, 64, "forty"), n=3179, p=64)


def test_a_sentence_broken_across_lines_is_still_read():
    text = ("A verification\nscript of 12\nchecks.\n(99 at the\nsubmission commit, 3 of them\n"
            "comparing a stored summary with a constant, about one check in\nfour).\n")
    issues = bpt._audit_count_issues("m.tex", text, n=12, p=3)
    assert len(issues) == 1 and "99" in issues[0] and "m.tex:4" in issues[0], issues


def test_check_manuscript_reports_a_stale_count(tmp_path, capsys):
    bad = _sentences(asr.CHECK_FLOOR - 1, asr.PIN_COUNT, _k())
    bpt.check_manuscript(_write(tmp_path, bad))
    out = capsys.readouterr().out
    assert f"the audit's check count reads {asr.CHECK_FLOOR - 1}" in out
    good = _sentences(asr.CHECK_FLOOR, asr.PIN_COUNT, _k())
    bpt.check_manuscript(_write(tmp_path, good))
    assert "the audit's" not in capsys.readouterr().out
