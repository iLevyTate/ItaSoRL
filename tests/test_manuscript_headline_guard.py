"""The abstract and the body quote numbers from about a dozen artifacts (H41).
build_paper_tables.check_manuscript() compares each with the artifact value that
headline_needles() names, the way the C1 and C2 headline numbers were compared before. The
fixtures are temporary .tex files; the manuscript itself is never read."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_paper_tables as bpt  # noqa: E402

# What each needle renders to on the committed artifacts, as the abstract and the body quote them
# in the manuscript revision of 2026-10-25. A refit that moves one of these must move the manuscript too.
EXPECTED = {
    "L1 oracle": "0.873", "L1 stream reader": "0.788", "corrected L1 survival": "0.551",
    "corrected L1 survival interval": r"\ci{0.531}{0.570}", "C1 survival": "0.733",
    "C1 survival interval": r"\ci{0.669}{0.797}", "C1 predictor": "0.589", "C1 untrained": "0.523",
    "drag fresh probe": "0.754", "drag margin over the untrained arm": "+0.142",
    "drag untrained floor": "0.612", "jitter maximum": "0.627", "sweep C follow-up contrast": "-0.017",
    "sweep C follow-up interval": r"\ci{-0.051}{+0.016}", "C1 L0 mean": "0.559",
    "T-touch survival": "0.796", "PB1 sensory-echo trained": "0.709", "PB2 sensory-echo trained": "0.743",
    "PB untrained floor slack": "0.002", "PB nonlinear and longer-history controls, min": "0.644",
    "PB nonlinear and longer-history controls, max": "0.725",
    "O2 long-history ridge": "0.737", "O2 long-history ridge with behavior": "0.691",
}


def _fixture(needles, swap: tuple[str, str] | None = None) -> str:
    abstract = " ".join(f"{w} reads ${t}$ here." for b, w, _, _, t in needles if b == "abstract")
    body = " ".join(f"{w} reads ${t}$ here." for b, w, _, _, t in needles if b == "body")
    # The Contributions list quotes no pinned number since 2026-10-25; the fixture keeps one to show
    # that the checker no longer reads it as a block of its own.
    text = ("\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
            "\\paragraph{Contributions.} Four.\n\\begin{enumerate}\n\\item The gap, made measurable."
            "\n\\end{enumerate}\n\n\\section{Results}\n" + body + "\n")
    if swap:
        assert swap[0] in text
        text = text.replace(swap[0], swap[1])
    return text


def test_every_needle_names_an_artifact_that_exists_and_a_key():
    needles = bpt.headline_needles()
    assert {n[1] for n in needles} == set(EXPECTED)
    for block, label, art, key, _ in needles:
        assert block in ("abstract", "body")
        assert key, label
        for path in art.split(" and "):
            assert os.path.isfile(os.path.join(ROOT, path)), (label, path)


def test_the_needles_render_the_numbers_the_manuscript_quotes():
    got = {label: text for _, label, _, _, text in bpt.headline_needles()}
    assert got == EXPECTED


def test_a_fixture_built_from_the_artifacts_passes():
    assert bpt._headline_block_issues("main.tex", _fixture(bpt.headline_needles())) == []


@pytest.mark.parametrize("swap,label", [
    (("$0.788$", "$0.798$"), "L1 stream reader"),
    (("\\ci{-0.051}{+0.016}", "\\ci{-0.052}{+0.016}"), "sweep C follow-up interval"),
    (("$+0.142$", "$+0.143$"), "drag margin over the untrained arm"),
    (("$0.743$", "$0.753$"), "PB2 sensory-echo trained"),
])
def test_one_moved_number_is_reported_by_name(swap, label):
    issues = bpt._headline_block_issues("main.tex", _fixture(bpt.headline_needles(), swap))
    assert len(issues) == 1 and f"({label};" in issues[0], issues
    art = next(n[2] for n in bpt.headline_needles() if n[1] == label)
    assert art in issues[0]


def test_a_file_without_an_abstract_is_not_read():
    assert bpt._headline_block_issues("appendix.tex", "Some 0.123 prose.\n") == []


def test_a_contributions_list_without_numbers_passes():
    text = "\\begin{abstract}\n" + " ".join(
        f"${t}$" for b, _, _, _, t in bpt.headline_needles() if b == "abstract") + "\n\\end{abstract}\n" + (
        "\\paragraph{Contributions.} Four things.\n\\begin{enumerate}\n\\item A ladder, a negative, a difference, a method.\n"
        "\\end{enumerate}\n") + " ".join(
        f"${t}$" for b, _, _, _, t in bpt.headline_needles() if b == "body") + "\n"
    assert bpt._headline_block_issues("main.tex", text) == []


def test_a_body_needle_missing_from_the_whole_file_is_reported():
    text = "\\begin{abstract}\n" + " ".join(
        f"${t}$" for b, _, _, _, t in bpt.headline_needles() if b == "abstract") + "\n\\end{abstract}\n"
    issues = bpt._headline_block_issues("main.tex", text)
    assert issues and all("the body does not quote" in i for i in issues)
    assert len(issues) == sum(1 for b, *_ in bpt.headline_needles() if b == "body")


def test_check_manuscript_reports_the_moved_abstract_number(tmp_path, capsys):
    (tmp_path / "main.tex").write_text(_fixture(bpt.headline_needles(), ("$0.873$", "$0.871$")),
                                       encoding="utf-8")
    bpt.check_manuscript(str(tmp_path))
    out = capsys.readouterr().out
    assert "the abstract does not quote 0.873 (L1 oracle; artifacts/expL1/organism_summary.json" in out
