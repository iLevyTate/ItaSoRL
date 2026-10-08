"""The manuscript checker (scripts/build_paper_tables.py --manuscript) must guard something:
it fails when no generated table is \\input, compares the headline numbers against the
verdict artifact, and sees retired wording through LaTeX markup (revision audit item 2)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_paper_tables as bpt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _headline_sentence() -> str:
    """A sentence quoting every headline number the checker expects, read from the artifact."""
    return " ".join(f"{v:.3f}" for _, v in bpt.headline_numbers())


def _write(d, text: str, name: str = "main.tex") -> str:
    p = d / name
    p.write_text(text, encoding="utf-8")
    return str(d)


GOOD = (r"\input{../paper_tables/corrected_runs}" + "\n"
        "The survival agent reads HEAD. Historical 0.752 is labeled historical here.\n")


def test_headline_numbers_come_from_the_verdict_artifact():
    with open(os.path.join(ROOT, "artifacts", "corrected_verdicts.json"), encoding="utf-8") as fh:
        v = json.load(fh)
    labels = dict(bpt.headline_numbers())
    assert labels["C1 survival mean"] == v["runs"]["C1"]["primary"]["survival"]["mean"]
    assert labels["C1 minus C2 paired mean"] == v["auxiliary_comparison"]["paired"]["mean"]
    assert len(labels) >= 8


def test_manuscript_that_inputs_no_generated_table_is_flagged(tmp_path, capsys):
    rc = bpt.check_manuscript(_write(tmp_path, "Just prose with " + _headline_sentence() + "\n"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "no generated table" in out


def test_manuscript_with_tables_and_numbers_and_clean_wording_passes(tmp_path, capsys):
    rc = bpt.check_manuscript(_write(tmp_path, GOOD.replace("HEAD", _headline_sentence())))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0 item(s) to review" in out


def test_missing_headline_number_is_reported(tmp_path, capsys):
    numbers = [f"{v:.3f}" for _, v in bpt.headline_numbers()]
    without_first = " ".join(numbers[1:])
    rc = bpt.check_manuscript(_write(tmp_path, GOOD.replace("HEAD", without_first)))
    out = capsys.readouterr().out
    assert rc == 1
    assert "headline number" in out and numbers[0] in out


def test_retired_wording_is_seen_through_latex_markup(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence()) + r"a survival-\emph{specific} signal" + "\n"
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert "survival" in out and "retired wording" in out


def test_newly_retired_phrases_are_flagged(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence())
    text += ("a stored component of the state\n"
             "it loads on the learned texture\n"
             "which objective does the encoding\n"
             "every number reported here is recomputed\n")
    bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert out.count("retired wording") == 4


def test_unlabeled_historical_headline_is_flagged(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence()) + "the signal reads 0.752 on GPU\n"
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert "0.752" in out and "label" in out


def test_strip_tex_removes_commands_but_keeps_their_text():
    assert bpt.strip_tex(r"survival-\emph{specific} and \textbf{bold} \cite{x}") == \
        "survival-specific and bold x"
