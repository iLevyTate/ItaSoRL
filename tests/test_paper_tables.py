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


# No 0.752 and no "historical" here: each test adds its own, in its own section, so the
# fixture cannot label a number that the test means to be unlabeled.
GOOD = (r"\input{../paper_tables/corrected_runs}" + "\n"
        "The survival agent reads HEAD.\n")


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
    text = GOOD.replace("HEAD", _headline_sentence())
    text += "\n\\subsection{Results}\nthe signal reads 0.752 on GPU\n"
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert "0.752" in out and "label" in out


def test_strip_tex_removes_commands_but_keeps_their_text():
    assert bpt.strip_tex(r"survival-\emph{specific} and \textbf{bold} \cite{x}") == \
        "survival-specific and bold x"


def test_historical_headline_is_accepted_when_its_paragraph_carries_the_label(tmp_path, capsys):
    """A paragraph headed "The historical headline" labels every number inside it; the label
    need not repeat on the line that quotes the number."""
    text = GOOD.replace("HEAD", _headline_sentence())
    text += ("\n\\paragraph{The historical headline.} The pre-registered run read\n"
             "survival & \\textbf{0.752} & 8/10\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0.752" not in out


def test_historical_headline_is_still_flagged_in_an_unlabeled_paragraph(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence())
    text += "\n\\subsection{Cross-recipe transfer}\nthe pools reproduce 0.752 exactly\n"
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert "0.752" in out


def test_a_retired_phrase_is_not_flagged_where_the_sentence_withdraws_it(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence())
    text += ("\nthe evidence does not support that it loads on the learned texture rather than\n"
             "on coherent deviation from the true law\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 0, out


def test_the_same_retired_phrase_is_flagged_when_asserted(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence())
    text += "\nthe signal loads on the learned texture\n"
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert "loads on the learned texture" in out


def test_recomputes_every_number_is_accepted_when_the_scope_is_named(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence())
    text += ("\na verification script recomputes every number in the project's findings document\n"
             "and claims inventory from those artifacts\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 0, out


def test_recomputes_every_number_is_flagged_when_unscoped(tmp_path, capsys):
    text = GOOD.replace("HEAD", _headline_sentence())
    text += "\na verification script recomputes every number in the project\n"
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert "recomputes every number" in out



def test_the_title_is_a_name_not_a_claim_and_is_not_matched_for_retired_wording(tmp_path, capsys):
    """"Detectable Is Not Encoded" names the paper's contrast; the prose guard still applies."""
    text = (r"\title{Detectable Is Not Encoded: Incidental Encoding \ of Simulator Authenticity%" + "\n"
            + GOOD.replace("HEAD", _headline_sentence())
            + "the L2 artifact is not encoded in the state\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("retired wording") == 1 and ":4:" in out


def test_the_generated_verdict_table_does_not_lead_with_met_for_a_run_whose_rule_is_not_met():
    """C1's verdict string starts "MET on the decodability clauses"; a reader skimming the
    rendered column sees MET for a run the registered rule is not met on. The table must say
    both things in their own columns, derived from the artifact's own fields."""
    with open(os.path.join(ROOT, "artifacts", "corrected_verdicts.json"), encoding="utf-8") as fh:
        v = json.load(fh)
    tex = bpt.corrected_table()
    assert tex is not None
    assert "Registered rule" in tex and "Decodability" in tex
    for name, r in v["runs"].items():
        if r.get("status") != "complete":
            continue
        row = next(ln for ln in tex.splitlines() if ln.startswith(name + " &"))
        assert not r["primary"]["met"], "fixture assumes neither corrected run meets the rule"
        assert "NOT MET" in row, name
        assert not row.split("&")[-2].strip().startswith("MET on"), name
        for g in r["gates"]["failed"]:
            assert g.upper() in row.upper(), (name, g)


def test_an_interval_bound_is_not_a_quoted_historical_headline(tmp_path, capsys):
    """A confidence bound that happens to equal a historical headline is not a claim about it.
    The drift-0.45 draw interval is [0.752, 0.808]; its lower bound collides with the historical
    hidden-8 headline 0.752 and must not be flagged."""
    text = (GOOD.replace("HEAD", _headline_sentence())
            + "\n\subsection{Draws}\nthe eight draw means give an interval of "
              "\ci{0.752}{0.808} over draws\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert "historical headline" not in out, out
    assert rc == 0


def test_a_bare_historical_headline_is_still_flagged(tmp_path, capsys):
    text = (GOOD.replace("HEAD", _headline_sentence())
            + "\n\subsection{Results}\nthe signal reads 0.752 on GPU\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    assert rc == 1 and "historical headline" in capsys.readouterr().out
