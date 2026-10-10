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


def _headline_sentence(drop: str | None = None) -> str:
    """A sentence quoting every headline number the checker expects, and every headline
    interval right after its mean, read from the artifact. `drop` names one headline number
    (a headline_numbers() label) to leave out, together with any interval it anchors."""
    numbers = dict(bpt.headline_numbers())
    gone = f"{abs(numbers[drop]):.3f}" if drop else None
    parts = [f"{v:.3f}" for k, v in bpt.headline_numbers() if f"{abs(v):.3f}" != gone]
    for a in bpt.anchored_intervals():
        if a["required"] and f"{abs(a['mean']):.3f}" != gone:
            parts.append(f"{a['mean']:.3f} (\\ci{{{a['lo']:.3f}}}{{{a['hi']:.3f}}})")
    return " ".join(parts)


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


def test_a_manuscript_path_with_no_tex_files_fails(tmp_path, capsys):
    """A mistyped or empty --manuscript directory checked nothing and used to return 0."""
    assert bpt.check_manuscript(str(tmp_path)) == 1
    assert bpt.check_manuscript(str(tmp_path / "no-such-dir")) == 1
    assert "no .tex files" in capsys.readouterr().out


def test_manuscript_with_tables_and_numbers_and_clean_wording_passes(tmp_path, capsys):
    rc = bpt.check_manuscript(_write(tmp_path, GOOD.replace("HEAD", _headline_sentence())))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0 item(s) to review" in out


def test_missing_headline_number_is_reported(tmp_path, capsys):
    label, value = bpt.headline_numbers()[0]
    rc = bpt.check_manuscript(_write(tmp_path, GOOD.replace("HEAD", _headline_sentence(drop=label))))
    out = capsys.readouterr().out
    assert rc == 1
    assert "headline number" in out and f"{value:.3f}" in out


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
    """A title line is a name, not a claim, so the retired-wording guard skips it even when it
    holds a retired phrase. The prose below the title is still checked."""
    text = (r"\title{A Placeholder Name That Is Not Encoded: A Neutral Subtitle%" + "\n"
            + GOOD.replace("HEAD", _headline_sentence())
            + "the L2 artifact is not encoded in the state\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert out.count("retired wording") == 1 and ":4:" in out


def _arm(mean, lo, hi):
    return {"mean": mean, "t90": [lo, hi]}


def _synthetic_verdicts(art_dir) -> None:
    """One run that meets the rule and one that does not because its L0 gate is open, written
    where corrected_table() reads them, so the test exercises the builder and not the record."""
    def run(met, failed, dec, surv):
        return {"status": "complete",
                "primary": {"met": met, "decodability": dec, "survival": _arm(surv, surv - 0.05, surv + 0.05),
                            "predictor": _arm(0.58, 0.56, 0.60), "untrained": _arm(0.52, 0.50, 0.54)},
                "gates": {"failed": failed, "l0": {"mean": 0.56 if failed else 0.51,
                                                   "tost_p": 0.94 if failed else 0.01, "pass": not failed}},
                "drift_zero": {"survival": _arm(0.56, 0.55, 0.57), "predictor": _arm(0.51, 0.47, 0.54),
                               "untrained": _arm(0.49, 0.47, 0.52)}}
    v = {"runs": {"RUNA": run(True, [], "MET", 0.80), "RUNB": run(False, ["l0"], "MET", 0.73)},
         "auxiliary_comparison": {"paired": {"mean": 0.07, "t90": [0.02, 0.12]}, "holds": False,
                                  "holds_on_decodability_clauses": True}}
    (art_dir / "corrected_verdicts.json").write_text(json.dumps(v), encoding="utf-8")


def test_the_generated_verdict_table_does_not_lead_with_met_for_a_run_whose_rule_is_not_met(
        tmp_path, monkeypatch):
    """A verdict string that starts "MET on the decodability clauses" reads as a pass at a
    glance. The table must say both things in their own columns, from the artifact's own
    `met`, `decodability` and `gates.failed` fields, whatever the record currently holds."""
    _synthetic_verdicts(tmp_path)
    monkeypatch.setattr(bpt, "ART", str(tmp_path))
    tex = bpt.corrected_table()
    assert tex is not None
    assert "Registered rule" in tex and "Decodability" in tex
    rows = {ln.split("&")[0].strip(): ln for ln in tex.splitlines()
            if ln.split("&")[0].strip() in ("RUNA", "RUNB") and " & 0.45 & " in ln}
    assert set(rows) == {"RUNA", "RUNB"}
    rule = {k: ln.split("&")[-2].strip() for k, ln in rows.items()}
    assert rule["RUNA"] == r"\textsc{met}"
    assert rule["RUNB"] == r"\textsc{not met} (L0 open)"
    assert all(ln.split("&")[-1].strip().startswith(r"\textsc{met}") for ln in rows.values())
    aux = next(ln for ln in tex.splitlines() if ln.startswith("C1 $-$ C2"))
    assert r"\textsc{not met} as registered" in aux and "the lead holds on the decodability clauses" in aux


def test_an_interval_bound_is_not_a_quoted_historical_headline(tmp_path, capsys):
    """A confidence bound that happens to equal a historical headline is not a claim about it.
    The drift-0.45 draw interval is [0.752, 0.808]; its lower bound collides with the historical
    hidden-8 headline 0.752 and must not be flagged."""
    text = (GOOD.replace("HEAD", _headline_sentence())
            + "\n\\subsection{Draws}\nthe eight draw means give an interval of "
              "\\ci{0.752}{0.808} over draws\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert "historical headline" not in out, out
    assert rc == 0


def test_a_bare_historical_headline_is_still_flagged(tmp_path, capsys):
    text = (GOOD.replace("HEAD", _headline_sentence())
            + "\n\\subsection{Results}\nthe signal reads 0.752 on GPU\n")
    rc = bpt.check_manuscript(_write(tmp_path, text))
    assert rc == 1 and "historical headline" in capsys.readouterr().out


# ------------------------------------------------------------------ location-aware headline guard
# A fixture .tex built from the committed artifact, so CI runs these without docs/paper. The
# paragraph mirrors the manuscript's corrected primary-run paragraph; each test changes one
# number in it, the way a retyping slip would, and the checker must name the line.

def _c1() -> dict:
    return bpt._verdicts()["runs"]["C1"]


def _primary_paragraph() -> str:
    c1 = _c1()
    p, g = c1["primary"], c1["gates"]
    s, pr, un = p["survival"], p["predictor"], p["untrained"]
    cp, cu = p["contrast_vs_predictor"], p["contrast_vs_untrained"]
    seeds = ", ".join(f"{x:.3f}" for x in s["per_seed"])
    return (r"\paragraph{The corrected primary run.} The survival agent reads "
            rf"\textbf{{{s['mean']:.3f}}} (t-based 90\% CI \ci{{{s['t90'][0]:.3f}}}{{{s['t90'][1]:.3f}}}; "
            rf"per seed {seeds}), the predictor {pr['mean']:.3f} (\ci{{{pr['t90'][0]:.3f}}}{{{pr['t90'][1]:.3f}}}), "
            rf"and the untrained arm {un['mean']:.3f} (\ci{{{un['t90'][0]:.3f}}}{{{un['t90'][1]:.3f}}}). "
            rf"The seed-paired margins are ${cp['mean']:+.3f}$ over the predictor "
            rf"(\ci{{{cp['t90'][0]:+.3f}}}{{{cp['t90'][1]:+.3f}}}) and ${cu['mean']:+.3f}$ over the untrained "
            rf"arm (\ci{{{cu['t90'][0]:+.3f}}}{{{cu['t90'][1]:+.3f}}}). The speed control reads at least "
            rf"{g['speed']['min']:.3f}." + "\n\n")


def _fixture() -> str:
    return GOOD.replace("HEAD", _headline_sentence()) + "\n" + _primary_paragraph()


def _off_value(x: float) -> str:
    """A three-decimal value near `x` that no field of the C1 entry holds."""
    allowed = bpt._floats(_c1(), set())
    for k in range(1, 50):
        cand = f"{x + k / 1000:.3f}"
        if cand not in allowed:
            return cand
    raise AssertionError("no free value")


def test_the_fixture_paragraph_passes_as_written(tmp_path, capsys):
    rc = bpt.check_manuscript(_write(tmp_path, _fixture()))
    out = capsys.readouterr().out
    assert rc == 0, out


def test_one_wrong_occurrence_of_the_headline_mean_is_caught(tmp_path, capsys):
    """The guard used to test presence only: the headline mean quoted correctly anywhere hid a
    wrong copy in the paragraph a reviewer reads first."""
    mean = f"{_c1()['primary']['survival']['mean']:.3f}"
    bad = _off_value(float(mean))
    text = _fixture().replace(rf"\textbf{{{mean}}}", rf"\textbf{{{bad}}}", 1)
    assert mean in text                       # still quoted correctly elsewhere
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    line = 1 + next(i for i, ln in enumerate(text.splitlines()) if ln.startswith(r"\paragraph"))
    assert f"main.tex:{line}: interval" in out and f"number before it reads {bad}" in out


def test_one_changed_per_seed_value_is_caught(tmp_path, capsys):
    seeds = [f"{x:.3f}" for x in _c1()["primary"]["survival"]["per_seed"]]
    wrong = list(seeds)
    wrong[3] = _off_value(float(seeds[3]))
    text = _fixture().replace(", ".join(seeds), ", ".join(wrong), 1)
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert "per-seed list differs from the C1 per-seed survival in 1 place" in out


def test_a_changed_number_without_an_interval_is_caught_in_the_primary_paragraph(tmp_path, capsys):
    speed = f"{_c1()['gates']['speed']['min']:.3f}"
    bad = _off_value(float(speed))
    text = _fixture().replace(f"at least {speed}", f"at least {bad}", 1)
    rc = bpt.check_manuscript(_write(tmp_path, text))
    out = capsys.readouterr().out
    assert rc == 1
    assert f"{bad} in the corrected primary-run paragraph" in out


def test_a_headline_interval_never_quoted_after_its_mean_is_reported(tmp_path, capsys):
    """The drift-zero baselines of the registered pair are headline readings; quoting the
    numbers somewhere is not enough, each must sit beside its interval."""
    a = next(x for x in bpt.anchored_intervals() if x["label"] == "C1 drift-zero untrained")
    sentence = _headline_sentence().replace(
        f"{a['mean']:.3f} (\\ci{{{a['lo']:.3f}}}{{{a['hi']:.3f}}})", f"{a['mean']:.3f}")
    rc = bpt.check_manuscript(_write(tmp_path, GOOD.replace("HEAD", sentence)))
    out = capsys.readouterr().out
    assert rc == 1
    assert "C1 drift-zero untrained" in out and "not quoted as its mean followed by its interval" in out


# ------------------------------------------------------------------ the generated tables

def test_the_corrected_table_carries_the_drift_zero_readings_of_every_arm():
    """On the registered pair the untrained and predictor arms read 0.491 and 0.508 at
    drift zero, each scored on episodes its own actor generates; the table prints them beside
    the survival arm's 0.559 and the L0 gate."""
    tex = bpt.corrected_table()
    c1_zero = next(ln for ln in tex.splitlines() if ln.startswith("C1 & 0 &"))
    cells = [c.strip() for c in c1_zero.split("&")]
    assert cells[2] == "0.559 [0.549, 0.570]"
    assert cells[3] == "0.508 [0.475, 0.541]"
    assert cells[4] == "0.491 [0.466, 0.515]"
    assert cells[5] == "L0 open (TOST $p = 0.939$)"


def test_the_corrected_table_uses_display_labels_and_centred_numbers():
    """Gates print as "L0", never the artifact key "l0"; registered verdict words
    are set in small caps; the numeric columns are centred."""
    tex = bpt.corrected_table()
    assert "l0 open" not in tex and "(L0 open)" in tex
    assert r"\textsc{not met}" in tex and "NOT MET" not in tex
    assert r"\begin{tabular}{llccccc}" in tex
    assert "Decodability clauses (joint)" in tex
    assert r"\begin{tabular}{llccc}" in bpt.contrast_table()
    assert r"\begin{tabular}{lllcccc}" in bpt.gate_table()


def test_every_run_identifier_in_the_tables_has_a_plain_label():
    """The run IDs are manifest identifiers no reader of the paper has seen; the legend
    names each one, and a run added to either artifact without a label fails here."""
    legend = bpt.run_legend_table()
    for name in ("gate_table.json", "contrast_intervals.json"):
        for r in bpt._load(name)["rows"]:
            assert r["run"] in bpt.RUN_LABELS, r["run"]
            assert bpt._tex_escape(r["run"]) + " & " + bpt.RUN_LABELS[r["run"]] in legend


def _write_json(d, name, obj):
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


def test_duplicate_contrast_rows_name_the_run_they_repeat(tmp_path, monkeypatch):
    """A readout that reuses another run's agents and pools repeats its numbers; the
    table says so instead of presenting it as a second run."""
    def row(run, part, diffs):
        return {"run": run, "partition": part, "metric": "target", "survival_mean": 0.75,
                "survival_t90": [0.70, 0.80],
                "vs_predictor": {"diff_per_seed": diffs, "mean": 0.18, "t90": [0.11, 0.25]},
                "vs_untrained": {"diff_per_seed": diffs, "mean": 0.26, "t90": [0.21, 0.32]}}
    _write_json(tmp_path, "contrast_intervals.json", {"rows": [
        row("FIRST", "legacy", [0.1, 0.2]), row("SECOND", "legacy", [0.1, 0.2]),
        row("THIRD", "legacy", [0.3, 0.2])]})
    monkeypatch.setattr(bpt, "ART", str(tmp_path))
    tex = bpt.contrast_table()
    assert "SECOND (= FIRST) & legacy" in tex
    assert "THIRD & legacy" in tex and "THIRD (=" not in tex
    assert "not independent evidence" in tex


def test_gate_rows_that_read_one_set_of_agents_share_a_letter(tmp_path, monkeypatch):
    """Several run names read one set of drift-zero agents; the Set column and
    the count of distinct sets say so."""
    def row(run, mean, p):
        return {"run": run, "status": "historical", "partition": "explicit",
                "l0": {"n": 10, "mean": mean, "sd": 0.04, "tost_p": p, "rope_interval": [0.5, 0.56],
                       "status": "inconclusive (not shown equivalent)"}}
    _write_json(tmp_path, "gate_table.json", {"rows": [
        row("A", 0.539256, 0.207055), row("B", 0.539256, 0.207055), row("C", 0.546364, 0.374007)]})
    monkeypatch.setattr(bpt, "ART", str(tmp_path))
    tex = bpt.gate_table()
    letters = [ln.split("&")[-1].strip(" \\") for ln in tex.splitlines() if ln[:2] in ("A ", "B ", "C ")]
    assert letters == ["a", "a", "b"]
    assert "The 3 rows read 2 distinct sets" in tex
