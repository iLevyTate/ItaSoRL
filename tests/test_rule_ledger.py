"""The committed rule ledger (docs/rule_ledger.json, rendered to docs/RULE_LEDGER.md by
scripts/build_paper_tables.py): one row per frozen decision rule adjudicated at the 0.65 bar
or the 0.05 margin. Every row must name a frozen file that exists, carry the frozen date that
file states, and record readings and an outcome its artifacts reproduce."""

from __future__ import annotations

import copy
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_paper_tables as bpt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def ledger():
    led = bpt.load_ledger()
    if led is None:
        # The anonymized supplement may ship without the ledger; a checkout must not.
        assert not os.path.exists(os.path.join(ROOT, ".git")), "docs/rule_ledger.json is missing"
        pytest.skip("no rule ledger in this tree")
    return led


@pytest.fixture(scope="module")
def evaluated(ledger):
    return bpt.evaluate_ledger(ledger)


def test_every_row_names_an_existing_frozen_file(ledger):
    frozen = re.compile(r"^docs/(specs/\d{4}-\d{2}-\d{2}-[\w.-]+\.md|PREREGISTRATION\w*\.md)$")
    for row in ledger["rows"]:
        assert frozen.match(row["source"]), row["id"]
        assert os.path.exists(os.path.join(ROOT, row["source"])), row["id"]


def test_every_frozen_date_is_taken_from_the_frozen_file(ledger):
    for row in ledger["rows"]:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["frozen"]), row["id"]
        with open(os.path.join(ROOT, row["source"]), encoding="utf-8") as fh:
            assert row["frozen"] in fh.read(), (row["id"], row["frozen"])


def test_row_ids_are_unique_and_each_row_reads_the_bar_or_the_margin(ledger):
    ids = [r["id"] for r in ledger["rows"]]
    assert len(ids) == len(set(ids))
    for row in ledger["rows"]:
        thresholds = {c["threshold"] for c in row["clauses"] if c["role"] == "rule"}
        assert thresholds & {0.65, 0.05}, row["id"]


def test_machine_checkable_outcomes_match_their_artifacts(evaluated):
    rows, problems = evaluated
    assert problems == []
    assert len(rows) >= 30


def test_the_published_rule_ledger_page_is_current(ledger):
    assert bpt.check_ledger() == []


def test_the_primary_rows_say_what_the_verdict_artifact_says(evaluated):
    """The ledger may not state the corrected primary result more favourably than
    artifacts/corrected_verdicts.json: C1's decodability clauses hold, its L0 gate does not,
    and the row reads 'threshold not met'."""
    rows = {r["id"]: r for r in evaluated[0]}
    with open(os.path.join(bpt.ART, "corrected_verdicts.json"), encoding="utf-8") as fh:
        v = json.load(fh)
    c1 = rows["C1-PRIMARY"]
    assert c1["clauses_hold"] is (v["runs"]["C1"]["primary"]["decodability"] == "MET")
    assert c1["gates_hold"] is False and v["runs"]["C1"]["gates"]["failed"] == ["l0"]
    assert c1["outcome_class"] == "threshold not met" and not v["runs"]["C1"]["primary"]["met"]
    assert rows["AUXILIARY-COMPARISON"]["outcome_class"] == "threshold not met"
    assert v["auxiliary_comparison"]["holds"] is False


def test_the_boundary_cases_include_the_three_named_and_not_hidden_7(evaluated):
    """The cross-recipe lower bound (+0.004), the seed-2 predictor margin (-0.0009) and the
    drift-0.45 rank at the bound are within 0.01 of a threshold. Hidden 7 misses its margin by
    0.027 and is not a boundary case."""
    hits = {(r["id"], c["clause"], c["basis"]): c["distance"] for r, c, _ in bpt.boundary_cases(evaluated[0])}
    assert hits[("CROSS-RECIPE", "bar", "t90 lower bound")] == pytest.approx(0.0036, abs=5e-4)
    assert hits[("HIDDEN8-SEED2", "margin over predictor", "difference of means")] == pytest.approx(-0.0009, abs=5e-5)
    assert hits[("DRIFT045-WORLD-SAMPLES", "registered draw at or below the median", "rank of nine, 1 = largest")] == 0
    assert not any(rid == "L3-PRIMARY-H7" for rid, _, _ in hits)


def _ledger_with(ledger, rid, mutate):
    led = copy.deepcopy(ledger)
    row = next(r for r in led["rows"] if r["id"] == rid)
    mutate(row)
    return led


def test_a_recorded_outcome_the_artifact_contradicts_is_reported(ledger):
    led = _ledger_with(ledger, "HIDDEN8-SEED2", lambda r: r.update(clauses_hold=True))
    _, problems = bpt.evaluate_ledger(led)
    assert any("HIDDEN8-SEED2" in p and "clauses_hold" in p for p in problems)


def test_a_reading_the_artifact_does_not_reproduce_is_reported(ledger):
    def bump(r):
        r["clauses"][0]["reading"] = "0.744"
    led = _ledger_with(ledger, "C1-PRIMARY", bump)
    _, problems = bpt.evaluate_ledger(led)
    assert any("C1-PRIMARY" in p and "recorded as 0.744" in p for p in problems)


def test_a_missing_frozen_file_is_reported(ledger):
    led = _ledger_with(ledger, "CROSS-RECIPE",
                       lambda r: r.update(source="docs/specs/2026-07-15-no-such-spec.md"))
    _, problems = bpt.evaluate_ledger(led)
    assert any("CROSS-RECIPE" in p and "does not exist" in p for p in problems)


def test_the_ledger_table_renders_from_the_committed_ledger(ledger):
    tex = bpt.rule_ledger_table()
    assert tex is not None and "do not edit" in tex
    assert "rule_ledger.tex" in bpt.tables()


def test_the_rules_adjudicated_on_2026_10_11_are_in_the_ledger_as_recorded(evaluated):
    """Goal and stakes (FINDINGS 18), the registered-trunk white-jitter run (18.6) and the three
    corrected-trainer reruns (20.2): each row's outcome follows from its frozen decision output
    or promoted summary."""
    rows = {r["id"]: r for r in evaluated[0]}
    with open(os.path.join(bpt.ART, "goal_stakes", "decision.json"), encoding="utf-8") as fh:
        dec = json.load(fh)
    want = {"GOAL-STAKES-TOUCH": "threshold met", "GOAL-STAKES-IMMORTAL": "threshold not met",
            "GOAL-STAKES-STAKES": "no verdict", "WHITE-JITTER-H96": "uninformative",
            "H7-CORRECTED": "uninformative", "H10GS1-CORRECTED": "threshold not met",
            "H8GS2-CORRECTED": "uninformative"}
    assert {k: rows[k]["outcome_class"] for k in want} == want
    assert dec["T-touch"]["primary"]["met"] is True and rows["GOAL-STAKES-TOUCH"]["gates_hold"] is True
    assert dec["stakes"]["testable"] is False and rows["GOAL-STAKES-STAKES"]["clauses_hold"] is None
    # The stakes contrast was never measured: it is listed, and it does not pass.
    stakes = rows["GOAL-STAKES-STAKES"]["clauses"][0]
    assert stakes["computed"] is None and stakes["passes"] is False and stakes["threshold"] == 0.05
    # Hidden 7 is off-design on the gate-0 floor of its own fit; the other two pass gate 0.
    floor = {rid: next(c for c in rows[rid]["clauses"] if c["clause"].startswith("gate 0: untrained floor"))
             for rid in ("H7-CORRECTED", "H10GS1-CORRECTED", "H8GS2-CORRECTED")}
    assert not floor["H7-CORRECTED"]["passes"] and floor["H7-CORRECTED"]["computed"] == pytest.approx(0.1011, abs=1e-4)
    assert floor["H10GS1-CORRECTED"]["passes"] and floor["H8GS2-CORRECTED"]["passes"]
    with open(os.path.join(bpt.ART, "texture", "T_gn_l3_h96_wm.json"), encoding="utf-8") as fh:
        assert json.load(fh)["gates_failed"] == ["l0_tost"]


def test_an_unmeasured_clause_with_a_reading_is_reported(ledger):
    def bad(r):
        r["clauses"][0]["reading"] = "0.060"
    led = _ledger_with(ledger, "GOAL-STAKES-STAKES", bad)
    _, problems = bpt.evaluate_ledger(led)
    assert any("GOAL-STAKES-STAKES" in p and "has no value" in p for p in problems)


def test_the_tally_counts_every_row_once(evaluated):
    rows = evaluated[0]
    t = bpt._tally(rows)
    assert sum(t.values()) == len(rows)
    assert bpt._tally_sentence(rows).startswith(f"{len(rows)} rule-and-run rows")


def test_the_balanced_rows_carry_the_promoted_reason_and_the_echo_rows_are_unchanged(ledger):
    """The two uninformative balanced rows read 'UNINFORMATIVE: ' plus the reason the promoter
    writes, which names every gate that fails (H78); the descriptive sensory-echo rows for the
    same runs keep their outcome text."""
    rows = {r["id"]: r for r in ledger["rows"]}
    for rid, art in (("B1-BALANCED", "b1.json"), ("B2-BALANCED", "b2.json")):
        with open(os.path.join(bpt.ART, "balanced_confirmation", art), encoding="utf-8") as fh:
            doc = json.load(fh)
        row = rows[rid]
        assert row["outcome_class"] == "uninformative"
        assert row["outcome"] == "UNINFORMATIVE: " + doc["reason"]
        assert "untrained floor" in row["outcome"] and "matched speed" in row["outcome"]
        assert "Speed and leakage were stored for the pooled worlds only." not in row["note"]
    for rid in ("B1-BALANCED-ECHO", "B2-BALANCED-ECHO"):
        assert rows[rid]["outcome_class"] == "threshold met"
        assert rows[rid]["outcome"] == "passes: sensory-independent under matched worlds"


def test_every_row_has_a_timing_class_from_the_allowed_set(ledger):
    """Third blind review (2026-10-20): every row states whether its rule text was committed
    before (prospective), with (simultaneous) or after (retrospective) the first result of the
    run it decides."""
    assert bpt.TIMING_CLASSES == ("prospective", "simultaneous", "retrospective")
    for row in ledger["rows"]:
        assert row.get("timing") in bpt.TIMING_CLASSES, row["id"]
        assert isinstance(row.get("timing_note", ""), str), row["id"]


def test_the_timing_anchors_the_registration_timeline_dates(ledger):
    """Rows whose timing the manuscript's registration timeline states outright: the corrected
    confirmation frozen before any cell, the historical sensory echo first committed with its
    result, and the matched-world sensory-echo rule frozen after the raw outcomes."""
    rows = {r["id"]: r for r in ledger["rows"]}
    assert rows["C1-PRIMARY"]["timing"] == rows["C2-PRIMARY"]["timing"] == "prospective"
    assert rows["SENSORY-ECHO-H8"]["timing"] == "simultaneous"
    assert rows["B1-BALANCED-ECHO"]["timing"] == rows["B2-BALANCED-ECHO"]["timing"] == "retrospective"


def test_a_missing_or_unknown_timing_is_reported(ledger):
    for bad in ("pre-registered", None):
        led = _ledger_with(ledger, "CROSS-RECIPE", lambda r: r.update(timing=bad))
        _, problems = bpt.evaluate_ledger(led)
        assert any("CROSS-RECIPE" in p and "timing" in p for p in problems)


def test_the_ledger_table_has_a_timing_column_within_the_old_width(ledger, evaluated):
    """The timing column is the fourth of seven, and the p{} widths still sum to 0.825 of the
    text width, so the table keeps its one float page; the timing tally is under the table."""
    tex = bpt.rule_ledger_table()
    spec = next(ln for ln in tex.splitlines() if ln.startswith(r"\begin{tabular}"))
    widths = [float(w) for w in re.findall(r"p\{([0-9.]+)\\textwidth\}", spec)]
    assert len(widths) == 7 and sum(widths) == pytest.approx(0.825, abs=1e-9)
    assert r"Rule & Frozen & Run & Timing & Bar on & Outcome & Deciding clause \\" in tex
    assert r"\multicolumn{7}" in tex and bpt._timing_sentence(evaluated[0]) in tex
    body = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and " & " in ln][1:]
    assert len(body) == len(ledger["rows"])
    assert all(ln.split(" & ")[3] in bpt.TIMING_CLASSES for ln in body)
