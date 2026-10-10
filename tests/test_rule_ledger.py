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
