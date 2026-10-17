"""scripts/audit_threshold_sensitivity.py (G06): the bar and margin sweep over the rule ledger is
labeled sensitivity only, reproduces the ledger at (0.65, 0.05), moves only bar and margin rule
clauses, and never touches gates or the excluded clauses."""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import audit_threshold_sensitivity as ats  # noqa: E402
import build_paper_tables as bpt  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return ats.sweep()


def test_the_reference_point_reproduces_the_ledger(res):
    ledger = bpt.load_ledger()
    adj = [r for r in ledger["rows"] if r["outcome_class"] in ats.ADJUDICATED]
    ref = res["reference"]
    assert ref["reproduces"] and ref["clauses_hold_mismatches"] == []
    # the split is read from the ledger, not typed
    assert ref["recorded"]["holds"] == sum(r["clauses_hold"] is True for r in adj)
    assert ref["recomputed"] == ref["recorded"]
    assert res["ledger_tally"] == bpt._tally(bpt.evaluate_ledger(ledger)[0])
    assert res["n_rows"] == len(ledger["rows"]) and res["n_adjudicated"] == len(adj)
    assert res["label"] == "sensitivity only"


def test_the_grid_and_monotonic_reading(res):
    assert [(p["bar"], p["margin"]) for p in res["points"]] == [(b, m) for b in ats.BARS for m in ats.MARGINS]
    by = {(p["bar"], p["margin"]): p["holds"] for p in res["points"]}
    for m in ats.MARGINS:
        assert by[(0.60, m)] >= by[(0.65, m)] >= by[(0.70, m)]
    for b in ats.BARS:
        assert by[(b, 0.03)] >= by[(b, 0.05)] >= by[(b, 0.10)]
    assert all(p["holds"] + p["fails"] == res["n_adjudicated"] for p in res["points"])
    assert next(p for p in res["points"] if (p["bar"], p["margin"]) == ats.REFERENCE)["flips"] == []


def test_gates_and_excluded_clauses_never_move():
    ledger = bpt.load_ledger()
    for r in ledger["rows"]:
        for c in r["clauses"]:
            kind = ats.clause_kind(c)
            if c["role"] != "rule":
                assert kind is None
            text = f"{c['clause']} {c['basis']}"
            if any(w in text for w in ("TOST", "C1 minus C2", "lead over")):
                assert kind is None, text


def _row(rid, clauses, outcome="threshold met", holds=True):
    return {"id": rid, "rule": rid, "short": rid, "source": "docs/rule_ledger.json", "source_part": "x",
            "frozen": "", "run": "R", "trainer": "corrected", "n": 10, "clauses": clauses,
            "clauses_hold": holds, "gates_hold": True, "outcome": "", "outcome_class": outcome,
            "recorded_in": ""}


def _cl(name, basis, op, thr, val, role="rule"):
    return {"clause": name, "basis": basis, "op": op, "threshold": thr, "value": {"const": val},
            "role": role}


def test_a_row_near_the_bar_flips_and_a_gate_does_not():
    ledger = {"about": "", "rows": [
        _row("NEAR", [_cl("bar", "mean", ">=", 0.65, 0.66), _cl("margin over untrained", "difference of means", ">=", 0.05, 0.06),
                      _cl("L0 gate", "TOST p, margin 0.05", "<", 0.05, 0.01, role="gate")]),
        _row("LEAD", [_cl("bar", "mean", ">=", 0.65, 0.9), _cl("lead over the no-auxiliary run", "difference of means", ">", 0.05, 0.04)],
             outcome="threshold not met", holds=False),
        _row("OPEN", [_cl("bar", "mean", ">=", 0.65, 0.5)], outcome="uninformative", holds=False),
    ]}
    res = ats.sweep(ledger)
    pts = {(p["bar"], p["margin"]): p for p in res["points"]}
    assert res["reference"]["reproduces"] and res["n_adjudicated"] == 2
    assert [f["id"] for f in pts[(0.70, 0.05)]["flips"]] == ["NEAR"]
    assert [f["id"] for f in pts[(0.65, 0.10)]["flips"]] == ["NEAR"]
    # the lead clause keeps its frozen 0.05 even at margin 0.03, so LEAD never holds
    assert all("LEAD" not in [f["id"] for f in p["flips"]] for p in res["points"])
    assert [r["id"] for r in res["not_adjudicated"]] == ["OPEN"]


def test_a_ledger_whose_recorded_hold_disagrees_fails_the_gate(tmp_path, monkeypatch, capsys):
    ledger = copy.deepcopy(bpt.load_ledger())
    row = next(r for r in ledger["rows"] if r["clauses_hold"] is True and r["outcome_class"] == "threshold met")
    row["clauses_hold"] = False
    row["outcome_class"] = "threshold not met"
    monkeypatch.setattr(bpt, "load_ledger", lambda: ledger)
    monkeypatch.setattr(bpt, "evaluate_ledger", lambda led: (
        [{**r, "clauses": [{**c, "computed": bpt.ledger_value(c["value"], {}) if c["value"] else None}
                           for c in r["clauses"]]} for r in led["rows"]], []))
    assert ats.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_out_writes_labeled_json_with_provenance(tmp_path, capsys):
    out = tmp_path / "bar_margin.json"
    assert ats.main(["--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["label"] == "sensitivity only"
    assert doc["provenance"]["generated_by"] == "scripts/audit_threshold_sensitivity.py"
    assert "git_commit" in doc["provenance"] and "git_commit_at_promotion" not in doc["provenance"]
