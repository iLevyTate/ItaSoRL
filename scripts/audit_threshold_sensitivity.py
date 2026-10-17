"""Threshold sensitivity of the rule ledger's bar and margin clauses (G06). Sensitivity only.

The 0.65 bar and the 0.05 margin were set by convention, not derived from power. This script
loads docs/rule_ledger.json through scripts/build_paper_tables.py, which recomputes every
clause value from the committed artifacts, and re-reads each row's rule clauses with the bar
clauses moved over {0.60, 0.65, 0.70} and the margin clauses over {0.03, 0.05, 0.10}. Gate
clauses never move, and neither do these rule clauses: TOST, gate 0, speed, leakage, "C1 minus
C2" and the lead clauses. A row holds at a grid point when every one of its rule clauses passes
there; a clause that was never measured counts as not passing, as in the ledger.

The rows counted are the adjudicated ones, outcome "threshold met" or "threshold not met"; the
uninformative and no-verdict rows are listed apart. At (0.65, 0.05) the recomputed hold flag of
every row must equal the ledger's recorded clauses_hold, and the hold and fail counts there are
read from the ledger, never typed here; the script exits 1 otherwise. No ledger reading,
outcome or verdict changes: the output is labeled "sensitivity only".

Usage:
    python scripts/audit_threshold_sensitivity.py [--out artifacts/threshold_sensitivity/bar_margin.json]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import re
import sys

import build_paper_tables as bpt

BARS = (0.60, 0.65, 0.70)
MARGINS = (0.03, 0.05, 0.10)
REFERENCE = (0.65, 0.05)
LABEL = "sensitivity only"
ADJUDICATED = ("threshold met", "threshold not met")
# Rule clauses that keep their frozen threshold whatever the grid point.
EXCLUDED = re.compile(r"TOST|gate 0|speed|leak|C1 minus C2|\blead\b", re.I)


def clause_kind(c: dict) -> str | None:
    """"bar", "margin", or None for a clause that does not move."""
    if c["role"] != "rule" or EXCLUDED.search(f"{c['clause']} {c['basis']}"):
        return None
    if c["threshold"] == REFERENCE[0] and c["op"] in (">=", ">"):
        return "bar"
    if c["threshold"] == REFERENCE[1] and c["op"] in (">=", ">"):
        return "margin"
    return None


def _passes(c: dict, bar: float, margin: float) -> bool:
    if c["computed"] is None:
        return False
    if c["op"] == "is_true":
        return bool(c["computed"])
    kind = clause_kind(c)
    thr = bar if kind == "bar" else margin if kind == "margin" else c["threshold"]
    return bool(bpt._OPS[c["op"]](c["computed"], thr))


def row_holds(row: dict, bar: float, margin: float) -> bool | None:
    """Whether every rule clause passes; None for a row with no rule clause."""
    rule = [c for c in row["clauses"] if c["role"] == "rule"]
    return all(_passes(c, bar, margin) for c in rule) if rule else None


def sweep(ledger: dict | None = None) -> dict:
    ledger = ledger if ledger is not None else bpt.load_ledger()
    if ledger is None:
        raise SystemExit("docs/rule_ledger.json is absent")
    rows, problems = bpt.evaluate_ledger(ledger)
    if problems:
        raise SystemExit("the ledger does not evaluate cleanly:\n  " + "\n  ".join(problems))
    adj = [r for r in rows if r["outcome_class"] in ADJUDICATED]
    ref = {r["id"]: row_holds(r, *REFERENCE) for r in rows}
    mismatches = [r["id"] for r in rows if r.get("clauses_hold") is not None
                  and ref[r["id"]] is not None and ref[r["id"]] != r["clauses_hold"]]
    recorded = {"holds": sum(r.get("clauses_hold") is True for r in adj),
                "fails": sum(r.get("clauses_hold") is not True for r in adj)}
    points = []
    for bar in BARS:
        for margin in MARGINS:
            h = {r["id"]: row_holds(r, bar, margin) for r in adj}
            points.append({
                "bar": bar, "margin": margin,
                "holds": sum(v is True for v in h.values()),
                "fails": sum(v is not True for v in h.values()),
                "flips": [{"id": r["id"], "rule": r["rule"], "run": r["run"],
                           "holds_at_reference": ref[r["id"]], "holds_here": h[r["id"]]}
                          for r in adj if h[r["id"]] != ref[r["id"]]]})
    at_ref = next(p for p in points if (p["bar"], p["margin"]) == REFERENCE)
    kinds = [clause_kind(c) for r in rows for c in r["clauses"]]
    return {
        "label": LABEL,
        "about": ("Each adjudicated rule-and-run row of docs/rule_ledger.json re-read with its bar "
                  "clauses at each bar and its margin clauses at each margin; gate clauses and the "
                  "excluded rule clauses keep their frozen thresholds. Descriptive: no ledger reading, "
                  "outcome or verdict changes."),
        "ledger": "docs/rule_ledger.json",
        "grid": {"bar": list(BARS), "margin": list(MARGINS), "reference": list(REFERENCE)},
        "excluded_rule_clauses": EXCLUDED.pattern,
        "clauses_moved": {"bar": kinds.count("bar"), "margin": kinds.count("margin")},
        "ledger_tally": bpt._tally(rows),
        "n_rows": len(rows), "n_adjudicated": len(adj),
        "reference": {"recorded": recorded, "recomputed": {"holds": at_ref["holds"], "fails": at_ref["fails"]},
                      "clauses_hold_mismatches": mismatches,
                      "reproduces": not mismatches and recorded["holds"] == at_ref["holds"]
                      and recorded["fails"] == at_ref["fails"] and not at_ref["flips"]},
        "points": points,
        "not_adjudicated": [{"id": r["id"], "outcome_class": r["outcome_class"],
                             "holds_at_reference": ref[r["id"]]} for r in rows if r not in adj],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write the sweep here (JSON)")
    a = ap.parse_args(argv)
    res = sweep()
    ref = res["reference"]
    print(f"threshold sensitivity ({LABEL}): {res['n_adjudicated']} adjudicated rows of {res['n_rows']}; "
          f"{res['clauses_moved']['bar']} bar and {res['clauses_moved']['margin']} margin clauses move")
    print(f"  at bar {REFERENCE[0]:.2f}, margin {REFERENCE[1]:.2f}: holds on {ref['recomputed']['holds']} "
          f"and fails on {ref['recomputed']['fails']} (ledger: {ref['recorded']['holds']} and "
          f"{ref['recorded']['fails']})")
    for p in res["points"]:
        names = ", ".join(f"{f['id']} {'->holds' if f['holds_here'] else '->fails'}" for f in p["flips"])
        print(f"  bar {p['bar']:.2f} margin {p['margin']:.2f}: holds {p['holds']:2d} fails {p['fails']:2d}"
              + (f"  flips: {names}" if names else ""))
    if a.out:
        from agent_free_common import provenance, write_json
        res["provenance"] = provenance("scripts/audit_threshold_sensitivity.py", commit_key="git_commit",
                                       output=a.out)
        write_json(a.out, res)
        print(f"wrote {a.out}")
    if not ref["reproduces"]:
        print("FAIL: the reference point does not reproduce the ledger's recorded clauses_hold "
              f"(rows {ref['clauses_hold_mismatches'] or 'counts differ'})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
