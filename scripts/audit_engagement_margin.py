"""Engagement-gate margin sensitivity, offline (no compute).

FINDINGS methods note 2: ENGAGE_MARGIN = 0.15 and LIFE_TOL = 2.0 were frozen from the
B-v2 de-risk and never swept. Every run's cell files persist the engagement inputs
(trained / random / scripted true returns and lifetimes), so the gate can be
re-adjudicated at other margins without rerunning anything. This prints, per run and
per margin, how many (drift, seed) cells pass, and writes a JSON summary.

A source is a run's cells/ directory, or a promoted JSON that carries the same per-cell
inputs as a list under "engagement" (artifacts/expL1/organism_summary.json). Paths are
recorded repo-relative (a path through fullruns/ as fullruns/<rest>).

Only the survival objective's rule is swept: trained return at least the better baseline plus
the margin, and trained lifetime no more than LIFE_TOL short of the random policy's. A cell
whose engagement block records another objective (the goal-and-stakes "touch" objective uses
a return ratio on a pellet count, itasorl.experiment_b2.engagement_rule) is not
re-adjudicated: it is counted under "n_skipped_objective" and left out of every pass count.

--with-inputs also writes each counted cell's engagement inputs, so a sweep over a
gitignored bundle can be recomputed from the committed JSON alone (rows_from_report, then
sweep).

Usage:
    python scripts/audit_engagement_margin.py --json fullruns/engagement_margin.json \\
        fullruns/l3_h8_heldout/cells fullruns/l3_h7_heldout/cells fullruns/07062026/cells
    python scripts/audit_engagement_margin.py --with-inputs \\
        --json artifacts/engagement_margin_gpu_runs.json \\
        fullruns/l3_n10/cells fullruns/l3_h8_traces/cells fullruns/l3_h7_traces/cells \\
        fullruns/l3_h4_traces/cells fullruns/l3_h8_heldout/cells fullruns/l3_h7_heldout/cells \\
        artifacts/reviewer_gaps_runs/l3_h10_gseed1_gpu/cells \\
        artifacts/reviewer_gaps_runs/l3_h8_gseed2_gpu/cells \\
        artifacts/reviewer_gaps_runs/l3_h8_nowm_skill_u450/cells \\
        artifacts/expL1/organism_summary.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from promote_bv3_gates import repo_rel  # noqa: E402

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

MARGINS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
LIFE_TOL = 2.0
SWEPT_OBJECTIVE = "survival"
INPUT_FIELDS = ("trained_return", "random_return", "scripted_return", "trained_len", "random_len")


def readjudicate(eng: dict, margin: float, life_tol: float = LIFE_TOL) -> bool:
    """The survival objective's engagement rule at `margin`. Refuses any other objective."""
    objective = eng.get("objective", SWEPT_OBJECTIVE)
    if objective != SWEPT_OBJECTIVE:
        raise ValueError(f"engagement objective {objective!r} has its own rule; this sweep "
                         f"re-adjudicates only {SWEPT_OBJECTIVE!r} cells")
    better = eng["trained_return"] >= max(eng["random_return"], eng["scripted_return"]) + margin
    not_worse = eng["trained_len"] >= eng["random_len"] - life_tol
    return bool(better and not_worse)


def rows_from_report(rep: dict) -> list[dict]:
    """[{drift, seed, eng}] back from one report written with --with-inputs."""
    return [{"drift": float(r["drift"]), "seed": int(r["seed"]),
             "eng": {k: r[k] for k in r if k not in ("drift", "seed")}} for r in rep["inputs"]]


def load_rows(source: str) -> list[dict]:
    """[{drift, seed, eng}] from a cells/ directory or a promoted JSON with an "engagement"
    list. A file this script wrote is read back per report with rows_from_report."""
    if os.path.isdir(source):
        rows = []
        for p in sorted(glob.glob(os.path.join(source, "cell_*.json"))):
            with open(p, encoding="utf-8") as fh:
                c = json.load(fh)["cell"]
            rows.append({"drift": float(c["drift"]), "seed": int(c["seed"]), "eng": c["eng"]})
        return rows
    with open(source, encoding="utf-8") as fh:
        doc = json.load(fh)
    if isinstance(doc, dict) and isinstance(doc.get("engagement"), list):
        return [{"drift": float(e["drift"]), "seed": int(e["seed"]), "eng": e}
                for e in doc["engagement"]]
    raise SystemExit(f"{source}: neither a cells/ directory nor a JSON with per-cell engagement")


def drift_key(d: float) -> str:
    """Two decimals for the registered drifts (0.00, 0.45); full precision otherwise, so the
    L1 grid 0.023 is not printed as 0.02."""
    return f"{d:.2f}" if abs(d - round(d, 2)) < 1e-12 else f"{d:g}"


def sweep(rows: list[dict], margins=MARGINS) -> dict:
    """{margin: {drift: {pass, n, min_gap}}} over survival-objective rows."""
    out: dict[str, dict] = {}
    for m in margins:
        by_drift: dict[str, dict] = {}
        for r in rows:
            k = drift_key(r["drift"])
            by_drift.setdefault(k, {"pass": 0, "n": 0, "min_gap": None})
            by_drift[k]["n"] += 1
            by_drift[k]["pass"] += int(readjudicate(r["eng"], m))
            gap = r["eng"]["trained_return"] - max(r["eng"]["random_return"], r["eng"]["scripted_return"])
            g0 = by_drift[k]["min_gap"]
            by_drift[k]["min_gap"] = gap if g0 is None else min(g0, gap)
        out[f"{m:.2f}"] = by_drift
    return out


def summarize(source: str, margins=MARGINS, *, with_inputs: bool = False) -> dict:
    rows = load_rows(source)
    counted = [r for r in rows if r["eng"].get("objective", SWEPT_OBJECTIVE) == SWEPT_OBJECTIVE]
    key = "cells_dir" if os.path.isdir(source) else "source"
    out = {key: repo_rel(source), "n_cells": len(counted), "by_margin": sweep(counted, margins)}
    if len(counted) != len(rows):
        out["objective"] = SWEPT_OBJECTIVE
        out["n_skipped_objective"] = len(rows) - len(counted)
    if with_inputs:
        out["inputs"] = [{"drift": r["drift"], "seed": r["seed"],
                          **{f: r["eng"][f] for f in INPUT_FIELDS}}
                         for r in sorted(counted, key=lambda r: (r["drift"], r["seed"]))]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", metavar="cells_dir_or_json")
    ap.add_argument("--json", default=None)
    ap.add_argument("--with-inputs", action="store_true",
                    help="record each counted cell's engagement inputs in the JSON")
    a = ap.parse_args(argv)
    report = [summarize(d, with_inputs=a.with_inputs) for d in a.sources]
    for rep in report:
        src = rep.get("cells_dir", rep.get("source"))
        skipped = (f"; {rep['n_skipped_objective']} non-survival cells skipped"
                   if rep.get("n_skipped_objective") else "")
        print(f"\n{src}  ({rep['n_cells']} cells{skipped})")
        for m, by_drift in rep["by_margin"].items():
            line = "  ".join(f"drift {d}: {v['pass']}/{v['n']} (min gap {v['min_gap']:+.3f})"
                             for d, v in sorted(by_drift.items()))
            print(f"  margin {m}: {line}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
