"""Engagement-gate margin sensitivity, offline (no compute).

FINDINGS methods note 2: ENGAGE_MARGIN = 0.15 and LIFE_TOL = 2.0 were frozen from the
B-v2 de-risk and never swept. Every run's cell files persist the engagement inputs
(trained / random / scripted true returns and lifetimes), so the gate can be
re-adjudicated at other margins without rerunning anything. This prints, per run and
per margin, how many (drift, seed) cells pass, and writes a JSON summary.

Usage:
    python scripts/audit_engagement_margin.py --json fullruns/engagement_margin.json \\
        fullruns/l3_h8_heldout/cells fullruns/l3_h7_heldout/cells fullruns/07062026/cells
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os

MARGINS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
LIFE_TOL = 2.0


def readjudicate(eng: dict, margin: float, life_tol: float = LIFE_TOL) -> bool:
    better = eng["trained_return"] >= max(eng["random_return"], eng["scripted_return"]) + margin
    not_worse = eng["trained_len"] >= eng["random_len"] - life_tol
    return bool(better and not_worse)


def summarize(cells_dir: str, margins=MARGINS) -> dict:
    rows = []
    for p in sorted(glob.glob(os.path.join(cells_dir, "cell_*.json"))):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        rows.append({"drift": float(c["drift"]), "seed": int(c["seed"]), "eng": c["eng"]})
    out = {"cells_dir": cells_dir.replace("\\", "/"), "n_cells": len(rows), "by_margin": {}}
    for m in margins:
        by_drift: dict[str, dict] = {}
        for r in rows:
            k = f"{r['drift']:.2f}"
            by_drift.setdefault(k, {"pass": 0, "n": 0, "min_gap": None})
            by_drift[k]["n"] += 1
            by_drift[k]["pass"] += int(readjudicate(r["eng"], m))
            gap = r["eng"]["trained_return"] - max(r["eng"]["random_return"], r["eng"]["scripted_return"])
            g0 = by_drift[k]["min_gap"]
            by_drift[k]["min_gap"] = gap if g0 is None else min(g0, gap)
        out["by_margin"][f"{m:.2f}"] = by_drift
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells_dirs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    report = [summarize(d) for d in a.cells_dirs]
    for rep in report:
        print(f"\n{rep['cells_dir']}  ({rep['n_cells']} cells)")
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
