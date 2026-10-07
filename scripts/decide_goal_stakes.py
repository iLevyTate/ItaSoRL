"""Apply the frozen wording rules of docs/specs/2026-10-07-goal-and-stakes-design.md to
the run artifacts. Prints the wording each result earns; changes nothing.

    python scripts/decide_goal_stakes.py --runs artifacts/goal_stakes --c1 artifacts/expB2/corrected_l3_h8_wm.json

Inputs per run directory: expB2_results.json (primary verdict via decide_h_b2 as the run
script prints it), mechanism.json (run_mechanism_readouts). The stakes contrast is the
seed-paired S-scarce minus S-immortal survival target (t-based 90% CI).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import re

import numpy as np

BAR = 0.65
SESOI = 0.05
INTERVENTION_MIN = 0.10
INTERVENTION_MIN_SEEDS = 7
SURPRISE_AUROC_MIN = 0.65
SURPRISE_CORR_MIN = 0.20

_CELL_NAME = re.compile(r"^cell_d(?P<drift>[0-9.]+)_s(?P<seed>\d+)\.json$")


def _excludes_zero(ci) -> bool:
    lo, hi = ci
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def intervention_verdict(summary: dict) -> dict:
    c = summary["intervention_real_minus_sham"]
    ok = (summary["n_informative"] >= INTERVENTION_MIN_SEEDS and c["mean"] >= INTERVENTION_MIN
          and _excludes_zero(c["ci90"]) and c["ci90"][0] > 0)
    wording = ("pushing the state along the decoded direction moves real-world behavior toward "
               "fake-world behavior; the direction is behaviorally live" if ok else
               "the decoded direction did not move behavior under the tested nudge")
    return {"pass": bool(ok), "wording": wording, "contrast": c,
            "n_informative": summary["n_informative"]}


def surprise_verdict(summary: dict) -> dict:
    a, c = summary["surprise_auroc"], summary["surprise_corr"]
    a_ok = a["mean"] >= SURPRISE_AUROC_MIN
    c_ok = abs(c["mean"]) >= SURPRISE_CORR_MIN and _excludes_zero(c["ci90"])
    if a_ok and c_ok:
        w = "the decoded direction tracks the agent's own prediction error"
    elif a_ok:
        w = ("the surrogate is surprising to the agent, but the decoded direction is not "
             "explained by that surprise")
    else:
        w = "prediction error does not separate the worlds for this agent"
    return {"auroc_met": bool(a_ok), "corr_met": bool(c_ok), "wording": w}


def adaptation_verdict(summary: dict) -> dict:
    g1, ad = summary["adaptation_gap_first"], summary["adaptation"]
    if not _excludes_zero(g1["ci90"]):
        return {"informative": False,
                "wording": "uninformative: the surrogate does not reduce intake in this run"}
    ok = ad["mean"] > 0 and _excludes_zero(ad["ci90"]) and ad["ci90"][0] > 0
    return {"informative": True, "pass": bool(ok),
            "wording": ("foraging in the fake world recovers within a lifetime" if ok else
                        "no within-lifetime recovery was detected")}


def stakes_verdict(means: dict, contrast: dict) -> dict:
    ordered = means["S-immortal"] < means["C1"] < means["S-scarce"]
    ok = ordered and contrast["mean"] >= SESOI and _excludes_zero(contrast["ci90"]) and contrast["ci90"][0] > 0
    return {"pass": bool(ok), "ordered": bool(ordered), "contrast": contrast,
            "wording": ("the reading grows with the stakes" if ok else
                        "within the tested range, stakes do not change the reading")}


def touch_wording(primary_met: bool, gates_pass: bool) -> str:
    if primary_met and gates_pass:
        return ("a pellet-directed goal without survival stakes produces walks from which the "
                "condition is decodable; survival is not required")
    return "pellet-seeking without stakes did not meet the registered criterion"


def _cell_seeds(cells_dir: str, drift: float) -> list[int] | None:
    """Seeds of the checkpointed cells at `drift`, ascending, or None when no cells dir.

    run_expB2 rebuilds its per-seed lists from sorted((drift, seed)), so ascending seed
    order at the chosen drift is exactly the list order in expB2_results.json.
    """
    if not os.path.isdir(cells_dir):
        return None
    seeds = []
    for path in glob.glob(os.path.join(cells_dir, "cell_d*_s*.json")):
        m = _CELL_NAME.match(os.path.basename(path))
        if m and abs(float(m.group("drift")) - drift) < 1e-9:
            seeds.append(int(m.group("seed")))
    return sorted(seeds)


def _survival_targets(results_path: str) -> dict:
    """Per-seed survival pooled target at the largest drift, keyed by seed.

    Two layouts are read. A run's expB2_results.json (scripts/run_expB2.py) is
    {str(drift): {arm: {metric: [per-seed list]}}} with no seed list; positions follow
    ascending seed, and the seed identities come from the sibling cells/ checkpoints when
    present, else 0..n-1. A promoted artifact (scripts/promote_reviewer_gaps_runs.py, e.g.
    artifacts/expB2/corrected_l3_h8_wm.json) copies that list by index into
    arms[dmax]["survival"]["pool_target"]["per_seed"]; its seeds are 0..n-1.
    """
    with open(results_path, encoding="utf-8") as fh:
        r = json.load(fh)
    if "arms" in r:
        dmax = str(r.get("dmax") or max(r["arms"], key=float))
        values = r["arms"][dmax]["survival"]["pool_target"]["per_seed"]
        seeds = None
    else:
        dmax = max(r, key=float)
        values = r[dmax]["survival"]["pool_target"]
        seeds = _cell_seeds(os.path.join(os.path.dirname(results_path), "cells"), float(dmax))
    if not seeds or len(seeds) != len(values):
        seeds = list(range(len(values)))
    return {int(s): float(v) for s, v in zip(seeds, values)}


def main() -> int:
    from itasorl.stats import t_ci90

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, help="directory holding T-touch/, S-immortal/, S-scarce/")
    ap.add_argument("--c1", required=True, help="artifacts/expB2/corrected_l3_h8_wm.json")
    a = ap.parse_args()
    report = {}
    for name in ("T-touch", "S-immortal", "S-scarce"):
        d = os.path.join(a.runs, name)
        if not os.path.isdir(d):
            report[name] = "not run"
            continue
        with open(os.path.join(d, "mechanism.json"), encoding="utf-8") as fh:
            mech = json.load(fh)["summary"]
        report[name] = {"intervention": intervention_verdict(mech),
                        "surprise": surprise_verdict(mech),
                        "adaptation": adaptation_verdict(mech)}
    im = os.path.join(a.runs, "S-immortal", "expB2_results.json")
    sc = os.path.join(a.runs, "S-scarce", "expB2_results.json")
    if os.path.exists(im) and os.path.exists(sc):
        t_im, t_sc = _survival_targets(im), _survival_targets(sc)
        t_c1 = _survival_targets(a.c1)
        seeds = sorted(set(t_im) & set(t_sc))
        diff = np.array([t_sc[s] - t_im[s] for s in seeds])
        means = {"S-immortal": float(np.mean([t_im[s] for s in seeds])),
                 "C1": float(np.mean([t_c1[s] for s in seeds if s in t_c1])),
                 "S-scarce": float(np.mean([t_sc[s] for s in seeds]))}
        report["stakes"] = stakes_verdict(means, {"mean": float(diff.mean()), "ci90": list(t_ci90(diff))})
    print(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
