"""Apply the frozen wording rules of docs/specs/2026-10-07-goal-and-stakes-design.md to
the run artifacts. Prints the wording each result earns; changes nothing.

    python scripts/decide_goal_stakes.py --runs artifacts/goal_stakes --c1 artifacts/expB2/corrected_l3_h8_wm.json \\
        --touch-promoted artifacts/expB2/goal_stakes_T-touch.json

Inputs per run directory: expB2_results.json (per-seed survival targets), mechanism.json
(run_mechanism_readouts, with the untrained and predictor floors). The T-touch primary
wording reads the promoted T-touch artifact's decision and gates blocks. The stakes
contrast is the seed-paired S-scarce minus S-immortal survival target (t-based 90% CI).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import re

import numpy as np

SESOI = 0.05
INTERVENTION_MIN = 0.10
INTERVENTION_MIN_SEEDS = 7
SURPRISE_AUROC_MIN = 0.65
SURPRISE_CORR_MIN = 0.20
SPEED_MIN = 0.75

_CELL_NAME = re.compile(r"^cell_d(?P<drift>[0-9.]+)_s(?P<seed>\d+)\.json$")
_PRIMARY_CLAUSES = ("pass_bar", "t90_excludes_bar", "pass_margin_predictor", "pass_margin_untrained")


def _excludes_zero(ci) -> bool:
    lo, hi = ci
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def _positive(ci) -> bool:
    """One-sided: the interval's lower bound is finite and above zero."""
    lo = ci[0]
    return bool(np.isfinite(lo) and lo > 0)


def _intervention_rule(block: dict) -> bool:
    """The 0.10 rule of readout 6 on one arm's block (the trained arm or a floor)."""
    c = block["intervention_real_minus_sham"]
    return bool(block["n_informative"] >= INTERVENTION_MIN_SEEDS and c["mean"] >= INTERVENTION_MIN
                and _positive(c["ci90"]))


def intervention_verdict(summary: dict) -> dict:
    """Readout 6 with the pre-launch amendment: the trained arm must meet the 0.10 rule and
    the untrained floor, scored the same way on its own direction, must not."""
    c = summary["intervention_real_minus_sham"]
    trained_ok = _intervention_rule(summary)
    floors = summary.get("floors") or {}
    floor_block = floors.get("untrained")
    pred_block = floors.get("predictor")
    floor_met = bool(floor_block and _intervention_rule(floor_block))
    ok = trained_ok and not floor_met
    if ok:
        wording = ("pushing the state along the decoded direction moves real-world behavior toward "
                   "fake-world behavior; the direction is behaviorally live")
    elif trained_ok:
        wording = "the trained arm's nudge moved behavior, but not separably from the untrained floor"
    else:
        wording = "the decoded direction did not move behavior under the tested nudge"
    out = {"pass": bool(ok), "wording": wording, "contrast": c,
           "n_informative": summary["n_informative"], "floor_met": floor_met,
           "floor": floor_block["intervention_real_minus_sham"] if floor_block else None,
           "predictor": pred_block["intervention_real_minus_sham"] if pred_block else None}
    if floor_block is None:
        out["note"] = "no floor recorded"
    return out


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
    ok = ad["mean"] > 0 and _positive(ad["ci90"])
    return {"informative": True, "pass": bool(ok),
            "wording": ("foraging in the fake world recovers within a lifetime" if ok else
                        "no within-lifetime recovery was detected")}


def stakes_verdict(means: dict, contrast: dict) -> dict:
    ordered = means["S-immortal"] < means["C1"] < means["S-scarce"]
    ok = ordered and contrast["mean"] >= SESOI and _positive(contrast["ci90"])
    return {"pass": bool(ok), "ordered": bool(ordered), "contrast": contrast,
            "wording": ("the reading grows with the stakes" if ok else
                        "within the tested range, stakes do not change the reading")}


def touch_wording(primary_met, gates_pass) -> str:
    """Spec: the pass wording needs the primary rule met with every gate passing; a gate or
    clause that is not shown (None) counts as open, so it earns the fail wording."""
    if primary_met and gates_pass:
        return ("a pellet-directed goal without survival stakes produces walks from which the "
                "condition is decodable; survival is not required")
    return "pellet-seeking without stakes did not meet the registered criterion"


def survival_note(touch_met, stakes_pass: bool):
    """The only survival-specific wording the spec allows: T-touch failed (known False, not
    unknown) while the stakes rule is met. None in every other combination."""
    if touch_met is False and stakes_pass:
        return "pellet-seeking alone was not enough in this world"
    return None


def _tri_all(flags: list):
    """All-true over tri-state flags: False if any is False, None if any is unknown."""
    if any(f is False for f in flags):
        return False
    if any(f is None for f in flags):
        return None
    return True


def primary_met(decision: dict):
    """The four clauses of the frozen primary rule as the promoted artifact records them
    (scripts/promote_reviewer_gaps_runs.py decision block): mean at the bar, t90 lower bound
    above the bar, and the two 0.05 margins. None if any clause is missing."""
    return _tri_all([None if decision.get(k) is None else bool(decision[k]) for k in _PRIMARY_CLAUSES])


def gates_pass(gates: dict) -> dict:
    """The six registered gates read from the promoted artifact's gates block, mirroring
    scripts/build_corrected_verdicts.py gates(); each is True, False, or None (not shown)."""
    def get(*path):
        cur = gates
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    eng = get("engagement")
    g_eng = None
    if isinstance(eng, dict) and eng and all(isinstance(v, dict) and "pass" in v and "n" in v
                                             for v in eng.values()):
        g_eng = all(v["pass"] == v["n"] for v in eng.values())
    tost = get("l0_tost", "equivalent")
    speed = get("speed_min")
    leak = get("pool_leak_clean_all")
    floor = get("untrained_floor_ok")
    deaths = get("deaths_total")
    per_gate = {
        "engagement": g_eng,
        "l0_equivalence": None if tost is None else bool(tost),
        "speed": None if speed is None else bool(speed >= SPEED_MIN),
        "reward_leak": None if leak is None else bool(leak),
        "untrained_floor": None if floor is None else bool(floor),
        "deaths": None if deaths is None else bool(deaths == 0),
    }
    return {"per_gate": per_gate, "all": _tri_all(list(per_gate.values())),
            "failed": [k for k, v in per_gate.items() if v is False],
            "not_shown": [k for k, v in per_gate.items() if v is None]}


def touch_primary(promoted_path: str) -> dict:
    """The T-touch primary wording from its promoted artifact's decision and gates blocks."""
    with open(promoted_path, encoding="utf-8") as fh:
        r = json.load(fh)
    decision = r.get("decision") or {}
    met = primary_met(decision)
    g = gates_pass(r.get("gates") or {})
    return {"met": met, "gates_pass": g["all"], "gates": g, "zone": decision.get("zone"),
            "wording": touch_wording(met, g["all"])}


def _cell_seeds(cells_dir: str, drift: float):
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
    ap.add_argument("--touch-promoted", default=None,
                    help="promoted T-touch artifact (promote_reviewer_gaps_runs.py format) for the primary wording")
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
    touch_met = None
    if a.touch_promoted:
        if not isinstance(report.get("T-touch"), dict):
            report["T-touch"] = {}
        report["T-touch"]["primary"] = touch_primary(a.touch_promoted)
        touch_met = report["T-touch"]["primary"]["met"]
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
        note = survival_note(touch_met, report["stakes"]["pass"])
        if note:
            report["survival_note"] = note
    print(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
