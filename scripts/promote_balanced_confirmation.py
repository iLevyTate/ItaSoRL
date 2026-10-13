"""Promote the 2026-10-12 balanced confirmation from fullruns/ cells.

The decision is the one frozen in docs/specs/2026-10-12-balanced-confirmation-design.md.
The runner's own "MET" line is the pooled 800000/850000 readout and is not this decision.
Speed and reward leakage were stored for that pooled readout only. The balanced readout
stores the target, the within-pair shuffle, dropped pairs, and identical_first_state.
"""

from __future__ import annotations

import json
import os

import numpy as np

from itasorl.stats import t_ci90

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = {
    "B1": "fullruns/balanced_confirmation_wm",
    "B2": "fullruns/balanced_confirmation_touch",
}
ARMS = ("survival", "predictor", "untrained")
BAR = 0.65
MARGIN = 0.05
FLOOR = 0.1


def _cells(run_dir: str) -> list[dict]:
    d = os.path.join(ROOT, run_dir, "cells")
    out = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(d, name), encoding="utf-8") as fh:
            out.append(json.load(fh)["cell"])
    return out


def _summ(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    lo, hi = t_ci90(a)
    return {"per_seed": [float(x) for x in a], "mean": float(a.mean()),
            "t90": [float(lo), float(hi)], "n": int(a.size)}


def promote(run_id: str) -> dict:
    cells = _cells(RUNS[run_id])
    d45 = [c for c in cells if abs(float(c["drift"]) - 0.45) < 1e-9]
    d45.sort(key=lambda c: int(c["seed"]))
    if len(d45) != 10 or len(cells) != 20:
        raise SystemExit(f"{run_id}: expected 20 cells and 10 at drift 0.45, got {len(cells)} and {len(d45)}")
    arms = {arm: _summ([c["agents"][arm]["balanced"]["target"] for c in d45]) for arm in ARMS}
    margins = {
        "predictor": _summ([s - p for s, p in zip(arms["survival"]["per_seed"], arms["predictor"]["per_seed"])]),
        "untrained": _summ([s - u for s, u in zip(arms["survival"]["per_seed"], arms["untrained"]["per_seed"])]),
    }
    shuf = [c["agents"][arm]["balanced"]["shuffled_within_pair"] for c in d45 for arm in ARMS]
    speeds = [c["agents"][arm]["pool"]["speed"] for c in d45 for arm in ARMS]
    leak_bad = []
    for c in d45:
        for arm in ARMS:
            lk = c["agents"][arm]["pool"]["pool_leakage"]
            for key in ("reward_sum", "pellets_consumed", "length", "lifetime"):
                if abs(float(lk[key]) - 0.5) >= FLOOR:
                    leak_bad.append(f"{c['seed']} {arm} {key}")
    floor_dev = abs(arms["untrained"]["mean"] - 0.5)
    gates = {
        "engagement_pass": int(sum(bool(c["eng"]["engaged"]) for c in cells)),
        "engagement_n": len(cells),
        "engagement_ok": all(c["eng"]["engaged"] for c in cells),
        "speed_min": float(min(speeds)),
        "speed_ok": bool(min(speeds) >= 0.75),
        "speed_scope": "pooled episodes at bases 800000 and 850000, not the balanced pairs",
        "pooled_leak_ok": not leak_bad,
        "pooled_leak_scope": "pooled episodes, not the balanced pairs",
        "identical_first_state_all": all(c["agents"][arm]["balanced"]["identical_first_state"]
                                         for c in d45 for arm in ARMS),
        "dropped_pairs_total": int(sum(c["agents"][arm]["balanced"]["dropped_pairs"]
                                       for c in d45 for arm in ARMS)),
        "shuffled_max_abs_dev": float(max(abs(x - 0.5) for x in shuf)),
        "shuffled_ok": bool(max(abs(x - 0.5) for x in shuf) < FLOOR),
        "untrained_mean": arms["untrained"]["mean"],
        "untrained_dev": float(floor_dev),
        "untrained_floor_ok": bool(floor_dev < FLOOR),
        "targets_finite": all(np.isfinite(arms[a]["per_seed"]).all() for a in ARMS),
    }
    score_ok = (arms["survival"]["t90"][0] >= BAR
                and margins["predictor"]["t90"][0] >= MARGIN
                and margins["untrained"]["t90"][0] >= MARGIN)
    gate_ok = all(gates[k] for k in (
        "engagement_ok", "speed_ok", "pooled_leak_ok", "identical_first_state_all",
        "shuffled_ok", "untrained_floor_ok", "targets_finite")) and gates["dropped_pairs_total"] == 0
    if not gate_ok:
        decision = "UNINFORMATIVE"
        reason = (f"the untrained balanced mean is {floor_dev:.3f} from 0.5, "
                  "outside the 0.1 floor")
    elif score_ok:
        decision = "MET"
        reason = "every recorded gate holds and both interval clauses clear"
    else:
        decision = "NOT MET"
        reason = "the gates that were recorded hold and an interval clause does not"
    return {
        "generated_by": "scripts/promote_balanced_confirmation.py",
        "spec": "docs/specs/2026-10-12-balanced-confirmation-design.md",
        "run": run_id,
        "git_commit_at_run": "2c170f0",
        "seed_base": 910000,
        "seeds": [int(c["seed"]) for c in d45],
        "stack": d45[0].get("stack"),
        "objective": d45[0].get("objective"),
        "mortal": d45[0].get("mortal"),
        "adv_norm": d45[0].get("adv_norm"),
        "decision": decision,
        "reason": reason,
        "score_clauses_hold": bool(score_ok),
        "balanced": {**arms, "margin_predictor": margins["predictor"],
                     "margin_untrained": margins["untrained"]},
        "gates": gates,
    }


def main() -> int:
    out_dir = os.path.join(ROOT, "artifacts", "balanced_confirmation")
    os.makedirs(out_dir, exist_ok=True)
    for run_id in RUNS:
        doc = promote(run_id)
        path = os.path.join(out_dir, f"{run_id.lower()}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")
        print(f"{run_id}: {doc['decision']}  survival t90 {doc['balanced']['survival']['t90']}  "
              f"untrained dev {doc['gates']['untrained_dev']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
