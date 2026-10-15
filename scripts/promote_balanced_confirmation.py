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
LEAK_KEYS = ("reward_sum", "pellets_consumed", "length", "lifetime")


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def decision_from_gates(gates: dict, score_ok: bool) -> tuple[str, str]:
    """MET needs every required gate on its own population, and every channel finite.

    Pooled speed and pooled leakage are recorded beside the decision. They do not
    satisfy the matched-world gates. A missing or non-finite matched gate cannot be MET.
    """
    required = (
        "engagement_ok", "matched_speed_ok", "matched_leak_ok", "identical_first_state_all",
        "shuffled_ok", "untrained_floor_ok", "targets_finite",
    )
    held = all(gates.get(name) is True for name in required) and gates.get("dropped_pairs_total") == 0
    if not held:
        if gates.get("untrained_floor_ok") is False:
            return ("UNINFORMATIVE",
                    f"the untrained balanced mean is {gates['untrained_dev']:.3f} from 0.5, "
                    "outside the 0.1 floor")
        if not gates.get("matched_speed_collected") or not gates.get("matched_leak_collected"):
            return ("UNINFORMATIVE",
                    "the balanced speed and leakage gates were not collected on the matched pairs")
        return "UNINFORMATIVE", "a required gate does not hold or is not finite"
    if score_ok:
        return "MET", "every required gate holds and both interval clauses clear"
    return "NOT MET", "the required gates hold and an interval clause does not"


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


def _attach_sidecar(run_dir: str, cells: list[dict]) -> list[dict]:
    """Copy a readout-only matched-gates file onto the in-memory cells.

    The historical cell files stay as recorded. The sidecar is optional; without
    it the matched gates are not collected and the decision cannot be MET.
    """
    path = os.path.join(ROOT, run_dir, "matched_gates.json")
    if not os.path.exists(path):
        return cells
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    by_key = {(int(row["seed"]), str(row["arm"])): row.get("matched_gates") for row in doc["rows"]}
    for cell in cells:
        if abs(float(cell["drift"]) - 0.45) > 1e-9:
            continue
        for arm in ARMS:
            block = by_key.get((int(cell["seed"]), arm))
            if isinstance(block, dict):
                cell["agents"][arm]["balanced"]["matched_gates"] = block
    return cells


def promote(run_id: str) -> dict:
    cells = _attach_sidecar(RUNS[run_id], _cells(RUNS[run_id]))
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
    matched = [c["agents"][arm]["balanced"].get("matched_gates") for c in d45 for arm in ARMS]
    present = bool(matched) and all(isinstance(block, dict) for block in matched)
    speed_vals = [block.get("speed") for block in matched] if present else []
    leak_vals = [(block.get("leakage") or {}) for block in matched] if present else []
    speed_finite = present and all(_finite(value) for value in speed_vals)
    leak_finite = present and all(all(_finite(leak.get(key)) for key in LEAK_KEYS) for leak in leak_vals)
    gates = {
        "engagement_pass": int(sum(bool(c["eng"]["engaged"]) for c in cells)),
        "engagement_n": len(cells),
        "engagement_ok": all(c["eng"]["engaged"] for c in cells),
        "engagement_population": (
            "objective-specific engagement protocol stored on the cell, "
            "64 episodes, not the balanced pairs at 910000"),
        "speed_min": float(min(speeds)),
        "speed_ok": bool(min(speeds) >= 0.75),
        "speed_scope": "pooled episodes at bases 800000 and 850000, not the balanced pairs",
        "pooled_leak_ok": not leak_bad,
        "pooled_leak_scope": "pooled episodes, not the balanced pairs",
        "matched_speed_collected": speed_finite,
        "matched_leak_collected": leak_finite,
        "matched_speed_min": float(min(float(value) for value in speed_vals)) if speed_finite else None,
        "matched_speed_ok": bool(speed_finite and min(float(value) for value in speed_vals) >= 0.75),
        "matched_leak_ok": bool(leak_finite and all(
            abs(float(leak[key]) - 0.5) < FLOOR for leak in leak_vals for key in LEAK_KEYS)),
        "matched_population": ("retained pairs at seed base 910000" if present else
                               "not collected; recorded speed and leakage checks used the pooled worlds"),
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
    decision, reason = decision_from_gates(gates, score_ok)
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
