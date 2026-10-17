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
# The registered population of the matched gates, from the frozen run specification.
MATCHED_CONTRACT = {
    "population": "retained pairs", "seed_base": 910000, "drift": 0.45,
    "seeds": tuple(range(10, 20)), "drifts": (0.0, 0.45), "n_pairs": 110, "dropped_pairs": 0,
    "runs": {"B1": {"objective": "survival", "mortal": True},
             "B2": {"objective": "touch", "mortal": False}},
}
RECON_TOL = 0.005   # matched-readout target against the balanced cell target, same pairs


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _all_finite(values) -> bool:
    values = list(values)
    return bool(values) and all(_finite(v) for v in values)


def decision_from_gates(gates: dict, score_ok: bool) -> tuple[str, str]:
    """MET needs every required gate on its own population, and every channel finite.

    Pooled speed and pooled leakage are recorded beside the decision. They do not
    satisfy the matched-world gates. A missing or non-finite matched gate cannot be MET.
    """
    required = (
        "engagement_ok", "matched_speed_ok", "matched_leak_ok", "identical_first_state_all",
        "shuffled_ok", "untrained_floor_ok", "targets_finite", "matched_reconstruction_ok",
    )
    held = all(gates.get(name) is True for name in required) and gates.get("dropped_pairs_total") == 0
    if not held:
        # Name every required gate that is not True, so the reason never stops at the first.
        parts = []
        for name in required:
            if gates.get(name) is True:
                continue
            if name == "untrained_floor_ok" and _finite(gates.get("untrained_dev")):
                parts.append(f"untrained floor ({gates['untrained_dev']:.3f} from 0.5, "
                             "outside the 0.1 floor)")
            elif name in ("matched_speed_ok", "matched_leak_ok"):
                channel = name.split("_")[1]
                if not gates.get(f"matched_{channel}_collected"):
                    parts.append(f"matched {channel} not collected on the matched pairs")
                elif channel == "speed" and _finite(gates.get("matched_speed_min")):
                    parts.append(f"matched speed (min {gates['matched_speed_min']:.3f} under 0.75)")
                elif channel == "leak":
                    parts.append("matched leak (a channel 0.1 or more from 0.5)")
                else:
                    parts.append(f"matched {channel} does not hold or is not finite")
            else:
                parts.append(f"{name} does not hold or is not finite")
        if gates.get("dropped_pairs_total") != 0:
            parts.append(f"dropped pairs ({gates.get('dropped_pairs_total')!r}, not 0)")
        return "UNINFORMATIVE", "gates not held: " + "; ".join(parts)
    if score_ok:
        return "MET", "every required gate holds and both interval clauses clear"
    return "NOT MET", "the required gates hold and an interval clause does not"


def _cells(run_dir: str) -> list[dict]:
    """The cells of one run, each with the file's recorded `git_commit` and `fingerprint`
    kept beside it under `_provenance`. A file without either key raises ValueError naming
    it; so does a run whose files record more than one commit or fingerprint."""
    d = os.path.join(ROOT, run_dir, "cells")
    out = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(d, name)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for key in ("git_commit", "fingerprint"):
            if not doc.get(key):
                raise ValueError(f"{path}: cell file has no top-level {key}")
        cell = doc["cell"]
        cell["_provenance"] = {"file": path, "git_commit": doc["git_commit"],
                               "fingerprint": doc["fingerprint"]}
        out.append(cell)
    return out


def _run_provenance(cells: list[dict]) -> dict:
    """The single commit and fingerprint a run's cell files record."""
    out = {}
    for key in ("git_commit", "fingerprint"):
        seen = {c["_provenance"][key]: c["_provenance"]["file"] for c in cells}
        if len(seen) != 1:
            raise ValueError(f"cells record more than one {key}: "
                             + ", ".join(f"{v} ({f})" for v, f in sorted(seen.items())))
        out[key] = next(iter(seen))
    return out


def _check_cells(run_id: str, cells: list[dict]) -> None:
    """The primary cell files hold exactly the expected seed and drift set, and each drift-0.45
    balanced block carries the contract's seed base, pair count and zero drops."""
    want = sorted((d, s) for d in MATCHED_CONTRACT["drifts"] for s in MATCHED_CONTRACT["seeds"])
    found = sorted((round(float(c["drift"]), 2), int(c["seed"])) for c in cells)
    if found != want:
        raise ValueError(f"{run_id}: cell seed and drift set differs from the contract: "
                         f"missing {sorted(set(want) - set(found))}, unexpected "
                         f"{sorted(set(found) - set(want))}, {len(found) - len(set(found))} duplicated")
    for c in cells:
        if round(float(c["drift"]), 2) != MATCHED_CONTRACT["drift"]:
            continue
        for arm in ARMS:
            bal = c["agents"][arm]["balanced"]
            for field in ("seed_base", "n_pairs", "dropped_pairs"):
                if bal.get(field) != MATCHED_CONTRACT[field]:
                    raise ValueError(f"{run_id}: cell seed {c['seed']} {arm} balanced {field} is "
                                     f"{bal.get(field)!r}, contract {MATCHED_CONTRACT[field]!r}")


def validate_sidecar(run_id: str, doc: dict) -> dict:
    """Rows of a matched-gates file keyed by (seed, arm), after checking the document and every
    row against MATCHED_CONTRACT. A mismatch raises ValueError naming the field."""
    want_doc = {"population": MATCHED_CONTRACT["population"], "seed_base": MATCHED_CONTRACT["seed_base"],
                **MATCHED_CONTRACT["runs"][run_id]}
    for field, value in want_doc.items():
        if doc.get(field) != value:
            raise ValueError(f"{run_id} matched sidecar: {field} is {doc.get(field)!r}, contract {value!r}")
    by_key: dict = {}
    for row in doc.get("rows", []):
        key = (int(row["seed"]), str(row["arm"]))
        if key in by_key:
            raise ValueError(f"{run_id} matched sidecar: duplicate row for seed {key[0]} {key[1]}")
        by_key[key] = row
        for field in ("seed_base", "drift", "n_pairs", "dropped_pairs"):
            value, got = MATCHED_CONTRACT[field], row.get(field)
            bad = (not _finite(got) or abs(float(got) - value) > 1e-9) if field == "drift" else got != value
            if bad:
                raise ValueError(f"{run_id} matched sidecar: seed {key[0]} {key[1]} {field} is {got!r}, "
                                 f"contract {value!r}")
        block = row.get("matched_gates")
        if isinstance(block, dict) and (block.get("population") != "matched pairs"
                                        or block.get("n_pairs") != MATCHED_CONTRACT["n_pairs"]):
            raise ValueError(f"{run_id} matched sidecar: seed {key[0]} {key[1]} matched_gates population "
                             f"or n_pairs differs from the contract")
    want = {(s, a) for s in MATCHED_CONTRACT["seeds"] for a in ARMS}
    if set(by_key) != want:
        raise ValueError(f"{run_id} matched sidecar: seed set differs from the contract; "
                         f"missing {sorted(want - set(by_key))}, unexpected {sorted(set(by_key) - want)}")
    return by_key


def _summ(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    lo, hi = t_ci90(a)
    return {"per_seed": [float(x) for x in a], "mean": float(a.mean()),
            "t90": [float(lo), float(hi)], "n": int(a.size)}


def _attach_sidecar(run_id: str, run_dir: str, cells: list[dict]) -> tuple[list[dict], dict]:
    """Copy a readout-only matched-gates file onto the in-memory cells, and report whether
    its readout reconstructs the balanced target of the same pairs.

    The historical cell files stay as recorded. The sidecar is optional; without
    it the matched gates are not collected and the decision cannot be MET. A sidecar off
    the registered contract raises. One without a reconstruction target is attached, but
    its reconstruction reads not recorded, which also keeps the decision from MET.
    """
    path = os.path.join(ROOT, run_dir, "matched_gates.json")
    if not os.path.exists(path):
        return cells, {"status": "no sidecar", "tolerance": RECON_TOL}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    by_key = validate_sidecar(run_id, doc)
    diffs, missing = [], 0
    for cell in cells:
        if abs(float(cell["drift"]) - MATCHED_CONTRACT["drift"]) > 1e-9:
            continue
        for arm in ARMS:
            row = by_key[(int(cell["seed"]), arm)]
            bal = cell["agents"][arm]["balanced"]
            if isinstance(row.get("matched_gates"), dict):
                bal["matched_gates"] = row["matched_gates"]
            if "target" not in row:
                missing += 1
                continue
            if not (_finite(row["target"]) and _finite(bal["target"])):
                raise ValueError(f"{run_id} matched sidecar: seed {cell['seed']} {arm} target is not finite")
            diff = abs(float(row["target"]) - float(bal["target"]))
            if diff > RECON_TOL:
                raise ValueError(f"{run_id} matched sidecar: seed {cell['seed']} {arm} target "
                                 f"{row['target']:.4f} differs from the balanced target {bal['target']:.4f} "
                                 f"by {diff:.4f}, over the {RECON_TOL} tolerance")
            diffs.append(diff)
    if missing:
        return cells, {"status": f"not recorded in the sidecar for {missing} of {missing + len(diffs)} rows",
                       "tolerance": RECON_TOL}
    return cells, {"status": "checked", "n": len(diffs), "max_abs_diff": float(max(diffs)),
                   "tolerance": RECON_TOL}


def promote(run_id: str) -> dict:
    cells = _cells(RUNS[run_id])
    prov = _run_provenance(cells)
    _check_cells(run_id, cells)
    cells, recon = _attach_sidecar(run_id, RUNS[run_id], cells)
    d45 = [c for c in cells if abs(float(c["drift"]) - 0.45) < 1e-9]
    d45.sort(key=lambda c: int(c["seed"]))
    arms = {arm: _summ([c["agents"][arm]["balanced"]["target"] for c in d45]) for arm in ARMS}
    margins = {
        "predictor": _summ([s - p for s, p in zip(arms["survival"]["per_seed"], arms["predictor"]["per_seed"])]),
        "untrained": _summ([s - u for s, u in zip(arms["survival"]["per_seed"], arms["untrained"]["per_seed"])]),
    }
    # every gate value is checked finite before any max, min or threshold reads it
    shuf = [c["agents"][arm]["balanced"].get("shuffled_within_pair") for c in d45 for arm in ARMS]
    shuf_finite = _all_finite(shuf)
    speeds = [c["agents"][arm]["pool"].get("speed") for c in d45 for arm in ARMS]
    speeds_finite = _all_finite(speeds)
    leak_bad = []
    for c in d45:
        for arm in ARMS:
            lk = c["agents"][arm]["pool"].get("pool_leakage") or {}
            for key in ("reward_sum", "pellets_consumed", "length", "lifetime"):
                if not _finite(lk.get(key)) or abs(float(lk[key]) - 0.5) >= FLOOR:
                    leak_bad.append(f"{c['seed']} {arm} {key}")
    floor_dev = abs(arms["untrained"]["mean"] - 0.5)
    targets_finite = bool(all(np.isfinite(arms[a]["per_seed"]).all() for a in ARMS))
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
        "speed_min": float(min(float(v) for v in speeds)) if speeds_finite else None,
        "speed_ok": bool(speeds_finite and min(float(v) for v in speeds) >= 0.75),
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
        "matched_reconstruction": recon,
        "matched_reconstruction_ok": recon["status"] == "checked",
        "identical_first_state_all": all(c["agents"][arm]["balanced"]["identical_first_state"]
                                         for c in d45 for arm in ARMS),
        "dropped_pairs_total": int(sum(c["agents"][arm]["balanced"]["dropped_pairs"]
                                       for c in d45 for arm in ARMS)),
        "shuffled_max_abs_dev": float(max(abs(float(x) - 0.5) for x in shuf)) if shuf_finite else None,
        "shuffled_ok": bool(shuf_finite and max(abs(float(x) - 0.5) for x in shuf) < FLOOR),
        "untrained_mean": arms["untrained"]["mean"],
        "untrained_dev": float(floor_dev),
        "untrained_floor_ok": bool(targets_finite and floor_dev < FLOOR),
        "targets_finite": targets_finite,
    }
    score_ok = targets_finite and (arms["survival"]["t90"][0] >= BAR
                and margins["predictor"]["t90"][0] >= MARGIN
                and margins["untrained"]["t90"][0] >= MARGIN)
    decision, reason = decision_from_gates(gates, score_ok)
    return {
        "generated_by": "scripts/promote_balanced_confirmation.py",
        "spec": "docs/specs/2026-10-12-balanced-confirmation-design.md",
        "run": run_id,
        "git_commit_at_run": prov["git_commit"],
        "fingerprint": prov["fingerprint"],
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
