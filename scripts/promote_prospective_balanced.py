"""Promote the prospective balanced confirmation (PB1, PB2) from fullruns/.

The decision is the one frozen in docs/specs/2026-10-20-prospective-balanced-confirmation-design.md
(first text 7d08686, amended before any cell existed in ea8ceea): the 2026-10-13 sensory-echo
rule on fresh seeds 20 to 29 at matched base 3000000, with the gates of 2026-10-12 that the
matched design can pass. This module reads, never writes, the run folders and writes
artifacts/prospective_balanced/{pb1,pb2,summary}.json.

Inputs per run: the 20 training cells (`cells/`), the matched-gates sidecar
(`matched_gates.json`, from scripts/audit_balanced_matched_gates.py), the primary sensory-echo
aggregate and cells (`sensory_echo/`), and, descriptive only, the second-draw aggregate
(`sensory_echo_draw2/`). The descriptive nonlinear readings of B1 and B2 are read from
`fullruns/balanced_confirmation_{wm,touch}/sensory_echo_nonlinear/`.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from itasorl.stats import t_ci90  # noqa: E402
from promote_balanced_confirmation import _cells, _finite, _run_provenance  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPEC = "docs/specs/2026-10-20-prospective-balanced-confirmation-design.md"
RUNS = {
    "PB1": {"dir": "fullruns/prospective_balanced_wm", "objective": "survival", "mortal": True},
    "PB2": {"dir": "fullruns/prospective_balanced_touch", "objective": "touch", "mortal": False},
}
B_RUNS = {"B1": "fullruns/balanced_confirmation_wm", "B2": "fullruns/balanced_confirmation_touch"}
ARMS = ("survival", "predictor", "untrained")
TRAINED = "survival"   # the trained arm's key in both runs (the touch run keeps the name)
SEEDS = tuple(range(20, 30))
DRIFTS = (0.0, 0.45)
SEED_BASE = 3_000_000
N_PAIRS = 110
BAR, MARGIN, DEV = 0.65, 0.05, 0.1
SPEED_MIN = 0.75
RECON_TOL = 0.005
LEAK_KEYS = ("reward_sum", "pellets_consumed", "length", "lifetime")
S2_METRICS = ("resid_obs_mlp", "resid_obs_int", "resid_obs_int_mlp", "resid_obs_beh_mlp")


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _summ(vals) -> dict:
    a = np.asarray([float(v) for v in vals], float)
    lo, hi = t_ci90(a)
    return {"per_seed": [float(x) for x in a], "mean": float(a.mean()),
            "t90": [float(lo), float(hi)], "n": int(a.size)}


def check_cells(run_id: str, cells: list[dict]) -> None:
    """Exactly drifts 0 and 0.45 by seeds 20 to 29, the run's objective and death setting, and
    each drift-0.45 balanced block at base 3000000 with 110 pairs and no drop."""
    want = sorted((d, s) for d in DRIFTS for s in SEEDS)
    found = sorted((round(float(c["drift"]), 2), int(c["seed"])) for c in cells)
    if found != want:
        raise ValueError(f"{run_id}: cell set differs from the spec: missing "
                         f"{sorted(set(want) - set(found))}, unexpected {sorted(set(found) - set(want))}")
    for c in cells:
        if c.get("objective") != RUNS[run_id]["objective"] or bool(c.get("mortal")) != RUNS[run_id]["mortal"]:
            raise ValueError(f"{run_id}: cell seed {c['seed']} objective or mortal differs from the spec")
        if round(float(c["drift"]), 2) != 0.45:
            continue
        for arm in ARMS:
            bal = c["agents"][arm]["balanced"]
            if bal.get("seed_base") != SEED_BASE or bal.get("n_pairs") != N_PAIRS:
                raise ValueError(f"{run_id}: cell seed {c['seed']} {arm} balanced block is off the spec")


def sidecar_rows(run_id: str, doc: dict) -> dict:
    """Rows of matched_gates.json keyed by (seed, arm), checked against the spec."""
    if (doc.get("seed_base") != SEED_BASE or doc.get("objective") != RUNS[run_id]["objective"]
            or bool(doc.get("mortal")) != RUNS[run_id]["mortal"]):
        raise ValueError(f"{run_id} matched sidecar: header differs from the spec")
    rows = {}
    for row in doc.get("rows", []):
        key = (int(row["seed"]), str(row["arm"]))
        if key in rows:
            raise ValueError(f"{run_id} matched sidecar: duplicate row {key}")
        rows[key] = row
    want = {(s, a) for s in SEEDS for a in ARMS}
    if set(rows) != want:
        raise ValueError(f"{run_id} matched sidecar: rows differ from the spec; missing "
                         f"{sorted(want - set(rows))}, unexpected {sorted(set(rows) - want)}")
    return rows


def speed_calls(blocks: dict) -> dict:
    """Gate 2 calls from {(seed, arm): matched_gates block}: every trained seed at least 0.75,
    and the untrained mean over its seeds at least 0.75."""
    trained = {s: blocks[(s, TRAINED)].get("speed") for s in SEEDS}
    untr = [blocks[(s, "untrained")].get("speed") for s in SEEDS]
    finite = all(_finite(v) for v in trained.values()) and all(_finite(v) for v in untr)
    if not finite:
        return {"finite": False}
    return {"finite": True,
            "trained": {s: bool(float(v) >= SPEED_MIN) for s, v in trained.items()},
            "untrained_mean": float(np.mean([float(v) for v in untr])),
            "untrained_mean_ok": bool(float(np.mean([float(v) for v in untr])) >= SPEED_MIN)}


def leak_calls(blocks: dict) -> dict | None:
    """Gate 3 calls: every channel of every seed and arm strictly less than 0.1 from 0.5."""
    out = {}
    for key, block in blocks.items():
        leak = block.get("leakage") or {}
        for ch in LEAK_KEYS:
            v = leak.get(ch)
            if not _finite(v):
                return None
            out[(key[0], key[1], ch)] = bool(abs(float(v) - 0.5) < DEV)
    return out


def matched_gates(run_id: str, d45: list[dict], rows: dict) -> dict:
    """Gates 2 and 3 from the sidecar, with the reconstruction check and the in-cell copy."""
    recon = [abs(float(rows[(int(c["seed"]), a)]["target"]) - float(c["agents"][a]["balanced"]["target"]))
             for c in d45 for a in ARMS]
    recon_ok = bool(recon) and all(_finite(x) and x <= RECON_TOL for x in recon)
    side = {k: r.get("matched_gates") or {} for k, r in rows.items()}
    cell = {(int(c["seed"]), a): c["agents"][a]["balanced"].get("matched_gates") or {} for c in d45 for a in ARMS}
    sp_side, sp_cell = speed_calls(side), speed_calls(cell) if all(cell.values()) else {"finite": False}
    lk_side, lk_cell = leak_calls(side), leak_calls(cell) if all(cell.values()) else None
    speed_agree = bool(sp_cell.get("finite")) and (
        sp_cell["trained"] == sp_side.get("trained")
        and sp_cell["untrained_mean_ok"] == sp_side.get("untrained_mean_ok"))
    leak_agree = lk_cell is not None and lk_cell == lk_side
    speed_ok = bool(sp_side.get("finite") and all(sp_side["trained"].values())
                    and sp_side["untrained_mean_ok"] and speed_agree and recon_ok)
    leak_ok = bool(lk_side is not None and all(lk_side.values()) and leak_agree and recon_ok)
    speeds = {f"{s} {a}": side[(s, a)].get("speed") for s in SEEDS for a in ARMS}
    return {
        "reconstruction_max_abs_diff": float(max(recon)) if recon else None,
        "reconstruction_ok": recon_ok,
        "speed": speeds,
        "trained_speed_min": (float(min(float(side[(s, TRAINED)]["speed"]) for s in SEEDS))
                              if sp_side.get("finite") else None),
        "untrained_speed_mean": sp_side.get("untrained_mean"),
        "in_cell_agrees_speed": speed_agree,
        "in_cell_agrees_leak": leak_agree,
        "leak_max_abs_dev": (float(max(abs(float((side[k].get("leakage") or {})[ch]) - 0.5)
                                       for k in side for ch in LEAK_KEYS)) if lk_side is not None else None),
        "gate2_speed_ok": speed_ok,
        "gate3_leak_ok": leak_ok,
    }


def s2_labels(agg: dict) -> dict:
    """S2: per control, SURVIVES / ATTENUATED / EXPLAINED from the aggregate means."""
    out = {}
    tr, un = agg.get(f"d=0.45 {TRAINED}", {}), agg.get("d=0.45 untrained", {})
    for m in S2_METRICS:
        t, u = (tr.get(m) or {}).get("mean"), (un.get(m) or {}).get("mean")
        if not (_finite(t) and _finite(u)):
            out[m] = {"label": "NOT COMPUTED"}
            continue
        t, u = float(t), float(u)
        label = ("SURVIVES" if t >= BAR and t > u + MARGIN else "ATTENUATED" if t >= 0.60 else "EXPLAINED")
        out[m] = {"trained": t, "untrained": u, "trained_t90": tr[m].get("t90"), "label": label}
    return out


def _by_seed(cells: list[dict], arm: str, metric: str) -> dict:
    return {int(c["seed"]): float(c[metric]) for c in cells
            if c.get("agent") == arm and str(c.get("drift")) in ("0.45", "0.450") and metric in c}


def strict_reading(agg: dict, se_cells: list[dict]) -> dict:
    """S1: the clauses read on intervals, paired by seed from cells.json."""
    tr = (agg.get(f"d=0.45 {TRAINED}") or {}).get("resid_obs") or {}
    un = (agg.get("d=0.45 untrained") or {}).get("resid_obs") or {}
    t, u = _by_seed(se_cells, TRAINED, "resid_obs"), _by_seed(se_cells, "untrained", "resid_obs")
    if not (tr.get("t90") and un.get("t90")) or set(t) != set(SEEDS) or set(u) != set(SEEDS):
        return {"label": "NOT COMPUTED"}
    diff = _summ([t[s] - u[s] for s in SEEDS])
    ok = (tr["t90"][0] >= BAR and diff["t90"][0] >= MARGIN
          and 0.40 <= un["t90"][0] and un["t90"][1] <= 0.60)
    return {"trained_t90": tr["t90"], "paired_margin": diff, "untrained_t90": un["t90"],
            "label": "STRICT MET" if ok else "STRICT NOT MET"}


def promote(run_id: str) -> dict:
    rd = os.path.join(ROOT, RUNS[run_id]["dir"])
    if not os.path.isdir(os.path.join(rd, "cells")) or len(
            [n for n in os.listdir(os.path.join(rd, "cells")) if n.endswith(".json")]) != len(DRIFTS) * len(SEEDS):
        return {"run": run_id, "spec": SPEC, "decision": "NOT RUN",
                "reason": "the run's 20 cells do not all exist"}
    cells = _cells(rd)   # absolute path: os.path.join keeps it as given
    prov = _run_provenance(cells)
    check_cells(run_id, cells)
    d45 = sorted((c for c in cells if round(float(c["drift"]), 2) == 0.45), key=lambda c: int(c["seed"]))
    rows = sidecar_rows(run_id, _load(os.path.join(rd, "matched_gates.json")))
    mg = matched_gates(run_id, d45, rows)
    agg = _load(os.path.join(rd, "sensory_echo", "aggregate.json"))
    se_cells = _load(os.path.join(rd, "sensory_echo", "cells.json"))
    db = agg.get("decision_balanced") or {}
    shuf = [c["agents"][a]["balanced"].get("shuffled_within_pair") for c in d45 for a in ARMS]
    targets = [c["agents"][a]["balanced"].get("target") for c in d45 for a in ARMS]
    resid = [row.get("resid_obs") for row in se_cells if str(row.get("drift")) in ("0.45", "0.450")]
    blockers = (agg.get("integrity") or {}).get("adjudication_blockers") or []
    withheld = str(db.get("zone", "")).startswith("WITHHELD")
    se_pairs_ok = all(row.get("identical_first_state") is True and row.get("dropped_pairs") == 0
                      and row.get("n_pairs") == N_PAIRS for row in se_cells)
    gates = {
        "gate1_engagement_ok": all(bool(c["eng"]["engaged"]) for c in cells),
        "engagement_pass": int(sum(bool(c["eng"]["engaged"]) for c in cells)), "engagement_n": len(cells),
        "gate2_speed_ok": mg["gate2_speed_ok"],
        "gate3_leak_ok": mg["gate3_leak_ok"],
        "gate4_shuffled_ok": bool(all(_finite(x) for x in shuf) and max(abs(float(x) - 0.5) for x in shuf) < DEV),
        "shuffled_max_abs_dev": (float(max(abs(float(x) - 0.5) for x in shuf)) if all(_finite(x) for x in shuf) else None),
        "gate5_pairs_ok": bool(all(c["agents"][a]["balanced"].get("identical_first_state") is True
                                   and c["agents"][a]["balanced"].get("dropped_pairs") == 0
                                   for c in d45 for a in ARMS) and se_pairs_ok and len(se_cells) == len(SEEDS) * len(ARMS)),
        "gate6_finite_ok": bool(all(_finite(x) for x in targets) and resid and all(_finite(x) for x in resid)
                                and not withheld and not blockers),
        "gate7_integrity_ok": bool((agg.get("integrity") or {}).get("all_match") is True
                                   and not (agg.get("integrity") or {}).get("mismatches")),
        "adjudication_blockers": blockers,
        "matched": mg,
    }
    gate_names = [k for k in gates if k.startswith("gate") and k.endswith("_ok")]
    gates_hold = all(gates[k] is True for k in gate_names)
    floor_ok = db.get("floor_explained") is True
    trained_ok = bool(db.get("pass_bar") and db.get("pass_margin") and db.get("t90_lo_above_untrained"))
    if not gates_hold:
        decision = "UNINFORMATIVE"
        reason = "gates not held: " + ", ".join(k for k in gate_names if gates[k] is not True)
    elif not floor_ok:
        decision, reason = "UNINFORMATIVE", "floor clause fails (untrained resid_obs more than 0.1 from 0.5)"
    elif trained_ok:
        decision, reason = "MET", "every gate, the floor clause and the trained clause hold"
    else:
        decision, reason = "NOT MET", f"the trained clause fails ({db.get('zone')})"
    raw = {a: _summ([c["agents"][a]["balanced"]["target"] for c in d45]) for a in ARMS}
    raw_margins = {b: _summ([s - o for s, o in zip(raw[TRAINED]["per_seed"], raw[b]["per_seed"])])
                   for b in ("predictor", "untrained")}
    out = {
        "generated_by": "scripts/promote_prospective_balanced.py", "spec": SPEC, "run": run_id,
        "git_commit_at_run": prov["git_commit"], "fingerprint": prov["fingerprint"],
        "seed_base": SEED_BASE, "seeds": list(SEEDS), "objective": RUNS[run_id]["objective"],
        "mortal": RUNS[run_id]["mortal"], "stack": d45[0].get("stack"),
        "decision": decision, "reason": reason, "gates": gates,
        "floor_clause": {"untrained_resid_obs": db.get("untrained_resid_obs"),
                         "untrained_dev": db.get("untrained_dev"), "holds": floor_ok,
                         "untrained_t90": ((agg.get("d=0.45 untrained") or {}).get("resid_obs") or {}).get("t90")},
        "trained_clause": {k: db.get(k) for k in ("trained_resid_obs", "trained_t90", "pass_bar",
                                                  "pass_margin", "t90_lo_above_untrained", "zone")},
        "S1_strict": strict_reading(agg, se_cells),
        "S2_controls": s2_labels(agg),
        "S3_raw_matched": {**raw, "margin_predictor": raw_margins["predictor"],
                           "margin_untrained": raw_margins["untrained"],
                           "raw_untrained_dev": float(abs(raw["untrained"]["mean"] - 0.5))},
        "S5_predictor_resid_obs": (agg.get("d=0.45 predictor") or {}).get("resid_obs"),
    }
    d2 = os.path.join(rd, "sensory_echo_draw2", "aggregate.json")
    if os.path.exists(d2):
        agg2 = _load(d2)
        c2 = _load(os.path.join(rd, "sensory_echo_draw2", "cells.json"))
        two = {}
        for arm in (TRAINED, "untrained"):
            a1, a2 = _by_seed(se_cells, arm, "resid_obs"), _by_seed(c2, arm, "resid_obs")
            if set(a1) == set(SEEDS) == set(a2):
                two[arm] = _summ([(a1[s] + a2[s]) / 2 for s in SEEDS])
        out["S4_draw2"] = {"decision_balanced": agg2.get("decision_balanced"),
                           "S2_controls": s2_labels(agg2), "two_draw_mean_resid_obs": two}
    return out


def b_nonlinear() -> dict:
    """Descriptive: S2 labels on the saved B1 and B2 agents at base 910000."""
    out = {}
    for run_id, d in B_RUNS.items():
        p = os.path.join(ROOT, d, "sensory_echo_nonlinear", "aggregate.json")
        out[run_id] = s2_labels(_load(p)) if os.path.exists(p) else {"status": "not run"}
    return out


def identical_untrained(pb1: dict, pb2: dict) -> bool | None:
    """Whether the two runs' untrained arms read identically (the spec's per-run floor rule)."""
    try:
        u1 = pb1["S3_raw_matched"]["untrained"]["per_seed"]
        u2 = pb2["S3_raw_matched"]["untrained"]["per_seed"]
    except (KeyError, TypeError):
        return None
    return bool(u1 == u2)


def main() -> int:
    out_dir = os.path.join(ROOT, "artifacts", "prospective_balanced")
    os.makedirs(out_dir, exist_ok=True)
    docs = {r: promote(r) for r in RUNS}
    for r, doc in docs.items():
        with open(os.path.join(out_dir, f"{r.lower()}.json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")
        print(f"{r}: {doc['decision']}  ({doc['reason']})")
    summary = {"generated_by": "scripts/promote_prospective_balanced.py", "spec": SPEC,
               "decisions": {r: d["decision"] for r, d in docs.items()},
               "untrained_identical_across_runs": identical_untrained(docs["PB1"], docs["PB2"]),
               "B_nonlinear_descriptive": b_nonlinear()}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
