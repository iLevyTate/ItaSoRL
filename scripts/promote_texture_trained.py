"""Promote a comparator-trained texture run to a committed artifact.

A comparator-trained run (a run of the registered L3 protocol with the surrogate family
switched) is adjudicated by the registered rule and by the registered gate battery. Whether it
is informative or not, the numbers reach the preregistration log and the manuscript, so they
must trace to a committed artifact rather than to a gitignored run bundle. This extracts the
per-seed decision values and the full gate battery, recomputes the rule and every gate from
them, and records the routing.

Usage:
    python scripts/promote_texture_trained.py \\
        --run fullruns/T_gn_l3_h8_wm/expB2_results.json \\
        --out artifacts/texture/T_gn_l3_h8_wm.json \\
        --family gn --family-param 0.01 --gate0 artifacts/expH2/texture_knockout_h8.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

from itasorl.results_io import git_head  # noqa: E402
from itasorl.stats import equivalence_test, paired_contrast, t_ci90  # noqa: E402

ARMS = ("untrained", "predictor", "survival")
# The full section 7 battery. Named here so a gate cannot go missing from a promoted artifact
# without the audit noticing: an absent gate is not a passed gate.
REGISTERED_GATES = ("gate0_oracle_band", "engagement", "l0_tost", "speed_positive_control",
                    "reward_leakage", "survivorship", "untrained_floor")
BAR = 0.65
MARGIN = 0.05
SPEED_GATE = 0.75          # preregistration section 7, gate 3
LEAK_TOL = 0.1             # gate 4
L0_ROPE = 0.05             # gate 2 margin


def speed_gate(run: dict, threshold: float = SPEED_GATE) -> dict:
    """Gate 3, scored over every pool at every drift.

    Preregistration section 7 gate 3 carries no arm restriction, and the project's convention
    (scripts/promote_bv3_gates.py) is "speed probe >= 0.75 in every pool". Scoring the survival
    arm alone understates a failure that sits in a baseline arm, and the baseline arms are what
    the registered margins are measured against, so an unprobeable baseline is exactly the case
    the positive control exists to catch.
    """
    pools, below = [], {}
    surv = []
    for drift, cells in run.items():
        for arm in ARMS:
            if arm not in cells:
                continue
            for x in cells[arm]["pool_speed"]:
                x = float(x)
                pools.append(x)
                if arm == "survival":
                    surv.append(x)
                if x < threshold:
                    below.setdefault(arm, {}).setdefault(str(float(drift)), 0)
                    below[arm][str(float(drift))] += 1
    n_below = sum(c for a in below.values() for c in a.values())
    return {"threshold": threshold, "n_pools": len(pools), "n_below": n_below,
            "min_all_pools": min(pools) if pools else float("nan"),
            "mean_all_pools": sum(pools) / len(pools) if pools else float("nan"),
            "min_survival": min(surv) if surv else float("nan"),
            "mean_survival": sum(surv) / len(surv) if surv else float("nan"),
            "below_by_arm_drift": below,
            "pass": bool(pools) and n_below == 0}


def engagement_gate(cells_dir: str | None) -> dict:
    """Gate 1, read from the run's per-cell records.

    Engagement is the one gate the registered matrix explicitly routes on, so it must be
    verifiable from the repository rather than asserted in prose. A missing cells directory is
    a failed gate, never an absent one.
    """
    out = {"cells_dir": cells_dir, "n_cells": 0, "n_engaged": 0, "per_cell": [], "pass": False}
    if not cells_dir or not os.path.isdir(cells_dir):
        out["note"] = "cells directory not found; engagement cannot be verified"
        return out
    for name in sorted(os.listdir(cells_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(cells_dir, name), encoding="utf-8") as fh:
            c = json.load(fh)
        c = c.get("cell", c)
        eng = c.get("eng") or {}
        ok = bool(eng.get("engaged"))
        out["per_cell"].append({"drift": c.get("drift"), "seed": c.get("seed"), "engaged": ok})
        out["n_cells"] += 1
        out["n_engaged"] += int(ok)
    out["pass"] = out["n_cells"] > 0 and out["n_engaged"] == out["n_cells"]
    return out


def _agg(v):
    v = [float(x) for x in v]
    out = {"per_seed": v, "mean": float(np.mean(v)), "n": len(v),
           "seeds_at_or_above_bar": int(sum(1 for x in v if x >= BAR))}
    if len(v) > 1:
        out["t90"] = [float(x) for x in t_ci90(v)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--family-param", type=float, required=True)
    ap.add_argument("--gate0", default=None, help="artifact holding this family's gate-0 row")
    ap.add_argument("--cells", default=None,
                    help="the run's cells/ directory, which carries the engagement records")
    ap.add_argument("--drift", type=float, default=0.45)
    a = ap.parse_args()

    with open(a.run, encoding="utf-8") as fh:
        run = json.load(fh)
    key = next(k for k in run if abs(float(k) - a.drift) < 1e-9)
    zero = next(k for k in run if abs(float(k)) < 1e-9)
    v, v0 = run[key], run[zero]

    primary = {arm: _agg(v[arm]["pool_target"]) for arm in ARMS if arm in v}
    l0 = {arm: _agg(v0[arm]["pool_target"]) for arm in ARMS if arm in v0}

    eq = equivalence_test(v0["survival"]["pool_target"])
    leak = max(abs(float(x) - 0.5) for arm in ARMS for x in v[arm]["pool_reward_leak"])
    deaths = sum(int(x) for arm in ARMS
                 for k2 in ("pool_deaths_auth", "pool_deaths_surr") for x in v[arm][k2])

    gates = {
        "l0_tost": {"mean": eq.mean, "p": eq.p_value, "margin": L0_ROPE,
                    "pass": bool(eq.equivalent)},
        "speed_positive_control": speed_gate(run),
        "engagement": engagement_gate(a.cells),
        "reward_leakage": {"tolerance": LEAK_TOL, "max_abs_dev": leak, "pass": leak <= LEAK_TOL},
        "survivorship": {"total_deaths": deaths, "pass": deaths == 0},
        "untrained_floor": {"mean": primary["untrained"]["mean"],
                            "pass": abs(primary["untrained"]["mean"] - 0.5) <= LEAK_TOL},
    }
    if a.gate0 and os.path.exists(a.gate0):
        with open(a.gate0, encoding="utf-8") as fh:
            g0doc = json.load(fh)
        row = (g0doc.get("gate0") or {}).get(a.family, {}).get("selected_row")
        if row:
            gates["gate0_oracle_band"] = {"oracle_auroc": row["oracle_auroc"],
                                          "in_band": row["in_band"], "floor": row["floor"],
                                          "mech_leak_pass": row["mech_leak_pass"],
                                          "pass": bool(row["in_band"] and row["mech_leak_pass"]),
                                          "source": a.gate0}
    missing = [k for k in REGISTERED_GATES if k not in gates]
    if missing:
        raise SystemExit(f"refusing to promote with gates unscored: {missing}")
    failed = sorted(k for k, g in gates.items() if not g["pass"])

    clauses = {
        "bar": primary["survival"]["mean"] >= BAR,
        "over_predictor": primary["survival"]["mean"] - primary["predictor"]["mean"] >= MARGIN,
        "over_untrained": primary["survival"]["mean"] - primary["untrained"]["mean"] >= MARGIN,
    }
    contrasts = {b: paired_contrast(v["survival"]["pool_target"], v[b]["pool_target"])
                 for b in ("predictor", "untrained")}

    # The registered matrix (section 8) names two reachable verdicts and both require every
    # gate to pass. With a gate open neither is reached, and the matrix routes only gate-0 and
    # engagement failures to "uninformative", so a failure of another gate has no cell at all.
    if failed:
        routing = ("UNINFORMATIVE: the registered matrix requires every gate to pass for either "
                   "verdict, and it names no cell for a failure of " + ", ".join(failed) +
                   "; the run is not promoted as a negative and strengthens no claim")
    elif all(clauses.values()):
        routing = "ENCODING INDUCED (conditional on this fingerprint)"
    else:
        routing = "STRENGTHENED NEGATIVE"

    out = {
        "source_run": a.run.replace("\\", "/"),
        "generated_by": "scripts/promote_texture_trained.py",
        "git_commit_at_promotion": git_head(),
        "spec": "docs/specs/2026-10-09-gn-comparator-trained-design.md",
        "family": a.family, "family_param": a.family_param, "drift": a.drift,
        "bars": {"bar": BAR, "margin": MARGIN},
        "primary": primary, "l0_control": l0,
        "clauses": clauses, "rule_met": bool(all(clauses.values()) and not failed),
        "contrasts": {k: {"mean": c["mean"], "t90": c["t90"]} for k, c in contrasts.items()},
        "gates": gates, "gates_failed": failed, "routing": routing,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out}")
    print(f"  survival {primary['survival']['mean']:.3f} "
          f"[{primary['survival']['t90'][0]:.3f}, {primary['survival']['t90'][1]:.3f}], "
          f"{primary['survival']['seeds_at_or_above_bar']}/10 at the bar")
    print(f"  gates failed: {failed or 'none'}")
    print(f"  routing: {routing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
