"""Rebuild the --dump-states pooled-state dumps of a finished run from its saved agents.

A run written with --save-agents but not --dump-states keeps every frozen agent, and the dumps
are nothing but the recurrent states of the standard pooled readout on those agents (world seed
bases 800000 authentic, 850000 surrogate; 110 episodes by 24 steps). This replays exactly that
readout, writes states_d{drift}_s{seed}_{arm}.npz in the layout run_expB2.py writes, and checks
each cell: the rebuilt pool's target must equal the target the run recorded in its cell to the
bit, or the dump is not the run's and the script exits non-zero. The dumps then feed
scripts/audit_behavior_mediation.py.

    python scripts/regenerate_state_dumps.py --run-dir artifacts/goal_stakes/T-touch \\
        --out-dir /tmp/T-touch_states --workers 4
    python scripts/audit_behavior_mediation.py /tmp/T-touch_states \\
        --json artifacts/goal_stakes/T-touch/behavior_audit.json

`--tolerance ABS` (closing item G57) is for a cross-stack replay, where the dumps are rebuilt on
another machine or library stack than the run's. The default stays the bit-exact check. With
the flag, the report marks each cell `reproduced_exact` and `within_tolerance`, records
`mode: cross_stack_replay`, the tolerance and the stack, and the script exits 0 only if every
|rebuilt target - recorded target| <= ABS. A non-finite difference never passes. The report
also carries a provenance block (closing item G46).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import math
import os
import re
import sys
import time

AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def tasks_for(run_dir: str) -> list:
    """One task per saved agent, carrying the run's own knobs from that agent's cell."""
    tasks = []
    for path in sorted(glob.glob(os.path.join(run_dir, "agents", "agent_d*_s*_*.pt"))):
        m = AGENT_RE.search(os.path.basename(path))
        if not m:
            continue
        drift, seed, arm = m.group(1), int(m.group(2)), m.group(3)
        with open(os.path.join(run_dir, "cells", f"cell_d{drift}_s{seed}.json"), encoding="utf-8") as fh:
            cell = json.load(fh)["cell"]
        tasks.append({"agent": path, "drift": float(drift), "drift_tag": drift, "seed": seed,
                      "arm": arm, "recorded_target": cell["agents"][arm]["pool"]["target"],
                      "objective": cell.get("objective", "survival"),
                      "mortal": bool(cell.get("mortal", True)), "knobs": cell.get("knobs") or {}})
    return tasks


def run_task(task: dict) -> dict:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    knobs = task["knobs"]
    b2.DRIFT_MODE = "l3"
    b2.OBJECTIVE = task["objective"]
    b2.MORTAL = task["mortal"]
    for key, field in (("n_pellets", "n_pellets"), ("reach", "reach")):
        if knobs.get(key) is not None:
            b2.SURVIVAL_FOOD[field] = knobs[key]
    if knobs.get("basal_e") is not None:
        b2.SURVIVAL_METAB["basal_E"] = knobs["basal_e"]
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=knobs.get("l3_hidden", 8), device="cpu",
                              seed=knobs.get("l3_seed", 0), params=P)
    t0 = time.time()
    agent, norm = b2.load_agent_bundle(task["agent"])
    dump = os.path.join(task["out_dir"], f"states_d{task['drift_tag']}_s{task['seed']}_{task['arm']}.npz")
    pool = b2.pooled_readout(agent, norm, P, task["drift"], n_eps=110, steps=24, ray_steps=5,
                             device="cpu", seed=task["seed"], dump_path=dump)
    return {"drift": task["drift"], "seed": task["seed"], "arm": task["arm"],
            "target": float(pool["target"]), "recorded_target": task["recorded_target"],
            "reproduced": float(pool["target"]) == task["recorded_target"],
            "seconds": round(time.time() - t0, 1)}


def judge(results: list, tolerance: float | None) -> tuple[dict, list]:
    """The report fields and the failing cells. Bit-exact without a tolerance (each cell's
    `reproduced`); with one, each cell gains abs_diff, reproduced_exact and within_tolerance,
    and a cell fails when its difference is not finite or exceeds the tolerance."""
    if tolerance is None:
        return {}, [r for r in results if not r["reproduced"]]
    bad = []
    for r in results:
        diff = abs(float(r["target"]) - float(r["recorded_target"]))
        ok = bool(math.isfinite(diff) and diff <= tolerance)
        r.update(abs_diff=diff, reproduced_exact=bool(r["reproduced"]), within_tolerance=ok)
        if not ok:
            bad.append(r)
    from agent_free_common import stack_record
    fields = {"mode": "cross_stack_replay", "tolerance": tolerance,
              "within_tolerance": len(results) - len(bad), "all_within_tolerance": not bad,
              "max_abs_diff": max((r["abs_diff"] for r in results), default=float("nan")),
              "stack": stack_record()}
    return fields, bad


def run_provenance(a) -> dict:
    """Closing item G46: commit, tracked changes, stack, agents dir and argv. A rebuild of a
    run's stored dumps, so the commit key is git_commit_at_promotion."""
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    p = provenance("scripts/regenerate_state_dumps.py", commit_key="git_commit_at_promotion",
                   output=repo_rel(a.report) if a.report else None, device="cpu")
    return {**p, "agents_dir": repo_rel(os.path.join(a.run_dir, "agents")),
            "argv": rel_flags(list(sys.argv[1:]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", default=None, help="also write the per-cell check here (json)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=None, metavar="ABS",
                    help="cross-stack replay: pass a cell when |rebuilt - recorded| <= ABS "
                         "(default: bit-exact)")
    a = ap.parse_args()
    if a.tolerance is not None and not (math.isfinite(a.tolerance) and a.tolerance >= 0):
        raise SystemExit("--tolerance must be a finite number >= 0")
    if os.environ.get("ITASORL_FOLDS") != "explicit":
        raise SystemExit("set ITASORL_FOLDS=explicit, the partition every recorded target uses")
    os.makedirs(a.out_dir, exist_ok=True)
    prov = run_provenance(a)
    tasks = [{**t, "out_dir": a.out_dir} for t in tasks_for(a.run_dir)]
    if not tasks:
        raise SystemExit(f"no saved agents under {a.run_dir}/agents")
    print(f"rebuilding {len(tasks)} dumps from {a.run_dir} into {a.out_dir}", flush=True)
    results = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_task, tasks):
                results.append(r)
                print(f"  d={r['drift']:.2f} s={r['seed']} {r['arm']:9s} reproduced={r['reproduced']} "
                      f"({r['seconds']} s)", flush=True)
    else:
        for t in tasks:
            r = run_task(t)
            results.append(r)
            print(f"  d={r['drift']:.2f} s={r['seed']} {r['arm']:9s} reproduced={r['reproduced']} "
                  f"({r['seconds']} s)", flush=True)
    results.sort(key=lambda r: (r["drift"], r["seed"], r["arm"]))
    tol_fields, bad = judge(results, a.tolerance)
    exact_bad = [r for r in results if not r["reproduced"]]
    summary = {"run_dir": a.run_dir, "n": len(results), "reproduced": len(results) - len(exact_bad),
               "all_reproduced": not exact_bad, **tol_fields, "cells": results, "provenance": prov}
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    print(f"reproduced {summary['reproduced']}/{summary['n']} recorded targets to the bit", flush=True)
    if a.tolerance is not None:
        print(f"within {a.tolerance:g}: {summary['within_tolerance']}/{summary['n']} "
              f"(max |diff| {summary['max_abs_diff']:.3g})", flush=True)
        if bad:
            print(f"OUTSIDE the {a.tolerance:g} tolerance; do not audit them:",
                  [(r["drift"], r["seed"], r["arm"]) for r in bad], file=sys.stderr)
            return 1
        return 0
    if bad:
        print("NOT the run's dumps; do not audit them:", [(r["drift"], r["seed"], r["arm"]) for r in bad],
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
