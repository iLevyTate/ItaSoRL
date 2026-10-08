"""Controlled persistence readout on saved agents (revision step 7, itasorl/persistence.py).

Readout-only. Scores every arm at the strongest drift of a run written with --save-agents
(and the drift-0 survival agents as the L0 sanity check, where every condition must read
exactly 0.5) under the four persistence conditions, and aggregates across agent seeds.

Usage:
    python scripts/run_persistence_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --out artifacts/persistence/corrected_l3_h8_wm.json [--workers 4] [--quick]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from itasorl.results_io import git_head  # noqa: E402

import argparse
import json
import os
import re
import time

import numpy as np

AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def run_one(task: dict) -> dict:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b2 import load_agent_bundle
    from itasorl.persistence import persistence_readout
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=task["l3_hidden"], device="cpu", seed=task["l3_seed"], params=P)
    t0 = time.time()
    agent, norm = load_agent_bundle(task["path"], "cpu")
    r = persistence_readout(agent, norm, P, task["drift"], n_pairs=task["n_pairs"],
                            prefix_steps=task["prefix"], tail_steps=task["tail"],
                            ray_steps=task["ray_steps"])
    return {"drift": task["drift"], "seed": task["seed"], "arm": task["arm"],
            "seconds": round(time.time() - t0, 1), "conditions": r}


def main() -> int:
    from itasorl import folds
    from itasorl.stats import t_ci90
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-pairs", type=int, default=110)
    ap.add_argument("--prefix", type=int, default=20)
    ap.add_argument("--tail", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="seeds 0-1, 12 pairs, short prefix/tail")
    a = ap.parse_args()
    if a.quick:
        a.n_pairs, a.prefix, a.tail, a.ray_steps = 12, 8, 10, 4
    cells = []
    for name in sorted(os.listdir(a.agents_dir)):
        m = AGENT_RE.match(name)
        if m:
            cells.append((float(m.group(1)), int(m.group(2)), m.group(3), name))
    if a.quick:
        cells = [c for c in cells if c[1] < 2]
    dmax = max(c[0] for c in cells)
    tasks = [{"path": os.path.join(a.agents_dir, nm), "drift": d, "seed": s, "arm": g,
              "n_pairs": a.n_pairs, "prefix": a.prefix, "tail": a.tail,
              "ray_steps": a.ray_steps, "l3_hidden": a.l3_hidden, "l3_seed": a.l3_seed}
             for d, s, g, nm in cells if d == dmax or (d == 0.0 and g == "survival")]
    results = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_one, tasks):
                results.append(r)
                print(f"  d{r['drift']} s{r['seed']} {r['arm']}: replay window "
                      f"{r['conditions']['replay:prefix']['window']:.3f} ({r['seconds']} s)", flush=True)
    else:
        for t in tasks:
            r = run_one(t)
            results.append(r)
            print(f"  d{r['drift']} s{r['seed']} {r['arm']}: replay window "
                  f"{r['conditions']['replay:prefix']['window']:.3f} ({r['seconds']} s)", flush=True)
    results.sort(key=lambda r: (r["drift"], r["arm"], r["seed"]))
    agg = {}
    for d in sorted({r["drift"] for r in results}):
        for g in ("untrained", "predictor", "survival"):
            rows = [r for r in results if r["drift"] == d and r["arm"] == g]
            if not rows:
                continue
            for key in rows[0]["conditions"]:
                win = [r["conditions"][key]["window"] for r in rows]
                late = [r["conditions"][key]["late"] for r in rows]
                curves = np.array([r["conditions"][key]["auc_by_t"] for r in rows], float)
                agg[f"d={d:.2f} {g} {key}"] = {
                    "window_per_seed": win, "window_mean": float(np.nanmean(win)),
                    "window_t90": [float(x) for x in t_ci90(win)] if len(win) > 1 else None,
                    "late_mean": float(np.nanmean(late)),
                    "auc_by_t_mean": curves.mean(0).tolist() if curves.size else []}
    out = {"generated_by": "scripts/run_persistence_readout.py", "git_commit": git_head(), "agents_dir": a.agents_dir,
           "config": {k: getattr(a, k) for k in ("n_pairs", "prefix", "tail", "ray_steps",
                                                 "l3_hidden", "l3_seed")},
           "seed_base": 980_000, "fold_scheme": folds.current_scheme(),
           "fold_version": folds.scheme_version(), "cells": results, "aggregate": agg}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k, v in agg.items():
        print(f"  {k:48s} window {v['window_mean']:.3f} {v['window_t90']}  late {v['late_mean']:.3f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
