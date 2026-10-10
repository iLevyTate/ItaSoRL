"""Policy- and exposure-controlled readouts on saved agents (revision step 6).

Readout-only for the saved arms; one retraining per seed for the exposure control. For every
seed at the strongest drift of a run written with --save-agents:

  1. Retrain the survival arm with the run's own configuration, logging every training batch,
     and require the result to be bit-identical to the saved survival agent (otherwise stop:
     the logged data would not be what the agent was trained on).
  2. Train `predictor_logged`: the prediction objective on exactly those batches.
  3. Score four arms (untrained, predictor, predictor_logged, survival) under three
     protocols, each with its own frozen normalizer:
       own       each arm's own actor head drives the episodes (the standard protocol; for the
                 untrained, predictor and predictor_logged arms that head never received an
                 actor loss);
       scripted  every arm follows the same scripted action sequence (the predictor's
                 training policy) in both pools;
       replay    the survival agent's own evaluation episodes are fed open loop to every arm.
     All three use the standard world seeds (800000 authentic, 850000 surrogate) and the
     standard [mean h, final h] probe on the active fold partition.

Usage:
    python scripts/run_policy_controlled_readouts.py --run-dir fullruns/corrected_l3_h8_wm \\
        --out artifacts/policy_controls/corrected_l3_h8_wm.json [--workers 4] [--quick]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from itasorl.results_io import git_head  # noqa: E402

import argparse
import json
import os
import time

import numpy as np

ARMS = ("untrained", "predictor", "predictor_logged", "survival")
PROTOCOLS = ("own", "scripted", "replay")


def _cfg_from_cells(run_dir: str) -> dict:
    import glob
    p = sorted(glob.glob(os.path.join(run_dir, "cells", "cell_d*_s*.json")))[0]
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def run_knobs(payload: dict) -> dict:
    """The trained arm's objective and mortality as recorded in a cell; cells written before
    the goal-and-stakes flags existed carry neither and get the registered defaults."""
    return {"objective": payload.get("objective", "survival"),
            "mortal": bool(payload.get("mortal", True))}


def run_seed(task: dict) -> dict:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.eval_protocols import (collect_pool_scripted, readout_from_states,
                                        record_trajectories, replay_states,
                                        train_predictor_on_logged)
    from itasorl.experiment_b2 import collect_pool, format_drift, load_agent_bundle
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    k, seed, d = task, task["seed"], task["drift"]
    b2.DRIFT_MODE = "l3"
    # The retrain must use the run's own trained-arm reward and mortality, or it cannot be
    # bit-identical to the saved agent (goal-and-stakes runs; absent keys are the defaults).
    b2.OBJECTIVE = k.get("objective", "survival")
    b2.MORTAL = bool(k.get("mortal", True))
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=k["l3_hidden"], device="cpu", seed=k["l3_seed"], params=P)
    agents_dir = os.path.join(k["run_dir"], "agents")
    arms = {g: load_agent_bundle(os.path.join(agents_dir, f"agent_d{format_drift(d)}_s{seed}_{g}.pt"))
            for g in ("untrained", "predictor", "survival")}
    t0 = time.time()
    log: list = []
    sa, sn, _ = b2.train_actor_critic(d, P, n_eps=k["n_eps"], updates=k["updates"],
                                      hidden=k["hidden"], max_steps=k["max_steps"],
                                      ray_steps=k["ray_steps"], seed=seed, device="cpu",
                                      shaping_coef=k["shaping_coef"],
                                      world_model=k["world_model"],
                                      gae_bootstrap=k["gae_bootstrap"], log_batches=log)
    saved = arms["survival"][0].state_dict()
    identical = all(torch.equal(saved[n], v) for n, v in sa.state_dict().items())
    out = {"seed": seed, "drift": d, "retrain_identical": bool(identical),
           "retrain_seconds": round(time.time() - t0, 1), "targets": {}}
    if not identical:
        out["error"] = "retrained survival agent differs from the saved one; logged data invalid"
        return out
    arms["predictor_logged"] = train_predictor_on_logged(log, hidden=k["hidden"], seed=seed)
    n, steps, rs = k["pool_n"], k["pool_steps"], k["ray_steps"]
    obs_a, act_a, _ = record_trajectories(*arms["survival"], P, 0.0, n, steps, "cpu", 800_000, rs)
    obs_s, act_s, _ = record_trajectories(*arms["survival"], P, d, n, steps, "cpu", 850_000, rs)
    for g in ARMS:
        ag, nm = arms[g]
        Ha, _ = collect_pool(ag, nm, P, 0.0, n, steps, "cpu", 800_000, rs)
        Hs, _ = collect_pool(ag, nm, P, d, n, steps, "cpu", 850_000, rs)
        Sa, _ = collect_pool_scripted(ag, nm, P, 0.0, n, steps, "cpu", 800_000, rs)
        Ss, _ = collect_pool_scripted(ag, nm, P, d, n, steps, "cpu", 850_000, rs)
        out["targets"][g] = {
            "own": readout_from_states(Ha, Hs, seed=seed),
            "scripted": readout_from_states(Sa, Ss, seed=seed),
            "replay": readout_from_states(replay_states(ag, nm, obs_a, act_a),
                                          replay_states(ag, nm, obs_s, act_s), seed=seed),
        }
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--no-world-model", dest="world_model", action="store_false",
                    help="the run's survival arm was trained without the decoder (C2)")
    for flag, default in (("--updates", 300), ("--n-eps", 16), ("--hidden", 96),
                          ("--max-steps", 80), ("--ray-steps", 5), ("--pool-n", 110),
                          ("--pool-steps", 24), ("--l3-hidden", 8), ("--l3-seed", 0)):
        ap.add_argument(flag, type=int, default=default)
    ap.add_argument("--quick", action="store_true",
                    help="the run was made with run_expB2.py --quick (sets its scale)")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_eps, a.hidden, a.max_steps, a.ray_steps = 60, 8, 64, 40, 4
        a.pool_n, a.pool_steps = 40, 16
    from itasorl import folds
    from itasorl.stats import t_ci90
    cell = _cfg_from_cells(a.run_dir)
    payload = cell["cell"]
    import glob
    seeds = sorted({int(os.path.basename(p).split("_s")[1].split(".json")[0])
                    for p in glob.glob(os.path.join(a.run_dir, "cells", "cell_d*_s*.json"))})
    if a.seeds is not None:
        seeds = a.seeds
    with open(os.path.join(a.run_dir, "expB2_results.json"), encoding="utf-8") as fh:
        dmax = max(float(x) for x in json.load(fh))
    base = {"run_dir": a.run_dir, "drift": dmax, "l3_hidden": a.l3_hidden, "l3_seed": a.l3_seed,
            "n_eps": a.n_eps, "updates": a.updates, "hidden": a.hidden,
            "max_steps": a.max_steps, "ray_steps": a.ray_steps, "shaping_coef": 1.0,
            "world_model": a.world_model,
            "gae_bootstrap": payload.get("gae_bootstrap", "pre_transition"),
            **run_knobs(payload),
            "pool_n": a.pool_n, "pool_steps": a.pool_steps}
    tasks = [{**base, "seed": s} for s in seeds]
    print(f"policy-controlled readouts: {a.run_dir} drift={dmax} seeds={seeds} "
          f"world_model={a.world_model} trainer={base['gae_bootstrap']}", flush=True)
    results = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_seed, tasks):
                results.append(r)
                print(f"  seed {r['seed']} done ({r.get('seconds')} s, identical={r['retrain_identical']})", flush=True)
    else:
        for t in tasks:
            r = run_seed(t)
            results.append(r)
            print(f"  seed {r['seed']} done ({r.get('seconds')} s, identical={r['retrain_identical']})", flush=True)
    results.sort(key=lambda r: r["seed"])
    agg = {}
    for g in ARMS:
        for pr in PROTOCOLS:
            v = [r["targets"][g][pr]["target"] for r in results if "targets" in r and g in r["targets"]]
            v = [x for x in v if np.isfinite(x)]
            if v:
                agg[f"{g} {pr}"] = {"per_seed": v, "mean": float(np.mean(v)),
                                   "t90": [float(x) for x in t_ci90(v)] if len(v) > 1 else None}
    out = {"generated_by": "scripts/run_policy_controlled_readouts.py", "git_commit": git_head(), "run_dir": a.run_dir,
           "config": base, "fold_scheme": folds.current_scheme(),
           "fold_version": folds.scheme_version(), "cells": results, "aggregate": agg}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for key, v in agg.items():
        print(f"  {key:28s} {v['mean']:.3f} {v['t90']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
