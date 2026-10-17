"""Texture comparators on saved agents: transfer of the original direction AND a fresh probe
(revision step 9; docs/specs/2026-10-06-texture-comparator-design.md).

Readout-only. Three different questions about a comparator perturbation (Gaussian velocity
jitter `gn`, or the hand-authored quadratic drag `qd`):
  1. transfer  Does the world-identity direction fit on the agent's own (learned-law) pools
               read the comparator? (the published 14.5 question)
  2. fresh     Can a probe fit FRESH, by grouped CV on authentic vs comparator pools of the
               same agent, decode the comparator?
  3. trained   Do agents trained under the comparator develop decodable states? Not here:
               that is `run_expB2.py --l3-family gn|qd` with the declared protocol.
A transfer failure answers only question 1.

Usage:
    python scripts/run_texture_fresh_probe.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --family gn --param 0.01 --out artifacts/texture/corrected_l3_h8_wm_gn.json

`--jitter-key {pool,agent}` (closing hook A7e, family arc only, default pool) sets the per-world
jitter seeds of sweep C. pool is the recorded run: the k-th surrogate world of every agent
reseeds from 2551000 + k, so all agents share one realization. agent gives each agent seed its
own block (jitter_seed), a second realization of the same family.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from itasorl.results_io import git_head  # noqa: E402

import argparse
import contextlib
import json
import os
import re
import time

import numpy as np

AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")
# World seed bases for the comparator pools: gn reuses the published 14.5 bases; qd is new and
# disjoint from every other base in the pipeline.
BASES = {"gn": (960_000, 970_000), "qd": (1_900_000, 1_950_000),
         # correlated jitter, spec 2026-10-15-coherence-integration-sweep-design.md
         "ar": (2_400_000, 2_450_000),
         # sweep C: the ar family drawn afresh at every world creation, spec
         # 2026-10-15-coherence-finite-horizon-companion-design.md
         "arc": (2_500_000, 2_550_000)}
JITTER_KEYS = ("pool", "agent")
AGENT_STRIDE = 1000   # surrogate worlds per agent under --jitter-key agent (a cell builds ~2 n_eps)


def jitter_seed(base: int, k: int, key: str = "pool", agent_seed: int = 0) -> int:
    """Family seed for the k-th surrogate world under per_episode_draw. pool (recorded):
    base + 1000 + k for every agent. agent: base + 1000 + AGENT_STRIDE * (agent_seed + 1) + k,
    one block per agent seed, disjoint from the pool block and from each other."""
    if key == "pool":
        return base + 1000 + k
    if key != "agent":
        raise ValueError(f"jitter key must be one of {JITTER_KEYS}, got {key!r}")
    if not 0 <= k < AGENT_STRIDE:
        raise ValueError(f"world {k} overruns the agent block of {AGENT_STRIDE} jitter seeds")
    return base + 1000 + AGENT_STRIDE * (int(agent_seed) + 1) + k


@contextlib.contextmanager
def per_episode_draw(fam, base: int, key: str = "pool", agent_seed: int = 0):
    """While active, every world created with a positive drift (through experiment_b2.make_world
    or l0_audit's imported name) reseeds `fam` from jitter_seed(base, k, key, agent_seed) for
    the k-th such world (base + 1000 + k under the default key), so each surrogate episode
    starts from its own stationary draw and nothing runs on from one episode to the next
    (sweep C spec). Yields the patched make_world."""
    jitter_seed(base, 0, key, agent_seed)   # an unknown key fails before anything is patched
    import itasorl.experiment_b2 as b2
    import itasorl.l0_audit as l0
    orig, count = b2.make_world, [0]

    def make_world(*args, **kw):
        drift = kw.get("drift_sigma", args[1] if len(args) > 1 else 0.0)
        if drift > 0:
            fam.reseed(jitter_seed(base, count[0], key, agent_seed))
            count[0] += 1
        return orig(*args, **kw)
    b2.make_world, l0.make_world = make_world, make_world
    try:
        yield make_world
    finally:
        b2.make_world, l0.make_world = orig, orig


def run_one(task: dict) -> dict:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b import episode_features
    from itasorl.experiment_b2 import collect_pool, load_agent_bundle, transfer_probe
    from itasorl.l0_audit import paired_pooled_readout
    from itasorl.eval_protocols import readout_from_states
    from itasorl.surrogate_l3_families import make_g_ar, make_g_gn, make_g_qd
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=task["l3_hidden"], device="cpu", seed=task["l3_seed"], params=P)
    G = b2._L3_GMOTION
    t0 = time.time()
    agent, norm = load_agent_bundle(task["path"], "cpu")
    d, s, n, steps, rs = task["drift"], task["seed"], task["n_eps"], task["steps"], task["ray_steps"]
    if task["family"] == "gn":
        fam = make_g_gn(sigma_v=task["param"], params=P, seed=0)
    elif task["family"] in ("ar", "arc"):
        fam = make_g_ar(sigma=task["param"], rho=task["rho"], params=P, seed=0)
    else:
        fam = make_g_qd(eps=task["param"], params=P)
    ba, bs = BASES[task["family"]]
    # arc: every surrogate world reseeds the family (sweep C); otherwise one draw per pool
    draw = (per_episode_draw(fam, bs, task.get("jitter_key", "pool"), s)
            if task["family"] == "arc" else contextlib.nullcontext())
    try:
        # the agent's own learned-law pools (standard seed bases): the transfer fit set
        Ha_tr, _ = collect_pool(agent, norm, P, 0.0, n, steps, "cpu", 800_000, rs)
        Hs_tr, _ = collect_pool(agent, norm, P, d, n, steps, "cpu", 850_000, rs)
        b2._L3_GMOTION = fam
        with draw:
            if hasattr(fam, "reseed"):
                fam.reseed(bs + s)
            Ha_c, _ = collect_pool(agent, norm, P, 0.0, n, steps, "cpu", ba, rs)
            Hs_c, _ = collect_pool(agent, norm, P, d, n, steps, "cpu", bs, rs)
            if hasattr(fam, "reseed"):
                fam.reseed(bs + 500 + s)
            paired = paired_pooled_readout(agent, norm, P, d, n_eps=n, steps=steps, ray_steps=rs,
                                           seed_base=ba, seed=s)
    finally:
        b2._L3_GMOTION = G
    Xtr = episode_features(np.concatenate([Ha_tr, Hs_tr]))
    ytr = np.r_[np.zeros(len(Ha_tr)), np.ones(len(Hs_tr))].astype(int)
    Xte = episode_features(np.concatenate([Ha_c, Hs_c]))
    yte = np.r_[np.zeros(len(Ha_c)), np.ones(len(Hs_c))].astype(int)
    return {"drift": d, "seed": s, "arm": task["arm"], "family": task["family"],
            "param": task["param"], "rho": task.get("rho"), "transfer": float(transfer_probe(Xtr, ytr, Xte, yte)),
            "fresh": readout_from_states(Ha_c, Hs_c, seed=s)["target"],
            "fresh_paired": paired["target"], "n_comparator": [int(len(Ha_c)), int(len(Hs_c))],
            "seconds": round(time.time() - t0, 1)}


def provenance_block(out_path: str, argv=None) -> dict:
    """Closing item G46: tracked changes, stack and argv beside the git_commit key the script
    already writes. A readout of saved agents, so the commit key is git_commit_at_promotion."""
    import sys

    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags
    prov = provenance("scripts/run_texture_fresh_probe.py", commit_key="git_commit_at_promotion",
                      output=out_path, device="cpu")
    prov.pop("generated_by")
    prov["argv"] = rel_flags(list(sys.argv[1:] if argv is None else argv))
    return prov


def main(argv=None) -> int:
    from itasorl import folds
    from itasorl.stats import t_ci90
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--family", choices=tuple(BASES), required=True)
    ap.add_argument("--param", type=float, required=True,
                    help="sigma_v (gn), eps (qd), from gate 0; or sigma per component (ar)")
    ap.add_argument("--rho", type=float, default=None, help="AR(1) coefficient, family ar only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jitter-key", choices=JITTER_KEYS, default="pool",
                    help="family arc: pool (recorded; every agent shares the per-world jitter "
                         "seeds) or agent (each agent seed draws its own block)")
    a = ap.parse_args(argv)
    if (a.family in ("ar", "arc")) != (a.rho is not None):
        raise SystemExit("--rho is required for families ar and arc and only for them")
    if a.jitter_key != "pool" and a.family != "arc":
        raise SystemExit("--jitter-key agent applies to family arc only")
    if a.quick:
        a.n_eps, a.steps, a.ray_steps = 30, 12, 4
    cells = []
    for name in sorted(os.listdir(a.agents_dir)):
        m = AGENT_RE.match(name)
        if m:
            cells.append((float(m.group(1)), int(m.group(2)), m.group(3), name))
    if a.quick:
        cells = [c for c in cells if c[1] < 2]
    dmax = max(c[0] for c in cells)
    tasks = [{"path": os.path.join(a.agents_dir, nm), "drift": d, "seed": s, "arm": g,
              "family": a.family, "param": a.param, "rho": a.rho, "n_eps": a.n_eps, "steps": a.steps,
              "ray_steps": a.ray_steps, "l3_hidden": a.l3_hidden, "l3_seed": a.l3_seed,
              "jitter_key": a.jitter_key}
             for d, s, g, nm in cells if d == dmax]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            results = list(pool.imap_unordered(run_one, tasks))
    else:
        results = [run_one(t) for t in tasks]
    results.sort(key=lambda r: (r["arm"], r["seed"]))
    agg = {}
    for g in ("untrained", "predictor", "survival"):
        for m in ("transfer", "fresh", "fresh_paired"):
            v = [r[m] for r in results if r["arm"] == g and np.isfinite(r[m])]
            if v:
                agg[f"{g} {m}"] = {"per_seed": v, "mean": float(np.mean(v)),
                                   "t90": [float(x) for x in t_ci90(v)] if len(v) > 1 else None}
    from promote_bv3_gates import repo_rel
    out = {"generated_by": "scripts/run_texture_fresh_probe.py", "git_commit": git_head(),
           "agents_dir": repo_rel(a.agents_dir),
           "family": a.family, "param": a.param, "rho": a.rho, "seed_bases": BASES[a.family],
           "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
           "cells": results, "aggregate": agg}
    if a.jitter_key != "pool":   # the recorded (pool) output gains no key here
        out["jitter_key"] = a.jitter_key
    out.update(provenance_block(a.out, argv))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k, v in agg.items():
        print(f"  {k:24s} {v['mean']:.3f} {v['t90']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
