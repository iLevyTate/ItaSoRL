"""Draw-crossed readout on the saved C1 and C2 agents (frozen 2026-11-12).

Readout-only: no agent is trained. The registered pooled readout scores every agent seed
on one shared pair of evaluation-world pools, so its interval over agent seeds is
conditional on that draw. This script scores every arm of the saved C1 and C2 agents on
eight fresh, independent draws at drift zero and at drift 0.45, with the registered probe
(C = 1) and with the penalty chosen inside each training fold, and summarizes each reading
over agent seeds and draws together. The design, its pre-stated labels and its integrity
gate are frozen in docs/specs/2026-11-12-draw-crossed-readout-design.md. It re-adjudicates
no stored run: the registered verdicts of C1 and C2 stay as recorded.

Usage:
    python scripts/run_draw_crossed_readout.py \\
        --c1-agents fullruns/branch_preserve/run/C1-rerun/artifacts/goal_stakes/C1-rerun/agents \\
        --c2-agents fullruns/branch_preserve/run/C2-rerun/artifacts/goal_stakes/C2-rerun/agents \\
        --checkpoint-dir fullruns/draw_crossed/cells \\
        --out artifacts/draw_crossed/corrected_c1_c2.json [--workers 4] [--registered-only]

The agents are the saved agents of the C1 and C2 reruns, whose cells equal C1's and C2's
to the bit; they sit on the origin branches run/C1-rerun and run/C2-rerun and are never
committed to another branch. One JSON per (run, drift, seed) is written to the checkpoint
directory after every draw, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import _bootstrap  # noqa: F401
import numpy as np

from itasorl import draw_crossed as dc

GENERATED_BY = "scripts/run_draw_crossed_readout.py"
SPEC = "docs/specs/2026-11-12-draw-crossed-readout-design.md"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTERED_BASES = (800_000, 850_000)
# Eight fresh pairs, disjoint from every evaluation-world seed base in the record (the
# registered pair, the eight audit pairs at 1000000 to 1750000, 910000, 2250000, 3000000,
# 3100000 and the bases of the closing readouts, all below 3200000).
FRESH_BASES = tuple((6_000_000 + 100_000 * k, 6_050_000 + 100_000 * k) for k in range(8))
DRIFTS = (0.0, 0.45)
SEEDS = tuple(range(10))
ARMS = ("survival", "predictor", "untrained")
RUNS = ("C1", "C2")
N_EPS, STEPS, RAY_STEPS = 110, 24, 5
L3_HIDDEN, L3_SEED = 8, 0
INTEGRITY_REF = "artifacts/corrected_verdicts.json"
INTEGRITY_TOL = 0.01


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_pair(Ha: np.ndarray, Hs: np.ndarray) -> dict:
    """Both probes on one pair of pools: the registered probe at C = 1 and the probe whose
    penalty is chosen inside each training fold. Features are [mean h, final h], the fold
    partition is the explicit one, and the episode is the group, all as registered."""
    from run_probe_penalty_sensitivity import cv_auroc, nested_auroc
    from threadpoolctl import threadpool_limits

    from itasorl.experiment_b import episode_features

    out = {"n_auth": len(Ha), "n_surr": len(Hs), "c1": None, "nested": None,
           "chosen_C": [], "convergence_warnings": 0}
    if len(Ha) < 5 or len(Hs) < 5:
        return out
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    g = np.arange(len(y))
    with threadpool_limits(1):
        c1, w1 = cv_auroc(X, y, g, 1.0)
        nst = nested_auroc(X, y, g)
    out.update({"c1": float(c1), "nested": float(nst["target"]),
                "chosen_C": [float(c) for c in nst["chosen_C"]],
                "convergence_warnings": int(w1 + nst["convergence_warnings"])})
    return out


def score_draw(arms: dict, params, drift: float, bases) -> dict:
    """Every feature set of one (run, drift, seed) cell on one pair of pools.

    own/<arm>: that arm's own actor drives its episodes, as in the registered readout.
    replay/<arm>: the survival agent's own episodes on this draw, fed open loop to that
    arm's core network with that arm's own frozen normalizer (the record's replay protocol).
    replay_snorm/<arm>: the same replay with the survival agent's normalizer.
    """
    from itasorl.eval_protocols import record_trajectories, replay_states
    from itasorl.experiment_b2 import collect_pool

    a, b = bases
    feats = {}
    for arm in ARMS:
        ag, nm = arms[arm]
        Ha, _ = collect_pool(ag, nm, params, 0.0, N_EPS, STEPS, "cpu", a, RAY_STEPS)
        Hs, _ = collect_pool(ag, nm, params, drift, N_EPS, STEPS, "cpu", b, RAY_STEPS)
        feats[f"own/{arm}"] = (Ha, Hs)
    sa, sn = arms["survival"]
    oa, aa, _ = record_trajectories(sa, sn, params, 0.0, N_EPS, STEPS, "cpu", a, RAY_STEPS)
    os_, as_, _ = record_trajectories(sa, sn, params, drift, N_EPS, STEPS, "cpu", b, RAY_STEPS)
    for arm in ("predictor", "untrained"):
        ag, nm = arms[arm]
        feats[f"replay/{arm}"] = (replay_states(ag, nm, oa, aa), replay_states(ag, nm, os_, as_))
        feats[f"replay_snorm/{arm}"] = (replay_states(ag, sn, oa, aa),
                                        replay_states(ag, sn, os_, as_))
    return {"bases": [int(a), int(b)],
            "readings": {name: probe_pair(Ha, Hs) for name, (Ha, Hs) in feats.items()}}


def run_cell(task: dict) -> dict:
    """One (run, drift, seed): the registered draw, then the fresh draws, checkpointed after
    every draw."""
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b2 import format_drift, load_agent_bundle
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    params = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=L3_HIDDEN, device="cpu", seed=L3_SEED, params=params)
    run, drift, seed = task["run"], task["drift"], task["seed"]
    cp = os.path.join(task["checkpoint_dir"], f"{run}_d{format_drift(drift)}_s{seed}.json")
    cell = {"run": run, "drift": drift, "seed": seed, "agent_sha256": {}, "draws": []}
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            cell = json.load(fh)
    arms = {}
    for arm in ARMS:
        path = os.path.join(task["agents_dir"], f"agent_d{format_drift(drift)}_s{seed}_{arm}.pt")
        arms[arm] = load_agent_bundle(path, "cpu")
        cell["agent_sha256"][arm] = _sha256(path)
    done = {tuple(d["bases"]) for d in cell["draws"]}
    t0 = time.time()
    for bases in task["bases"]:
        if tuple(bases) in done:
            continue
        cell["draws"].append(score_draw(arms, params, drift, bases))
        tmp = cp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cell, fh, indent=1)
        os.replace(tmp, cp)
    cell["seconds_this_pass"] = round(time.time() - t0, 1)
    return cell


def load_reference(path: str) -> dict:
    """Recorded per-seed targets of C1 and C2 for the integrity gate, by run, drift and arm."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    out = {}
    for run in RUNS:
        r = doc["runs"][run]
        out[run] = {"0.45": {a: list(r["primary"][a]["per_seed"]) for a in ARMS},
                    "0.00": {a: list(r["drift_zero"][a]["per_seed"]) for a in ARMS}}
    return out


def stack() -> dict:
    import platform

    import scipy
    import sklearn
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "device": "cpu",
            "cpu_count": os.cpu_count()}


def build_artifact(cells: list, a, wall_seconds: float | None) -> dict:
    from promote_bv3_gates import rel_flags, repo_rel

    from itasorl import folds
    from itasorl.results_io import git_head

    reference = load_reference(os.path.join(ROOT, INTEGRITY_REF))
    integrity = dc.integrity_gate(cells, reference, REGISTERED_BASES, tol=INTEGRITY_TOL)
    fresh = [list(b) for b in FRESH_BASES]
    complete = all(len({tuple(d["bases"]) for d in c["draws"]} & set(FRESH_BASES)) == len(FRESH_BASES)
                   for c in cells) and len(cells) == len(RUNS) * len(DRIFTS) * len(SEEDS)
    out = {
        "generated_by": GENERATED_BY, "git_commit": git_head(), "spec": SPEC,
        "status": ("draw-crossed readout of saved agents; descriptive labels; re-adjudicates "
                   "no stored run and changes no stored value or verdict"),
        "runs": list(RUNS), "drifts": list(DRIFTS), "seeds": list(SEEDS), "arms": list(ARMS),
        "feature_sets": list(dc.FEATURE_SETS), "probes": list(dc.PROBES),
        "primary_probe": "nested",
        "registered_bases": list(REGISTERED_BASES), "fresh_bases": fresh,
        "n_eps": N_EPS, "steps": STEPS, "ray_steps": RAY_STEPS,
        "surrogate": {"hidden": L3_HIDDEN, "g_seed": L3_SEED},
        "fold_scheme": "explicit", "fold_version": folds.scheme_version(),
        "bar": dc.BAR, "margin": dc.MARGIN, "band": list(dc.BAND), "interval_level": dc.LEVEL,
        "agents_dirs": {"C1": repo_rel(a.c1_agents), "C2": repo_rel(a.c2_agents)},
        "stack": stack(), "argv": rel_flags(list(sys.argv[1:])),
        "complete": bool(complete),
        "integrity": {"reference": INTEGRITY_REF, **integrity},
        "cells": sorted(({k: v for k, v in c.items() if k != "seconds_this_pass"} for c in cells),
                        key=lambda c: (c["run"], c["drift"], c["seed"])),
    }
    out["summary"] = dc.summarize(out["cells"], RUNS, DRIFTS, fresh, list(REGISTERED_BASES),
                                  integrity)
    out["c1_minus_c2"] = dc.summarize_between_runs(out["cells"], "C1", "C2", DRIFTS, fresh,
                                                   integrity)
    if wall_seconds is not None:
        out["wall_seconds"] = round(wall_seconds, 1)
    return out


def _clean(x):
    """Non-finite floats become null, so the artifact is strict JSON. A null reading or bound
    is one that was not computable."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def print_summary(out: dict) -> None:
    for run in out["runs"]:
        g = out["integrity"][run]
        print(f"{run} integrity gate: {'PASS' if g['pass'] else 'FAIL'} "
              f"({g['n_checked']} of {g['n_expected']} agents, worst |dev| {g['worst_abs_dev']})")
        for dkey, by_probe in out["summary"][run].items():
            for probe in out["probes"]:
                blk = by_probe[probe]
                r = blk["readings"]["own/survival"]
                iv = (f"[{r['lo']:.3f}, {r['hi']:.3f}]" if r["computable"] else "[not computable]")
                reg = blk["registered_draw"]["own/survival"]["mean"]
                mean = "n/a" if r["mean"] is None else f"{r['mean']:.3f}"
                reg = "n/a" if reg is None else f"{reg:.3f}"
                print(f"  {run} drift {dkey} {probe:6s} survival {mean} {iv} "
                      f"label: {r['label']}; registered draw {reg}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c1-agents", required=True)
    ap.add_argument("--c2-agents", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--registered-only", action="store_true",
                    help="score the registered draw alone (the integrity gate; no fresh draw is read)")
    ap.add_argument("--seeds", default=None, help="comma separated subset, for a partial pass")
    a = ap.parse_args(argv)

    os.makedirs(a.checkpoint_dir, exist_ok=True)
    seeds = SEEDS if a.seeds is None else tuple(int(s) for s in a.seeds.split(","))
    bases = [REGISTERED_BASES] + ([] if a.registered_only else list(FRESH_BASES))
    agents = {"C1": a.c1_agents, "C2": a.c2_agents}
    tasks = [{"run": run, "drift": d, "seed": s, "agents_dir": agents[run],
              "checkpoint_dir": a.checkpoint_dir, "bases": bases}
             for s in seeds for run in RUNS for d in DRIFTS]
    t0 = time.time()
    cells = []
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_cell, t): t for t in tasks}
            for f in as_completed(futs):
                c = f.result()
                cells.append(c)
                print(f"done {c['run']} d{c['drift']:.2f} s{c['seed']} "
                      f"({len(c['draws'])} draws, {time.time() - t0:.0f}s elapsed)", flush=True)
    else:
        for t in tasks:
            cells.append(run_cell(t))
            print(f"done {t['run']} d{t['drift']:.2f} s{t['seed']} ({time.time() - t0:.0f}s elapsed)",
                  flush=True)

    out = _clean(build_artifact(cells, a, time.time() - t0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print_summary(out)
    print(f"saved {a.out} (complete: {out['complete']})")
    return 0 if all(out["integrity"][r]["pass"] for r in RUNS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
