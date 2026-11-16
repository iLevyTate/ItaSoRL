"""Extension of the draw-crossed readout on the saved C1 agents (frozen 2026-11-16).

Readout-only: no agent is trained. FINDINGS 25 scored the saved C1 and C2 agents on eight
fresh draws of evaluation worlds and left the C1 survival arm's authentic-versus-authentic
reading straddling the edge of the registered band, with an interval whose width is set by the
number of draws. This script scores the three arms of the saved C1 agents, each on its own
walk, on fifty-six further draws at drift zero and at drift 0.45, with the registered probe
(C = 1) and with the penalty chosen inside each training fold, and summarizes each reading over
agent seeds and draws together. It adds one contrast, an arm's reading at drift 0.45 minus the
same arm's reading at drift zero, paired by agent seed and by draw. The design, its pre-stated
labels and its integrity gate are frozen in
docs/specs/2026-11-16-draw-crossed-extension-design.md. It re-adjudicates no stored run: the
registered verdict of C1 stays as recorded.

Usage:
    python scripts/run_draw_crossed_extension.py \\
        --c1-agents fullruns/branch_preserve/run/C1-rerun/artifacts/goal_stakes/C1-rerun/agents \\
        --checkpoint-dir fullruns/draw_crossed_extension/cells \\
        --out artifacts/draw_crossed/c1_extension.json [--workers 4] [--registered-only]

One JSON per (drift, seed) is written to the checkpoint directory after every draw, so an
interrupted run resumes where it stopped. The eight draws of the first readout are read from
its committed artifact and are not rescored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import _bootstrap  # noqa: F401
import run_draw_crossed_readout as first

from itasorl import draw_crossed as dc
from itasorl import draw_crossed_ext as ext

GENERATED_BY = "scripts/run_draw_crossed_extension.py"
SPEC = "docs/specs/2026-11-16-draw-crossed-extension-design.md"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST_ARTIFACT = "artifacts/draw_crossed/corrected_c1_c2.json"

REGISTERED_BASES = first.REGISTERED_BASES
FIRST_BASES = first.FRESH_BASES                # the eight pairs of 2026-11-12, k from 0 to 7
# Fifty-six further pairs, k from 8 to 63, continuing the sequence of the first readout. No
# evaluation-world seed base in the code lies at or above 6800000.
FURTHER_BASES = tuple((6_000_000 + 100_000 * k, 6_050_000 + 100_000 * k) for k in range(8, 64))
DRIFTS = (ext.DRIFT_ZERO, ext.DRIFT_SURROGATE)
SEEDS = first.SEEDS
ARMS = ext.ARMS
N_EPS, STEPS, RAY_STEPS = first.N_EPS, first.STEPS, first.RAY_STEPS
L3_HIDDEN, L3_SEED = first.L3_HIDDEN, first.L3_SEED
INTEGRITY_REF = first.INTEGRITY_REF
INTEGRITY_TOL = first.INTEGRITY_TOL


def score_draw(arms: dict, params, drift: float, bases) -> dict:
    """The three own-walk feature sets of one (drift, seed) cell on one pair of pools, scored
    exactly as the first readout scores them: each arm's own actor drives its episodes."""
    from itasorl.experiment_b2 import collect_pool

    a, b = bases
    readings = {}
    for arm in ARMS:
        ag, nm = arms[arm]
        Ha, _ = collect_pool(ag, nm, params, 0.0, N_EPS, STEPS, "cpu", a, RAY_STEPS)
        Hs, _ = collect_pool(ag, nm, params, drift, N_EPS, STEPS, "cpu", b, RAY_STEPS)
        readings[f"own/{arm}"] = first.probe_pair(Ha, Hs)
    return {"bases": [int(a), int(b)], "readings": readings}


def run_cell(task: dict) -> dict:
    """One (drift, seed): the registered draw, then the further draws, checkpointed after
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
    drift, seed = task["drift"], task["seed"]
    cp = os.path.join(task["checkpoint_dir"], f"{ext.RUN}_d{format_drift(drift)}_s{seed}.json")
    cell = {"run": ext.RUN, "drift": drift, "seed": seed, "agent_sha256": {}, "draws": []}
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            cell = json.load(fh)
    arms = {}
    for arm in ARMS:
        path = os.path.join(task["agents_dir"], f"agent_d{format_drift(drift)}_s{seed}_{arm}.pt")
        arms[arm] = load_agent_bundle(path, "cpu")
        cell["agent_sha256"][arm] = first._sha256(path)
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
    """Recorded per-seed targets of C1 for the integrity gate, by drift and arm."""
    return {ext.RUN: first.load_reference(path)[ext.RUN]}


def build_artifact(cells: list, a, wall_seconds: float | None) -> dict:
    from promote_bv3_gates import rel_flags, repo_rel

    from itasorl import folds
    from itasorl.results_io import git_head

    reference = load_reference(os.path.join(ROOT, INTEGRITY_REF))
    integrity = dc.integrity_gate(cells, reference, REGISTERED_BASES, tol=INTEGRITY_TOL)
    ok = bool(integrity[ext.RUN]["pass"])
    first_path = os.path.join(ROOT, FIRST_ARTIFACT)
    with open(first_path, encoding="utf-8") as fh:
        first_cells = ext.first_cells(json.load(fh))
    further = [list(b) for b in FURTHER_BASES]
    eight = [list(b) for b in FIRST_BASES]
    complete = (len(cells) == len(DRIFTS) * len(SEEDS)
                and all({tuple(d["bases"]) for d in c["draws"]} >= set(FURTHER_BASES) for c in cells))
    out = {
        "generated_by": GENERATED_BY, "git_commit": git_head(), "spec": SPEC,
        "status": ("extension of the draw-crossed readout of saved agents; descriptive labels; "
                   "re-adjudicates no stored run and changes no stored value or verdict"),
        "run": ext.RUN, "drifts": list(DRIFTS), "seeds": list(SEEDS), "arms": list(ARMS),
        "feature_sets": list(ext.FEATURE_SETS), "probes": list(ext.PROBES),
        "primary_probe": "nested",
        "registered_bases": list(REGISTERED_BASES), "first_bases": eight, "further_bases": further,
        "first_artifact": {"path": FIRST_ARTIFACT, "sha256": first._sha256(first_path)},
        "n_eps": N_EPS, "steps": STEPS, "ray_steps": RAY_STEPS,
        "surrogate": {"hidden": L3_HIDDEN, "g_seed": L3_SEED},
        "fold_scheme": "explicit", "fold_version": folds.scheme_version(),
        "bar": dc.BAR, "margin": dc.MARGIN, "band": list(dc.BAND), "net_floor": ext.NET_FLOOR,
        "interval_level": dc.LEVEL,
        "agents_dir": repo_rel(a.c1_agents),
        "stack": first.stack(), "argv": rel_flags(list(sys.argv[1:])),
        "complete": bool(complete),
        "integrity": {"reference": INTEGRITY_REF, **integrity},
        "cells": sorted(({k: v for k, v in c.items() if k != "seconds_this_pass"} for c in cells),
                        key=lambda c: (c["drift"], c["seed"])),
    }
    out["summary"] = ext.summarize_all(out["cells"], first_cells, further, eight,
                                       list(REGISTERED_BASES), ok)
    if wall_seconds is not None:
        out["wall_seconds"] = round(wall_seconds, 1)
    return out


def print_summary(out: dict) -> None:
    g = out["integrity"][out["run"]]
    print(f"{out['run']} integrity gate: {'PASS' if g['pass'] else 'FAIL'} "
          f"({g['n_checked']} of {g['n_expected']} agents, worst |dev| {g['worst_abs_dev']})")
    for name in ("further", "all_fresh"):
        for probe in out["probes"]:
            blk = out["summary"][name][probe]
            for dkey in ("0.00", "0.45"):
                r = blk["readings"][dkey]["own/survival"]
                print(f"  {name:9s} drift {dkey} {probe:6s} survival {dc.fmt_ci(r)} label: {r['label']}")
            n = blk["net_of_control"][ext.net_name("survival")]
            print(f"  {name:9s} net of control {probe:6s} survival {dc.fmt_ci(n, signed=True)} "
                  f"label: {n['label']}")
    for probe in out["probes"]:
        n = out["summary"]["first_eight"][probe]["net_of_control"][ext.net_name("survival")]
        print(f"  first_eight net of control {probe:6s} survival {dc.fmt_ci(n, signed=True)} (post hoc)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c1-agents", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--registered-only", action="store_true",
                    help="score the registered draw alone (the integrity gate; no further draw is read)")
    ap.add_argument("--seeds", default=None, help="comma separated subset, for a partial pass")
    a = ap.parse_args(argv)

    os.makedirs(a.checkpoint_dir, exist_ok=True)
    seeds = SEEDS if a.seeds is None else tuple(int(s) for s in a.seeds.split(","))
    bases = [REGISTERED_BASES] + ([] if a.registered_only else list(FURTHER_BASES))
    tasks = [{"drift": d, "seed": s, "agents_dir": a.c1_agents,
              "checkpoint_dir": a.checkpoint_dir, "bases": bases}
             for s in seeds for d in DRIFTS]
    t0 = time.time()
    cells = []
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_cell, t): t for t in tasks}
            for f in as_completed(futs):
                c = f.result()
                cells.append(c)
                print(f"done d{c['drift']:.2f} s{c['seed']} "
                      f"({len(c['draws'])} draws, {time.time() - t0:.0f}s elapsed)", flush=True)
    else:
        for t in tasks:
            cells.append(run_cell(t))
            print(f"done d{t['drift']:.2f} s{t['seed']} ({time.time() - t0:.0f}s elapsed)",
                  flush=True)

    out = first._clean(build_artifact(cells, a, time.time() - t0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print_summary(out)
    print(f"saved {a.out} (complete: {out['complete']})")
    return 0 if out["integrity"][out["run"]]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
