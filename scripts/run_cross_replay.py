"""Cross-run replay: does the next-observation decoder change the state or the trajectories?

EXPLORATORY, written after the corrected runs C1 (decoder on) and C2 (decoder off) were read;
not part of any frozen protocol (revision step 11). For each agent seed at the strongest
drift, the survival agent of each run is driven by its own policy through the standard pools
(seed bases 800000 / 850000), its raw observations and actions are recorded, and both
survival trunks are then run open loop on both sets of streams. The standard pooled probe
reads each of the four (trunk, streams) combinations:

  C1 trunk on C1 streams   reproduces C1's own readout (up to open-loop replay)
  C2 trunk on C2 streams   reproduces C2's own readout
  C2 trunk on C1 streams   would the decoder-less trunk read the world from the streams the
                           decoder-carrying policy generates?
  C1 trunk on C2 streams   and the reverse

If the trunk matters, rows differ by trunk at fixed streams; if the trajectories matter, they
differ by streams at fixed trunk. Seed-paired contrasts with t-based 90% CIs.

Usage:
    python scripts/run_cross_replay.py --a fullruns/corrected_l3_h8_wm \
        --b fullruns/corrected_l3_h8_nowm --out artifacts/cross_replay/corrected_c1_c2.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np

COMBOS = (("a", "a"), ("b", "a"), ("a", "b"), ("b", "b"))   # (trunk, streams)


def run_seed(task: dict) -> dict:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.eval_protocols import readout_from_states, record_trajectories, replay_states
    from itasorl.experiment_b2 import format_drift, load_agent_bundle
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=task["l3_hidden"], device="cpu", seed=task["l3_seed"], params=P)
    s, d, n, steps, rs = task["seed"], task["drift"], task["pool_n"], task["pool_steps"], task["ray_steps"]
    t0 = time.time()
    agents = {k: load_agent_bundle(os.path.join(task[k], "agents", f"agent_d{format_drift(d)}_s{s}_survival.pt"))
              for k in ("a", "b")}
    streams = {}
    for k, (ag, nm) in agents.items():
        oa, aa, _ = record_trajectories(ag, nm, P, 0.0, n, steps, "cpu", 800_000, rs)
        os_, as_, _ = record_trajectories(ag, nm, P, d, n, steps, "cpu", 850_000, rs)
        streams[k] = (oa, aa, os_, as_)
    out = {"seed": s, "drift": d, "targets": {}}
    for trunk, src in COMBOS:
        ag, nm = agents[trunk]
        oa, aa, os_, as_ = streams[src]
        r = readout_from_states(replay_states(ag, nm, oa, aa), replay_states(ag, nm, os_, as_), seed=s)
        out["targets"][f"{trunk}_on_{src}"] = r
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="run dir A (decoder on)")
    ap.add_argument("--b", required=True, help="run dir B (decoder off)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--drift", type=float, default=0.45)
    a = ap.parse_args()
    from itasorl import folds
    from itasorl.stats import paired_contrast, t_ci90
    base = {"a": os.path.abspath(a.a), "b": os.path.abspath(a.b), "drift": a.drift, "pool_n": 110,
            "pool_steps": 24, "ray_steps": 5, "l3_hidden": 8, "l3_seed": 0}
    tasks = [{**base, "seed": s} for s in a.seeds]
    results = []
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_seed, tasks):
            results.append(r)
            print(f"  seed {r['seed']}: " + "  ".join(f"{k} {v['target']:.3f}" for k, v in r["targets"].items())
                  + f"  ({r['seconds']} s)", flush=True)
    results.sort(key=lambda r: r["seed"])
    v = {f"{t}_on_{s}": [r["targets"][f"{t}_on_{s}"]["target"] for r in results] for t, s in COMBOS}
    agg = {k: {"per_seed": x, "mean": float(np.mean(x)), "t90": [float(y) for y in t_ci90(x)]}
           for k, x in v.items()}
    contrasts = {
        "trunk_effect_on_a_streams (a_on_a - b_on_a)": paired_contrast(v["a_on_a"], v["b_on_a"], 0.0),
        "trunk_effect_on_b_streams (a_on_b - b_on_b)": paired_contrast(v["a_on_b"], v["b_on_b"], 0.0),
        "stream_effect_in_a_trunk (a_on_a - a_on_b)": paired_contrast(v["a_on_a"], v["a_on_b"], 0.0),
        "stream_effect_in_b_trunk (b_on_a - b_on_b)": paired_contrast(v["b_on_a"], v["b_on_b"], 0.0),
    }
    out = {"generated_by": "scripts/run_cross_replay.py", "status": "exploratory (post hoc)",
           "config": base, "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
           "cells": results, "aggregate": agg, "contrasts": contrasts}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k, x in agg.items():
        print(f"{k:10s} {x['mean']:.3f} [{x['t90'][0]:.3f}, {x['t90'][1]:.3f}]")
    for k, c in contrasts.items():
        print(f"{k:48s} {c['mean']:+.3f} [{c['t90'][0]:+.3f}, {c['t90'][1]:+.3f}]")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
