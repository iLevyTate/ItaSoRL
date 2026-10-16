"""O1. Fold-local speed thresholds on the dumps that produced the recorded gates.

The recorded call is each cell's pool.speed, a global-median probe. This does
not rewrite those cells. Matched pairs are a separate pass: both members of a
pair share a group.

    python scripts/run_optional_o1.py --pooled
    python scripts/run_optional_o1.py --matched --device cuda
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys

import numpy as np

from itasorl.experiment_b import episode_features
from itasorl.optional_readouts import SPEED_BAR, provenance, speed_threshold_probes
from itasorl.stats import t_ci90

RUNS = {
    "B1": "fullruns/balanced_confirmation_wm",
    "B2": "fullruns/balanced_confirmation_touch",
}
ARMS = ("survival", "predictor", "untrained")
DRIFTS = (0.0, 0.45)


def _call(value: float) -> bool | None:
    if not np.isfinite(value):
        return None
    return bool(value >= SPEED_BAR)


def _row(run: str, drift: float, seed: int, arm: str, recorded: float, probes: dict,
         population: str) -> dict:
    return {
        "run": run, "drift": drift, "seed": seed, "arm": arm, "population": population,
        "recorded_global": recorded,
        "recomputed_global": probes["global"],
        "fold_local": probes["fold_local"],
        "recorded_call": _call(recorded),
        "recomputed_call": _call(probes["global"]),
        "fold_local_call": _call(probes["fold_local"]),
        "call_changed": _call(recorded) != _call(probes["fold_local"]),
        "n_folds_unscored": probes["n_folds_unscored"],
    }


def pooled() -> list[dict]:
    rows = []
    for run, root in RUNS.items():
        for drift in DRIFTS:
            for seed in range(10, 20):
                cell_path = os.path.join(root, "cells", f"cell_d{drift:.2f}_s{seed}.json")
                with open(cell_path, encoding="utf-8") as fh:
                    cell = json.load(fh)["cell"]
                for arm in ARMS:
                    dump = os.path.join(root, "states", f"states_d{drift:.2f}_s{seed}_{arm}.npz")
                    with np.load(dump) as z:
                        H = np.concatenate([z["Ha"], z["Hs"]])
                        speeds = np.concatenate([z["spa"], z["sps"]])
                    groups = np.arange(len(speeds))
                    probes = speed_threshold_probes(episode_features(H), speeds, groups)
                    recorded = float(cell["agents"][arm]["pool"]["speed"])
                    rows.append(_row(run, drift, seed, arm, recorded, probes, "pooled"))
                    print(f"{run} d={drift:.2f} s={seed} {arm} "
                          f"recorded={recorded:.3f} global={probes['global']:.3f} "
                          f"fold={probes['fold_local']:.3f}", flush=True)
    return rows


def matched(device: str) -> list[dict]:
    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b2 import load_agent_bundle, setup_l3_surrogate
    from itasorl.world import WorldParams

    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    setup_l3_surrogate(hidden=8, device=device, seed=0, params=P)
    rows = []
    for run, root in RUNS.items():
        b2.OBJECTIVE = "touch" if run == "B2" else "survival"
        b2.MORTAL = run != "B2"
        for seed in range(10, 20):
            for arm in ARMS:
                path = os.path.join(root, "agents", f"agent_d0.45_s{seed}_{arm}.pt")
                agent, norm = load_agent_bundle(path, device)
                Ha, spa, *_rest_a, ia = b2.collect_pool(
                    agent, norm, P, 0.0, 110, 24, device, 910000, 5,
                    return_anchors=True, return_index=True)
                Hs, sps, *_rest_s, is_ = b2.collect_pool(
                    agent, norm, P, 0.45, 110, 24, device, 910000, 5,
                    return_anchors=True, return_index=True)
                common = np.intersect1d(ia, is_)
                sel_a = np.searchsorted(ia, common)
                sel_s = np.searchsorted(is_, common)
                H = np.concatenate([Ha[sel_a], Hs[sel_s]])
                speeds = np.concatenate([spa[sel_a], sps[sel_s]])
                groups = np.r_[common, common]
                probes = speed_threshold_probes(episode_features(H), speeds, groups)
                gate_path = os.path.join(root, "matched_gates.json")
                recorded = float("nan")
                if os.path.exists(gate_path):
                    with open(gate_path, encoding="utf-8") as fh:
                        doc = json.load(fh)
                    for item in doc["rows"]:
                        if int(item["seed"]) == seed and item["arm"] == arm:
                            recorded = float(item["matched_gates"]["speed"])
                rows.append(_row(run, 0.45, seed, arm, recorded, probes, "matched"))
                print(f"{run} matched s={seed} {arm} recorded={recorded:.3f} "
                      f"global={probes['global']:.3f} fold={probes['fold_local']:.3f}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Rows share seeds, arms and baseline readings, so they are not independent replications.
    The uncertainty unit is the seed: differences are averaged within each seed first and the
    t interval is taken over the seed-block means (2026-10-14 review, C05)."""
    changed = [r for r in rows if r["call_changed"]]
    ok = [r for r in rows if np.isfinite(r["fold_local"]) and np.isfinite(r["recomputed_global"])]
    by_seed: dict = {}
    for r in ok:
        by_seed.setdefault(r["seed"], []).append(r["fold_local"] - r["recomputed_global"])
    blocks = {str(k): float(np.mean(v)) for k, v in sorted(by_seed.items())}
    bm = list(blocks.values())
    lo, hi = t_ci90(bm) if len(bm) > 1 else (float("nan"), float("nan"))
    gaps = [abs(r["recomputed_global"] - r["recorded_global"])
            for r in rows if np.isfinite(r["recorded_global"]) and np.isfinite(r["recomputed_global"])]
    return {
        "n": len(rows),
        "n_call_changed": len(changed),
        "changed": [{"run": r["run"], "population": r["population"], "drift": r["drift"],
                     "seed": r["seed"], "arm": r["arm"], "recorded": r["recorded_global"],
                     "fold_local": r["fold_local"]} for r in changed],
        "unit": "seed",
        "n_blocks": len(bm),
        "estimand": ("mean over seeds of the within-seed mean of fold-local minus recomputed global "
                     "speed accuracy; the 90% t interval is taken over the seed-block means"),
        "block_means": blocks,
        "fold_minus_global_mean": float(np.mean(bm)) if bm else float("nan"),
        "fold_minus_global_t90": [float(lo), float(hi)],
        "fold_minus_global_row_mean": float(np.mean([r["fold_local"] - r["recomputed_global"] for r in ok]))
        if ok else float("nan"),
        "max_abs_recomputed_vs_recorded": float(max(gaps)) if gaps else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--matched", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="docs/optional/o1_fold_local_speed.json")
    args = ap.parse_args()
    if not args.pooled and not args.matched:
        raise SystemExit("pass --pooled, --matched, or both")
    rows = []
    if args.pooled:
        rows.extend(pooled())
    if args.matched:
        rows.extend(matched(args.device))
    inputs = {"runs": dict(RUNS), "seeds": list(range(10, 20)), "arms": list(ARMS)}
    if args.pooled:
        inputs["pooled"] = {"drifts": list(DRIFTS), "cells": "<run>/cells/cell_d{drift}_s{seed}.json",
                            "state_dumps": "<run>/states/states_d{drift}_s{seed}_{arm}.npz",
                            "recorded_call": "cell.agents.<arm>.pool.speed"}
    if args.matched:
        inputs["matched"] = {"drift": 0.45, "seed_base": 910000, "n_eps": 110, "steps": 24,
                             "agents": "<run>/agents/agent_d0.45_s{seed}_{arm}.pt",
                             "recorded_call": "<run>/matched_gates.json rows[].matched_gates.speed"}
    command = "python " + " ".join(["scripts/run_optional_o1.py", *sys.argv[1:]])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doc = {"provenance": provenance("scripts/run_optional_o1.py", command, inputs,
                                    device=args.device if args.matched else "cpu"),
           "summary": summarize(rows), "rows": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
