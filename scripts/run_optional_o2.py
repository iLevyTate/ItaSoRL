"""O2. Long-history ridge penalty chosen inside each outer training fold.

Readout only, on the saved C1 survival agents at drift 0.45. The two bases are
the ones whose fixed penalty of 1e-3 left a negative held-out fit: observation
history through t-7 with two causal traces and the actions, and that basis
plus the behavior trace. Nothing here replaces the recorded control.

    python scripts/run_optional_o2.py --device cuda
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from itasorl.control_diagnostics import history_basis
from itasorl.experiment_b2 import collect_pool, load_agent_bundle, setup_l3_surrogate
from itasorl.optional_readouts import long_history_ridge, provenance
from itasorl.stats import t_ci90
from itasorl.world import WorldParams
import itasorl.experiment_b2 as b2

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
LOG = "fullruns/optional/o2_progress.txt"


def _log(message: str) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print(message, flush=True)


def _mean_ci(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    lo, hi = t_ci90(a) if a.size > 1 else (float("nan"), float("nan"))
    return {"per_seed": [float(v) for v in a], "mean": float(a.mean()) if a.size else float("nan"),
            "t90": [float(lo), float(hi)], "n": int(a.size)}


def run_seed(seed: int, device: str) -> dict:
    _log(f"s={seed} collecting pools")
    agent, norm = load_agent_bundle(
        f"fullruns/corrected_l3_h8_wm/agents/agent_d0.45_s{seed}_survival.pt", device)
    pools = {}
    for label, drift, base in (("a", 0.0, 800_000), ("s", 0.45, 850_000)):
        H, _spd, _en, _fd, _dg, _rw, Bt, Ot, At = collect_pool(
            agent, norm, P, drift, 110, 24, device, base, 5,
            return_anchors=True, return_obs=True, return_actions=True)
        pools[label] = (H, Bt, Ot, At)
    H = np.concatenate([pools["a"][0], pools["s"][0]])
    Bt = np.concatenate([pools["a"][1], pools["s"][1]])
    Ot = np.concatenate([pools["a"][2], pools["s"][2]])
    At = np.concatenate([pools["a"][3], pools["s"][3]])
    y = np.r_[np.zeros(len(pools["a"][0])), np.ones(len(pools["s"][0]))].astype(int)
    bases = {
        "resid_obs_hist": history_basis(Ot, lags=7, ema_taus=(4, 16), At=At),
        "resid_obs_hist_beh": history_basis(Ot, lags=7, ema_taus=(4, 16), At=At, Bt=Bt),
    }
    out = {"seed": seed, "n": int(len(y))}
    for name, phi in bases.items():
        t0 = time.time()
        _log(f"s={seed} fitting {name} dim={phi.shape[1]}")
        out[name] = long_history_ridge(H, phi, y)
        out[name]["seconds"] = time.time() - t0
        _log(f"s={seed} {name} auroc={out[name]['auroc']:.3f} "
             f"r2={out[name]['nuisance_r2_heldout']:.3f} "
             f"fixed_r2={out[name]['fixed_alpha']['nuisance_r2_heldout']:.3f} "
             f"alphas={out[name]['alphas']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="docs/optional/o2_long_history_ridge.json")
    ap.add_argument("--ckpt", default="fullruns/optional/o2")
    args = ap.parse_args()
    os.makedirs(args.ckpt, exist_ok=True)
    b2.DRIFT_MODE = "l3"
    b2.OBJECTIVE = "survival"
    b2.MORTAL = True
    _log(f"training surrogate on {args.device}")
    setup_l3_surrogate(hidden=8, device=args.device, seed=0, params=P)
    _log("surrogate ready")
    cells = []
    for seed in args.seeds:
        path = os.path.join(args.ckpt, f"s{seed}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                cells.append(json.load(fh))
            print(f"s={seed} already saved", flush=True)
            continue
        cell = run_seed(seed, args.device)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cell, fh, indent=1)
        cells.append(cell)
    summary = {}
    for name in ("resid_obs_hist", "resid_obs_hist_beh"):
        summary[name] = {
            "auroc": _mean_ci([c[name]["auroc"] for c in cells]),
            "nuisance_r2_heldout": _mean_ci([c[name]["nuisance_r2_heldout"] for c in cells]),
            "fixed_auroc": _mean_ci([c[name]["fixed_alpha"]["auroc"] for c in cells]),
            "fixed_r2": _mean_ci([c[name]["fixed_alpha"]["nuisance_r2_heldout"] for c in cells]),
        }
    inputs = {"agents": "fullruns/corrected_l3_h8_wm/agents/agent_d0.45_s{seed}_survival.pt",
              "seeds": list(args.seeds), "n_eps": 110, "steps": 24,
              "pools": {"authentic": {"drift": 0.0, "seed_base": 800000},
                        "surrogate": {"drift": 0.45, "seed_base": 850000}},
              "bases": {"resid_obs_hist": "observation history through t-7, EMA traces (4, 16), actions",
                        "resid_obs_hist_beh": "the same plus the behavior trace"},
              "checkpoints": args.ckpt}
    command = "python " + " ".join(["scripts/run_optional_o2.py", *sys.argv[1:]])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"provenance": provenance("scripts/run_optional_o2.py", command, inputs,
                                            device=args.device),
                   "summary": summary, "cells": cells}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
