"""Readout-only matched speed and leakage gates for the 2026-10-12 runs.

The October 12 rule asks for these gates on the retained pairs at base 910000.
The cells recorded the pooled checks at 800000 and 850000 instead. This script
does not retrain and does not rewrite those cells. It writes matched_gates.json
beside the run, one block per drift-0.45 seed and arm, with the readout's own target
beside the balanced cell target of the same pairs (the reconstruction check) and the
agent file hash. A file written under another configuration is refused, never resumed.

Usage:
    python scripts/audit_balanced_matched_gates.py --run fullruns/balanced_confirmation_wm --objective survival --mortal on
    python scripts/audit_balanced_matched_gates.py --run fullruns/balanced_confirmation_touch --objective touch --mortal off
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import os

import itasorl.experiment_b2 as b2
from itasorl.experiment_b2 import default_device, load_agent_bundle, setup_l3_surrogate
from itasorl.l0_audit import paired_pooled_readout
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
READOUT = {"drift": 0.45, "n_eps": 110, "steps": 24, "ray_steps": 5}
SURROGATE = "registered L3 surrogate, hidden 8, seed 0"
RECON_TOL = 0.005   # same value as promote_balanced_confirmation.RECON_TOL


def invocation_config(seed_base: int, objective: str, mortal: bool, device: str) -> dict:
    """The document-level configuration a resumed file must match."""
    return {"population": "retained pairs", "seed_base": seed_base, "objective": objective,
            "mortal": mortal, "config": {**READOUT, "device": device}}


def check_resume(stored: dict, want: dict) -> None:
    """Refuse to add rows to a file written under another configuration."""
    for field, value in want.items():
        if field not in stored:
            raise SystemExit(f"matched_gates.json has no {field} block; move it aside to recollect")
        if field == "config":
            bad = sorted(k for k in value if stored["config"].get(k) != value[k])
            if bad:
                raise SystemExit(f"matched_gates.json was written with a different config: {bad}")
        elif stored[field] != value:
            raise SystemExit(f"matched_gates.json was written with {field}={stored[field]!r}, "
                             f"this invocation has {value!r}")


def _sha16(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _balanced_target(run: str, seed: int, arm: str) -> float:
    with open(os.path.join(run, "cells", f"cell_d0.45_s{seed}.json"), encoding="utf-8") as fh:
        return float(json.load(fh)["cell"]["agents"][arm]["balanced"]["target"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--objective", choices=("survival", "touch"), required=True)
    ap.add_argument("--mortal", choices=("on", "off"), required=True)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--seed-base", type=int, default=910000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    ap.add_argument("--arms", nargs="+", default=["survival", "predictor", "untrained"])
    args = ap.parse_args()
    device = default_device() if args.device == "auto" else args.device
    b2.DRIFT_MODE = "l3"
    b2.OBJECTIVE = args.objective
    b2.MORTAL = args.mortal == "on"
    setup_l3_surrogate(hidden=8, device=device, seed=0, params=P)
    header = invocation_config(args.seed_base, args.objective, args.mortal == "on", device)
    rows = []
    out_path = os.path.join(args.run, "matched_gates.json")
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            stored = json.load(fh)
        check_resume(stored, header)
        rows = stored["rows"]
    done = {(row["seed"], row["arm"]) for row in rows}
    agents = os.path.join(args.run, "agents")
    for seed in args.seeds:
        for arm in args.arms:
            if (seed, arm) in done:
                continue
            path = os.path.join(agents, f"agent_d0.45_s{seed}_{arm}.pt")
            agent, norm = load_agent_bundle(path, device)
            readout = paired_pooled_readout(
                agent, norm, P, READOUT["drift"], n_eps=READOUT["n_eps"], steps=READOUT["steps"],
                ray_steps=READOUT["ray_steps"], device=device, seed_base=args.seed_base)
            original = _balanced_target(args.run, seed, arm)
            rows.append({
                "seed": seed, "arm": arm, "drift": READOUT["drift"],
                "seed_base": args.seed_base,
                "n_pairs": readout["n_pairs"],
                "dropped_pairs": readout["dropped_pairs"],
                "target": readout["target"],
                "reconstruction": {
                    "balanced_target": original,
                    "abs_diff": abs(float(readout["target"]) - original),
                    "tolerance": RECON_TOL,
                    "agent_sha16": _sha16(path),   # the bundle carries the normalizer
                    "surrogate": SURROGATE,
                },
                "matched_gates": readout.get("matched_gates"),
            })
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({**header, "rows": rows}, fh, indent=1)
            speed = (readout.get("matched_gates") or {}).get("speed", float("nan"))
            print(f"s={seed} {arm} pairs={readout['n_pairs']} speed={speed:.3f}", flush=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
