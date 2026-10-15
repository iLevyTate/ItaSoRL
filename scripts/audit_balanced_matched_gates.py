"""Readout-only matched speed and leakage gates for the 2026-10-12 runs.

The October 12 rule asks for these gates on the retained pairs at base 910000.
The cells recorded the pooled checks at 800000 and 850000 instead. This script
does not retrain and does not rewrite those cells. It writes matched_gates.json
beside the run, one block per drift-0.45 seed and arm.

Usage:
    python scripts/audit_balanced_matched_gates.py --run fullruns/balanced_confirmation_wm --objective survival --mortal on
    python scripts/audit_balanced_matched_gates.py --run fullruns/balanced_confirmation_touch --objective touch --mortal off
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os

import itasorl.experiment_b2 as b2
from itasorl.experiment_b2 import default_device, load_agent_bundle, setup_l3_surrogate
from itasorl.l0_audit import paired_pooled_readout
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


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
    rows = []
    out_path = os.path.join(args.run, "matched_gates.json")
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
    done = {(row["seed"], row["arm"]) for row in rows}
    agents = os.path.join(args.run, "agents")
    for seed in args.seeds:
        for arm in args.arms:
            if (seed, arm) in done:
                continue
            path = os.path.join(agents, f"agent_d0.45_s{seed}_{arm}.pt")
            agent, norm = load_agent_bundle(path, device)
            readout = paired_pooled_readout(
                agent, norm, P, 0.45, n_eps=110, steps=24, ray_steps=5,
                device=device, seed_base=args.seed_base)
            rows.append({
                "seed": seed, "arm": arm, "drift": 0.45,
                "seed_base": args.seed_base,
                "n_pairs": readout["n_pairs"],
                "dropped_pairs": readout["dropped_pairs"],
                "matched_gates": readout.get("matched_gates"),
            })
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"population": "retained pairs", "seed_base": args.seed_base,
                           "objective": args.objective, "mortal": args.mortal == "on",
                           "rows": rows}, fh, indent=1)
            speed = (readout.get("matched_gates") or {}).get("speed", float("nan"))
            print(f"s={seed} {arm} pairs={readout['n_pairs']} speed={speed:.3f}", flush=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
