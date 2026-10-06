"""Agent-based L0 and balanced-sampling audit on saved agents (revision step 5).

Readout-only. For a run directory written with --save-agents:
  * every drift-0 survival agent is rescored on independent pairs of evaluation-world
    samples (itasorl.l0_audit.AUDIT_BASES), alongside the standard pair, and pool
    membership is decoded from h_1, the state after the reset observation only;
  * every arm at the strongest drift is scored with the balanced (paired-seed) pooled
    readout, whose twins share an initial state and a CV group.
The standard pooled readout of the run stays the frozen primary; these are diagnostics.

Usage:
    python scripts/run_l0_audit.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --out artifacts/l0_audit/corrected_l3_h8_wm.json [--pairs 8] [--quick]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re
import time

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl import folds
from itasorl.experiment_b2 import load_agent_bundle
from itasorl.l0_audit import AUDIT_BASES, STANDARD_BASES, l0_world_samples, paired_pooled_readout
from itasorl.stats import equivalence_test, t_ci90
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=8, help="independent world-sample pairs")
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 pairs, 20 episodes, 2 seeds")
    a = ap.parse_args()
    if a.quick:
        a.pairs, a.n_eps = 2, 20
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=a.l3_hidden, device="cpu", seed=a.l3_seed, params=P)
    cells = []
    for name in sorted(os.listdir(a.agents_dir)):
        m = AGENT_RE.match(name)
        if m:
            cells.append((float(m.group(1)), int(m.group(2)), m.group(3), name))
    if a.quick:
        cells = [c for c in cells if c[1] < 2]
    dmax = max(c[0] for c in cells)
    bases = (STANDARD_BASES,) + AUDIT_BASES[: a.pairs]
    out = {"agents_dir": a.agents_dir, "generated_by": "scripts/run_l0_audit.py",
           "bases": [list(b) for b in bases], "n_eps": a.n_eps, "steps": a.steps,
           "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
           "l0": [], "paired": []}
    t0 = time.time()
    for drift, seed, arm, name in cells:
        agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), "cpu")
        if drift == 0.0 and arm == "survival":
            rows = l0_world_samples(agent, norm, P, bases=bases, n_eps=a.n_eps, steps=a.steps,
                                    ray_steps=a.ray_steps, seed=seed)
            out["l0"].append({"seed": seed, "rows": rows})
            print(f"L0 seed {seed}: " + " ".join(f"{r['target']:.3f}" for r in rows), flush=True)
        if drift == dmax:
            r = paired_pooled_readout(agent, norm, P, drift, n_eps=a.n_eps, steps=a.steps,
                                      ray_steps=a.ray_steps, seed=seed)
            out["paired"].append({"seed": seed, "arm": arm, **r})
            print(f"paired d{drift} s{seed} {arm}: {r['target']:.3f} ({r['n_pairs']} pairs)", flush=True)
    # Summaries: per world-sample pair, the across-seed L0 mean and TOST; across pairs, the spread.
    by_pair = []
    for k, b in enumerate(bases):
        v = [c["rows"][k]["target"] for c in out["l0"]]
        h1 = [c["rows"][k]["first_state_target"] for c in out["l0"]]
        eq = equivalence_test(v) if len(v) > 1 else None
        by_pair.append({"bases": list(b), "standard": k == 0, "mean": float(np.mean(v)),
                        "tost_p": eq.p_value if eq else float("nan"),
                        "equivalent": bool(eq.equivalent) if eq else False,
                        "first_state_mean": float(np.mean(h1))})
    pair_means = [p["mean"] for p in by_pair[1:]]
    out["l0_summary"] = {
        "by_pair": by_pair,
        "independent_pair_means": pair_means,
        "independent_sd_of_pair_means": float(np.std(pair_means, ddof=1)) if len(pair_means) > 1 else float("nan"),
        "tost_over_independent_pairs": (
            {"mean": equivalence_test(pair_means).mean, "p": equivalence_test(pair_means).p_value,
             "equivalent": bool(equivalence_test(pair_means).equivalent)} if len(pair_means) > 1 else None),
    }
    summ = {}
    for arm in ("untrained", "predictor", "survival"):
        v = [p["target"] for p in out["paired"] if p["arm"] == arm and np.isfinite(p["target"])]
        if v:
            summ[arm] = {"per_seed": v, "mean": float(np.mean(v)),
                         "t90": [float(x) for x in t_ci90(v)] if len(v) > 1 else None}
    out["paired_summary"] = summ
    out["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out["l0_summary"], indent=1, default=float))
    print(json.dumps({k: (v["mean"], v["t90"]) for k, v in summ.items()}, default=float))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
