"""Agent-based L0 and balanced-sampling audit on saved agents (revision step 5).

Readout-only. For a run directory written with --save-agents:
  * every drift-0 survival agent is rescored on independent pairs of evaluation-world
    samples (itasorl.l0_audit.AUDIT_BASES), alongside the standard pair, and pool
    membership is decoded from h_1, the state after the reset observation only;
  * every arm at the strongest drift is scored with the balanced (paired-seed) pooled
    readout, whose twins share an initial state and a CV group.
The standard pooled readout of the run stays the frozen primary; these are diagnostics.

G is retrained on one CPU thread, as the run workers fit it. An integrity gate then checks
that the standard pair reproduces the drift-0 survival targets the run recorded in its
cells/ (beside the agents directory, or --integrity-cells) within --integrity-tol, and
reports whether they match bit for bit. A failed gate voids the readout (exit code 1).
Without recorded cells the gate is reported as not checked.

Usage:
    python scripts/run_l0_audit.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --out artifacts/l0_audit/corrected_l3_h8_wm.json [--pairs 8] [--quick]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from itasorl.results_io import git_head  # noqa: E402

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

import itasorl.experiment_b2 as b2
from itasorl import folds
from itasorl.experiment_b2 import load_agent_bundle
from itasorl.l0_audit import AUDIT_BASES, STANDARD_BASES, l0_world_samples, paired_pooled_readout
from itasorl.stats import equivalence_test, t_ci90
from itasorl.world import WorldParams
from promote_bv3_gates import repo_rel

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def recorded_l0_targets(cells_dir: str) -> dict:
    """Drift-0 survival pooled targets the run recorded, keyed by seed ({} if none)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(cells_dir, "cell_d0.00_s*.json"))):
        with open(path, encoding="utf-8") as fh:
            cell = json.load(fh)["cell"]
        out[int(cell["seed"])] = float(cell["agents"]["survival"]["pool"]["target"])
    return out


def integrity_gate(l0_rows: list, recorded: dict, tol: float) -> dict:
    """Standard-pair L0 targets (rows[0]) against the run's recorded drift-0 targets."""
    checks = [{"seed": c["seed"], "recorded": recorded[c["seed"]], "rescored": c["rows"][0]["target"],
               "abs_dev": abs(c["rows"][0]["target"] - recorded[c["seed"]])}
              for c in l0_rows if c["seed"] in recorded]
    if not recorded:
        return {"pass": None, "note": "no recorded cells found; integrity not checked",
                "tolerance": tol, "n_checked": 0, "per_seed": []}
    worst = max((c["abs_dev"] for c in checks), default=float("nan"))
    return {"pass": bool(checks) and worst <= tol, "tolerance": tol, "n_checked": len(checks),
            "worst_abs_dev": worst,
            "bit_identical": bool(checks) and all(c["abs_dev"] == 0.0 for c in checks),
            "per_seed": checks}


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
    ap.add_argument("--integrity-cells", default=None,
                    help="cells/ of the run that saved the agents (default: beside --agents-dir)")
    ap.add_argument("--integrity-tol", type=float, default=0.01)
    ap.add_argument("--quick", action="store_true", help="2 pairs, 20 episodes, 2 seeds")
    a = ap.parse_args()
    if a.quick:
        a.pairs, a.n_eps = 2, 20
        print("--quick: 20 episodes per pool. The integrity gate compares against targets recorded "
              "at 110 episodes, so it is expected to fail; quick mode never gives a reported number.",
              flush=True)
    torch.set_num_threads(1)     # the run workers fit G single-threaded; match their bits
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
    out = {"agents_dir": repo_rel(a.agents_dir), "generated_by": "scripts/run_l0_audit.py", "git_commit": git_head(),
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
    cells_dir = a.integrity_cells or os.path.join(os.path.dirname(os.path.normpath(a.agents_dir)), "cells")
    out["integrity"] = {"reference": repo_rel(cells_dir),
                        **integrity_gate(out["l0"], recorded_l0_targets(cells_dir), a.integrity_tol)}
    out["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out["l0_summary"], indent=1, default=float))
    print(json.dumps({k: (v["mean"], v["t90"]) for k, v in summ.items()}, default=float))
    ig = out["integrity"]
    if ig["pass"] is None:
        print(f"integrity gate: NOT CHECKED ({ig['note']})")
    else:
        print(f"integrity gate: {'PASS' if ig['pass'] else 'FAIL (readout void)'} "
              f"(worst |dev| {ig['worst_abs_dev']:.4f} against {a.integrity_tol}, "
              f"bit-identical {ig['bit_identical']})")
    print(f"wrote {a.out}")
    return 1 if ig["pass"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
