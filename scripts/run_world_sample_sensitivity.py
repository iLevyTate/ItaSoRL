"""Drift-0.45 world-sample sensitivity on saved agents (frozen 2026-10-08).

Readout-only, and a diagnostic rather than a gate. The pooled readout draws its authentic
pool from one evaluation-world seed base and its surrogate pool from another, so its
reading contains a world-sample component as well as a dynamics component. That component
has been measured at drift 0 only. This carries the same independent pairs to a chosen
drift, defaulting to 0.45, and adjudicates them under the rule frozen before the run in
docs/specs/2026-10-08-drift-045-world-sample-sensitivity-design.md.

Two properties of that rule are deliberate and are enforced here rather than at analysis
time. The per-draw means are signed and never folded about chance. And the registered draw
is rescored alongside the independent ones as an integrity gate: it must reproduce the
run's recorded per-seed targets within --integrity-tol, or the run is void.

Usage:
    python scripts/run_world_sample_sensitivity.py \\
        --agents-dir fullruns/l3_h8_heldout/agents \\
        --out fullruns/d045_world_samples/d045_world_samples_l3_h8_heldout.json \\
        [--drift 0.45] [--pairs 8] [--arms survival] [--quick]

The output path sits inside a run directory on purpose: the Colab notebook mirrors one
tree to Drive, so writing there is what survives a dropped session. Copy the finished file
into artifacts/l0_audit/ locally, where the manifest owns it.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from itasorl.results_io import git_head  # noqa: E402

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import itasorl.experiment_b2 as b2  # noqa: E402
from itasorl import folds  # noqa: E402
from itasorl.experiment_b2 import load_agent_bundle  # noqa: E402
from itasorl.l0_audit import (  # noqa: E402
    AUDIT_BASES, STANDARD_BASES, world_sample_scan, world_sample_summary,
)
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reference(path: str, drift: float, arm: str) -> dict:
    """Recorded per-seed targets for the integrity gate, keyed by seed.

    artifacts/fold_rescore/*.json stores drift as a STRING, the arm under "agent", and the
    target nested under the fold scheme's name.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    scheme = folds.current_scheme()
    out = {}
    for c in doc.get("cells", []):
        if float(c["drift"]) == drift and c.get("agent") == arm and scheme in c:
            out[int(c["seed"])] = float(c[scheme]["target"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drift", type=float, default=0.45)
    ap.add_argument("--pairs", type=int, default=8, help="independent world-sample pairs")
    ap.add_argument("--arms", default="survival", help="comma separated")
    ap.add_argument("--bar", type=float, default=0.65)
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--integrity-ref", default="artifacts/fold_rescore/l3_h8_heldout.json")
    ap.add_argument("--integrity-tol", type=float, default=0.01)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="one JSON per (arm, seed); resumes if present")
    ap.add_argument("--quick", action="store_true", help="2 pairs, 20 episodes, 2 seeds")
    a = ap.parse_args()
    if a.quick:
        a.pairs, a.n_eps = 2, 20

    torch.set_num_threads(1)
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=a.l3_hidden, device="cpu", seed=a.l3_seed, params=P)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]

    cells = []
    for name in sorted(os.listdir(a.agents_dir)):
        m = AGENT_RE.match(name)
        if m and float(m.group(1)) == a.drift and m.group(3) in arms:
            cells.append((int(m.group(2)), m.group(3), name))
    if a.quick:
        cells = [c for c in cells if c[0] < 2]
    if not cells:
        raise SystemExit(f"no agents at drift {a.drift} for arms {arms} in {a.agents_dir}")

    bases = (STANDARD_BASES,) + AUDIT_BASES[: a.pairs]
    ckpt = a.checkpoint_dir
    if ckpt:
        os.makedirs(ckpt, exist_ok=True)

    out = {
        "generated_by": "scripts/run_world_sample_sensitivity.py",
        "git_commit": git_head(),
        "spec": "docs/specs/2026-10-08-drift-045-world-sample-sensitivity-design.md",
        "agents_dir": a.agents_dir, "drift": a.drift, "arms": arms, "bar": a.bar,
        "bases": [list(b) for b in bases], "registered_bases": list(STANDARD_BASES),
        "n_eps": a.n_eps, "steps": a.steps,
        "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
        "cells": [],
    }

    t0 = time.time()
    for seed, arm, name in cells:
        path = os.path.join(a.agents_dir, name)
        cp = os.path.join(ckpt, f"{arm}_s{seed}.json") if ckpt else None
        if cp and os.path.exists(cp):
            with open(cp, encoding="utf-8") as fh:
                out["cells"].append(json.load(fh))
            print(f"resumed {arm} s{seed} from checkpoint", flush=True)
            continue
        agent, norm = load_agent_bundle(path, "cpu")
        rows = world_sample_scan(agent, norm, P, a.drift, bases=bases, n_eps=a.n_eps,
                                 steps=a.steps, ray_steps=a.ray_steps, seed=seed)
        cell = {"seed": seed, "arm": arm, "agent_sha256": _sha256(path), "rows": rows}
        out["cells"].append(cell)
        if cp:
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(cell, fh, indent=1)
        print(f"{arm} s{seed}: " + " ".join(f"{r['target']:.3f}" for r in rows)
              + f"   ({time.time() - t0:.0f}s elapsed)", flush=True)

    # ---- integrity gate on the registered draw, before anything is adjudicated ----
    ref = _reference(a.integrity_ref, a.drift, "survival")
    checks = []
    for c in out["cells"]:
        if c["arm"] == "survival" and c["seed"] in ref:
            got, want = c["rows"][0]["target"], ref[c["seed"]]
            checks.append({"seed": c["seed"], "recorded": want, "rescored": got,
                           "abs_dev": abs(got - want)})
    worst = max((c["abs_dev"] for c in checks), default=float("nan"))
    passed = bool(checks) and worst <= a.integrity_tol
    out["integrity"] = {"reference": a.integrity_ref, "tolerance": a.integrity_tol,
                        "n_checked": len(checks), "worst_abs_dev": worst,
                        "pass": passed, "per_seed": checks}

    # ---- adjudication under the frozen rule ----
    out["summary"] = {}
    for arm in arms:
        per_arm = [c for c in out["cells"] if c["arm"] == arm]
        if not per_arm:
            continue
        draw_means = [float(np.mean([c["rows"][k]["target"] for c in per_arm]))
                      for k in range(len(bases))]
        s = world_sample_summary(draw_means[1:], draw_means[0], bar=a.bar)
        s["per_draw_bases"] = [list(b) for b in bases[1:]]
        out["summary"][arm] = s

    out["wall_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"\nintegrity gate: {'PASS' if passed else 'FAIL'} "
          f"(worst |dev| {worst:.4f} against {a.integrity_tol})")
    for arm, s in out["summary"].items():
        print(f"{arm}: registered {s['registered']:.3f} (rank {s['registered_rank']} of "
              f"{s['n_draws'] + 1}); independent draws {s['min']:.3f} to {s['max']:.3f}, "
              f"sd {s['between_draw_sd']:.4f}, {s['n_at_or_above_bar']}/{s['n_draws']} at "
              f"or above {a.bar}")
        print(f"  verdict under the frozen rule: {s['verdict']}"
              + ("" if passed else "  (VOID: integrity gate failed)"))
    print(f"saved {a.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
