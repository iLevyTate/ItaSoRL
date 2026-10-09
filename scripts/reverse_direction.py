"""Reverse-direction readout on saved drift-0 agents (frozen 2026-10-09).

Readout-only, and a diagnostic rather than a gate. Every positive on record is in one
direction: the agent is raised inside the surrogate and the probe separates the law it was
raised under from the authentic law. This scores the other direction, an agent raised in the
authentic world probed against a learned surrogate it never lived in, under the rule frozen
before the run in docs/specs/2026-10-09-reverse-direction-readout-design.md.

The direction is the whole point, so it is enforced here rather than at analysis time: the
agents selected are the drift-0 ones and the readout drift is 0.45, the opposite pairing from
every other cell in the project. The registered-estimand standard readout is what the verdict
is adjudicated on; the balanced same-world-seed readout is reported beside it and never
substituted for it.

Usage:
    python scripts/reverse_direction.py \\
        --agents-dir fullruns/l3_h8_heldout/agents \\
        --out fullruns/reverse_direction/reverse_direction_l3_h8_heldout.json \\
        [--drift 0.45] [--quick]

The output path sits inside a run directory on purpose: the Colab notebook mirrors one tree to
Drive, so writing there is what survives a dropped session. Copy the finished file into
artifacts/l0_audit/ locally, where the manifest owns it.
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
from itasorl.experiment_b2 import load_agent_bundle, pooled_readout  # noqa: E402
from itasorl.l0_audit import paired_pooled_readout  # noqa: E402
from itasorl.stats import t_ci90  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")
ARMS = ("survival", "predictor", "untrained")


def adjudicate(*, survival: float, untrained: float, bar: float = 0.65,
               margin: float = 0.05) -> dict:
    """The rule frozen in the spec. The bar is checked before the margin."""
    m = float(survival) - float(untrained)
    if float(survival) < bar:
        verdict = "NOT CARRIED"
    elif m >= margin:
        verdict = "CARRIED"
    else:
        verdict = "AMBIGUOUS"
    return {"verdict": verdict, "survival": float(survival), "untrained": float(untrained),
            "margin_over_untrained": m, "bar": bar, "margin": margin}


def select_agents(agents_dir: str, *, arms=ARMS) -> list[tuple[int, str, str]]:
    """The drift-0 agents only. A drift-0.45 agent would be the forward direction."""
    out = []
    for name in sorted(os.listdir(agents_dir)):
        m = AGENT_RE.match(name)
        if m and float(m.group(1)) == 0.0 and m.group(3) in arms:
            out.append((int(m.group(2)), m.group(3), name))
    return out


def score_one(agent, norm, params, *, drift: float, n_eps: int, steps: int, ray_steps: int,
              seed: int, balanced: bool = True, device: str = "cpu") -> dict:
    """The two readouts at `drift`, which must reach the pooled readout by keyword."""
    row = {"target": pooled_readout(agent, norm, params, drift_sigma=drift, n_eps=n_eps,
                                    steps=steps, ray_steps=ray_steps, device=device,
                                    seed=seed)["target"]}
    if balanced:
        row["balanced_target"] = paired_pooled_readout(agent, norm, params, drift, n_eps=n_eps,
                                                       steps=steps, ray_steps=ray_steps,
                                                       device=device, seed=seed)["target"]
    return row


def integrity(got: dict, ref: dict, *, tol: float = 0.01) -> dict:
    """The drift-0 rescore must reproduce the run's recorded drift-0 targets."""
    checks = []
    for key, want in sorted(ref.items()):
        if key in got:
            checks.append({"arm": key[0], "seed": key[1], "recorded": float(want),
                           "rescored": float(got[key]), "abs_dev": abs(float(got[key]) - float(want))})
    worst = max((c["abs_dev"] for c in checks), default=float("nan"))
    return {"tolerance": tol, "n_checked": len(checks),
            "worst_abs_dev": worst, "pass": bool(checks) and worst <= tol, "per_cell": checks}


def _agg(v: list[float]) -> dict:
    v = [float(x) for x in v]
    return {"per_seed": v, "mean": float(np.mean(v)), "n": len(v),
            "t90": [float(x) for x in t_ci90(v)] if len(v) > 1 else None}


def summarize(standard: dict, balanced: dict, *, bar: float = 0.65, margin: float = 0.05) -> dict:
    out = {"standard": {a: _agg(v) for a, v in standard.items() if v},
           "balanced": {a: _agg(v) for a, v in balanced.items() if v}}
    # Adjudicated on the standard readout, the registered estimand. The balanced readout was
    # frozen after the registered rule and is reported, never substituted.
    out.update(adjudicate(survival=out["standard"]["survival"]["mean"],
                          untrained=out["standard"]["untrained"]["mean"], bar=bar, margin=margin))
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reference(path: str, arms) -> dict:
    """Recorded drift-0 per-seed targets, keyed by (arm, seed), for the integrity gate."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    scheme = folds.current_scheme()
    out = {}
    for c in doc.get("cells", []):
        if float(c["drift"]) == 0.0 and c.get("agent") in arms and scheme in c:
            out[(c["agent"], int(c["seed"]))] = float(c[scheme]["target"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drift", type=float, default=0.45)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--bar", type=float, default=0.65)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--integrity-ref", default="artifacts/fold_rescore/l3_h8_heldout.json")
    ap.add_argument("--integrity-tol", type=float, default=0.01)
    ap.add_argument("--checkpoint-dir", default=None, help="one JSON per (arm, seed); resumes")
    ap.add_argument("--quick", action="store_true", help="20 episodes, 2 seeds")
    a = ap.parse_args()
    if a.quick:
        a.n_eps = 20
        print("--quick: 20 episodes per pool and 2 seeds. The integrity gate compares against "
              "targets recorded at 110 episodes, so it is expected to FAIL and void the run. "
              "Quick mode is for exercising the pipeline, never for a reported number.", flush=True)

    torch.set_num_threads(1)
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=a.l3_hidden, device="cpu", seed=a.l3_seed, params=P)
    arms = tuple(s.strip() for s in a.arms.split(",") if s.strip())

    cells = select_agents(a.agents_dir, arms=arms)
    if a.quick:
        cells = [c for c in cells if c[0] < 2]
    if not cells:
        raise SystemExit(f"no drift-0 agents for arms {arms} in {a.agents_dir}")

    ckpt = a.checkpoint_dir
    if ckpt:
        os.makedirs(ckpt, exist_ok=True)

    out = {
        "generated_by": "scripts/reverse_direction.py",
        "git_commit": git_head(),
        "spec": "docs/specs/2026-10-09-reverse-direction-readout-design.md",
        "agents_dir": a.agents_dir, "readout_drift": a.drift, "agent_drift": 0.0,
        "arms": list(arms), "bar": a.bar, "margin": a.margin,
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
        row = score_one(agent, norm, P, drift=a.drift, n_eps=a.n_eps, steps=a.steps,
                        ray_steps=a.ray_steps, seed=seed)
        # The integrity gate: the same agent at its own drift must reproduce the record.
        row["drift0_target"] = pooled_readout(agent, norm, P, drift_sigma=0.0, n_eps=a.n_eps,
                                              steps=a.steps, ray_steps=a.ray_steps,
                                              device="cpu", seed=seed)["target"]
        cell = {"seed": seed, "arm": arm, "agent_sha256": _sha256(path), **row}
        out["cells"].append(cell)
        if cp:
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(cell, fh, indent=1)
        print(f"{arm} s{seed}: reverse {row['target']:.3f}  balanced "
              f"{row.get('balanced_target', float('nan')):.3f}  drift0 {row['drift0_target']:.3f}"
              f"   ({time.time() - t0:.0f}s elapsed)", flush=True)

    ref = _reference(a.integrity_ref, arms)
    got = {(c["arm"], c["seed"]): c["drift0_target"] for c in out["cells"]}
    out["integrity"] = {"reference": a.integrity_ref, **integrity(got, ref, tol=a.integrity_tol)}

    std = {arm: [c["target"] for c in out["cells"] if c["arm"] == arm] for arm in arms}
    bal = {arm: [c["balanced_target"] for c in out["cells"]
                 if c["arm"] == arm and "balanced_target" in c] for arm in arms}
    out["summary"] = summarize(std, bal, bar=a.bar, margin=a.margin)
    out["wall_seconds"] = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    ig = out["integrity"]
    print(f"\nintegrity gate: {'PASS' if ig['pass'] else 'FAIL'} "
          f"(worst abs dev {ig['worst_abs_dev']:.4f} against {a.integrity_tol}, "
          f"{ig['n_checked']} cells)")
    s = out["summary"]
    for arm in arms:
        st = s["standard"].get(arm)
        if st:
            t = st["t90"] or [float("nan")] * 2
            print(f"{arm}: reverse {st['mean']:.3f} [{t[0]:.3f}, {t[1]:.3f}]"
                  + (f"   balanced {s['balanced'][arm]['mean']:.3f}" if arm in s["balanced"] else ""))
    print(f"  margin over untrained {s['margin_over_untrained']:+.3f}")
    print(f"  verdict under the frozen rule: {s['verdict']}"
          + ("" if ig["pass"] else "  (VOID: integrity gate failed)"))
    print(f"saved {a.out}")
    return 0 if ig["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
