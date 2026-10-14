"""Promote the L3 sensory-echo control (readout-only) to a committed artifact.

Spec: docs/specs/2026-09-26-l3-sensory-echo-control-design.md. The run bundle is
gitignored (`fullruns/l3_h8_sensory_echo/{cells,aggregate}.json`); this copies the
decision-relevant per-seed values and aggregates into `artifacts/expB2/` with
provenance so every published number recomputes in-repo via
scripts/audit_stats_recheck.py.

Balanced runs (spec docs/specs/2026-10-13-balanced-sensory-echo-design.md, aggregate
`mode == "balanced"`) promote the same way; the artifact then records the pair bookkeeping
per cell, the recorded-target integrity gap, the stack, and `decision_balanced`.

Usage:
    python scripts/promote_sensory_echo.py \
        --run fullruns/l3_h8_sensory_echo --out artifacts/expB2/sensory_echo_l3_h8.json
    python scripts/promote_sensory_echo.py --run fullruns/balanced_confirmation_wm/sensory_echo \
        --out artifacts/balanced_confirmation/sensory_echo_b1.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import audit_sensory_echo as ase

from promote_bv3_gates import repo_rel

METRICS = ("target", "obs_trace_only", "resid_trace", "resid_obs", "resid_obs_int", "resid_obs_beh",
           "resid_obs_mlp", "resid_obs_beh_mlp")   # the last two only in --nonlinear runs
BALANCED_CELL_KEYS = ("n_pairs", "dropped_pairs", "identical_first_state", "recorded_target",
                      "target_abs_diff")


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def promote(run_dir: str, out_path: str, head: str | None = None) -> dict:
    with open(os.path.join(run_dir, "aggregate.json"), encoding="utf-8") as fh:
        agg = json.load(fh)
    with open(os.path.join(run_dir, "cells.json"), encoding="utf-8") as fh:
        cells = json.load(fh)
    keep = {k: v for k, v in agg.items() if k.startswith("d=")}
    balanced = agg.get("mode") == "balanced"
    cell_keys = ("drift", "seed", "agent", "n_auth", "n_surr", "integrity_match", *METRICS)
    if balanced:
        agents_dir = os.path.basename(os.path.dirname(repo_rel(run_dir).rstrip("/\\")))
        cell_keys = cell_keys + BALANCED_CELL_KEYS
    out = {
        "source_run": repo_rel(run_dir),
        "spec": ("docs/specs/2026-10-13-balanced-sensory-echo-design.md" if balanced
                 else "docs/specs/2026-09-26-l3-sensory-echo-control-design.md"),
        "world": "WorldParams(k_land=1.5, k_water=1.5, gravity=0.4) [P]",
        "surrogate": f"L3 GMotion hidden={agg['hidden']} seed={agg['g_seed']} trained on P",
        "agents": (f"saved fullruns/{agents_dir}/agents bundles (readout-only, no training)" if balanced
                   else f"saved fullruns/l3_h{agg['hidden']}_heldout bundles (readout-only, no training)"),
        "git_commit_at_promotion": head or git_head(),
        "generated_by": "scripts/promote_sensory_echo.py",
        "bars": {"auroc_floor": agg["bar"], "margin": agg["margin"]},
        "config": {"n_eps": agg["n_eps"], "steps": agg["steps"], "device": agg["device"],
                   "primary_basis": "[x_t, x_{t-1}] (instantaneous; spec amendment 2026-09-26)",
                   "secondary_basis": "[x_t, x_{t-1}, cummean(x)] (resid_obs_int)"},
        "metrics": list(METRICS),
        "integrity": agg["integrity"],
        "decision": agg.get("decision"),
        "decision_nonlinear": agg.get("decision_nonlinear"),
        "aggregate": keep,
        "cells": [{k: c[k] for k in cell_keys if k in c} for c in cells],
    }
    if balanced:
        out["mode"] = "balanced"
        out["balanced_seed_base"] = agg["balanced_seed_base"]
        out["objective"] = agg["objective"]
        out["mortal"] = agg["mortal"]
        out["stack"] = agg.get("stack")
        recomputed = ase.aggregate(
            cells, [0.45], ["survival", "predictor", "untrained"],
            hidden=int(agg["hidden"]), g_seed=int(agg["g_seed"]),
            n_eps=int(agg["n_eps"]), steps=int(agg["steps"]),
            quick=bool(agg.get("quick")), device=str(agg.get("device") or "cpu"),
            mismatches=list((agg.get("integrity") or {}).get("mismatches") or []),
            balanced_seed_base=int(agg.get("balanced_seed_base") or 0),
            objective=str(agg.get("objective") or "survival"),
            mortal=bool(agg.get("mortal", True)),
            expected_keys=set(ase.BALANCED_SPEC_KEYS))
        out["decision_balanced"] = recomputed.get("decision_balanced")
        out["aggregate"] = {key: value for key, value in recomputed.items() if key.startswith("d=")}
        out["integrity"]["adjudication_blockers"] = recomputed["integrity"]["adjudication_blockers"]
    else:
        raw_keys = agg.get("declared_keys")
        expected = ({(str(drift), int(seed), str(arm)) for drift, seed, arm in raw_keys}
                    if raw_keys else None)
        blockers = ase.adjudication_blockers(
            cells, quick=bool(agg.get("quick")),
            mismatches=list((agg.get("integrity") or {}).get("mismatches") or []),
            balanced=False, expected_keys=expected)
        if blockers:
            out["decision"] = ase._withhold(out.get("decision"), blockers)
            if out.get("decision_nonlinear"):
                out["decision_nonlinear"] = ase._withhold(out["decision_nonlinear"], blockers)
            out["integrity"]["adjudication_blockers"] = blockers
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fullruns/l3_h8_sensory_echo")
    ap.add_argument("--out", default="artifacts/expB2/sensory_echo_l3_h8.json")
    a = ap.parse_args()
    out = promote(a.run, a.out)
    if out.get("mode") == "balanced":
        dec = out["decision_balanced"] or {}
        print(f"wrote {a.out}: trained resid_obs {dec.get('trained_resid_obs', float('nan')):.3f} "
              f"vs untrained {dec.get('untrained_resid_obs', float('nan')):.3f} "
              f"(floor explained: {dec.get('floor_explained')}) -> {dec.get('zone')}")
        return 0
    dec = out["decision"] or {}
    print(f"wrote {a.out}: survival resid_obs {dec.get('survival_resid_obs', float('nan')):.3f} "
          f"vs untrained {dec.get('untrained_resid_obs', float('nan')):.3f} -> {dec.get('zone')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
