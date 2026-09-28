"""Promote the L3 sensory-echo control (readout-only) to a committed artifact.

Spec: docs/specs/2026-09-26-l3-sensory-echo-control-design.md. The run bundle is
gitignored (`fullruns/l3_h8_sensory_echo/{cells,aggregate}.json`); this copies the
decision-relevant per-seed values and aggregates into `artifacts/expB2/` with
provenance so every published number recomputes in-repo via
scripts/audit_stats_recheck.py.

Usage:
    python scripts/promote_sensory_echo.py \
        --run fullruns/l3_h8_sensory_echo --out artifacts/expB2/sensory_echo_l3_h8.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

METRICS = ("target", "obs_trace_only", "resid_trace", "resid_obs", "resid_obs_int", "resid_obs_beh")


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
    out = {
        "source_run": run_dir.replace("\\", "/"),
        "spec": "docs/specs/2026-09-26-l3-sensory-echo-control-design.md",
        "world": "WorldParams(k_land=1.5, k_water=1.5, gravity=0.4) [P]",
        "surrogate": f"L3 GMotion hidden={agg['hidden']} seed={agg['g_seed']} trained on P",
        "agents": "saved fullruns/l3_h8_heldout bundles (readout-only, no training)",
        "git_commit_at_promotion": head or git_head(),
        "generated_by": "scripts/promote_sensory_echo.py",
        "bars": {"auroc_floor": agg["bar"], "margin": agg["margin"]},
        "config": {"n_eps": agg["n_eps"], "steps": agg["steps"], "device": agg["device"],
                   "primary_basis": "[x_t, x_{t-1}] (instantaneous; spec amendment 2026-09-26)",
                   "secondary_basis": "[x_t, x_{t-1}, cummean(x)] (resid_obs_int)"},
        "metrics": list(METRICS),
        "integrity": agg["integrity"],
        "decision": agg.get("decision"),
        "aggregate": keep,
        "cells": [{k: c[k] for k in ("drift", "seed", "agent", "n_auth", "n_surr",
                                     "integrity_match", *METRICS)} for c in cells],
    }
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
    dec = out["decision"] or {}
    print(f"wrote {a.out}: survival resid_obs {dec.get('survival_resid_obs', float('nan')):.3f} "
          f"vs untrained {dec.get('untrained_resid_obs', float('nan')):.3f} -> {dec.get('zone')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
