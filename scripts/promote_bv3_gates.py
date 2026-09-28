"""Promote the B-v3 n=10 run's gate values to a committed artifact.

PREREGISTRATION_Bv3.md section 12 (2026-07-18 audit entry) records that the
n = 10 run's gate values - engagement, L0 TOST, speed positive control, and the
matched-pair leakage audit - were never promoted alongside the survival target
in `artifacts/expB2/bv3_n10_summary.json`. This script copies them, per seed,
out of the (gitignored) `fullruns/07062026` bundle (commit 820849f) into a
compact committed JSON with provenance, and records the gate verdicts as the
pre-registration section 7 defines them:

  (1) engagement: trained TRUE return beats random and scripted by the margin,
      lifetime not worse (cell["eng"]["engaged"]);
  (2) L0 control: drift-0 pooled survival target equivalent to 0.5 by TOST
      (margin 0.05; ROPE reported alongside);
  (3) positive control: speed probe >= 0.75 in every pool;
  (4) leakage audit: matched-pair reward-sum / length / lifetime each within
      0.1 of 0.5 (cell["agents"][arm]["mp"]["leakage"]).

Values are copied, not recomputed; the only derived fields are the L0
equivalence tests and the seed-level CI over the copied per-seed lists.

Usage:
    python scripts/promote_bv3_gates.py \
        --run-dir fullruns/07062026 \
        --out artifacts/expB2/bv3_n10_gates.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itasorl.stats import equivalence_test, mean_ci, rope_test, t_ci90  # noqa: E402

GENERATED_BY = "scripts/promote_bv3_gates.py"
BAR = 0.65
SPEED_THRESHOLD = 0.75
TOST_MARGIN = 0.05
ARMS = ("survival", "untrained", "predictor")
RESULT_KEYS = ("pool_target", "pool_target_lo", "pool_target_hi", "pool_speed",
               "pool_shuffled", "pool_anchor_energy", "pool_anchor_food",
               "pool_ceiling_drag", "mp_target", "mp_leak_clean")
MP_LEAK_CHANNELS = ("reward_sum", "length", "lifetime")


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def nan_to_none(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def seed_stats(vals: list[float]) -> dict:
    mean, lo, hi = mean_ci(vals)
    tlo, thi = t_ci90(vals)
    return {"mean": mean, "boot90": [lo, hi], "tci90": [tlo, thi],
            "n_ge_065": int(sum(v >= BAR for v in vals)), "n_seeds": len(vals)}


def source_label(run_dir: str) -> str:
    """Canonical `fullruns/<name>` label for a bundle that may live in another checkout."""
    absdir = os.path.abspath(run_dir)
    if os.path.basename(os.path.dirname(absdir)) == "fullruns":
        return "fullruns/" + os.path.basename(absdir)
    return run_dir.replace("\\", "/")


def drift_key(value) -> str:
    """Bundle results are keyed by str(float) ('0.0', '0.45'); cells carry floats."""
    return str(float(value))


def build_gates_doc(results: dict, cell_files: list[dict], manifest: dict | None,
                    flags: list | None) -> dict:
    fingerprints = {c.get("fingerprint") for c in cell_files}
    commits = {c.get("git_commit") for c in cell_files}
    if len(fingerprints) != 1:
        raise ValueError(f"mixed fingerprints in bundle: {sorted(map(str, fingerprints))}")

    drifts = sorted(results.keys(), key=float)
    arms_out: dict[str, dict] = {}
    for drift in drifts:
        arms_out[drift] = {}
        for arm in ARMS:
            payload = results[drift][arm]
            row = {k: payload[k] for k in RESULT_KEYS if k in payload}
            row["aggregate"] = {"pool_target": seed_stats([float(v) for v in payload["pool_target"]])}
            arms_out[drift][arm] = row

    # per-cell receipts: engagement + matched-pair leakage detail
    engagement, mp_leakage, pools = [], [], []
    for c in sorted(cell_files, key=lambda c: (float(c["cell"]["drift"]), c["cell"]["seed"])):
        cell = c["cell"]
        d, s = drift_key(cell["drift"]), cell["seed"]
        engagement.append({"drift": d, "seed": s, **cell["eng"]})
        for arm in ARMS:
            agent = cell["agents"][arm]
            mp = agent.get("mp", {})
            leak = mp.get("leakage", {})
            mp_leakage.append({"drift": d, "seed": s, "agent": arm,
                               "mp_target": mp.get("target"),
                               "leakage_clean": mp.get("leakage_clean"),
                               "leakage_max_dev": mp.get("leakage_max_dev"),
                               "margin": leak.get("margin"),
                               **{ch: leak.get(ch) for ch in MP_LEAK_CHANNELS}})
            pool = agent.get("pool", {})
            pools.append({"drift": d, "seed": s, "agent": arm,
                          "target": pool.get("target"), "speed": pool.get("speed"),
                          "n_auth": pool.get("n_auth"), "n_surr": pool.get("n_surr"),
                          "too_few_survivors": pool.get("too_few_survivors")})

    # gate verdicts (PREREGISTRATION_Bv3 section 7)
    l0 = drifts[0]
    l0_vals = [float(v) for v in results[l0]["survival"]["pool_target"]]
    tost = equivalence_test(l0_vals, margin=TOST_MARGIN)
    rope = rope_test(l0_vals)
    eng_by_drift = {}
    for d in drifts:
        rows = [e for e in engagement if e["drift"] == d]
        eng_by_drift[d] = {"n_cells": len(rows),
                           "n_engaged": int(sum(bool(e["engaged"]) for e in rows)),
                           "engaged_per_seed": [bool(e["engaged"]) for e in rows]}
    speed = {}
    speed_pass = True
    for d in drifts:
        for arm in ARMS:
            ms = float(min(results[d][arm]["pool_speed"]))
            speed[f"d={d} {arm}"] = ms
            speed_pass = speed_pass and ms >= SPEED_THRESHOLD
    leak_counts = {}
    leak_failures = []
    for d in drifts:
        for arm in ARMS:
            rows = [r for r in mp_leakage if r["drift"] == d and r["agent"] == arm]
            n_clean = int(sum(bool(r["leakage_clean"]) for r in rows))
            leak_counts[f"d={d} {arm}"] = {"n_clean": n_clean, "n_cells": len(rows)}
            for r in rows:
                if not r["leakage_clean"]:
                    worst = max(MP_LEAK_CHANNELS,
                                key=lambda ch: abs((r[ch] if r[ch] is not None else 0.5) - 0.5))
                    leak_failures.append({"drift": d, "seed": r["seed"], "agent": arm,
                                          "channel": worst, "value": r[worst],
                                          "max_abs_dev": r["leakage_max_dev"],
                                          "margin": r["margin"]})
    gates = {
        "definitions": "PREREGISTRATION_Bv3.md section 7 (identical to B-v2)",
        "engagement": {"per_drift": eng_by_drift,
                       "n_engaged": int(sum(v["n_engaged"] for v in eng_by_drift.values())),
                       "n_cells": len(engagement),
                       "pass": all(bool(e["engaged"]) for e in engagement)},
        "l0_control": {"drift": l0, "survival_pool_target_per_seed": l0_vals,
                       "mean": tost.mean,
                       "tost": {"margin": tost.margin, "p_value": tost.p_value,
                                "equivalent": tost.equivalent},
                       "rope": {"rope": list(rope.rope), "hdi": list(rope.hdi),
                                "p_in_rope": rope.p_in_rope, "accept": rope.accept},
                       "pass": bool(tost.equivalent)},
        "speed_positive_control": {"threshold": SPEED_THRESHOLD, "min_speed": speed,
                                   "pass": speed_pass},
        "leakage": {"channel": "matched-pair (mp) reward_sum / length / lifetime, margin 0.1",
                    "per_arm": leak_counts,
                    "n_clean": int(sum(v["n_clean"] for v in leak_counts.values())),
                    "n_cells": int(sum(v["n_cells"] for v in leak_counts.values())),
                    "failures": leak_failures,
                    "pass": not leak_failures},
    }
    gates["all_pass"] = all(gates[k]["pass"] for k in
                            ("engagement", "l0_control", "speed_positive_control", "leakage"))

    return {
        "fingerprint": sorted(map(str, fingerprints))[0],
        "git_commit_at_run": sorted(map(str, commits)),
        "run_id": (manifest or {}).get("run_id"),
        "environment": (manifest or {}).get("environment"),
        "flags": flags,
        "drifts": drifts,
        "arms": arms_out,
        "engagement": engagement,
        "mp_leakage": mp_leakage,
        "pools": pools,
        "gates": gates,
    }


def write_json(path: str, doc: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(nan_to_none(doc), fh, indent=1, allow_nan=False)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="fullruns/07062026")
    ap.add_argument("--out", default="artifacts/expB2/bv3_n10_gates.json")
    args = ap.parse_args(argv)

    results_path = os.path.join(args.run_dir, "artifacts", "expB2_results.json")
    cell_paths = sorted(glob.glob(os.path.join(args.run_dir, "artifacts", "cells", "cell_*.json")))
    if not cell_paths:
        print(f"ERROR: no cell files under {args.run_dir}/artifacts/cells", file=sys.stderr)
        return 1
    manifest_path = os.path.join(args.run_dir, "manifest.json")
    flags_path = os.path.join(args.run_dir, "b2_flags.json")
    manifest = load_json(manifest_path) if os.path.exists(manifest_path) else None
    flags = load_json(flags_path) if os.path.exists(flags_path) else None

    doc = {"source_run": source_label(args.run_dir),
           "source_files": {"results": "artifacts/expB2_results.json",
                            "cells": "artifacts/cells/cell_*.json",
                            "manifest": "manifest.json", "flags": "b2_flags.json"},
           "git_commit_at_promotion": git_head(),
           "generated_by": GENERATED_BY,
           "bar": BAR,
           "companion": "artifacts/expB2/bv3_n10_summary.json (survival target, FINDINGS 7.1)"}
    doc.update(build_gates_doc(load_json(results_path),
                               [load_json(p) for p in cell_paths], manifest, flags))
    write_json(args.out, doc)
    g = doc["gates"]
    print(f"wrote {args.out}  (engaged {g['engagement']['n_engaged']}/{g['engagement']['n_cells']}, "
          f"L0 TOST p={g['l0_control']['tost']['p_value']:.4f} equiv={g['l0_control']['tost']['equivalent']}, "
          f"speed min {min(g['speed_positive_control']['min_speed'].values()):.3f}, "
          f"mp leak clean {g['leakage']['n_clean']}/{g['leakage']['n_cells']}, "
          f"all_pass={g['all_pass']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
