"""Promote the H2 ablation batteries (FINDINGS 14.5, 14.6, 14.7) to committed artifacts.

The texture-knockout / capacity-ladder numbers (14.5), the observation-channel
localization tables (14.6), and the L1 substrate-grounding battery (14.7) were
traceable only to gitignored `fullruns/` bundles. This script copies the
decision-relevant per-seed values, aggregates, gate-0 calibration rows, and
integrity-gate receipts out of those bundles into compact committed JSONs with
provenance, so every published 14.5-14.7 number is re-verified in-repo by
scripts/audit_stats_recheck.py. Nothing is recomputed from raw pools; values
are copied verbatim from the bundle aggregates (4 dp) and cell files (full
precision), and the only derived fields are the seed-level CIs and the L0
equivalence tests over the copied per-seed lists.

Sources (local run bundles, gitignored):
    fullruns/l3_h2_ablations/        hidden=8 texture knockout + capacity ladder
                                     (aggregate.json, cells.json, gate0_gn.json,
                                     gate0_ladder.json, ablations.log)
    fullruns/l3_h7_h2_ablations/     hidden=7 replication (shares the gate-0 files)
    fullruns/l3_h8_obs_localization/ hidden=8 observation-channel masks
    fullruns/l3_h7_obs_localization/ hidden=7 observation-channel masks
    fullruns/l1_heldout/             L1 organism run (expB2_results.json + cells/)
    fullruns/l1_h2_ablations/        L1 A1 graded seam + A2 noise knockout
    fullruns/l1_obs_localization/    L1 observation-channel masks
    fullruns/l1_calib.json           L1 gate-0 (delta) calibration
    fullruns/l1_noise_calib.json     L1 noise comparator gate-0 calibration

Outputs:
    artifacts/expH2/texture_knockout_h8.json
    artifacts/expH2/texture_knockout_h7.json
    artifacts/expH2/obs_localization_h8.json
    artifacts/expH2/obs_localization_h7.json
    artifacts/expL1/organism_summary.json
    artifacts/expL1/h2_ablations.json
    artifacts/expL1/obs_localization.json

Usage:
    python scripts/promote_h2_batteries.py            # defaults relative to repo root
    python scripts/promote_h2_batteries.py --fullruns D:/bundles --out-root artifacts
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itasorl.stats import equivalence_test, mean_ci, rope_test, t_ci90  # noqa: E402

GENERATED_BY = "scripts/promote_h2_batteries.py"
BAR = 0.65
ARMS = ("survival", "untrained", "predictor")
KNOCKOUT_CHANNELS = ("gn", "h16", "h32", "h64")
ORGANISM_POOL_KEYS = (
    "pool_target", "pool_target_lo", "pool_target_hi", "pool_speed", "pool_shuffled",
    "pool_anchor_energy", "pool_anchor_food", "pool_ceiling_drag", "pool_reward_leak",
    "pool_leak_clean", "pool_deaths_auth", "pool_deaths_surr", "mp_target", "mp_leak_clean",
)

INTEGRITY_RE = re.compile(
    r"integrity gate PASSED: (?:headline )?survival mean ([\d.]+)(?: == ([\d.]+))?"
    r"(?: \(determinism check #(\d+)\))?")
MASK_DIMS_RE = re.compile(r"mask=(\w+): (\d+)/(\d+) dimensions zeroed")


# ---------------------------------------------------------------- helpers ---
def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def read_text_any(path: str) -> str | None:
    """Read a run log written by either a UTF-8 or a PowerShell UTF-16 redirect."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def nan_to_none(obj):
    """JSON has no NaN; the bundles carry NaN for undefined ceiling-drag cells."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def rel_source(path: str, fullruns: str) -> str:
    """Record a bundle path as the canonical `fullruns/<name>` label.

    The bundles are gitignored and may live in another checkout (e.g. the main
    checkout while promoting from a worktree), so the label is taken relative to
    the --fullruns directory rather than to the repo root; a path outside that
    directory is recorded as given.
    """
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(fullruns))
    except ValueError:  # different drive on Windows
        return path.replace("\\", "/")
    if rel.startswith(".."):
        return path.replace("\\", "/")
    return "fullruns/" + rel.replace("\\", "/")


def file_mtime_iso(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    return _dt.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")


def parse_integrity(log_text: str | None) -> dict:
    if log_text is None:
        return {"log_found": False}
    out: dict = {"log_found": True,
                 "n_integrity_ok_lines": len(re.findall(r"integrity ok:", log_text))}
    m = INTEGRITY_RE.search(log_text)
    if m:
        out["gate_passed"] = True
        out["survival_mean_reproduced"] = float(m.group(1))
        if m.group(2):
            out["published_target"] = float(m.group(2))
        if m.group(3):
            out["determinism_check"] = int(m.group(3))
        out["log_line"] = m.group(0)
    else:
        out["gate_passed"] = None
    return out


def parse_mask_dims(log_text: str | None) -> dict:
    if log_text is None:
        return {}
    return {m.group(1): {"zeroed": int(m.group(2)), "total": int(m.group(3))}
            for m in MASK_DIMS_RE.finditer(log_text)}


def seed_sorted(cells: list[dict], arm: str, key: str, **filt) -> list[float]:
    rows = [c for c in cells if c.get("arm") == arm
            and all(c.get(k) == v for k, v in filt.items()) and key in c]
    rows.sort(key=lambda c: c["seed"])
    return [float(c[key]) for c in rows]


def seed_stats(vals: list[float]) -> dict:
    mean, lo, hi = mean_ci(vals)
    tlo, thi = t_ci90(vals)
    return {"mean": mean, "boot90": [lo, hi], "tci90": [tlo, thi],
            "n_ge_065": int(sum(v >= BAR for v in vals)), "n_seeds": len(vals)}


def write_json(path: str, doc: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(nan_to_none(doc), fh, indent=1, allow_nan=False)
    os.replace(tmp, path)
    print(f"wrote {path}")


def provenance(source_dir: str, fullruns: str, head: str, **extra) -> dict:
    return {"source_run": rel_source(source_dir, fullruns),
            "git_commit_at_promotion": head,
            "generated_by": GENERATED_BY,
            "bar": BAR, **extra}


# ------------------------------------------------ 14.5 texture knockout ---
def texture_knockout_summary(agg: dict, cells: list[dict], gate0_gn: dict | None,
                             gate0_ladder: dict | None, log_text: str | None,
                             hidden: int) -> dict:
    channels: dict[str, dict] = {}
    for ch in KNOCKOUT_CHANNELS:
        channels[ch] = {}
        for arm in ARMS:
            per_seed = agg[f"{ch}_{arm}_per_seed"]
            entry = {"per_seed": per_seed,
                     "mean": agg[f"{ch}_{arm}_mean"],
                     "n_ge_065": agg[f"{ch}_{arm}_n_ge_065"]}
            full = seed_sorted(cells, arm, f"transfer_{ch}_target")
            if full:
                entry["per_seed_full"] = full
                entry["deaths_auth"] = seed_sorted(cells, arm, f"transfer_{ch}_deaths_auth")
                entry["deaths_surr"] = seed_sorted(cells, arm, f"transfer_{ch}_deaths_surr")
                entry["n_auth"] = seed_sorted(cells, arm, f"transfer_{ch}_n_auth")
                entry["n_surr"] = seed_sorted(cells, arm, f"transfer_{ch}_n_surr")
            channels[ch][arm] = entry
    gate0: dict = {}
    if gate0_gn is not None:
        sel = gate0_gn.get("selected") or {}
        rows = gate0_gn["rows"]
        chosen = [r for r in rows if r.get("sigma_v") == sel.get("sigma_v")]
        gate0["gn"] = {"band": gate0_gn["band"], "floor_tol": gate0_gn["floor_tol"],
                       "drift": gate0_gn["drift"], "sigma_meas": gate0_gn["sigma_meas"],
                       "world": gate0_gn["world"], "selected": sel,
                       "selected_row": chosen[0] if chosen else None, "rows": rows}
    if gate0_ladder is not None:
        gate0["ladder"] = {"band": gate0_ladder["band"], "floor_tol": gate0_ladder["floor_tol"],
                           "rows": gate0_ladder["rows"]}
    return {
        "hidden": hidden,
        "config": {"quick": agg["quick"], "n_eps": agg["n_eps"], "steps": agg["steps"],
                   "drift": agg.get("drift", 0.45), "gn_sigma_v": agg["gn_sigma_v"],
                   "published_target_check": agg["published_target_check"],
                   "n_seeds": len(agg["gn_survival_per_seed"])},
        "integrity": parse_integrity(log_text),
        "gate0": gate0,
        "ladder_oracles": agg["ladder_oracles"],
        "gn_dropped_at_gate0": agg["gn_dropped_at_gate0"],
        "channels": channels,
        "gn_rule": {"pass": agg["gn_rule_pass"], "margin": agg["gn_rule_margin"],
                    "verdict": agg["gn_verdict"]},
        "ladder_promotion_eligible": agg["ladder_promotion_eligible"],
        "cells": cells,
    }


# ------------------------------------------ 14.6 observation localization ---
def obs_localization_summary(agg: dict, cells: list[dict], log_text: str | None) -> dict:
    masks: dict[str, dict] = {}
    dims = parse_mask_dims(log_text)
    for mask in agg["masks"]:
        masks[mask] = {"zeroed_dims": dims.get(mask)}
        for arm in ARMS:
            entry = {"per_seed": agg[f"{mask}_{arm}_per_seed"],
                     "mean": agg[f"{mask}_{arm}_mean"],
                     "n_ge_065": agg[f"{mask}_{arm}_n_ge_065"]}
            full = seed_sorted(cells, arm, "target", mask=mask)
            if full:
                entry["per_seed_full"] = full
                entry["pool_leak_clean"] = [bool(v) for v in seed_sorted(cells, arm, "pool_leak_clean", mask=mask)]
                entry["deaths_auth"] = seed_sorted(cells, arm, "deaths_auth", mask=mask)
                entry["deaths_surr"] = seed_sorted(cells, arm, "deaths_surr", mask=mask)
            masks[mask][arm] = entry
    config = {k: v for k, v in agg.items() if not isinstance(v, list) or k == "masks"}
    return {"config": config, "masks": masks, "cells": cells}


# ------------------------------------------------- 14.7 L1 organism run ---
def l1_organism_summary(results: dict, cell_files: list[dict], run_log: str | None,
                        calib: dict, noise_calib: dict) -> dict:
    fingerprints = {c["fingerprint"] for c in cell_files}
    commits = {c["git_commit"] for c in cell_files}
    if len(fingerprints) > 1:
        raise ValueError(f"mixed fingerprints in L1 bundle: {sorted(fingerprints)}")
    per_drift: dict[str, dict] = {}
    for drift, arms in results.items():
        per_drift[drift] = {}
        for arm in ARMS:
            payload = arms[arm]
            row = {k: payload[k] for k in ORGANISM_POOL_KEYS if k in payload}
            row["aggregate"] = {"pool_target": seed_stats([float(v) for v in payload["pool_target"]])}
            per_drift[drift][arm] = row
    engagement = []
    for c in sorted(cell_files, key=lambda c: (float(c["cell"]["drift"]), c["cell"]["seed"])):
        cell = c["cell"]
        engagement.append({"drift": cell["drift"], "seed": cell["seed"], **cell["eng"]})
    drifts = sorted(results.keys(), key=float)
    l0_key = drifts[0]
    headline_key = drifts[-1]
    l0_vals = [float(v) for v in results[l0_key]["survival"]["pool_target"]]
    tost = equivalence_test(l0_vals, margin=0.05)
    rope = rope_test(l0_vals)
    gates = {
        "engagement": {"n_cells": len(engagement),
                       "n_engaged": int(sum(bool(e["engaged"]) for e in engagement)),
                       "pass": all(bool(e["engaged"]) for e in engagement)},
        "l0_control": {"drift": l0_key, "survival_pool_target_per_seed": l0_vals,
                       "mean": tost.mean,
                       "tost": {"margin": tost.margin, "p_value": tost.p_value,
                                "equivalent": tost.equivalent},
                       "rope": {"rope": list(rope.rope), "hdi": list(rope.hdi),
                                "p_in_rope": rope.p_in_rope, "accept": rope.accept}},
        "speed_positive_control": {"threshold": 0.75, "min_speed": {}, "pass": True},
        "leakage": {"pool_leak_clean_all": True, "mp_leak_clean_all": True},
        "survivorship": {"deaths_total": 0},
    }
    for drift in drifts:
        for arm in ARMS:
            payload = results[drift][arm]
            ms = float(min(payload["pool_speed"]))
            gates["speed_positive_control"]["min_speed"][f"d={drift} {arm}"] = ms
            if ms < 0.75:
                gates["speed_positive_control"]["pass"] = False
            if "pool_leak_clean" in payload and not all(payload["pool_leak_clean"]):
                gates["leakage"]["pool_leak_clean_all"] = False
            if "mp_leak_clean" in payload and not all(payload["mp_leak_clean"]):
                gates["leakage"]["mp_leak_clean_all"] = False
            gates["survivorship"]["deaths_total"] += int(sum(payload.get("pool_deaths_auth", [])))
            gates["survivorship"]["deaths_total"] += int(sum(payload.get("pool_deaths_surr", [])))
    chosen_delta = calib["chosen_delta"]
    delta_rows = [r for r in calib.get("fine_sweep", []) + calib.get("deltas_sweep", [])
                  if r.get("delta") == chosen_delta]
    chosen_sigma = noise_calib["chosen_sigma"]
    noise_rows = [r for r in noise_calib.get("noise_sigmas_sweep", [])
                  if r.get("noise_sigma") == chosen_sigma]
    header = (run_log or "").splitlines()[:3]
    return {
        "fingerprint": sorted(fingerprints)[0] if fingerprints else None,
        "git_commit_at_run": sorted(commits),
        "run_log_header": header,
        "drifts": drifts,
        "headline_drift": headline_key,
        "gate0": {"delta": {"chosen_delta": chosen_delta, "chosen_auroc": calib["chosen_auroc"],
                            "sensor_sigma": calib["sensor_sigma"],
                            "chosen_row": delta_rows[0] if delta_rows else None,
                            "fine_sweep": calib.get("fine_sweep", [])},
                  "noise": {"chosen_sigma": chosen_sigma,
                            "headline_delta": noise_calib["headline_delta"],
                            "chosen_row": noise_rows[0] if noise_rows else None,
                            "sweep": noise_calib.get("noise_sigmas_sweep", [])}},
        "arms": per_drift,
        "engagement": engagement,
        "gates": gates,
    }


def l1_h2_ablations_summary(a1: dict, noise: dict, log_text: str | None,
                            noise_calib: dict) -> dict:
    chosen_sigma = noise_calib["chosen_sigma"]
    noise_rows = [r for r in noise_calib.get("noise_sigmas_sweep", [])
                  if r.get("noise_sigma") == chosen_sigma]
    return {
        "integrity": parse_integrity(log_text),
        "a1_graded_seam": a1,
        "a2_noise_knockout": noise,
        "gate0_noise": {"chosen_sigma": chosen_sigma,
                        "chosen_row": noise_rows[0] if noise_rows else None},
    }


# -------------------------------------------------------------------- main ---
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--root", default=default_root,
                    help="repo root; source_run paths are recorded relative to it")
    ap.add_argument("--fullruns", default=None,
                    help="bundle directory (default: <root>/fullruns)")
    ap.add_argument("--out-root", default=None,
                    help="artifact root (default: <root>/artifacts)")
    ap.add_argument("--h8-dir", default=None, help="override fullruns/l3_h2_ablations")
    ap.add_argument("--h7-dir", default=None, help="override fullruns/l3_h7_h2_ablations")
    ap.add_argument("--gate0-dir", default=None,
                    help="dir holding gate0_gn.json / gate0_ladder.json (default: --h8-dir; "
                         "the hidden=7 replication reuses the hidden=8 gate-0 calibration)")
    ap.add_argument("--h8-obs-dir", default=None)
    ap.add_argument("--h7-obs-dir", default=None)
    ap.add_argument("--l1-obs-dir", default=None)
    ap.add_argument("--l1-heldout-dir", default=None)
    ap.add_argument("--l1-ablations-dir", default=None)
    ap.add_argument("--l1-calib", default=None)
    ap.add_argument("--l1-noise-calib", default=None)
    args = ap.parse_args(argv)

    root = args.root
    fr = args.fullruns or os.path.join(root, "fullruns")
    out_root = args.out_root or os.path.join(root, "artifacts")
    h8_dir = args.h8_dir or os.path.join(fr, "l3_h2_ablations")
    h7_dir = args.h7_dir or os.path.join(fr, "l3_h7_h2_ablations")
    gate0_dir = args.gate0_dir or h8_dir
    h8_obs = args.h8_obs_dir or os.path.join(fr, "l3_h8_obs_localization")
    h7_obs = args.h7_obs_dir or os.path.join(fr, "l3_h7_obs_localization")
    l1_obs = args.l1_obs_dir or os.path.join(fr, "l1_obs_localization")
    l1_held = args.l1_heldout_dir or os.path.join(fr, "l1_heldout")
    l1_abl = args.l1_ablations_dir or os.path.join(fr, "l1_h2_ablations")
    l1_calib = args.l1_calib or os.path.join(fr, "l1_calib.json")
    l1_noise_calib = args.l1_noise_calib or os.path.join(fr, "l1_noise_calib.json")
    head = git_head()

    def gate0(name: str):
        p = os.path.join(gate0_dir, name)
        return load_json(p) if os.path.exists(p) else None

    # -- 14.5 texture knockout + capacity ladder ------------------------------
    for hidden, src in ((8, h8_dir), (7, h7_dir)):
        agg_path = os.path.join(src, "aggregate.json")
        doc = provenance(src, fr, head,
                         source_files={"aggregate": "aggregate.json", "cells": "cells.json",
                                       "log": "ablations.log",
                                       "gate0_gn": rel_source(os.path.join(gate0_dir, "gate0_gn.json"), fr),
                                       "gate0_ladder": rel_source(os.path.join(gate0_dir, "gate0_ladder.json"), fr)},
                         gate0_shared_from_hidden8=(gate0_dir != src),
                         git_commit_at_run=None,
                         git_commit_at_run_note="bundle does not record the run commit; "
                                                "aggregate mtime recorded instead",
                         aggregate_mtime=file_mtime_iso(agg_path),
                         runner="scripts/run_l3_h2_ablations.py")
        doc.update(texture_knockout_summary(
            load_json(agg_path), load_json(os.path.join(src, "cells.json")),
            gate0("gate0_gn.json"), gate0("gate0_ladder.json"),
            read_text_any(os.path.join(src, "ablations.log")), hidden))
        write_json(os.path.join(out_root, "expH2", f"texture_knockout_h{hidden}.json"), doc)

    # -- 14.6 / 14.7 observation-channel localization -------------------------
    for out_rel, src in (("expH2/obs_localization_h8.json", h8_obs),
                         ("expH2/obs_localization_h7.json", h7_obs),
                         ("expL1/obs_localization.json", l1_obs)):
        agg_path = os.path.join(src, "aggregate.json")
        doc = provenance(src, fr, head,
                         source_files={"aggregate": "aggregate.json", "cells": "cells.json",
                                       "log": "ablations.log"},
                         git_commit_at_run=None,
                         git_commit_at_run_note="bundle does not record the run commit; "
                                                "aggregate mtime recorded instead",
                         aggregate_mtime=file_mtime_iso(agg_path),
                         runner="scripts/run_l3_obs_localization.py")
        doc.update(obs_localization_summary(
            load_json(agg_path), load_json(os.path.join(src, "cells.json")),
            read_text_any(os.path.join(src, "ablations.log"))))
        write_json(os.path.join(out_root, out_rel), doc)

    # -- 14.7 L1 organism run + A1/A2 batteries -------------------------------
    cell_paths = sorted(glob.glob(os.path.join(l1_held, "cells", "cell_*.json")))
    if not cell_paths:
        print(f"ERROR: no cell files under {l1_held}/cells", file=sys.stderr)
        return 1
    calib = load_json(l1_calib)
    noise_calib = load_json(l1_noise_calib)
    doc = provenance(l1_held, fr, head,
                     source_files={"results": "expB2_results.json", "cells": "cells/cell_*.json",
                                   "log": "run.log",
                                   "calib": rel_source(l1_calib, fr),
                                   "noise_calib": rel_source(l1_noise_calib, fr)},
                     runner="scripts/run_expB2.py --drift-mode l1")
    doc.update(l1_organism_summary(
        load_json(os.path.join(l1_held, "expB2_results.json")),
        [load_json(p) for p in cell_paths],
        read_text_any(os.path.join(l1_held, "run.log")), calib, noise_calib))
    write_json(os.path.join(out_root, "expL1", "organism_summary.json"), doc)

    a1_path = os.path.join(l1_abl, "a1_aggregate.json")
    doc = provenance(l1_abl, fr, head,
                     source_files={"a1_aggregate": "a1_aggregate.json",
                                   "noise_aggregate": "noise_aggregate.json",
                                   "log": "ablations.log",
                                   "noise_calib": rel_source(l1_noise_calib, fr)},
                     git_commit_at_run=None,
                     git_commit_at_run_note="bundle does not record the run commit; "
                                            "aggregate mtime recorded instead",
                     aggregate_mtime=file_mtime_iso(a1_path),
                     runner="scripts/run_l1_h2_ablations.py")
    doc.update(l1_h2_ablations_summary(
        load_json(a1_path), load_json(os.path.join(l1_abl, "noise_aggregate.json")),
        read_text_any(os.path.join(l1_abl, "ablations.log")), noise_calib))
    write_json(os.path.join(out_root, "expL1", "h2_ablations.json"), doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
