"""Promote an off-design or failed-gates L3 run to a descriptive committed artifact.

Two local runs are quoted in the record without a committed artifact of their run record:

  T-qd  (fullruns/T_qd_l3_h8_wm, finding R72): a survival run in the hand-authored
        quadratic-drag family at eps 6.0, corrected trainer, 2026-10-06, commit 2606e7f. Its
        family failed gate 0, its untrained arm reads above the survival arm, and its 60
        saved agents carry an 8-unit GRU trunk, not the registered 96. OFF-DESIGN.
  L3-H4 (fullruns/l3_h4_traces): the hidden-4 run of 2026-07-13, historical trainer, whose
        capacity was frozen from a pre-bugfix calibration on the wrong world. It failed its
        gates (untrained floor, reward leakage, engagement). UNINFORMATIVE. Its re-scored
        state targets are already committed (artifacts/fold_rescore/l3_h4_traces.json); its
        run record (speed control, engagement, leakage, survivorship) was not.

Neither run bears on any rule. This writes what each run measured, in the conventions of
scripts/promote_texture_trained.py (whose gate helpers it reuses): per arm and drift, the
per-seed pooled targets with mean, the t-based 90% interval and the seed-bootstrap 90%
interval the run printed (itasorl.stats.mean_ci, 10000 draws, seed 0); the speed positive
control in every pool; engagement per cell; the L0 TOST; reward leakage; survivorship; the
untrained floor; and the agent trunk width, read from the saved agents when they exist and
otherwise from the dumped state dimension.

Usage:
    python scripts/promote_off_design_runs.py --which tqd
    python scripts/promote_off_design_runs.py --which h4
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

from agent_free_common import provenance, write_json  # noqa: E402
from itasorl.stats import equivalence_test, mean_ci, t_ci90  # noqa: E402
from promote_bv3_gates import repo_rel  # noqa: E402
from promote_texture_trained import (ARMS, BAR, LEAK_TOL, MARGIN, REGISTERED_TRUNK,  # noqa: E402
                                     engagement_gate, speed_gate, trunk_check)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_FULLRUNS = os.environ.get("ITASORL_FULLRUNS",
                                  os.path.join(os.path.dirname(os.path.dirname(ROOT)), "fullruns")
                                  if os.path.basename(os.path.dirname(ROOT)) == ".worktrees"
                                  else os.path.join(ROOT, "fullruns"))

RUNS = {
    "tqd": {
        "run_dir": "T_qd_l3_h8_wm",
        "out": "artifacts/texture/T_qd_l3_h8_wm.json",
        "status": "OFF-DESIGN",
        "label": ("OFF-DESIGN, descriptive only: comparator-trained qd run (finding R72). The "
                  "qd family failed gate 0, the untrained arm reads above the survival arm, "
                  "and the agent trunk is 8 units, not the registered 96. Bears on no rule."),
        "family": "qd", "family_param": 6.0, "surrogate_hidden": None,
        "record": "FINDINGS 17.9 addendum (2026-10-07) and its note of 2026-10-10; CORRECTIONS "
                  "2026-10-10; RESULTS_MANIFEST T-qd note",
        "gate0": {"source": "artifacts/texture/gate0_qd.json", "kind": "family_sweep"},
        "recorded": {"survival_mean": 0.580, "survival_boot90": [0.542, 0.615],
                     "survival_t90": [0.537, 0.624], "predictor_mean": 0.573,
                     "predictor_boot90": [0.549, 0.595], "untrained_mean": 0.608,
                     "untrained_boot90": [0.584, 0.630], "l0_mean": 0.523, "l0_tost_p": 0.020,
                     "speed_pools_below": 8, "speed_worst": 0.7018, "engaged_cells": 16,
                     "trunk": 8},
    },
    "h4": {
        "run_dir": "l3_h4_traces",
        "out": "artifacts/expB2/failed_gates_l3_h4_traces.json",
        "status": "FAILED GATES",
        "label": ("FAILED GATES, UNINFORMATIVE (historical trainer): the hidden-4 L3 run. Its "
                  "capacity was frozen from a pre-bugfix calibration on the wrong world; on "
                  "world P its untrained floor fails gate 0. Bears on no rule."),
        "family": "mlp", "family_param": None, "surrogate_hidden": 4,
        "record": "FINDINGS 10.5 ('hidden = 4 was uninformative, not negative'); "
                  "PREREGISTRATION_L3 sec. 12 entries 2026-07-13/14",
        "gate0": {"source": "artifacts/agent_free/gate0_pair_intervals.json",
                  "instance": "mlp_h4_s0", "kind": "instance",
                  "floor_source": "fullruns/l3_gate0_recal/calibration.json"},
        "recorded": {"untrained_floor": 0.891, "reward_leak_clean_seeds": 0,
                     "engagement_share": 0.30},
    },
}


def _agg(values) -> dict:
    v = [float(x) for x in values]
    out = {"per_seed": v, "mean": float(np.mean(v)), "n": len(v),
           "seeds_at_or_above_bar": int(sum(1 for x in v if x >= BAR))}
    if len(v) > 1:
        out["t90"] = [float(x) for x in t_ci90(v)]
        _, lo, hi = mean_ci(v, level=0.90)
        out["seed_boot90"] = [lo, hi]
    return out


def speed_pools(run: dict) -> list[dict]:
    rows = []
    for drift, cells in run.items():
        for arm in ARMS:
            for seed, x in enumerate(cells.get(arm, {}).get("pool_speed", [])):
                rows.append({"drift": f"{float(drift):.2f}", "arm": arm, "seed": seed,
                             "speed": float(x)})
    return rows


def state_dim_check(states_dir: str) -> dict:
    """Trunk width read from the dumped recurrent states (Ha: n_eps x steps x hidden) when no
    agents were kept."""
    files = sorted(glob.glob(os.path.join(states_dir, "states_*.npz")))
    dims = set()
    n = 0
    for f in files:
        with np.load(f) as z:
            if "Ha" not in z:
                continue
            dims.add(int(z["Ha"].shape[2]))
            n += 1
    return {"states_dir": repo_rel(states_dir), "n_dumps": n, "state_dims": sorted(dims),
            "verified": n > 0 and len(dims) == 1}


def cell_provenance(cells_dir: str) -> dict:
    commits, fps, boots = set(), set(), set()
    for f in sorted(glob.glob(os.path.join(cells_dir, "cell_*.json"))):
        with open(f, encoding="utf-8") as fh:
            c = json.load(fh)
        commits.add(c.get("git_commit"))
        fps.add(c.get("fingerprint"))
        boots.add(c.get("cell", {}).get("gae_bootstrap", "absent"))
    return {"git_commit_at_run": sorted(x for x in commits if x),
            "config_fingerprint": sorted(x for x in fps if x),
            "gae_bootstrap": sorted(boots)}


def gate0_block(spec: dict) -> dict:
    src = os.path.join(ROOT, spec["source"])
    with open(src, encoding="utf-8") as fh:
        doc = json.load(fh)
    if spec["kind"] == "family_sweep":
        rows = [{"eps": r["eps"], "oracle_auroc": r["oracle_auroc"], "in_band": r["in_band"],
                 "floor": r["floor"], "passes_gate0": r["passes_gate0"]} for r in doc["rows"]]
        return {"source": spec["source"], "sweep": rows,
                "any_in_band": any(r["in_band"] for r in rows),
                "note": "No eps in the frozen qd sweep reaches the band (highest oracle "
                        f"{max(r['oracle_auroc'] for r in rows):.3f}); eps 6.0 was matched to "
                        "the learned map's one-step magnitude, not calibrated through gate 0.",
                "pass": False}
    inst = next(i for i in doc["instances"] if i["id"] == spec["instance"])
    leg = inst["schemes"]["legacy"]
    out = {"source": spec["source"], "instance": spec["instance"],
           "oracle_auroc_legacy": leg["oracle_auroc"], "in_band": leg["in_band_point"],
           "pair_cluster_ci95_legacy": leg["pair_cluster_ci95"],
           "mech_leak_pass": leg["mech_leak_pass"]}
    fsrc = os.path.join(PRIMARY_FULLRUNS, spec["floor_source"][len("fullruns/"):])
    if os.path.exists(fsrc):
        with open(fsrc, encoding="utf-8") as fh:
            row = next(r for r in json.load(fh)["rows"] if r.get("hidden") == 4)
        out.update({"floor": row["floor"], "floor_per_seed": row["floor_per_seed"],
                    "floor_ok": row["floor_ok"], "floor_source": spec["floor_source"]})
        out["pass"] = bool(row["passes_gate0"])
    else:
        out["floor_source_missing"] = spec["floor_source"]
        out["pass"] = False
    out["note"] = ("Measured on world P after the run: the oracle is in band, the untrained "
                   "floor is not. The run itself used a capacity frozen from the pre-bugfix "
                   "calibration on the wrong world (oracle 0.940 at sigma 0.05).")
    return out


def promote(which: str) -> dict:
    spec = RUNS[which]
    run_dir = os.path.join(PRIMARY_FULLRUNS, spec["run_dir"])
    with open(os.path.join(run_dir, "expB2_results.json"), encoding="utf-8") as fh:
        run = json.load(fh)
    drifts = sorted(run, key=float)
    per_drift = {f"{float(d):.2f}": {arm: _agg(run[d][arm]["pool_target"])
                                     for arm in ARMS if arm in run[d]} for d in drifts}
    secondary = {f"{float(d):.2f}": {arm: {k: float(np.nanmean(run[d][arm][k]))
                                           for k in ("pool_target_var", "pool_target_full",
                                                     "mp_target", "pool_shuffled")}
                                     for arm in ARMS if arm in run[d]} for d in drifts}
    d45 = next(d for d in drifts if abs(float(d) - 0.45) < 1e-9)
    d0 = next(d for d in drifts if abs(float(d)) < 1e-9)
    v, v0 = run[d45], run[d0]
    eq = equivalence_test(v0["survival"]["pool_target"])
    leak = {arm: [float(x) for x in v[arm]["pool_reward_leak"]] for arm in ARMS}
    leak_dev = max(abs(x - 0.5) for xs in leak.values() for x in xs)
    leak_clean = {arm: [bool(x) for x in v[arm]["pool_leak_clean"]] for arm in ARMS}
    mp_clean = {arm: [bool(x) for x in v[arm]["mp_leak_clean"]] for arm in ARMS}
    deaths = sum(int(x) for arm in ARMS for k in ("pool_deaths_auth", "pool_deaths_surr")
                 for x in v[arm][k])
    cells_dir = os.path.join(run_dir, "cells")
    untrained = per_drift[f"{float(d45):.2f}"]["untrained"]["mean"]
    gates = {
        "gate0_oracle_band": gate0_block(spec["gate0"]),
        "engagement": engagement_gate(cells_dir),
        "l0_tost": {"mean": eq.mean, "p": eq.p_value, "margin": 0.05, "pass": bool(eq.equivalent)},
        "speed_positive_control": speed_gate(run),
        "reward_leakage": {"tolerance": LEAK_TOL, "max_abs_dev": leak_dev,
                           "pass": leak_dev <= LEAK_TOL, "per_seed": leak,
                           "pool_leak_clean": leak_clean,
                           "seeds_clean_survival": int(sum(leak_clean["survival"])),
                           "matched_pair_leak_clean": mp_clean},
        "survivorship": {"total_deaths": deaths, "pass": deaths == 0},
        "untrained_floor": {"mean": untrained, "abs_dev": abs(untrained - 0.5),
                            "pass": abs(untrained - 0.5) <= LEAK_TOL},
    }
    by_drift: dict = {}
    for c in gates["engagement"]["per_cell"]:
        k = f"{float(c['drift']):.2f}"
        by_drift.setdefault(k, [0, 0])
        by_drift[k][0] += int(c["engaged"])
        by_drift[k][1] += 1
    gates["engagement"]["engaged_by_drift"] = {k: {"engaged": e, "cells": n}
                                               for k, (e, n) in sorted(by_drift.items())}
    failed = sorted(k for k, g in gates.items() if not g["pass"])
    agents_dir = os.path.join(run_dir, "agents")
    trunk = trunk_check(agents_dir)
    out = {
        "status": spec["status"], "label": spec["label"],
        **provenance("scripts/promote_off_design_runs.py"),
        "source_run": repo_rel(os.path.join(run_dir, "expB2_results.json")),
        "run": cell_provenance(cells_dir),
        "record": spec["record"], "recorded_in_prose": spec["recorded"],
        "family": spec["family"], "family_param": spec["family_param"],
        "surrogate_hidden": spec["surrogate_hidden"],
        "bars": {"bar": BAR, "margin": MARGIN},
        "pooled_target": per_drift,
        "secondary_means": secondary,
        "clauses_descriptive": {
            "bar": per_drift[f"{float(d45):.2f}"]["survival"]["mean"] >= BAR,
            "over_predictor": (per_drift[f"{float(d45):.2f}"]["survival"]["mean"]
                               - per_drift[f"{float(d45):.2f}"]["predictor"]["mean"]) >= MARGIN,
            "over_untrained": (per_drift[f"{float(d45):.2f}"]["survival"]["mean"]
                               - untrained) >= MARGIN},
        "rule_met": False,
        "rule_note": "Not adjudicated: the run is not the registered protocol and failed gates; "
                     "the clauses above are descriptive.",
        "gates": gates, "gates_failed": failed,
        "speed_pools": speed_pools(run),
        "trunk_check": trunk,
        "registered_trunk_hidden": REGISTERED_TRUNK,
    }
    if trunk["verified"]:
        out["trunk_hidden"] = trunk["hidden_values"][0]
        out["trunk_source"] = "saved agents (ctor record and weight_hh shape)"
    else:
        sd = state_dim_check(os.path.join(run_dir, "states"))
        out["state_dim_check"] = sd
        out["trunk_hidden"] = sd["state_dims"][0] if sd["verified"] else None
        out["trunk_source"] = ("dumped recurrent state dimension (no agents were kept)"
                               if sd["verified"] else "not verified")
    if out["trunk_hidden"] != REGISTERED_TRUNK:
        out["configuration_deviation"] = (
            f"The agent trunk was {out['trunk_hidden']} units, not the registered "
            f"{REGISTERED_TRUNK} ({out['trunk_source']}).")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", choices=sorted(RUNS), required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    doc = promote(a.which)
    out = a.out or RUNS[a.which]["out"]
    out_path = out if os.path.isabs(out) else os.path.join(ROOT, out)
    write_json(out_path, doc)
    s = doc["pooled_target"]["0.45"]
    print(f"wrote {os.path.relpath(out_path, ROOT)}  [{doc['status']}]")
    for arm in ARMS:
        e = s[arm]
        print(f"  d=0.45 {arm:9s} {e['mean']:.3f} t90 [{e['t90'][0]:.3f}, {e['t90'][1]:.3f}] "
              f"boot90 [{e['seed_boot90'][0]:.3f}, {e['seed_boot90'][1]:.3f}] "
              f"{e['seeds_at_or_above_bar']}/10")
    sp = doc["gates"]["speed_positive_control"]
    print(f"  speed: {sp['n_below']} of {sp['n_pools']} pools below 0.75 (min "
          f"{sp['min_all_pools']:.4f}); engagement {doc['gates']['engagement']['n_engaged']}/"
          f"{doc['gates']['engagement']['n_cells']}; trunk {doc['trunk_hidden']} "
          f"({doc['trunk_source']}); gates failed {doc['gates_failed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
