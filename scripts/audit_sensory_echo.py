"""L3 sensory-echo control, READOUT-ONLY.

Spec: docs/specs/2026-09-26-l3-sensory-echo-control-design.md.

Reuses the saved L3 agents (default: fullruns/l3_h8_heldout); NO training. For every
(drift, seed, arm) agent the standard pools are regenerated with the SAME seed bases as
`pooled_readout` (800_000 authentic, 850_000 surrogate) so the recurrent states must
bit-match the saved dumps (integrity gate), this time also recording the normalized
observation trace the trunk received. Four readouts per cell:

  target          the headline probe on h_t (must reproduce the published aggregate)
  obs_trace_only  can the observation trace alone decode the world? (ceiling)
  resid_trace     the published per-timestep BEHAVIOR control, seven-channel basis
  resid_obs       the SENSORY control (PRIMARY): regress [x_t, x_{t-1}] out of h_t
  resid_obs_int   integrated variant with cummean(x) added (SECONDARY, over-strict when
                  the inputs carry the label; see the spec amendment)
  resid_obs_beh   instantaneous sensory + full behavior traces jointly

Frozen decision rule (drift 0.45, survival arm, n = 10): resid_obs mean >= 0.65 AND
> untrained resid_obs + 0.05 -> sensory-independent world representation;
[0.60, 0.65) -> attenuated; < 0.60 -> largely sensory-mediated.

Usage:
    python scripts/audit_sensory_echo.py --agents-dir fullruns/l3_h8_heldout/agents \\
        --states-dir fullruns/l3_h8_heldout/states --out-dir fullruns/l3_h8_sensory_echo \\
        --device cuda
    ... --quick   (seed 0 only, tiny pools, no bit compare)
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
from itasorl.behavior_audit import (sensory_residual_probe_auroc,
                                    trace_residual_probe_auroc)
from itasorl.experiment_b import episode_features, episode_features_full, probe_auroc
from itasorl.experiment_b2 import collect_pool, default_device, load_agent_bundle
from itasorl.stats import mean_ci, t_ci90
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)   # frozen organism world
PUBLISHED = {8: 0.752, 7: 0.737}                         # drift-0.45 survival means
BAR = 0.65
MARGIN = 0.05
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")
METRICS = ("target", "obs_trace_only", "resid_trace", "resid_obs", "resid_obs_int", "resid_obs_beh",
           "resid_obs_mlp", "resid_obs_beh_mlp")


def cfg():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--g-seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["survival", "untrained", "predictor"])
    ap.add_argument("--drifts", type=float, nargs="+", default=[0.45, 0.0])
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--no-abort", action="store_true",
                    help="record an integrity mismatch instead of aborting (diagnostic only)")
    ap.add_argument("--quick", action="store_true", help="seed 0 only, tiny pools, no bit compare")
    ap.add_argument("--reaggregate", action="store_true",
                    help="skip collection; rebuild aggregate.json from the saved cells.json")
    ap.add_argument("--nonlinear", action="store_true",
                    help="also run the MLP residualizers resid_obs_mlp and resid_obs_beh_mlp "
                         "(spec 2026-09-27-local-strengthening-probes, probe B)")
    ap.add_argument("--save-traces", action="store_true",
                    help="save the regenerated H, observation and behavior traces per cell "
                         "as traces_d<drift>_s<seed>_<arm>.npz under --out-dir")
    return ap.parse_args()


def fmt_drift(d: float) -> str:
    return f"{d:.2f}"


def aggregate(cells: list[dict], drifts, arms, *, hidden: int, g_seed: int, n_eps: int,
              steps: int, quick: bool, device: str, mismatches: list[str]) -> dict:
    """Across-seed aggregation plus the frozen decision rule. Every value is a plain
    Python float/bool/list so the result is JSON-serializable."""
    agg: dict = {"hidden": hidden, "g_seed": g_seed, "n_eps": n_eps, "steps": steps,
                 "quick": quick, "device": device, "bar": BAR, "margin": MARGIN,
                 "integrity": {"checked": not quick, "mismatches": list(mismatches),
                               "all_match": (not mismatches) if not quick else None}}
    for d in drifts:
        for arm in arms:
            rows = [c for c in cells if c["drift"] == fmt_drift(d) and c["agent"] == arm]
            if not rows:
                continue
            key = f"d={fmt_drift(d)} {arm}"
            agg[key] = {}
            for met in METRICS:
                vals = [float(r[met]) for r in rows if met in r and np.isfinite(r[met])]
                if not vals:
                    continue
                mean, lo, hi = mean_ci(vals)
                tlo, thi = t_ci90(vals) if len(vals) > 1 else (float("nan"), float("nan"))
                agg[key][met] = {"per_seed": vals, "mean": float(mean),
                                 "boot90": [float(lo), float(hi)],
                                 "t90": [float(tlo), float(thi)], "n_seeds": len(vals),
                                 "n_ge_065": int(sum(v >= BAR for v in vals))}
    surv = agg.get("d=0.45 survival", {})
    untr = agg.get("d=0.45 untrained", {})
    if "target" in surv and not quick and hidden in PUBLISHED:
        agg["integrity"]["survival_target_mean"] = round(surv["target"]["mean"], 3)
        agg["integrity"]["published_target"] = PUBLISHED[hidden]
        agg["integrity"]["target_reproduced"] = bool(
            abs(surv["target"]["mean"] - PUBLISHED[hidden]) < 5e-4)
    if "resid_obs" in surv and "resid_obs" in untr:
        s, u = surv["resid_obs"]["mean"], untr["resid_obs"]["mean"]
        rule = {"survival_resid_obs": s, "untrained_resid_obs": u,
                "survival_t90": surv["resid_obs"]["t90"],
                "pass_bar": bool(s >= BAR), "pass_margin": bool(s > u + MARGIN),
                "t90_lo_above_untrained": bool(surv["resid_obs"]["t90"][0] > u)}
        if s >= BAR and s > u + MARGIN:
            rule["zone"] = "SENSORY-INDEPENDENT world representation (rule passes)"
        elif s >= 0.60:
            rule["zone"] = "ATTENUATED: below-bar trace survives the sensory control"
        else:
            rule["zone"] = "LARGELY SENSORY-MEDIATED"
        agg["decision"] = rule
    if "resid_obs_beh_mlp" in surv and "resid_obs_beh_mlp" in untr:
        s, u = surv["resid_obs_beh_mlp"]["mean"], untr["resid_obs_beh_mlp"]["mean"]
        nl = {"survival_resid_obs_beh_mlp": s, "untrained_resid_obs_beh_mlp": u,
              "survival_t90": surv["resid_obs_beh_mlp"]["t90"],
              "pass_bar": bool(s >= BAR), "pass_margin": bool(s > u + MARGIN)}
        if s >= BAR and s > u + MARGIN:
            nl["zone"] = "SURVIVES the nonlinear joint control (rule passes)"
        elif s >= 0.60:
            nl["zone"] = "ATTENUATED under the nonlinear joint control"
        else:
            nl["zone"] = "LARGELY EXPLAINED by nonlinear mirroring of inputs and behavior"
        agg["decision_nonlinear"] = nl
    return agg


def main() -> int:
    a = cfg()
    dev = default_device() if a.device == "auto" else a.device
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)
    os.makedirs(a.out_dir, exist_ok=True)
    cells, mismatches = [], []
    t0 = time.time()
    if a.reaggregate:
        with open(os.path.join(a.out_dir, "cells.json"), encoding="utf-8") as fh:
            cells = json.load(fh)
        mismatches = [f"{c['drift']}_s{c['seed']}_{c['agent']}" for c in cells
                      if c.get("integrity_match") is False]
        names = []
        print(f"re-aggregating {len(cells)} saved cells from {a.out_dir}/cells.json")
    else:
        b2.DRIFT_MODE = "l3"
        b2.setup_l3_surrogate(hidden=a.hidden, device=dev, seed=a.g_seed, params=P)
        print(f"sensory-echo control  device={dev} hidden={a.hidden} g_seed={a.g_seed} "
              f"n_eps={n_eps} steps={steps} quick={a.quick}")
        names = sorted(n for n in os.listdir(a.agents_dir) if AGENT_RE.search(n))
    for name in names:
        m = AGENT_RE.search(name)
        drift, seed, arm = float(m.group(1)), int(m.group(2)), m.group(3)
        if arm not in a.arms or drift not in a.drifts or (a.quick and seed != 0):
            continue
        agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
        Ha, _, _, _, _, _, bta, Oa = collect_pool(agent, norm, P, 0.0, n_eps, steps, dev, 800_000, 5,
                                                  return_anchors=True, return_obs=True)
        Hs, _, _, _, _, _, bts, Os = collect_pool(agent, norm, P, drift, n_eps, steps, dev, 850_000, 5,
                                                  return_anchors=True, return_obs=True)
        integrity = None
        if not a.quick:
            saved = os.path.join(a.states_dir, f"states_d{fmt_drift(drift)}_s{seed}_{arm}.npz")
            if os.path.exists(saved):
                z = np.load(saved)
                integrity = bool(np.array_equal(z["Ha"], Ha) and np.array_equal(z["Hs"], Hs))
                if not integrity:
                    mismatches.append(name)
                    msg = f"INTEGRITY MISMATCH: regenerated pools differ from {saved}"
                    if not a.no_abort:
                        raise SystemExit(msg)
                    print("  " + msg)
        if len(Ha) < 5 or len(Hs) < 5:
            print(f"  {name}: too few survivors ({len(Ha)}/{len(Hs)}), skipped")
            continue
        H = np.concatenate([Ha, Hs])
        Ot = np.concatenate([Oa, Os])
        Bt = np.concatenate([bta, bts])
        y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
        cell = {"drift": fmt_drift(drift), "seed": seed, "agent": arm,
                "n_auth": int(len(Ha)), "n_surr": int(len(Hs)), "integrity_match": integrity,
                "target": probe_auroc(episode_features(H), y),
                "obs_trace_only": probe_auroc(episode_features_full(Ot), y),
                "resid_trace": trace_residual_probe_auroc(H, Bt, y),
                "resid_obs": sensory_residual_probe_auroc(H, Ot, y),
                "resid_obs_int": sensory_residual_probe_auroc(H, Ot, y, integrated=True),
                "resid_obs_beh": sensory_residual_probe_auroc(H, Ot, y, Bt=Bt)}
        if a.nonlinear:
            cell["resid_obs_mlp"] = sensory_residual_probe_auroc(H, Ot, y, nonlinear=True)
            cell["resid_obs_beh_mlp"] = sensory_residual_probe_auroc(H, Ot, y, Bt=Bt, nonlinear=True)
        if a.save_traces:
            np.savez_compressed(os.path.join(a.out_dir, f"traces_d{fmt_drift(drift)}_s{seed}_{arm}.npz"),
                                Ha=Ha, Hs=Hs, Oa=Oa, Os=Os, bta=bta, bts=bts)
        cells.append(cell)
        print(f"  d={cell['drift']} s={seed} {arm:9s} target={cell['target']:.3f} "
              f"obs_only={cell['obs_trace_only']:.3f} resid_trace={cell['resid_trace']:.3f} "
              f"resid_obs={cell['resid_obs']:.3f} resid_obs_int={cell['resid_obs_int']:.3f} "
              f"resid_obs_beh={cell['resid_obs_beh']:.3f} "
              f"integrity={integrity}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        with open(os.path.join(a.out_dir, "cells.json"), "w", encoding="utf-8") as fh:
            json.dump(cells, fh, indent=1)

    agg = aggregate(cells, a.drifts, a.arms, hidden=a.hidden, g_seed=a.g_seed, n_eps=n_eps,
                    steps=steps, quick=a.quick, device=dev, mismatches=mismatches)
    if "decision" in agg:
        print("\nDECISION (frozen rule, drift 0.45):", json.dumps(agg["decision"], indent=1))
    with open(os.path.join(a.out_dir, "aggregate.json"), "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1)
    print(f"wrote {a.out_dir}/aggregate.json  ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
