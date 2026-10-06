"""Behavior and sensory controls with diagnostics, on saved agents (revision step 8).

Readout-only. For every arm at the strongest drift of a run written with --save-agents and
--dump-states, regenerate the standard pools (seed bases 800000 / 850000) with observation,
action, and behavior traces, require the recurrent states to bit-match the run's own dump,
and report, each labeled by the alternative it addresses:

  target                 the headline probe on h_t
  resid_trace            published behavior control (seven behavior channels: b_t, b_{t-1},
                         running mean), with held-out nuisance R^2 and nuisance-from-residual R^2
  resid_trace_act        the same basis plus the env action a_t and the previous action a_{t-1}
                         the GRU received
  resid_obs              published sensory control [x_t, x_{t-1}] with diagnostics
  resid_obs_hist         x_t ... x_{t-7}, causal EMA traces (tau 4, 16), a_t, a_{t-1}
  resid_obs_hist_beh     resid_obs_hist plus the behavior trace
  resid_joint_mlp        [x_t, x_{t-1}, EMA 4, EMA 16, a_t, a_{t-1}, b_t, b_{t-1}] removed by a
                         64-unit MLP, with its optimizer diagnostics
  obs_summary_only       SUMMARY-FEATURE comparator: the linear probe on [mean, final, sd,
                         mean |delta|] of the observation trace (not a sequence readout)
  seq_gru                SEQUENCE readout: supervised GRU (64 / 96, the trunk's capacity) on
                         the (observation, previous action) sequence, per fold
  seq_flat_linear        SEQUENCE readout: linear probe on the flattened sequence after in-fold
                         PCA to 192 components

Usage:
    python scripts/run_control_diagnostics.py --run-dir fullruns/corrected_l3_h8_wm \\
        --out artifacts/control_diagnostics/corrected_l3_h8_wm.json [--workers 4] [--quick]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re
import time

import numpy as np

AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def run_one(task: dict) -> dict:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.behavior_audit import _trace_phi, sensory_residual_probe_auroc, trace_residual_probe_auroc
    from itasorl.control_diagnostics import (flat_sequence_linear_auroc, history_basis,
                                             residual_probe_with_diagnostics, sequence_gru_auroc)
    from itasorl.experiment_b import episode_features, episode_features_full, probe_auroc
    from itasorl.experiment_b2 import collect_pool, format_drift, load_agent_bundle
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=task["l3_hidden"], device="cpu", seed=task["l3_seed"], params=P)
    t0 = time.time()
    d, s, g = task["drift"], task["seed"], task["arm"]
    agent, norm = load_agent_bundle(task["path"], "cpu")
    pools = {}
    for lab, dd, base in (("a", 0.0, 800_000), ("s", d, 850_000)):
        H, *_, Bt, Ot, At = collect_pool(agent, norm, P, dd, task["n_eps"], task["steps"], "cpu", base,
                                         task["ray_steps"], return_anchors=True, return_obs=True,
                                         return_actions=True)
        pools[lab] = (H, Bt, Ot, At)
    out = {"drift": d, "seed": s, "arm": g}
    dump = os.path.join(task["states_dir"], f"states_d{format_drift(d)}_s{s}_{g}.npz") \
        if task.get("states_dir") else None
    if dump and os.path.exists(dump):
        with np.load(dump) as z:
            out["dump_bit_match"] = bool(np.array_equal(z["Ha"], pools["a"][0])
                                         and np.array_equal(z["Hs"], pools["s"][0]))
    H = np.concatenate([pools["a"][0], pools["s"][0]])
    Bt = np.concatenate([pools["a"][1], pools["s"][1]])
    Ot = np.concatenate([pools["a"][2], pools["s"][2]])
    At = np.concatenate([pools["a"][3], pools["s"][3]])
    y = np.r_[np.zeros(len(pools["a"][0])), np.ones(len(pools["s"][0]))].astype(int)
    out["n"] = int(len(y))
    if min(len(pools["a"][0]), len(pools["s"][0])) < 5:
        out["error"] = "too few survivors"
        return out
    out["target"] = probe_auroc(episode_features(H), y)
    out["resid_trace_published"] = trace_residual_probe_auroc(H, Bt, y)
    out["resid_obs_published"] = sensory_residual_probe_auroc(H, Ot, y)
    out["resid_trace"] = residual_probe_with_diagnostics(H, _trace_phi(Bt, quad=False), y)
    a_now = At.reshape(-1, At.shape[-1])
    a_prev = np.concatenate([np.zeros_like(At[:, :1]), At[:, :-1]], axis=1)
    out["resid_trace_act"] = residual_probe_with_diagnostics(
        H, np.concatenate([_trace_phi(Bt, quad=False), a_now,
                           a_prev.reshape(-1, At.shape[-1])], axis=1), y)
    out["resid_obs"] = residual_probe_with_diagnostics(H, history_basis(Ot, lags=1), y)
    out["resid_obs_hist"] = residual_probe_with_diagnostics(
        H, history_basis(Ot, lags=7, ema_taus=(4, 16), At=At), y)
    out["resid_obs_hist_beh"] = residual_probe_with_diagnostics(
        H, history_basis(Ot, lags=7, ema_taus=(4, 16), At=At, Bt=Bt), y)
    out["resid_joint_mlp"] = residual_probe_with_diagnostics(
        H, history_basis(Ot, lags=1, ema_taus=(4, 16), At=At, Bt=Bt), y, model="mlp", seed=s)
    out["obs_summary_only"] = probe_auroc(episode_features_full(Ot), y)
    seq = np.concatenate([Ot, a_prev], axis=2)
    out["seq_gru"] = sequence_gru_auroc(seq, y, epochs=task["gru_epochs"], seed=s)
    out["seq_flat_linear"] = flat_sequence_linear_auroc(seq, y)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _scalar(v):
    return v["auroc"] if isinstance(v, dict) else v


def main() -> int:
    from itasorl import folds
    from itasorl.stats import t_ci90
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--gru-epochs", type=int, default=60)
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_eps, a.steps, a.ray_steps, a.gru_epochs = 40, 16, 4, 10
    agents_dir = os.path.join(a.run_dir, "agents")
    cells = []
    for name in sorted(os.listdir(agents_dir)):
        m = AGENT_RE.match(name)
        if m:
            cells.append((float(m.group(1)), int(m.group(2)), m.group(3), name))
    if a.quick:
        cells = [c for c in cells if c[1] < 2]
    dmax = max(c[0] for c in cells)
    tasks = [{"path": os.path.join(agents_dir, nm), "drift": d, "seed": s, "arm": g,
              "states_dir": os.path.join(a.run_dir, "states"), "n_eps": a.n_eps,
              "steps": a.steps, "ray_steps": a.ray_steps, "gru_epochs": a.gru_epochs,
              "l3_hidden": a.l3_hidden, "l3_seed": a.l3_seed}
             for d, s, g, nm in cells if d == dmax]
    results = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_one, tasks):
                results.append(r)
                print(f"  s{r['seed']} {r['arm']}: target {r.get('target', float('nan')):.3f} "
                      f"bit_match={r.get('dump_bit_match')} ({r.get('seconds')} s)", flush=True)
    else:
        for t in tasks:
            r = run_one(t)
            results.append(r)
            print(f"  s{r['seed']} {r['arm']}: target {r.get('target', float('nan')):.3f} "
                  f"bit_match={r.get('dump_bit_match')} ({r.get('seconds')} s)", flush=True)
    results.sort(key=lambda r: (r["arm"], r["seed"]))
    metrics = ["target", "resid_trace_published", "resid_obs_published", "resid_trace",
               "resid_trace_act", "resid_obs", "resid_obs_hist", "resid_obs_hist_beh",
               "resid_joint_mlp", "obs_summary_only", "seq_gru", "seq_flat_linear"]
    agg = {}
    for g in ("untrained", "predictor", "survival"):
        rows = [r for r in results if r["arm"] == g and "target" in r]
        for m in metrics:
            v = [_scalar(r[m]) for r in rows if m in r]
            v = [x for x in v if np.isfinite(x)]
            if not v:
                continue
            entry = {"per_seed": v, "mean": float(np.mean(v)),
                     "t90": [float(x) for x in t_ci90(v)] if len(v) > 1 else None}
            diag = [r[m] for r in rows if isinstance(r.get(m), dict)]
            if diag and "nuisance_r2_heldout" in diag[0]:
                entry["nuisance_r2_heldout_mean"] = float(np.mean([x["nuisance_r2_heldout"] for x in diag]))
                entry["nuisance_from_residual_r2_mean"] = float(
                    np.mean([x["nuisance_from_residual_r2_heldout"] for x in diag]))
            if diag and "optimizer" in diag[0]:
                entry["converged_folds"] = int(sum(x["optimizer"]["converged_folds"] for x in diag))
                entry["folds"] = int(sum(x["optimizer"]["folds"] for x in diag))
            agg[f"d={dmax:.2f} {g} {m}"] = entry
    out = {"generated_by": "scripts/run_control_diagnostics.py", "run_dir": a.run_dir,
           "config": {k: getattr(a, k) for k in ("n_eps", "steps", "ray_steps", "gru_epochs")},
           "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
           "all_dumps_bit_match": all(r.get("dump_bit_match", False) for r in results),
           "cells": results, "aggregate": agg}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k, v in agg.items():
        extra = ""
        if "nuisance_r2_heldout_mean" in v:
            extra = (f"  R2 {v['nuisance_r2_heldout_mean']:.2f} back "
                     f"{v['nuisance_from_residual_r2_mean']:.2f}")
        print(f"  {k:44s} {v['mean']:.3f} {v['t90']}{extra}")
    print(f"dumps bit-match: {out['all_dumps_bit_match']}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
