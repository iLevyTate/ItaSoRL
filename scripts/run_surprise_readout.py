"""Matched-input surprise against retention on saved agents.

Frozen design: docs/specs/2026-10-15-surprise-dissociation-design.md. Readout-only. Every arm
reads the SAME recorded episodes (the trained survival agent's own, and a scripted walk), so
the agent's model error (S) and its state probe (R) are compared on identical input.

    python scripts/run_surprise_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --drift 0.45 --out artifacts/surprise/corrected_l3_h8_wm.json
    python scripts/run_surprise_readout.py --agents-dir fullruns/corrected_l1_heldout/agents \\
        --drift-mode l1 --drift 0.023 --out artifacts/surprise/corrected_l1_heldout.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

AUTH_BASE, SURR_BASE = 800_000, 850_000   # the registered pools
ARMS = ("survival", "predictor", "untrained", "survival_d0")


def _setup(t: dict):
    import itasorl.experiment_b2 as b2
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = t["drift_mode"]
    if t["drift_mode"] == "l1":
        b2.L1_DELTA, b2.SENSOR_SIGMA = t["l1_delta"], t["l1_sigma"]
    elif b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=8, device="cpu", seed=0, params=P)
    return P


def run_seed(t: dict) -> dict:
    import torch

    from itasorl import mechanism_readouts as mr
    from itasorl.eval_protocols import readout_from_states, record_trajectories, replay_states
    from itasorl.experiment_b2 import load_agent_bundle

    torch.set_num_threads(1)
    P = _setup(t)
    s, d, n, T, rs = t["seed"], t["drift"], t["n_eps"], t["steps"], t["ray_steps"]
    dtag = f"{d:.2f}" if t["drift_mode"] == "l3" else f"{d:.4f}"
    paths = {"survival": f"agent_d{dtag}_s{s}_survival.pt", "predictor": f"agent_d{dtag}_s{s}_predictor.pt",
             "untrained": f"agent_d{dtag}_s{s}_untrained.pt", "survival_d0": f"agent_d0.00_s{s}_survival.pt"}
    agents = {g: load_agent_bundle(os.path.join(t["agents_dir"], p), "cpu") for g, p in paths.items()}
    t0 = time.time()
    surv, snorm = agents["survival"]
    streams = {
        "own": (record_trajectories(surv, snorm, P, 0.0, n, T, "cpu", AUTH_BASE, rs),
                record_trajectories(surv, snorm, P, d, n, T, "cpu", SURR_BASE, rs)),
        "scripted": (mr.record_scripted_trajectories(P, 0.0, n_eps=n, steps=T, seed_base=AUTH_BASE, ray_steps=rs),
                     mr.record_scripted_trajectories(P, d, n_eps=n, steps=T, seed_base=SURR_BASE, ray_steps=rs)),
    }
    out = {"seed": s, "drift": d}
    for proto, ((Oa, Aa, Ka), (Os, As, Ks)) in streams.items():
        blk = {"n": [int(len(Ka)), int(len(Ks))]}
        for g, (ag, nm) in agents.items():
            Ea, Es = mr.replay_prediction_error(ag, nm, Oa, Aa), mr.replay_prediction_error(ag, nm, Os, As)
            Ha, Hs = replay_states(ag, nm, Oa, Aa), replay_states(ag, nm, Os, As)
            ma, ms = float(np.nanmean(Ea)), float(np.nanmean(Es))
            blk[g] = {"S": mr.surprise_auroc(Ea, Es), "R": readout_from_states(Ha, Hs, seed=s)["target"],
                      "mean_err_auth": ma, "mean_err_surr": ms,
                      "log_ratio_auth_over_surr": float(np.log(ma / ms)) if np.isfinite(ma) and ms > 0 else float("nan")}
        out[proto] = blk
    # L0 analogue: the trained decoder on its own episodes in two AUTHENTIC pools
    O1, A1, _ = record_trajectories(surv, snorm, P, 0.0, n, T, "cpu", AUTH_BASE, rs)
    O2, A2, _ = record_trajectories(surv, snorm, P, 0.0, n, T, "cpu", SURR_BASE, rs)
    out["l0_S"] = mr.surprise_auroc(mr.replay_prediction_error(surv, snorm, O1, A1),
                                    mr.replay_prediction_error(surv, snorm, O2, A2))
    out["seconds"] = round(time.time() - t0, 1)
    print(f"seed {s}: own S surv {out['own']['survival']['S']:.3f} pred {out['own']['predictor']['S']:.3f} "
          f"untr {out['own']['untrained']['S']:.3f} | R surv {out['own']['survival']['R']:.3f} | L0 S {out['l0_S']:.3f} "
          f"({out['seconds']}s)", flush=True)
    return out


def _paired(a, b) -> dict:
    from itasorl.stats import t_ci90
    x = np.asarray(a, float) - np.asarray(b, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float("nan"), "t90": [float("nan")] * 2, "n": int(len(x))}
    lo, hi = t_ci90(x)
    return {"mean": float(x.mean()), "t90": [float(lo), float(hi)], "n": int(len(x))}


def _call(c: dict, threshold: float) -> str:
    lo, hi = c["t90"]
    if not np.isfinite(lo):
        return "not computable"
    if lo > threshold:
        return "met"
    if hi < (0.0 if threshold == 0.0 else -threshold):
        return "reversed"
    return "not met"


def summarize(rows: list[dict]) -> dict:
    from itasorl.stats import t_ci90
    summ = {}
    for proto in ("own", "scripted"):
        blk = {}
        for g in ARMS:
            for q in ("S", "R", "log_ratio_auth_over_surr"):
                v = np.array([r[proto][g][q] for r in rows], float)
                f = v[np.isfinite(v)]
                blk[f"{g} {q}"] = {"per_seed": v.tolist(), "mean": float(f.mean()) if len(f) else float("nan"),
                                   "t90": [float(x) for x in t_ci90(f)] if len(f) > 1 else None, "n": int(len(f))}
        get = lambda g, q: [r[proto][g][q] for r in rows]  # noqa: E731
        c1 = _paired(get("survival", "S"), get("untrained", "S"))
        c2 = _paired(get("survival", "S"), get("survival", "R"))
        c3 = _paired(get("predictor", "S"), get("untrained", "S"))
        pos = int(np.sum(np.array(get("survival", "log_ratio_auth_over_surr")) > 0))
        neg = int(np.sum(np.array(get("survival_d0", "log_ratio_auth_over_surr")) < 0))
        blk["readings"] = {
            "1 model detects beyond the input floor (S surv - S untr, LB > 0.05)": {**c1, "call": _call(c1, 0.05)},
            "2 detection exceeds retention (S surv - R surv, LB > 0)": {**c2, "call": _call(c2, 0.0)},
            "3 prediction training alone detects (S pred - S untr, LB > 0.05)": {**c3, "call": _call(c3, 0.05)},
            "4 error tracks the law raised under (sign, >= 8 of 10 each)": {
                "raised_in_surrogate_positive": pos, "raised_in_authentic_negative": neg,
                "call": "met" if pos >= 8 and neg >= 8 else "not met"},
        }
        summ[proto] = blk
    l0 = np.array([r["l0_S"] for r in rows], float)
    summ["l0_S"] = {"per_seed": l0.tolist(), "mean": float(np.nanmean(l0)),
                    "within_0.1_of_0.5": bool(abs(np.nanmean(l0) - 0.5) < 0.1)}
    return summ


def main() -> int:
    from itasorl import folds
    from itasorl.optional_readouts import provenance
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--drift", type=float, required=True)
    ap.add_argument("--drift-mode", choices=("l3", "l1"), default="l3")
    ap.add_argument("--l1-delta", type=float, default=0.023)
    ap.add_argument("--l1-sigma", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="directory of per-seed JSON; finished seeds are skipped")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [{"agents_dir": a.agents_dir, "seed": s, "drift": a.drift, "drift_mode": a.drift_mode,
              "l1_delta": a.l1_delta, "l1_sigma": a.l1_sigma, "n_eps": a.n_eps, "steps": a.steps,
              "ray_steps": a.ray_steps} for s in a.seeds]
    rows, todo = [], []
    for t in tasks:
        p = a.ckpt and os.path.join(a.ckpt, f"s{t['seed']}.json")
        if p and os.path.exists(p):
            rows.append(json.load(open(p, encoding="utf-8")))
        else:
            todo.append(t)
    def _save(r):
        rows.append(r)
        if a.ckpt:
            os.makedirs(a.ckpt, exist_ok=True)
            with open(os.path.join(a.ckpt, f"s{r['seed']}.json"), "w", encoding="utf-8") as fh:
                json.dump(r, fh, default=float)
    if a.workers > 1 and todo:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_seed, todo):
                _save(r)
    else:
        for t in todo:
            _save(run_seed(t))
    rows.sort(key=lambda r: r["seed"])
    command = "python " + " ".join(["scripts/run_surprise_readout.py", *sys.argv[1:]])
    inputs = {"agents_dir": a.agents_dir, "drift": a.drift, "drift_mode": a.drift_mode, "seeds": a.seeds,
              "pools": {"authentic": AUTH_BASE, "surrogate": SURR_BASE}, "n_eps": a.n_eps, "steps": a.steps}
    prov = provenance("scripts/run_surprise_readout.py", command, inputs, device="cpu")
    prov["status"] = ("readout under docs/specs/2026-10-15-surprise-dissociation-design.md; "
                      "diagnostic, not an adjudication; changes no registered rule or verdict")
    doc = {"provenance": prov, "fold_scheme": folds.current_scheme(), "summary": summarize(rows), "rows": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=float)
    for proto in ("own", "scripted"):
        print(f"== {proto}")
        for k, v in doc["summary"][proto]["readings"].items():
            print(f"  {k}: {v}")
    print("L0 S:", doc["summary"]["l0_S"]["mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
