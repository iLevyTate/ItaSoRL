"""Matched-input surprise against retention on saved agents.

Frozen design: docs/specs/2026-10-15-surprise-dissociation-design.md. Readout-only. Every arm
reads the SAME recorded episodes (the trained survival agent's own, and a scripted walk), so
the agent's model error (S) and its state probe (R) are compared on identical input.

    python scripts/run_surprise_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --drift 0.45 --out artifacts/surprise/corrected_l3_h8_wm.json
    python scripts/run_surprise_readout.py --agents-dir fullruns/corrected_l1_heldout/agents \\
        --drift-mode l1 --drift 0.023 --out artifacts/surprise/corrected_l1_heldout.json

`--snapshots U [U ...]` (closing item X04, E05; post hoc, descriptive, L3 only) adds the arms
survival_u<U>, the frozen snapshots agent_d<drift>_s<seed>_survival_u<U>.pt, read on the same
recorded streams. The output then gains a `snapshots` block: per budget b (0 is the untrained
arm, 300 the survival arm) the prediction-error AUROC S_b and both mean errors, the per-seed
Spearman between the authentic mean error and the stored fixed-walk target (`<b> replay_target`
of --development-ref) across the budgets, and the S_450 - S_100 label. Gate: S_300 and both mean
errors per seed equal --snapshot-ref rows[*].own.survival within 1e-6, and the b = 0 row equals
rows[*].own.untrained, else nothing is written and the script exits 1. Without the flag the
output is unchanged.
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
SNAPSHOT_REF = "artifacts/surprise/corrected_l3_h8_wm.json"
DEVELOPMENT_REF = "artifacts/development/corrected_l3_h8_wm.json"
SNAPSHOT_TOL = 1e-6


def snapshot_paths(agents_dir: str, drift: float, seed: int, snapshots) -> dict:
    """{"survival_u<U>": path} for each snapshot, by the `_u{updates}` suffix rule of
    _readout_common.agent_path."""
    from _readout_common import agent_path
    return {f"survival_u{u}": agent_path(agents_dir, drift, seed, "survival", u) for u in snapshots}


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
    if t.get("snapshots"):
        paths.update({g: os.path.basename(p) for g, p in
                      snapshot_paths(t["agents_dir"], d, s, t["snapshots"]).items()})
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
        # "reversed" for readings 1 and 3 means the upper bound lies below -0.05 (the mirror of their
        # 0.05 bound), added with the artifact and not in the frozen spec text; reading 2 uses the
        # spec's bound of 0. See the 2026-10-17 note in docs/specs/2026-10-15-surprise-dissociation-design.md.
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


def _close(a, b, tol: float = SNAPSHOT_TOL) -> bool:
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(np.isnan(a) and np.isnan(b))
    return abs(a - b) <= tol


def snapshot_gate(rows: list[dict], ref: dict) -> list[str]:
    """Failures of the X04 gate: S, mean_err_auth and mean_err_surr of own.survival (b = 300)
    and of own.untrained (b = 0) per seed against the stored readout."""
    stored = {int(r["seed"]): r for r in ref["rows"]}
    fails = []
    for r in rows:
        s = int(r["seed"])
        if s not in stored:
            fails.append(f"seed {s}: not in the stored readout")
            continue
        for arm in ("survival", "untrained"):
            for q in ("S", "mean_err_auth", "mean_err_surr"):
                got, want = r["own"][arm][q], stored[s]["own"][arm][q]
                if not _close(got, want):
                    fails.append(f"seed {s} own.{arm}.{q}: {got!r} vs stored {want!r}")
    return fails


def snapshot_summary(rows: list[dict], snapshots, dev: dict) -> dict:
    """Per budget S, mean errors and the stored fixed-walk target; per-seed Spearman of the
    authentic mean error against that target; the S_450 - S_100 label."""
    from scipy.stats import spearmanr

    from itasorl.novel_readouts import paired_t90
    arm_of = {0: "untrained", 300: "survival", **{int(u): f"survival_u{u}" for u in snapshots}}
    budgets = sorted(arm_of)
    dev_seeds = sorted(int(r["seed"]) for r in dev["rows"])
    agg = dev["summary"]["aggregate"]
    R = {b: dict(zip(dev_seeds, agg[f"{b} replay_target"]["per_seed"])) for b in budgets
         if f"{b} replay_target" in agg}
    per_budget = {}
    for b in budgets:
        blk = {"arm": arm_of[b]}
        for q in ("S", "mean_err_auth", "mean_err_surr"):
            v = [r["own"][arm_of[b]][q] for r in rows]
            blk[q] = {**paired_t90(v), "per_seed": v}
        if b in R:
            blk["R_stored_fixed_walk_target"] = [R[b].get(int(r["seed"]), float("nan")) for r in rows]
        per_budget[str(b)] = blk
    rb = [b for b in budgets if b in R]
    rho = []
    for r in rows:
        L = [r["own"][arm_of[b]]["mean_err_auth"] for b in rb]
        T = [R[b].get(int(r["seed"]), float("nan")) for b in rb]
        ok = len(rb) >= 3 and np.all(np.isfinite(L)) and np.all(np.isfinite(T))
        rho.append(float(spearmanr(L, T)[0]) if ok else float("nan"))
    out = {"budgets": budgets, "per_budget": per_budget,
           "spearman_L_vs_R": {"budgets": rb, "per_seed": rho, **paired_t90(rho),
                               "note": "descriptive, unthresholded"}}
    if {100, 450} <= set(budgets):
        d = paired_t90(per_budget["450"]["S"]["per_seed"], per_budget["100"]["S"]["per_seed"])
        lo, hi = d["t90"]
        lab = ("not computable" if not np.isfinite(lo) else "detection grows" if lo > 0.05
               else "detection falls" if hi < -0.05 else "detection flat")
        out["S_450_minus_S_100"] = {**d, "label": lab}
    else:
        out["S_450_minus_S_100"] = {"label": "not computable (needs snapshots 100 and 450)"}
    return out


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
    ap.add_argument("--snapshots", type=int, nargs="+", default=None, metavar="U",
                    help="also read the survival_u<U> snapshots (X04; L3 only)")
    ap.add_argument("--snapshot-ref", default=SNAPSHOT_REF,
                    help="stored readout whose own.survival and own.untrained rows gate --snapshots")
    ap.add_argument("--development-ref", default=DEVELOPMENT_REF,
                    help="stored development readout holding the fixed-walk targets (<b> replay_target)")
    ap.add_argument("--dry-run", action="store_true", help="check the agent files, print the plan and exit")
    a = ap.parse_args()
    if a.snapshots:
        if a.drift_mode != "l3":
            ap.error("--snapshots reads the L3 survival snapshots; the L1 runs have none")
        if folds.current_scheme() != "explicit":
            ap.error("--snapshots is read with the explicit fold scheme; unset ITASORL_FOLDS or set it to explicit")
        missing = [p for s in a.seeds for p in snapshot_paths(a.agents_dir, a.drift, s, a.snapshots).values()
                   if not os.path.exists(p)]
        if missing:
            print(f"snapshots {a.snapshots}: missing agent files {missing}", file=sys.stderr)
            return 1
    tasks = [{"agents_dir": a.agents_dir, "seed": s, "drift": a.drift, "drift_mode": a.drift_mode,
              "l1_delta": a.l1_delta, "l1_sigma": a.l1_sigma, "n_eps": a.n_eps, "steps": a.steps,
              "ray_steps": a.ray_steps} for s in a.seeds]
    if a.snapshots:
        tasks = [{**t, "snapshots": list(a.snapshots)} for t in tasks]
    if a.dry_run:
        print(f"{len(tasks)} seed task(s), arms {list(ARMS)} plus snapshots {a.snapshots or []}; would write {a.out}")
        return 0
    from _readout_common import agent_file_hashes, fingerprint_extra, run_tasks
    if a.ckpt:
        arms = [("survival", 0.0) if g == "survival_d0" else (g, a.drift) for g in ARMS]
        fpx = fingerprint_extra(__file__, a.agents_dir, a.seeds, arms)
        for u in a.snapshots or ():
            fpx["agents"].update(agent_file_hashes(a.agents_dir, a.seeds, [("survival", a.drift)], updates=u))
        rows = run_tasks(run_seed, tasks, a.ckpt, lambda t: f"s{t['seed']}", a.workers, fpx)
    elif a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            rows = list(pool.imap_unordered(run_seed, tasks))
    else:
        rows = [run_seed(t) for t in tasks]
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    rows.sort(key=lambda r: r["seed"])
    command = "python " + " ".join(["scripts/run_surprise_readout.py", *sys.argv[1:]])
    inputs = {"agents_dir": a.agents_dir, "drift": a.drift, "drift_mode": a.drift_mode, "seeds": a.seeds,
              "pools": {"authentic": AUTH_BASE, "surrogate": SURR_BASE}, "n_eps": a.n_eps, "steps": a.steps}
    prov = provenance("scripts/run_surprise_readout.py", command, inputs, device="cpu")
    prov["status"] = ("readout under docs/specs/2026-10-15-surprise-dissociation-design.md; "
                      "diagnostic, not an adjudication; changes no registered rule or verdict")
    doc = {"provenance": prov, "fold_scheme": folds.current_scheme(), "summary": summarize(rows), "rows": rows}
    if a.snapshots:
        with open(a.snapshot_ref, encoding="utf-8") as fh:
            fails = snapshot_gate(rows, json.load(fh))
        if fails:
            print("snapshot gate failed; nothing written:\n  " + "\n  ".join(fails), file=sys.stderr)
            return 1
        with open(a.development_ref, encoding="utf-8") as fh:
            dev = json.load(fh)
        from agent_free_common import provenance as provenance_14
        from promote_bv3_gates import repo_rel
        p14 = provenance_14("scripts/run_surprise_readout.py", commit_key="git_commit_at_promotion",
                            output=repo_rel(a.out), device="cpu")
        prov.update({k: v for k, v in p14.items() if k != "generated_by"})
        prov["agents_dir"] = repo_rel(a.agents_dir)
        doc["snapshots"] = {"status": "post hoc, descriptive (X04); changes no stored value or verdict",
                            "snapshot_ref": repo_rel(a.snapshot_ref), "development_ref": repo_rel(a.development_ref),
                            "gate": "PASS", **snapshot_summary(rows, a.snapshots, dev)}
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
