"""What develops with training: the physical trace or the memory?

Frozen design: docs/specs/2026-10-15-what-develops-design.md (with its pre-code amendment:
the battery without its two sequence decoders). Readout-only. One task per (run, seed)
scores the snapshots at 100, 200, 300 and 450 updates and the untrained floor.

    python scripts/run_development_readout.py --run wm=fullruns/corrected_l3_h8_wm/agents \\
        --run nowm=fullruns/corrected_l3_h8_nowm/agents --out-dir artifacts/development
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from _readout_common import AUTH_BASE, DRIFT, SURR_BASE, load, run_tasks, setup_l3, world_params, write_output

SPEC = "docs/specs/2026-10-15-what-develops-design.md"
BUDGETS = (100, 200, 300, 450)
CURVE_KEY = {"wm": "decoder on", "nowm": "decoder off"}


def battery(agent, norm, P, t: dict, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """The control battery's quantities this readout reads, computed by the battery's own
    functions on the registered pools (scripts/run_control_diagnostics.py run_one)."""
    from itasorl.behavior_audit import _trace_phi
    from itasorl.control_diagnostics import history_basis, residual_probe_with_diagnostics
    from itasorl.experiment_b import episode_features, episode_features_full, probe_auroc
    from itasorl.experiment_b2 import collect_pool

    pools = {}
    for lab, dd, base in (("a", 0.0, AUTH_BASE), ("s", DRIFT, SURR_BASE)):
        H, *_, Bt, Ot, At = collect_pool(agent, norm, P, dd, t["n_eps"], t["steps"], "cpu", base,
                                         t["ray_steps"], return_anchors=True, return_obs=True,
                                         return_actions=True)
        pools[lab] = (H, Bt, Ot, At)
    H = np.concatenate([pools["a"][0], pools["s"][0]])
    Bt = np.concatenate([pools["a"][1], pools["s"][1]])
    Ot = np.concatenate([pools["a"][2], pools["s"][2]])
    At = np.concatenate([pools["a"][3], pools["s"][3]])
    y = np.r_[np.zeros(len(pools["a"][0])), np.ones(len(pools["s"][0]))].astype(int)
    au = lambda d: d["auroc"] if isinstance(d, dict) else d  # noqa: E731
    out = {"n": [int(len(pools["a"][0])), int(len(pools["s"][0]))],
           "target": float(probe_auroc(episode_features(H), y)),
           "resid_trace": au(residual_probe_with_diagnostics(H, _trace_phi(Bt, quad=False), y)),
           "resid_obs": au(residual_probe_with_diagnostics(H, history_basis(Ot, lags=1), y)),
           "resid_joint_mlp": au(residual_probe_with_diagnostics(
               H, history_basis(Ot, lags=1, ema_taus=(4, 16), At=At, Bt=Bt), y, model="mlp", seed=seed)),
           "obs_summary_only": float(probe_auroc(episode_features_full(Ot), y))}
    return out, pools["a"][0], pools["s"][0]


def run_task(t: dict) -> dict:
    from itasorl import mechanism_readouts as mr
    from itasorl.eval_protocols import readout_from_states, record_trajectories, replay_states
    from itasorl.experiment_b import episode_features
    from itasorl.experiment_b2 import transfer_probe
    from itasorl.novel_readouts import cosine

    setup_l3()
    P = world_params()
    s, n, T, rs = t["seed"], t["n_eps"], t["steps"], t["ray_steps"]
    t0 = time.time()
    a300, n300 = load(t["agents_dir"], DRIFT, s, "survival", 300)
    Oa, Aa, _ = record_trajectories(a300, n300, P, 0.0, n, T, "cpu", AUTH_BASE, rs)
    Os, As, _ = record_trajectories(a300, n300, P, DRIFT, n, T, "cpu", SURR_BASE, rs)
    out = {"run": t["run"], "seed": s, "budgets": {}}
    feats, dirs = {}, {}
    for b in (0,) + BUDGETS:
        ag, nm = load(t["agents_dir"], DRIFT, s, "untrained") if b == 0 else \
            load(t["agents_dir"], DRIFT, s, "survival", b)
        cell, Ha, Hs = battery(ag, nm, P, t, s)
        cell["replay_target"] = readout_from_states(replay_states(ag, nm, Oa, Aa),
                                                    replay_states(ag, nm, Os, As), seed=s)["target"]
        dirs[b], _ = mr.probe_direction(Ha, Hs)
        feats[b] = (episode_features(np.concatenate([Ha, Hs])),
                    np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int))
        out["budgets"][str(b)] = cell
    for b in BUDGETS:
        out["budgets"][str(b)]["cos_to_300"] = cosine(dirs[300], dirs[b])
        out["budgets"][str(b)]["transfer_from_300"] = float(transfer_probe(*feats[300], *feats[b]))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(rows: list, runs: list) -> dict:
    from itasorl.novel_readouts import paired_t90
    curve = json.load(open("artifacts/budget_curve.json", encoding="utf-8"))["series"]
    out = {}
    for run in runs:
        rr = sorted([r for r in rows if r["run"] == run], key=lambda r: r["seed"])
        get = lambda b, k: [r["budgets"][str(b)][k] for r in rr]  # noqa: E731
        integ = {}
        if run in CURVE_KEY:
            for pt in curve[CURVE_KEY[run]]:
                rec = dict(zip(pt["seeds"], pt["target"]["per_seed"]))
                diffs = [abs(r["budgets"][str(pt["updates"])]["target"] - rec[r["seed"]]) for r in rr]
                integ[str(pt["updates"])] = {"max_abs_diff": float(max(diffs)), "ok": bool(max(diffs) <= 0.001)}
        ok = all(v["ok"] for v in integ.values()) if integ else None
        agg = {f"{b} {k}": {**paired_t90(get(b, k)), "per_seed": get(b, k)}
               for b in (0,) + BUDGETS
               for k in ("target", "resid_trace", "resid_obs", "resid_joint_mlp", "obs_summary_only", "replay_target")}
        for b in BUDGETS:
            for k in ("cos_to_300", "transfer_from_300"):
                agg[f"{b} {k}"] = {**paired_t90(get(b, k)), "per_seed": get(b, k)}
        d_obs = paired_t90(get(450, "resid_obs"), get(100, "resid_obs"))
        d_tr = paired_t90(get(450, "resid_trace"), get(100, "resid_trace"))
        d_t = paired_t90(get(450, "target"), get(100, "target"))
        cos, tr = agg["450 cos_to_300"], agg["450 transfer_from_300"]
        axis = ("one axis sharpens" if cos["mean"] >= 0.7 and cos["t90"][0] > 0.5 and tr["mean"] >= 0.65
                else "reorganizes" if cos["t90"][1] < 0.5 else "partly")
        floor = agg["0 target"]["mean"]
        first = next((b for b in BUDGETS if agg[f"{b} target"]["t90"][0] > floor + 0.05), None)
        out[run] = {"integrity": integ, "integrity_ok": ok, "aggregate": agg, "readings": {
            "1 memory grows beyond the senses (resid_obs 450 - 100, LB > 0.05)": {
                **d_obs, "met": bool(d_obs["t90"][0] > 0.05),
                "only the trace sharpens": bool(d_obs["t90"][0] <= 0.05 and d_t["t90"][0] > 0.05)},
            "target 450 - 100": d_t,
            "2 stored component beyond the trace (resid_trace 450 - 100, LB > 0.05)": {
                **d_tr, "met": bool(d_tr["t90"][0] > 0.05)},
            "3 direction": axis,
            "descriptive: first budget with target LB > untrained mean + 0.05": first}}
    if {"wm", "nowm"} <= set(out):
        by = {r["run"]: {} for r in rows}
        for r in rows:
            by[r["run"]][r["seed"]] = r
        seeds = sorted(set(by["wm"]) & set(by["nowm"]))
        c = {str(b): paired_t90([by["wm"][s]["budgets"][str(b)]["resid_obs"] for s in seeds],
                                [by["nowm"][s]["budgets"][str(b)]["resid_obs"] for s in seeds]) for b in BUDGETS}
        out["4 auxiliary: C1 minus C2 resid_obs"] = {**c, "met at 450": bool(c["450"]["t90"][0] > 0)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="name=agents_dir (wm and nowm)")
    ap.add_argument("--out-dir", default="artifacts/development")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (default OUT_DIR/ckpt)")
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.run)
    tasks = [{"run": nm, "agents_dir": d, "seed": s, "n_eps": a.n_eps, "steps": a.steps,
              "ray_steps": a.ray_steps} for nm, d in runs.items() for s in a.seeds]
    rows = run_tasks(run_task, tasks, a.ckpt or os.path.join(a.out_dir, "ckpt"),
                     lambda t: f"{t['run']}_s{t['seed']}", a.workers)
    summ = summarize(rows, list(runs))
    for nm, d in runs.items():
        name = {"wm": "corrected_l3_h8_wm", "nowm": "corrected_l3_h8_nowm"}.get(nm, nm)
        body = {"spec": SPEC, "run": nm, "summary": summ[nm],
                "auxiliary_contrast": summ.get("4 auxiliary: C1 minus C2 resid_obs"),
                "rows": sorted([r for r in rows if r["run"] == nm], key=lambda r: r["seed"])}
        write_output(os.path.join(a.out_dir, f"{name}.json"), "scripts/run_development_readout.py",
                     sys.argv[1:], {"agents_dir": d, "seeds": a.seeds, "budgets": list(BUDGETS)}, SPEC, body)
        print(nm, "integrity", summ[nm]["integrity_ok"], json.dumps(summ[nm]["readings"], default=float)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
