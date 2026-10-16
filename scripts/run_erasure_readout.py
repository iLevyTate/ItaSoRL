"""The removal test: erase the decoded subspace while the agent acts.

Frozen design: docs/specs/2026-10-15-online-erasure-design.md (with its pre-code amendment:
rank-two erasure). Readout-only.

    python scripts/run_erasure_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --out artifacts/mechanism/erasure/corrected_l3_h8_wm.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import sys
import time

import numpy as np

from _readout_common import AUTH_BASE, DRIFT, SURR_BASE, fingerprint_extra, load, run_tasks, setup_l3, world_params, write_output

SPEC = "docs/specs/2026-10-15-online-erasure-design.md"
WORLD_BASE = 1_400_000      # the nudge readout's rollout base
GAP_MIN = 0.25
WIN = 24                    # the registered feature window


def run_cell(t: dict) -> dict:
    from itasorl import mechanism_readouts as mr
    from itasorl.eval_protocols import readout_from_states
    from itasorl.experiment_b import episode_features
    from itasorl.experiment_b2 import collect_pool, transfer_probe
    from itasorl.novel_readouts import erase, probe_block_directions, projector, sham_projector

    setup_l3()
    P = world_params()
    s, g = t["seed"], t["arm"]
    t0 = time.time()
    ag, nm = load(t["agents_dir"], DRIFT, s, g)
    Ha, _ = collect_pool(ag, nm, P, 0.0, t["pool_n"], WIN, "cpu", AUTH_BASE, t["ray_steps"])
    Hs, _ = collect_pool(ag, nm, P, DRIFT, t["pool_n"], WIN, "cpu", SURR_BASE, t["ray_steps"])
    hid = Ha.shape[-1]
    Pr = projector(probe_block_directions(Ha, Hs), rank=2)
    Pv = sham_projector(Pr, 2, s)
    m = 0.5 * (Ha.reshape(-1, hid).mean(0) + Hs.reshape(-1, hid).mean(0))
    # sanity: erasing the pools themselves pins every linear score in the subspace
    resid_pool = float(np.abs((erase(Ha, Pr, m) - m) @ Pr).max())

    def roll(drift, ers=None):
        return mr.rollout_behavior(ag, nm, P, drift, n_eps=t["n_eps"], steps=t["steps"],
                                   seed_base=WORLD_BASE, ray_steps=t["ray_steps"], erase=ers)

    v = {"a": roll(0.0), "s": roll(DRIFT), "ea": roll(0.0, (Pr, m)), "es": roll(DRIFT, (Pr, m)),
         "sa": roll(0.0, (Pv, m)), "ss": roll(DRIFT, (Pv, m))}
    common = set(v["a"]["kept"].tolist())
    for x in v.values():
        common &= set(x["kept"].tolist())
    common = sorted(common)
    # Every attempted episode, before the common-survivor intersection (third amendment,
    # review N04): termination count, mean return summed through termination, mean steps.
    attempted = {"n_attempted": int(t["n_eps"]),
                 "terminated": {k: int(x["terminated"].sum()) for k, x in v.items()},
                 "pop_return": {k: float(x["ret_all"].mean()) for k, x in v.items()},
                 "pop_steps_alive": {k: float(x["steps_alive"].mean()) for k, x in v.items()}}

    def sub(x, key):
        idx = [list(x["kept"]).index(i) for i in common]
        return x[key][idx]

    B = {k: sub(x, "B") for k, x in v.items()}
    R = {k: sub(x, "ret") for k, x in v.items()}
    Hx = {k: sub(x, "H") for k, x in v.items()}
    out = {"seed": s, "arm": g, "n_common": len(common), "pool_resid_max": resid_pool, **attempted}
    if len(common) < 5:
        out.update(informative=False, reason="fewer than 5 common survivors",
                   seconds=round(time.time() - t0, 1))
        return out
    scale = mr.behavior_scale(B["a"], B["s"])
    real_s, gap = mr.gap_closed(B["s"], B["a"], B["es"], scale)
    sham_s, _ = mr.gap_closed(B["s"], B["a"], B["ss"], scale)
    real_a, _ = mr.gap_closed(B["a"], B["s"], B["ea"], scale)
    sham_a, _ = mr.gap_closed(B["a"], B["s"], B["sa"], scale)
    ii = mr.BEHAVIOR_NAMES.index("intake_rate")
    Xtr = episode_features(np.concatenate([Ha, Hs]))
    ytr = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    yte = np.r_[np.zeros(len(common)), np.ones(len(common))].astype(int)

    from sklearn.metrics import roc_auc_score
    _, _, p_ref = transfer_probe(Xtr, ytr, episode_features(np.concatenate(
        [Hx["a"][:, :WIN], Hx["s"][:, :WIN]])), yte, return_scores=True)
    sd_ref = float(np.std(p_ref))

    def frozen(ka, ks):
        """Frozen-probe AUROC with scores quantized to 1/1000 of the unerased rollouts' score
        spread, so a pinned score (equal up to float32 noise) reads as the tie it is, 0.5,
        rather than a ranking of rounding noise; and the score spread relative to the
        unerased rollouts."""
        _, _, p = transfer_probe(Xtr, ytr, episode_features(np.concatenate(
            [Hx[ka][:, :WIN], Hx[ks][:, :WIN]])), yte, return_scores=True)
        q = np.round(p / (1e-3 * sd_ref)) if sd_ref > 0 else p
        return float(roc_auc_score(yte, q)), float(np.std(p) / sd_ref) if sd_ref > 0 else float("nan")

    out.update({
        "gap": gap, "informative": bool(np.isfinite(gap) and gap >= GAP_MIN),
        "surr_real": real_s, "surr_sham": sham_s, "auth_real": real_a, "auth_sham": sham_a,
        "surr_real_minus_sham": real_s - sham_s, "auth_real_minus_sham": real_a - sham_a,
        "primary": 0.5 * ((real_s - sham_s) + (real_a - sham_a)),
        "return": {k: float(R[k].mean()) for k in R},
        "intake": {k: float(B[k][:, ii].mean()) for k in B},
        "frozen_probe_unerased": frozen("a", "s")[0], "frozen_probe_erased": frozen("ea", "es")[0],
        "frozen_probe_sham": frozen("sa", "ss")[0],
        "score_spread_erased_over_unerased": frozen("ea", "es")[1],
        "score_spread_sham_over_unerased": frozen("sa", "ss")[1],
        "refit_probe_erased": readout_from_states(Hx["ea"][:, :WIN], Hx["es"][:, :WIN], seed=s)["target"],
        "seconds": round(time.time() - t0, 1)})
    return out


def summarize(rows: list) -> dict:
    from itasorl.novel_readouts import paired_t90
    out = {}
    for g in ("survival", "untrained"):
        rr = [r for r in rows if r["arm"] == g]
        inf = [r for r in rr if r.get("informative")]
        f = lambda key: paired_t90([r[key] for r in inf])  # noqa: E731
        conds = ("a", "s", "ea", "es", "sa", "ss")
        blk = {"n_informative": len(inf), "scope": "common survivors across all six conditions "
                                                   "(behavior, return and probe readings below)",
               "attrition": {
                   "excluded_fraction": paired_t90([(r["n_attempted"] - r["n_common"]) / r["n_attempted"]
                                                    for r in rr if "n_attempted" in r]),
                   "terminated_per_condition": {k: paired_t90([r["terminated"][k] for r in rr if "terminated" in r])
                                                for k in conds},
                   "note": "attempted episodes per condition and how many ended before the horizon; "
                           "the common-survivor readings drop every episode that ended in any condition"},
               "population": {
                   "surr_return_erased_minus_sham": paired_t90([r["pop_return"]["es"] for r in rr if "pop_return" in r],
                                                               [r["pop_return"]["ss"] for r in rr if "pop_return" in r]),
                   "auth_return_erased_minus_sham": paired_t90([r["pop_return"]["ea"] for r in rr if "pop_return" in r],
                                                               [r["pop_return"]["sa"] for r in rr if "pop_return" in r]),
                   "surr_terminated_erased_minus_sham": paired_t90([r["terminated"]["es"] for r in rr if "terminated" in r],
                                                                   [r["terminated"]["ss"] for r in rr if "terminated" in r]),
                   "auth_terminated_erased_minus_sham": paired_t90([r["terminated"]["ea"] for r in rr if "terminated" in r],
                                                                   [r["terminated"]["sa"] for r in rr if "terminated" in r]),
                   "note": "every attempted episode, return summed through termination; descriptive, "
                           "no pre-stated reading (third amendment)"},
               "primary": f("primary"),
               "surr_real_minus_sham": f("surr_real_minus_sham"),
               "auth_real_minus_sham": f("auth_real_minus_sham"),
               "surr_return_erased_minus_sham": paired_t90([r["return"]["es"] for r in inf],
                                                           [r["return"]["ss"] for r in inf]),
               "auth_return_erased_minus_sham": paired_t90([r["return"]["ea"] for r in inf],
                                                           [r["return"]["sa"] for r in inf]),
               "frozen_probe_erased": paired_t90([r.get("frozen_probe_erased", np.nan) for r in rr]),
               "frozen_probe_unerased": paired_t90([r.get("frozen_probe_unerased", np.nan) for r in rr]),
               "refit_probe_erased": paired_t90([r.get("refit_probe_erased", np.nan) for r in rr])}
        held = sum(abs(r.get("frozen_probe_erased", np.nan) - 0.5) < 0.1 for r in rr)
        enough = len(inf) >= 7
        p, sw, aw = blk["primary"], blk["surr_real_minus_sham"], blk["auth_real_minus_sham"]
        blk["readings"] = {
            "enough informative seeds (>= 7)": enough,
            "1 used under removal": bool(enough and p["mean"] >= 0.10 and p["t90"][0] > 0),
            "2 costly (surrogate return erased minus sham, UB < 0)":
                bool(enough and blk["surr_return_erased_minus_sham"]["t90"][1] < 0),
            "3 not used under removal": bool(enough and -0.10 <= sw["t90"][0] and sw["t90"][1] <= 0.10
                                             and -0.10 <= aw["t90"][0] and aw["t90"][1] <= 0.10),
            "4 erasure held (frozen probe within 0.1 of 0.5 in >= 8 seeds)": {"seeds": int(held),
                                                                              "met": bool(held >= 8)},
            "partial erasure (refit probe >= 0.65)": bool(blk["refit_probe_erased"]["mean"] >= 0.65)}
        out[g] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--arms", nargs="+", default=["survival", "untrained"])
    ap.add_argument("--n-eps", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--pool-n", type=int, default=110)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    ckpt = a.ckpt or a.out + ".ckpt"
    tasks = [{"agents_dir": a.agents_dir, "seed": s, "arm": g, "n_eps": a.n_eps, "steps": a.steps,
              "pool_n": a.pool_n, "ray_steps": a.ray_steps} for g in a.arms for s in a.seeds]
    rows = sorted(run_tasks(run_cell, tasks, ckpt, lambda t: f"{t['arm']}_s{t['seed']}", a.workers,
                            fingerprint_extra(__file__, a.agents_dir, a.seeds, a.arms)),
                  key=lambda r: (r["arm"], r["seed"]))
    body = {"spec": SPEC, "summary": summarize(rows), "rows": rows}
    write_output(a.out, "scripts/run_erasure_readout.py", sys.argv[1:],
                 {"agents_dir": a.agents_dir, "seeds": a.seeds, "world_base": WORLD_BASE,
                  "n_eps": a.n_eps, "steps": a.steps}, SPEC, body)
    for k, v in body["summary"]["survival"]["readings"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
