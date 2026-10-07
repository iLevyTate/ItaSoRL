"""Mechanism readouts 6 to 9 of the goal-and-stakes spec on one run's saved trained arm,
seed by seed, plus the scripted-walk stream decoder (the missing cell of FINDINGS 17.8)
and the value of world information from the run's cells.

    python scripts/run_mechanism_readouts.py --run-dir artifacts/goal_stakes/T-touch \
        --out artifacts/goal_stakes/T-touch/mechanism.json --workers 4

Fresh evaluation worlds use seed bases disjoint from the pooled (800000 / 850000) and
matched-pair bases: authentic 1_400_000 + i, surrogate 1_450_000 + i. Reads every number
from the run's cells and agents; trains nothing.

Survivor pairing: a nudge can change which episodes die, so every comparison between
variants of the same world (plain, real nudge, sham nudge) is taken on the intersection
of their surviving episodes (`common_rows`). Authentic and surrogate worlds use different
seed bases, so each world is intersected among its own variants and the two means are
then compared; `n_common_auth` / `n_common_surr` record the episode counts used.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np

AUTH_BASE, SURR_BASE = 1_400_000, 1_450_000
NUDGE_SD = 1.0          # spec: +1.0 s u
GAP_MIN = 0.25          # spec: a seed with a smaller standardized behavior gap is uninformative


def common_rows(*rollouts: dict) -> list[np.ndarray]:
    """Each rollout's `B` restricted to the episodes (world seed indices in `kept`) that
    survived in EVERY rollout, rows in ascending episode order, so row j of every output
    is the same world. Empty (0, 7) arrays when nothing is shared."""
    common = set(int(e) for e in rollouts[0]["kept"])
    for r in rollouts[1:]:
        common &= set(int(e) for e in r["kept"])
    idx = sorted(common)
    out = []
    for r in rollouts:
        pos = {int(e): j for j, e in enumerate(r["kept"])}
        out.append(r["B"][[pos[e] for e in idx]] if idx else r["B"][:0])
    return out


def _cfg_from_cells(run_dir: str) -> dict:
    cells = {}
    for p in glob.glob(os.path.join(run_dir, "cells", "cell_d*_s*.json")):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        cells[(float(c["drift"]), int(c["seed"]))] = c
    if not cells:
        raise SystemExit(f"no cells under {run_dir}")
    return cells


def _nan_intervention(gap: float, n_auth: int, n_surr: int, s: float, reason: str) -> dict:
    nan = float("nan")
    return {"score_real": nan, "score_sham": nan, "gap": gap,
            "score_real_reverse": nan, "score_sham_reverse": nan,
            "informative": False, "reason": reason, "nudge_sd": NUDGE_SD, "s": s,
            "n_common_auth": int(n_auth), "n_common_surr": int(n_surr)}


def run_seed(task: dict) -> dict:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl import mechanism_readouts as mr
    from itasorl.control_diagnostics import sequence_gru_auroc
    from itasorl.experiment_b2 import collect_pool, format_drift, load_agent_bundle
    from itasorl.experiment_c import value_of_world_information
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    k, seed, d = task, task["seed"], task["drift"]
    b2.DRIFT_MODE = "l3"
    b2.OBJECTIVE = k["objective"]
    b2.MORTAL = k["mortal"]
    if k.get("n_pellets") is not None:
        b2.SURVIVAL_FOOD["n_pellets"] = k["n_pellets"]
    if k.get("basal_e") is not None:
        b2.SURVIVAL_METAB["basal_E"] = k["basal_e"]
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=8, device="cpu", seed=0, params=P)
    agents_dir = os.path.join(k["run_dir"], "agents")
    arms = {g: load_agent_bundle(os.path.join(agents_dir, f"agent_d{format_drift(d)}_s{seed}_{g}.pt"))
            for g in ("untrained", "predictor", "survival")}
    n, steps, rs, eps, T = k["pool_n"], k["pool_steps"], k["ray_steps"], k["n_eps"], k["steps"]
    t0 = time.time()
    out = {"seed": seed, "drift": d}

    for g in ("survival", "untrained", "predictor"):
        ag, nm = arms[g]
        Ha, _ = collect_pool(ag, nm, P, 0.0, n, steps, "cpu", 800_000, rs)
        Hs, _ = collect_pool(ag, nm, P, d, n, steps, "cpu", 850_000, rs)
        u, s = mr.probe_direction(Ha, Hs)
        ra = mr.rollout_behavior(ag, nm, P, 0.0, n_eps=eps, steps=T, seed_base=AUTH_BASE, ray_steps=rs)
        rsu = mr.rollout_behavior(ag, nm, P, d, n_eps=eps, steps=T, seed_base=SURR_BASE, ray_steps=rs)
        if np.isfinite(s):
            v = mr.sham_direction(u, seed)
            rn = mr.rollout_behavior(ag, nm, P, 0.0, n_eps=eps, steps=T, seed_base=AUTH_BASE,
                                     ray_steps=rs, nudge=NUDGE_SD * s * u)
            rsh = mr.rollout_behavior(ag, nm, P, 0.0, n_eps=eps, steps=T, seed_base=AUTH_BASE,
                                      ray_steps=rs, nudge=NUDGE_SD * s * v)
            rnr = mr.rollout_behavior(ag, nm, P, d, n_eps=eps, steps=T, seed_base=SURR_BASE,
                                      ray_steps=rs, nudge=-NUDGE_SD * s * u)
            rshr = mr.rollout_behavior(ag, nm, P, d, n_eps=eps, steps=T, seed_base=SURR_BASE,
                                       ray_steps=rs, nudge=-NUDGE_SD * s * v)
            # pair survivors within each world: the authentic variants among themselves,
            # the surrogate variants among themselves (different seed bases across worlds)
            Ba, Bn, Bsh = common_rows(ra, rn, rsh)
            Bs, Bnr, Bshr = common_rows(rsu, rnr, rshr)
            scale = mr.behavior_scale(Ba, Bs)
            real, gap = mr.gap_closed(Ba, Bs, Bn, scale)
            sham, _ = mr.gap_closed(Ba, Bs, Bsh, scale)
            real_r, _ = mr.gap_closed(Bs, Ba, Bnr, scale)
            sham_r, _ = mr.gap_closed(Bs, Ba, Bshr, scale)
            intervention = {"score_real": real, "score_sham": sham, "gap": gap,
                            "score_real_reverse": real_r, "score_sham_reverse": sham_r,
                            "informative": bool(np.isfinite(gap) and gap >= GAP_MIN),
                            "nudge_sd": NUDGE_SD, "s": s,
                            "n_common_auth": int(len(Ba)), "n_common_surr": int(len(Bs))}
            if not intervention["informative"]:
                intervention["reason"] = "gap below threshold" if np.isfinite(gap) else "empty pool"
        else:
            Ba, Bs = ra["B"], rsu["B"]
            scale = mr.behavior_scale(Ba, Bs)
            _, gap = mr.gap_closed(Ba, Bs, Ba, scale)
            intervention = _nan_intervention(gap, len(Ba), len(Bs), s, "degenerate probe")
        block = {
            "intervention": intervention,
            "behavior": {"difference": mr.behavior_difference(Ba, Bs, scale),
                         "auth_mean": Ba.mean(0).tolist() if len(Ba) else [float("nan")] * Ba.shape[1],
                         "surr_mean": Bs.mean(0).tolist() if len(Bs) else [float("nan")] * Bs.shape[1],
                         "n_auth": int(len(ra["B"])), "n_surr": int(len(rsu["B"])),
                         "names": list(mr.BEHAVIOR_NAMES)},
            "surprise": {"auroc": mr.surprise_auroc(ra["E"], rsu["E"]) if ag.world_model else float("nan"),
                         "corr_auth": mr.direction_error_correlation(ra["H"], ra["E"], u),
                         "corr_surr": mr.direction_error_correlation(rsu["H"], rsu["E"], u)},
            "adaptation": mr.adaptation(ra["halves"], rsu["halves"]),
        }
        block["surprise"]["corr_mean"] = float(np.nanmean([block["surprise"]["corr_auth"],
                                                           block["surprise"]["corr_surr"]]))
        if g == "survival":
            out.update(block)
        else:
            out[f"{g}_arm"] = block

    nm = arms["survival"][1]
    Oa, Aa = mr.scripted_observation_streams(nm, P, 0.0, n_eps=n, steps=steps, seed_base=800_000, ray_steps=rs)
    Os, As = mr.scripted_observation_streams(nm, P, d, n_eps=n, steps=steps, seed_base=850_000, ray_steps=rs)
    seq = np.concatenate([np.concatenate([Oa, Aa], -1), np.concatenate([Os, As], -1)])
    y = np.r_[np.zeros(len(Oa)), np.ones(len(Os))].astype(int)
    out["scripted_stream"] = sequence_gru_auroc(seq, y, epochs=k["gru_epochs"], seed=seed)

    cells = k["cells"]
    out["value_of_world_information"] = value_of_world_information(
        cells[f"0.00/{seed}"]["xeval"], cells[f"{d:.2f}/{seed}"]["xeval"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(rows: list[dict]) -> dict:
    from itasorl.stats import t_ci90

    def paired(key_fn):
        v = np.array([key_fn(r) for r in rows], float)
        v = v[np.isfinite(v)]
        return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else float("nan"),
                "ci90": list(t_ci90(v)) if len(v) > 1 else [float("nan")] * 2}

    inf = [r for r in rows if r["intervention"]["informative"]]
    return {
        "n_seeds": len(rows), "n_informative": len(inf),
        "intervention_real_minus_sham": paired(lambda r: (r["intervention"]["score_real"] - r["intervention"]["score_sham"])
                                               if r["intervention"]["informative"] else np.nan),
        "intervention_reverse_real_minus_sham": paired(lambda r: (r["intervention"]["score_real_reverse"] - r["intervention"]["score_sham_reverse"])
                                                       if r["intervention"]["informative"] else np.nan),
        "behavior_difference": {name: paired(lambda r, nm=name: r["behavior"]["difference"][nm])
                                for name in rows[0]["behavior"]["names"]},
        "surprise_auroc": paired(lambda r: r["surprise"]["auroc"]),
        "surprise_corr": paired(lambda r: r["surprise"]["corr_mean"]),
        "adaptation_gap_first": paired(lambda r: r["adaptation"]["gap_first"]),
        "adaptation": paired(lambda r: r["adaptation"]["adaptation"]),
        "scripted_stream_auroc": paired(lambda r: r["scripted_stream"]["auroc"]),
        "value_of_world_information": paired(lambda r: r["value_of_world_information"]["value_of_information"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-eps", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--pool-n", type=int, default=110)
    ap.add_argument("--pool-steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--gru-epochs", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_eps, a.steps, a.pool_n, a.pool_steps, a.gru_epochs = 6, 12, 8, 6, 2
    cells = _cfg_from_cells(a.run_dir)
    dmax = max(d for d, _ in cells)
    any_cell = next(iter(cells.values()))
    seeds = sorted({s for _, s in cells}) if a.seeds is None else a.seeds
    # run knobs live in the cell record ("knobs", written by run_expB2 since the goal-and-stakes
    # revision); older cells lack it and fall back to the module defaults
    knobs = any_cell.get("knobs") or {}
    task_base = {"run_dir": a.run_dir, "drift": dmax, "n_eps": a.n_eps, "steps": a.steps,
                 "pool_n": a.pool_n, "pool_steps": a.pool_steps, "ray_steps": a.ray_steps,
                 "gru_epochs": a.gru_epochs, "objective": any_cell.get("objective", "survival"),
                 "mortal": bool(any_cell.get("mortal", True)),
                 "n_pellets": knobs.get("n_pellets"), "basal_e": knobs.get("basal_e"),
                 "cells": {f"{d:.2f}/{s}": {"xeval": c["xeval"]} for (d, s), c in cells.items()}}
    tasks = [{**task_base, "seed": s} for s in seeds]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            rows = pool.map(run_seed, tasks)
    else:
        rows = [run_seed(t) for t in tasks]
    rows.sort(key=lambda r: r["seed"])
    out = {"run_dir": a.run_dir, "drift": dmax, "objective": task_base["objective"],
           "mortal": task_base["mortal"], "knobs": knobs,
           "seed_bases": {"auth": AUTH_BASE, "surr": SURR_BASE},
           "settings": {"n_eps": a.n_eps, "steps": a.steps, "pool_n": a.pool_n,
                        "pool_steps": a.pool_steps, "nudge_sd": NUDGE_SD, "gap_min": GAP_MIN},
           "per_seed": rows, "summary": summarize(rows)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    s = out["summary"]
    print(f"seeds={s['n_seeds']} informative={s['n_informative']}  "
          f"intervention real-sham={s['intervention_real_minus_sham']['mean']:+.3f} "
          f"surprise auroc={s['surprise_auroc']['mean']:.3f} corr={s['surprise_corr']['mean']:+.3f}  "
          f"adaptation={s['adaptation']['mean']:+.4f}  scripted stream={s['scripted_stream_auroc']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
