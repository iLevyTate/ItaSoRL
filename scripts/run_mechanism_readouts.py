"""Mechanism readouts 6 to 9 of the goal-and-stakes spec on one run's saved trained arm,
seed by seed, plus the scripted-walk stream decoder (the missing cell of FINDINGS 17.8)
and the value of world information from the run's cells.

    python scripts/run_mechanism_readouts.py --run-dir artifacts/goal_stakes/T-touch \
        --out artifacts/goal_stakes/T-touch/mechanism.json --workers 4

Fresh evaluation worlds use one seed base, WORLD_BASE + i, shared by the authentic and the
surrogate rollouts (the spec compares unnudged authentic and unnudged surrogate episodes on
the same world seeds) and disjoint from the pooled (800000 / 850000) and matched-pair bases.
Reads every number from the run's cells and agents; trains nothing except the shared L3
velocity law G_motion, which each worker process retrains once from the run's knobs (the
same frozen recipe run_expB2 used, so it is the run's surrogate world).

Survivor pairing: a nudge can change which episodes die, so the intervention block (readout 6)
is taken on the episodes that survived in ALL six rollout variants (`n_common`; the per-world
counts before the cross-world intersection are `n_common_auth` / `n_common_surr`), and the
behavior and adaptation blocks (readouts 7 and 9) on the episodes that survived in both
unnudged rollouts (`n_paired`). The behavior scale is computed on the rows it standardizes.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np

WORLD_BASE = 1_400_000
NUDGE_SD = 1.0          # spec: +1.0 s u
GAP_MIN = 0.25          # spec: a seed with a smaller standardized behavior gap is uninformative
ARMS = ("survival", "untrained", "predictor")
FLOOR_ARMS = ("untrained", "predictor")


def common_rows(*rollouts: dict, key: str = "B") -> list[np.ndarray]:
    """Each rollout's `key` array (default `B`) restricted to the episodes (world seed
    indices in `kept`) that survived in EVERY rollout, rows in ascending episode order, so
    row j of every output is the same world. Empty (0, ...) arrays when nothing is shared."""
    common = set(int(e) for e in rollouts[0]["kept"])
    for r in rollouts[1:]:
        common &= set(int(e) for e in r["kept"])
    idx = sorted(common)
    out = []
    for r in rollouts:
        pos = {int(e): j for j, e in enumerate(r["kept"])}
        out.append(r[key][[pos[e] for e in idx]] if idx else r[key][:0])
    return out


def _n_common(*rollouts: dict) -> int:
    return len(common_rows(*rollouts)[0])


def _cfg_from_cells(run_dir: str) -> dict:
    cells = {}
    for p in glob.glob(os.path.join(run_dir, "cells", "cell_d*_s*.json")):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        cells[(float(c["drift"]), int(c["seed"]))] = c
    if not cells:
        raise SystemExit(f"no cells under {run_dir}")
    return cells


def _finite_mean(a: float, b: float) -> float:
    return float(0.5 * (a + b)) if np.isfinite(a) and np.isfinite(b) else float("nan")


def _nan_intervention(gap: float, n_auth: int, n_surr: int, n_common: int, u: np.ndarray,
                      s: float, reason: str) -> dict:
    nan = float("nan")
    return {"score_real": nan, "score_sham": nan, "gap": gap,
            "score_real_reverse": nan, "score_sham_reverse": nan,
            "informative": False, "reason": reason, "nudge_sd": NUDGE_SD, "s": s,
            "u": u.tolist(), "v": None,
            "n_common_auth": int(n_auth), "n_common_surr": int(n_surr), "n_common": int(n_common)}


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
    knobs = k["knobs"]
    b2.DRIFT_MODE = "l3"
    b2.OBJECTIVE = k["objective"]
    b2.MORTAL = k["mortal"]
    if knobs.get("n_pellets") is not None:
        b2.SURVIVAL_FOOD["n_pellets"] = knobs["n_pellets"]
    if knobs.get("reach") is not None:
        b2.SURVIVAL_FOOD["reach"] = knobs["reach"]
    if knobs.get("basal_e") is not None:
        b2.SURVIVAL_METAB["basal_E"] = knobs["basal_e"]
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=knobs.get("l3_hidden") or 8, device="cpu",
                              seed=knobs.get("l3_seed") or 0, params=P)
    agents_dir = os.path.join(k["run_dir"], "agents")
    arms = {g: load_agent_bundle(os.path.join(agents_dir, f"agent_d{format_drift(d)}_s{seed}_{g}.pt"))
            for g in ARMS}
    pool_n, pool_steps, ray_steps = k["pool_n"], k["pool_steps"], k["ray_steps"]
    n_eps, roll_steps = k["n_eps"], k["steps"]
    t0 = time.time()
    out = {"seed": seed, "drift": d}

    def roll(ag, nm, drift, nudge=None):
        return mr.rollout_behavior(ag, nm, P, drift, n_eps=n_eps, steps=roll_steps,
                                   seed_base=WORLD_BASE, ray_steps=ray_steps, nudge=nudge)

    for g in ARMS:
        ag, nm = arms[g]
        Ha, _ = collect_pool(ag, nm, P, 0.0, pool_n, pool_steps, "cpu", 800_000, ray_steps)
        Hs, _ = collect_pool(ag, nm, P, d, pool_n, pool_steps, "cpu", 850_000, ray_steps)
        u, s = mr.probe_direction(Ha, Hs)
        ra = roll(ag, nm, 0.0)
        rsu = roll(ag, nm, d)
        # readouts 7 and 9: the two unnudged rollouts, paired on shared surviving worlds
        Pa, Ps = common_rows(ra, rsu)
        Hpa, Hps = common_rows(ra, rsu, key="halves")
        n_paired = int(len(Pa))
        pair_scale = mr.behavior_scale(Pa, Ps)
        if np.isfinite(s):
            v = mr.sham_direction(u, seed)
            rn = roll(ag, nm, 0.0, nudge=NUDGE_SD * s * u)
            rsh = roll(ag, nm, 0.0, nudge=NUDGE_SD * s * v)
            rnr = roll(ag, nm, d, nudge=-NUDGE_SD * s * u)
            rshr = roll(ag, nm, d, nudge=-NUDGE_SD * s * v)
            # readout 6: every variant on the episodes that survived in all six rollouts
            Ba, Bs, Bn, Bsh, Bnr, Bshr = common_rows(ra, rsu, rn, rsh, rnr, rshr)
            scale = mr.behavior_scale(Ba, Bs)
            real, gap = mr.gap_closed(Ba, Bs, Bn, scale)
            sham, _ = mr.gap_closed(Ba, Bs, Bsh, scale)
            real_r, _ = mr.gap_closed(Bs, Ba, Bnr, scale)
            sham_r, _ = mr.gap_closed(Bs, Ba, Bshr, scale)
            intervention = {"score_real": real, "score_sham": sham, "gap": gap,
                            "score_real_reverse": real_r, "score_sham_reverse": sham_r,
                            "informative": bool(np.isfinite(gap) and gap >= GAP_MIN),
                            "nudge_sd": NUDGE_SD, "s": s, "u": u.tolist(), "v": v.tolist(),
                            "n_common_auth": _n_common(ra, rn, rsh),
                            "n_common_surr": _n_common(rsu, rnr, rshr),
                            "n_common": int(len(Ba))}
            if not intervention["informative"]:
                intervention["reason"] = "gap below threshold" if np.isfinite(gap) else "empty pool"
        else:
            _, gap = mr.gap_closed(Pa, Ps, Pa, pair_scale)
            intervention = _nan_intervention(gap, len(ra["B"]), len(rsu["B"]), n_paired, u, s,
                                             "degenerate probe")
        corr_auth = mr.direction_error_correlation(ra["H"], ra["E"], u)
        corr_surr = mr.direction_error_correlation(rsu["H"], rsu["E"], u)
        block = {
            "intervention": intervention,
            "behavior": {"difference": mr.behavior_difference(Pa, Ps, pair_scale),
                         "auth_mean": Pa.mean(0).tolist() if n_paired else [float("nan")] * Pa.shape[1],
                         "surr_mean": Ps.mean(0).tolist() if n_paired else [float("nan")] * Ps.shape[1],
                         "n_paired": n_paired,
                         "n_auth": int(len(ra["B"])), "n_surr": int(len(rsu["B"])),
                         "names": list(mr.BEHAVIOR_NAMES)},
            "surprise": {"auroc": mr.surprise_auroc(ra["E"], rsu["E"]) if ag.world_model else float("nan"),
                         "corr_auth": corr_auth, "corr_surr": corr_surr,
                         "corr_mean": _finite_mean(corr_auth, corr_surr)},
            "adaptation": {**mr.adaptation(Hpa, Hps), "n_paired": n_paired},
        }
        if g == "survival":
            out.update(block)
        else:
            out[f"{g}_arm"] = block

    nm = arms["survival"][1]
    Oa, Aa = mr.scripted_observation_streams(nm, P, 0.0, n_eps=pool_n, steps=pool_steps,
                                             seed_base=800_000, ray_steps=ray_steps)
    Os, As = mr.scripted_observation_streams(nm, P, d, n_eps=pool_n, steps=pool_steps,
                                             seed_base=850_000, ray_steps=ray_steps)
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

    def block_summary(get):
        """The seed-paired summaries of one arm's block (the trained arm or a floor)."""
        def real_minus_sham(b, fwd=True):
            i = b["intervention"]
            if not i["informative"]:
                return np.nan
            return (i["score_real"] - i["score_sham"]) if fwd else \
                (i["score_real_reverse"] - i["score_sham_reverse"])
        return {
            "n_informative": sum(bool(get(r)["intervention"]["informative"]) for r in rows),
            "intervention_real_minus_sham": paired(lambda r: real_minus_sham(get(r))),
            "intervention_reverse_real_minus_sham": paired(lambda r: real_minus_sham(get(r), False)),
            "surprise_auroc": paired(lambda r: get(r)["surprise"]["auroc"]),
            "surprise_corr": paired(lambda r: get(r)["surprise"]["corr_mean"]),
            "adaptation_gap_first": paired(lambda r: get(r)["adaptation"]["gap_first"]),
            "adaptation": paired(lambda r: get(r)["adaptation"]["adaptation"]),
        }

    return {
        "n_seeds": len(rows),
        **block_summary(lambda r: r),
        "behavior_difference": {name: paired(lambda r, nm=name: r["behavior"]["difference"][nm])
                                for name in rows[0]["behavior"]["names"]},
        "scripted_stream_auroc": paired(lambda r: r["scripted_stream"]["auroc"]),
        "value_of_world_information": paired(lambda r: r["value_of_world_information"]["value_of_information"]),
        "floors": {g: block_summary(lambda r, g=g: r[f"{g}_arm"]) for g in FLOOR_ARMS},
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
    have = sorted({s for d, s in cells if d == dmax})
    seeds = have if a.seeds is None else list(a.seeds)
    missing = [s for s in seeds if s not in have]
    if missing:
        raise SystemExit(f"no cell at drift {dmax:.2f} for requested seed(s) {missing}; "
                         f"cells exist for seeds {have}")
    # run knobs live in the cell record ("knobs", written by run_expB2 since the goal-and-stakes
    # revision); older cells lack it and fall back to the module defaults
    knobs = any_cell.get("knobs") or {}
    if knobs.get("drift_mode") not in (None, "l3"):
        raise SystemExit(f"run was trained with drift_mode={knobs['drift_mode']!r}; the mechanism "
                         "readouts are defined for the L3 (learned velocity law) surrogate only")
    if knobs.get("l3_family") not in (None, "gmotion"):
        raise SystemExit(f"run was trained with l3_family={knobs['l3_family']!r}; the mechanism "
                         "readouts rebuild only the default G_motion surrogate")
    task_base = {"run_dir": a.run_dir, "drift": dmax, "n_eps": a.n_eps, "steps": a.steps,
                 "pool_n": a.pool_n, "pool_steps": a.pool_steps, "ray_steps": a.ray_steps,
                 "gru_epochs": a.gru_epochs, "objective": any_cell.get("objective", "survival"),
                 "mortal": bool(any_cell.get("mortal", True)), "knobs": knobs,
                 "cells": {f"{d:.2f}/{s}": {"xeval": c["xeval"]} for (d, s), c in cells.items()}}
    tasks = [{**task_base, "seed": s} for s in seeds]
    if a.workers > 1:
        ctx = mp.get_context("spawn")            # spawn: safe with torch, as run_expB2 does
        with ctx.Pool(a.workers) as pool:
            rows = pool.map(run_seed, tasks)
    else:
        rows = [run_seed(t) for t in tasks]
    rows.sort(key=lambda r: r["seed"])
    out = {"run_dir": a.run_dir, "drift": dmax, "objective": task_base["objective"],
           "mortal": task_base["mortal"], "knobs": knobs,
           "seed_bases": {"world": WORLD_BASE,
                          "note": "authentic and surrogate share world seeds; disjoint from the "
                                  "800000/850000 pools and the matched-pair bases"},
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
