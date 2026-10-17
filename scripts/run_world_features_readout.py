"""Which features of its own world does the state keep? Instrumental against incidental.

Frozen design: docs/specs/2026-10-15-world-draw-features-design.md (with its pre-code
amendment). Readout-only. Two phases: the reach radius from the headline agent's
calibration episodes pooled over seeds, then the per-seed readout at that radius.

    python scripts/run_world_features_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --out artifacts/world_features/corrected_l3_h8_wm.json

`--protocol own` (closing item G07, post hoc) runs the spec's secondary protocol: every arm is
read on its own episodes at the same seed base instead of on the headline agent's replayed
episodes. Each row then carries per-arm counts, feature summaries and echo, and every recorded
episode's feature values (H24). The default, replay, is the registered protocol and its
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

from _readout_common import DRIFT, fingerprint_extra, load, run_tasks, setup_l3, world_params, write_output

SPEC = "docs/specs/2026-10-15-world-draw-features-design.md"
READ_BASE = 2_600_000       # listed in itasorl.post_review.BASES_IN_USE
CALIB_BASE = 2_650_000
ARMS = (("survival", DRIFT), ("predictor", DRIFT), ("untrained", DRIFT), ("survival", 0.0))
PROTOCOLS = ("replay", "own")


def record_with_worlds(agent, norm, P, n_eps: int, steps: int, seed_base: int, ray_steps: int,
                       r: float | None):
    """The authentic-world (drift 0) episodes of eval_protocols.record_trajectories, with the
    draw features computed at reset (no random draw consumed) and the largest and final
    toroidal distance from the start. Full-length episodes only."""
    import torch

    from itasorl.experiment_b2 import _seeds, make_world
    from itasorl.novel_readouts import arena_dist, draw_features

    O, A, K, F, dmax, dfin = [], [], [], [], [], []
    for i in range(n_eps):
        w = make_world(P, 0.0, ray_steps)
        w.reset(_seeds(seed_base + i))
        feats = draw_features(w, r) if r is not None else None
        start = np.asarray(w.pos, float).copy()
        h = agent.initial_state(1, "cpu")
        prev = torch.zeros(1, agent.act_dim)
        obs = w.observe().astype(np.float64)
        orow, arow, dist, died = [], [], [], False
        for _ in range(steps):
            o = torch.as_tensor(norm(obs)[None], dtype=torch.float32)
            _, env_act, _, _, h = agent.act(o, prev, h, deterministic=True)
            a = env_act[0].detach().cpu().numpy().astype(np.float32)
            orow.append(obs.astype(np.float32))
            arow.append(a)
            res = w.step(a)
            dist.append(float(arena_dist(np.asarray(w.pos, float), start)))
            prev = env_act
            obs = res.obs.astype(np.float64)
            if res.terminated:
                died = True
                break
        if not died and len(orow) == steps:
            O.append(np.stack(orow))
            A.append(np.stack(arow))
            K.append(i)
            F.append(feats)
            dmax.append(max(dist))
            dfin.append(dist[-1])
    return np.stack(O), np.stack(A), np.asarray(K), F, np.asarray(dmax), np.asarray(dfin)


def calibrate(t: dict) -> dict:
    setup_l3()
    t0 = time.time()
    ag, nm = load(t["agents_dir"], DRIFT, t["seed"], "survival")
    *_, dmax, dfin = record_with_worlds(ag, nm, world_params(), t["n_calib"], t["steps"],
                                        CALIB_BASE, t["ray_steps"], None)
    return {"seed": t["seed"], "max_disp": dmax.tolist(), "final_disp": dfin.tolist(),
            "seconds": round(time.time() - t0, 1)}


def run_seed(t: dict) -> dict:
    if t.get("protocol", "replay") == "own":
        return run_seed_own(t)
    from itasorl.eval_protocols import replay_states
    from itasorl.experiment_a import grouped_auroc
    from itasorl.experiment_b import episode_features
    from itasorl.novel_readouts import (INCIDENTAL, INSTRUMENTAL, exceedance_rate, obs_summary,
                                        residual_r2_cv, residual_targets, residualize_incidental,
                                        ridge_r2_cv)

    setup_l3()
    P = world_params()
    s = t["seed"]
    t0 = time.time()
    ag, nm = load(t["agents_dir"], DRIFT, s, "survival")
    O, A, K, F, dmax_read, _ = record_with_worlds(ag, nm, P, t["n_eps"], t["steps"], READ_BASE,
                                                  t["ray_steps"], t["r"])
    raw = {k: np.array([f[k] for f in F], float) for k in INSTRUMENTAL + INCIDENTAL}
    ok = np.all(np.isfinite(np.column_stack(list(raw.values()))), axis=1)
    feats = {k: v[ok] for k, v in raw.items()}
    full = residualize_incidental(feats)      # full-sample residual: the descriptive median split only
    tgt = {k: residual_targets(feats, k) for k in feats}   # far targets residualized in fold
    Xo = obs_summary(O[ok])
    out = {"seed": s, "n": int(ok.sum()), "n_recorded": int(len(K)),
           "feature_means": {k: float(v.mean()) for k, v in feats.items()},
           "feature_sds": {k: float(v.std()) for k, v in feats.items()},
           "echo_r2": {k: ridge_r2_cv(Xo, feats[k], target=tgt[k]) for k in feats}, "arms": {},
           "targets": "far features residualized on the near ones inside each training fold "
                      "(third amendment); auroc_median_split uses the full-sample residual",
           "exceedance_rate": exceedance_rate(dmax_read[ok], t["r"])}
    rng = np.random.default_rng(10_000 + s)
    for g, d in ARMS[: t["n_arms"]]:
        a2, n2 = (ag, nm) if (g, d) == ("survival", DRIFT) else load(t["agents_dir"], d, s, g)
        Xh = episode_features(replay_states(a2, n2, O[ok], A[ok])).astype(np.float64)
        blk = {}
        for k, v in feats.items():
            yb = (full[k] > np.median(full[k])).astype(int)
            blk[k] = {"r2_state": ridge_r2_cv(Xh, v, target=tgt[k]),
                      "r2_resid": residual_r2_cv(Xh, Xo, v, target=tgt[k]),
                      "auroc_median_split": float(grouped_auroc(Xh, yb, np.arange(len(yb))))}
            if (g, d) == ("survival", DRIFT):
                perm = {**feats, k: rng.permutation(v)}
                blk[k]["r2_permuted"] = ridge_r2_cv(Xh, perm[k], target=residual_targets(perm, k))
        out["arms"][f"{g}_d{d:.2f}"] = blk
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run_seed_own(t: dict) -> dict:
    """The secondary protocol (G07, post hoc): every arm acts on its own episodes at READ_BASE
    and is read on them, with the replay protocol's decoders, targets and permutation draws.
    Each arm keeps its own episode count, feature summaries and echo under `per_arm`, with
    every recorded episode's index, finite-feature flag and feature values. The row-level
    n, feature summaries, echo and exceedance are the headline arm's, so `summarize` reads
    an own-episode row as it reads a replay row."""
    from itasorl.eval_protocols import replay_states
    from itasorl.experiment_a import grouped_auroc
    from itasorl.experiment_b import episode_features
    from itasorl.novel_readouts import (INCIDENTAL, INSTRUMENTAL, exceedance_rate, obs_summary,
                                        residual_r2_cv, residual_targets, residualize_incidental,
                                        ridge_r2_cv)

    setup_l3()
    P = world_params()
    s = t["seed"]
    t0 = time.time()
    head = f"survival_d{DRIFT:.2f}"
    out = {"seed": s, "protocol": "own", "arms": {}, "per_arm": {}}
    rng = np.random.default_rng(10_000 + s)
    for g, d in ARMS[: t["n_arms"]]:
        arm = f"{g}_d{d:.2f}"
        ag, nm = load(t["agents_dir"], d, s, g)
        O, A, K, F, dmax, _ = record_with_worlds(ag, nm, P, t["n_eps"], t["steps"], READ_BASE,
                                                 t["ray_steps"], t["r"])
        raw = {k: np.array([f[k] for f in F], float) for k in INSTRUMENTAL + INCIDENTAL}
        ok = np.all(np.isfinite(np.column_stack(list(raw.values()))), axis=1)
        feats = {k: v[ok] for k, v in raw.items()}
        full = residualize_incidental(feats)
        tgt = {k: residual_targets(feats, k) for k in feats}
        Xo = obs_summary(O[ok])
        Xh = episode_features(replay_states(ag, nm, O[ok], A[ok])).astype(np.float64)
        blk = {}
        for k, v in feats.items():
            yb = (full[k] > np.median(full[k])).astype(int)
            blk[k] = {"r2_state": ridge_r2_cv(Xh, v, target=tgt[k]),
                      "r2_resid": residual_r2_cv(Xh, Xo, v, target=tgt[k]),
                      "auroc_median_split": float(grouped_auroc(Xh, yb, np.arange(len(yb))))}
            if arm == head:
                perm = {**feats, k: rng.permutation(v)}
                blk[k]["r2_permuted"] = ridge_r2_cv(Xh, perm[k], target=residual_targets(perm, k))
        out["arms"][arm] = blk
        out["per_arm"][arm] = {
            "n": int(ok.sum()), "n_recorded": int(len(K)),
            "feature_means": {k: float(v.mean()) for k, v in feats.items()},
            "feature_sds": {k: float(v.std()) for k, v in feats.items()},
            "echo_r2": {k: ridge_r2_cv(Xo, feats[k], target=tgt[k]) for k in feats},
            "exceedance_rate": exceedance_rate(dmax[ok], t["r"]),
            "episode_index": [int(i) for i in K], "episode_ok": [bool(x) for x in ok],
            "episode_features": {k: [float(x) for x in v] for k, v in raw.items()}}
    hp = out["per_arm"][head]
    out.update({k: hp[k] for k in ("n", "n_recorded", "feature_means", "feature_sds", "echo_r2",
                                   "exceedance_rate")})
    out["targets"] = ("far features residualized on the near ones inside each training fold "
                      "(third amendment); auroc_median_split uses the full-sample residual; "
                      "each arm on its own episodes")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(rows: list, n_arms: int) -> dict:
    from itasorl.novel_readouts import INCIDENTAL, INSTRUMENTAL, paired_t90
    head, untr = f"survival_d{DRIFT:.2f}", f"untrained_d{DRIFT:.2f}"
    get = lambda arm, k, q: [r["arms"][arm][k][q] for r in rows]  # noqa: E731
    agg = {}
    for g, d in ARMS[:n_arms]:
        arm = f"{g}_d{d:.2f}"
        for k in INSTRUMENTAL + INCIDENTAL:
            for q in ("r2_state", "r2_resid", "auroc_median_split"):
                agg[f"{arm} {k} {q}"] = {**paired_t90(get(arm, k, q)), "per_seed": get(arm, k, q)}
    for k in INSTRUMENTAL + INCIDENTAL:
        agg[f"echo {k} r2"] = paired_t90([r["echo_r2"][k] for r in rows])
        agg[f"permuted {k} r2"] = paired_t90(get(head, k, "r2_permuted"))
    r1 = {}
    for k in INSTRUMENTAL:
        c = paired_t90(get(head, k, "r2_state"), get(untr, k, "r2_state")) if n_arms >= 3 else None
        res = agg[f"{head} {k} r2_resid"]
        r1[k] = {"surv_minus_untr_r2_state": c, "resid": res,
                 "met": bool(c is not None and c["t90"][0] > 0 and res["t90"][0] > 0.05)}
    r2 = {k: {"resid": agg[f"{head} {k} r2_resid"],
              "met": bool(agg[f"{head} {k} r2_resid"]["t90"][1] < 0.05)} for k in INCIDENTAL}
    mi = [np.mean([r["arms"][head][k]["r2_resid"] for k in INSTRUMENTAL]) for r in rows]
    mn = [np.mean([r["arms"][head][k]["r2_resid"] for k in INCIDENTAL]) for r in rows]
    lad = paired_t90(mi, mn)
    echo = {k: bool(agg[f"echo {k} r2"]["mean"] > agg[f"{head} {k} r2_state"]["mean"])
            for k in INSTRUMENTAL + INCIDENTAL}
    perm_ok = all(agg[f"permuted {k} r2"]["mean"] <= 0.02 for k in INSTRUMENTAL + INCIDENTAL)
    all1 = all(v["met"] for v in r1.values())
    none1 = not any(v["met"] for v in r1.values())
    n2_fail = sum(not v["met"] for v in r2.values())
    if all1 and all(v["met"] for v in r2.values()):
        label = "follows usefulness"
    elif all1 and n2_fail >= 2:
        label = "keeps everything"
    elif none1:
        label = "keeps nothing beyond the senses"
    else:
        label = "mixed"
    return {"aggregate": agg, "readings": {
        "1 instrumental kept": r1, "2 incidental not kept": r2,
        "3 ladder (mean resid I minus mean resid N)": {**lad, "met": bool(lad["t90"][0] > 0)},
        "4 carried by the senses": echo, "sanity: permuted R2 <= 0.02": perm_ok, "label": label}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-eps", type=int, default=1100)
    ap.add_argument("--n-calib", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--arms", type=int, default=len(ARMS), help="first N of survival, predictor, untrained, survival d0")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="replay",
                    help="replay (registered): every arm reads the headline agent's episodes; "
                         "own (secondary, post hoc): every arm reads its own episodes")
    a = ap.parse_args()
    own = a.protocol == "own"
    prov = run_provenance(a)
    ckpt = a.ckpt or a.out + ".ckpt"
    base = {"agents_dir": a.agents_dir, "steps": a.steps, "ray_steps": a.ray_steps}
    fpx = fingerprint_extra(__file__, a.agents_dir, a.seeds, ARMS[: a.arms])
    cal = run_tasks(calibrate, [{**base, "seed": s, "n_calib": a.n_calib} for s in a.seeds],
                    os.path.join(ckpt, "calib"), lambda t: f"s{t['seed']}", a.workers, fpx)
    dmax = np.concatenate([c["max_disp"] for c in cal])
    dfin = np.concatenate([c["final_disp"] for c in cal])
    r = float(np.percentile(dmax, 95))
    print(f"reach radius r = {r:.4f} (95th pct of max displacement; final-step pct "
          f"{np.percentile(dfin, 95):.4f})", flush=True)
    tasks = [{**base, "seed": s, "n_eps": a.n_eps, "r": r, "n_arms": a.arms} for s in a.seeds]
    if own:     # the replay task dict, and so its checkpoint fingerprint, gains no key
        tasks = [{**t, "protocol": "own"} for t in tasks]
    rows = sorted(run_tasks(run_seed, tasks, os.path.join(ckpt, "read_own" if own else "read"),
                            lambda t: f"s{t['seed']}", a.workers, fpx), key=lambda x: x["seed"])
    body = {"spec": SPEC, "reach": {"r": r, "pct95_max_disp": r,
                                    "pct95_final_disp": float(np.percentile(dfin, 95)),
                                    "readout_exceedance_rate": float(np.nanmean([r_["exceedance_rate"] for r_ in rows])),
                                    "note": "r is the 95th percentile of the calibration episodes' largest "
                                            "displacement, a percentile and not a bound on where an episode "
                                            "can reach (third amendment)",
                                    "n_calib_episodes": int(len(dmax))},
            **summarize(rows, a.arms), "rows": rows}
    inputs = {"agents_dir": a.agents_dir, "seeds": a.seeds, "n_eps": a.n_eps,
              "seed_bases": {"readout": READ_BASE, "calibration": CALIB_BASE}}
    if own:
        body = {"protocol": "own (secondary protocol of the spec, post hoc)", **body}
        inputs["protocol"] = "own"
    write_output(a.out, "scripts/run_world_features_readout.py", sys.argv[1:], inputs, SPEC, body)
    add_provenance(a.out, prov)
    print("  label:", body["readings"]["label"])
    return 0


def run_provenance(a) -> dict:
    """Closing item G46: the keys the shared readout provenance lacks. A readout of saved
    agents, so the commit key is git_commit_at_promotion."""
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    p = provenance("scripts/run_world_features_readout.py", commit_key="git_commit_at_promotion",
                   output=repo_rel(a.out), device="cpu")
    return {**p, "agents_dir": repo_rel(a.agents_dir), "argv": rel_flags(list(sys.argv[1:]))}


def add_provenance(path: str, extra: dict) -> None:
    """Merge `extra` into the written file's provenance block and rewrite it the way
    write_output wrote it. Every key write_output recorded keeps its value, the stack block
    included; nothing outside the provenance block changes."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    prov = doc["provenance"]
    for k, v in extra.items():
        if k == "stack":
            prov["stack"] = {**v, **prov.get("stack", {})}
        elif k not in prov:
            prov[k] = v
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=float)


if __name__ == "__main__":
    raise SystemExit(main())
