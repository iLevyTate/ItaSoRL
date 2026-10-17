"""Walk speed versus goal direction (closing item B2, runbook 1.6 and R13).

The survival walk carries the world signal (main text: the scripted walk reads 0.551 to 0.577
on every trunk), and it is also the fastest walk (mean speed 0.158 against 0.147 to 0.150).
This readout asks whether speed alone carries the signal or the goal-directed walk does. Every
trunk of the corrected C1 agents (survival, predictor, untrained, drift 0.45, seeds 0 to 9) is
read with the registered pooled probe ([mean h, h_T], scaler and logistic regression, in-fold)
on four walks through the same fresh worlds, B2-auth 2800000 (authentic) and B2-surr 2850000
(surrogate), both registered in itasorl.post_review.REGISTERED_LATER:

  own            each trunk's own deterministic policy, closed loop (collect_pool), the
                 registered protocol at fresh bases; the survival trunk's own walk sets the
                 target speed
  scripted       the scripted walk of eval_protocols (action stream i = default_rng(600000 + i)
                 in both pools), unscaled
  speed_matched  arm A: the same scripted walk with its thrust multiplied by one per-seed
                 thrust_scale (eval_protocols.collect_pool_scripted(thrust_scale=...)), chosen
                 by bisection on [0.5, 1/0.6] so the walk's mean speed over both pools matches
                 the survival trunk's own walk within 0.001; headings and draws are unchanged
  replay         arm B: the survival agent's recorded actions, open loop. The actions come from
                 its own closed-loop episodes in the registered authentic worlds (800000 + j,
                 eval_protocols.record_trajectories), so neither B2 pool was the world they were
                 chosen in; sequence j is replayed in B2-auth + j and in B2-surr + j, and every
                 trunk receives each observation and the replayed action as its previous action

Speed is the per-episode mean of |velocity| after each step (collect_pool's speed), averaged over
the full-length episodes of both pools. The scripted, speed-matched and replay walks give pool
index i the same action stream in both pools, so the probe groups the two rows of index i
together (CV groups = episode index); the own walks are independent pools (one group per row).

Design decisions made here, to be frozen in the Stage 2 spec
(docs/specs/2026-10-17-walk-speed-vs-goal-readout-design.md): the four walks above; the speed
measure and the 0.001 match tolerance; the bisection bracket (0.5 to 1/0.6, where the scripted
thrust of uniform(0, 0.6) reaches the world's cap of 1); the action source of arm B; the paired
grouping; seed-paired contrasts per trunk (speed_matched - scripted, own - speed_matched,
replay - speed_matched, own - replay) labeled ABOVE, BELOW or INCLUDES by their t-based 90%
interval. A seed whose speed cannot be matched inside the bracket is kept and flagged.

The script refuses to start when the B2 bases collide with a base in use
(post_review.base_collisions against post_review.all_bases_in_use()) or when --n-eps exceeds
their registered footprint. Readout only, post hoc and descriptive: it changes no stored value
or verdict. Narrows C1 only.

    python scripts/run_walk_speed_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --workers 2 --out fullruns/closing_2026-10/b2/walk_speed_corrected_l3_h8_wm.json [--dry-run]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import sys
import time

import numpy as np

from _readout_common import DRIFT, agent_path, fingerprint_extra, load, run_tasks, setup_l3, world_params

GENERATED_BY = "scripts/run_walk_speed_readout.py"
ARMS = ("survival", "predictor", "untrained")
WALKS = ("own", "scripted", "speed_matched", "replay")
STEPS, RAY_STEPS = 24, 5
SCALE_LO, SCALE_HI = 0.5, 1.0 / 0.6
SPEED_TOL, MAX_BISECT = 1e-3, 14
CONTRASTS = (("speed_matched", "scripted"), ("own", "speed_matched"),
             ("replay", "speed_matched"), ("own", "replay"))
STATUS = "post hoc, descriptive; narrows C1 only; changes no stored value or verdict"


def b2_bases(n_eps: int) -> tuple[int, int]:
    """The registered B2 bases, after the collision and footprint checks."""
    from itasorl import post_review as pr
    own = [pr.registered_later("B2-auth"), pr.registered_later("B2-surr")]
    too_big = [name for name, _, n in own if n_eps > n]
    if too_big:
        raise SystemExit(f"--n-eps {n_eps} exceeds the registered footprint of {too_big}")
    hits = pr.base_collisions(new=tuple((name, base, n_eps) for name, base, _ in own),
                              in_use=pr.all_bases_in_use())
    if hits:
        raise SystemExit(f"B2 bases collide with bases in use: {hits}")
    return own[0][1], own[1][1]


def scaled(a: np.ndarray, thrust_scale: float) -> np.ndarray:
    """The thrust scaling of eval_protocols.collect_pool_scripted, applied in place."""
    if thrust_scale != 1.0:
        a[0] = np.float32(min(max(float(a[0]) * thrust_scale, 0.0), 1.0))
    return a


def scripted_actions(policy_seed_base: int, thrust_scale: float):
    """Action factory of the (scaled) scripted walk: stream i is default_rng(base + i)."""
    from itasorl.experiment_b import scripted_policy

    def make(i):
        rng = np.random.default_rng(policy_seed_base + i)
        return lambda t: scaled(scripted_policy(rng), thrust_scale)
    return make


def replay_actions(actions: np.ndarray):
    """Action factory of the open-loop replay: episode j steps the recorded actions of j."""
    return lambda j: (lambda t: np.array(actions[j, t], dtype=np.float32))


def walk_speed(P, drift: float, base: int, n: int, make_actions) -> tuple[np.ndarray, np.ndarray]:
    """World-only walk under an action stream that ignores the agent: (kept indices, per
    episode mean |velocity| after each step) over the full-length episodes. The worlds and
    actions are those of the corresponding pool, so the episodes are the pool's episodes."""
    from itasorl.experiment_b2 import _seeds, make_world
    kept, speed = [], []
    for i in range(n):
        w = make_world(P, drift, RAY_STEPS)
        w.reset(_seeds(base + i))
        w.observe()                              # as the pools do before their first step
        act, sp, died = make_actions(i), [], False
        for t in range(STEPS):
            r = w.step(act(t))
            sp.append(float(np.linalg.norm(w.vel)))
            if r.terminated:
                died = True
                break
        if not died and len(sp) == STEPS:
            kept.append(i)
            speed.append(float(np.mean(sp)))
    return np.asarray(kept, int), np.asarray(speed)


def pooled_speed(P, bases, n_auth_surr, factory) -> float:
    (ba, bs), (na, ns) = bases, n_auth_surr
    sa = walk_speed(P, 0.0, ba, na, factory)[1]
    ss = walk_speed(P, DRIFT, bs, ns, factory)[1]
    v = np.r_[sa, ss]
    return float(v.mean()) if v.size else float("nan")


def calibrate(speed_of, target: float, lo: float = SCALE_LO, hi: float = SCALE_HI) -> dict:
    """Bisection for the thrust scale whose walk speed matches `target` within SPEED_TOL;
    speed_of(scale) -> mean speed. Outside the bracket the nearer end is kept and flagged."""
    hist = [(lo, speed_of(lo)), (hi, speed_of(hi))]
    if not np.isfinite(target):
        return {"thrust_scale": float("nan"), "speed": float("nan"), "matched": False, "history": hist}
    if target <= hist[0][1] or target >= hist[1][1]:
        s, v = hist[0] if target <= hist[0][1] else hist[1]
        return {"thrust_scale": s, "speed": v, "matched": bool(abs(v - target) <= SPEED_TOL),
                "history": hist, "note": "target speed outside the bracket"}
    a, b = lo, hi
    for _ in range(MAX_BISECT):
        m = 0.5 * (a + b)
        v = speed_of(m)
        hist.append((m, v))
        if abs(v - target) <= SPEED_TOL:
            break
        a, b = (m, b) if v < target else (a, m)
    s, v = min(hist[2:], key=lambda h: abs(h[1] - target))
    return {"thrust_scale": float(s), "speed": float(v), "matched": bool(abs(v - target) <= SPEED_TOL),
            "history": hist}


def collect_pool_replayed(agent, norm, P, drift: float, actions: np.ndarray, base: int):
    """collect_pool_scripted with the recorded actions in place of the scripted policy: episode
    j runs in world base + j and steps actions[j], the trunk receiving each observation and the
    previous replayed action. Returns (H, kept indices) for full-length episodes."""
    import torch

    from itasorl.experiment_b2 import _seeds, make_world
    Hs, kept = [], []
    for j in range(len(actions)):
        w = make_world(P, drift, RAY_STEPS)
        w.reset(_seeds(base + j))
        h = agent.initial_state(1, "cpu")
        prev = torch.zeros(1, agent.act_dim)
        obs = w.observe().astype(np.float64)
        row, died = [], False
        with torch.no_grad():
            for t in range(STEPS):
                o = torch.as_tensor(norm(obs)[None], dtype=torch.float32)
                h = agent.step_state(o, prev, h)
                row.append(h[0].cpu().numpy())
                a = np.array(actions[j, t], dtype=np.float32)
                r = w.step(a)
                prev = torch.as_tensor(a[None], dtype=torch.float32)
                obs = r.obs.astype(np.float64)
                if r.terminated:
                    died = True
                    break
        if not died and len(row) == STEPS:
            Hs.append(np.asarray(row, np.float32))
            kept.append(j)
    H = np.stack(Hs) if Hs else np.zeros((0, STEPS, agent.hidden), np.float32)
    return H, np.asarray(kept, int)


def read(Ha, Hs, seed: int, groups) -> dict:
    from itasorl.eval_protocols import readout_from_states
    r = readout_from_states(Ha, Hs, seed=seed, groups=groups)
    return {"target": float(r["target"]), "n_auth": r["n_auth"], "n_surr": r["n_surr"],
            "oof_ci95": [float(x) for x in r["oof_ci95"]]}


def run_task(t: dict) -> dict:
    from itasorl.eval_protocols import POLICY_SEED_BASE, collect_pool_scripted, record_trajectories
    from itasorl.experiment_b2 import collect_pool
    from itasorl.post_review import REGISTERED_AUTH_BASE

    setup_l3()
    P = world_params()
    s, n, (ba, bs) = t["seed"], t["n_eps"], t["bases"]
    t0 = time.time()
    agents = {arm: load(t["agents_dir"], DRIFT, s, arm) for arm in ARMS}
    out = {"seed": s, "arms": {arm: {} for arm in ARMS}, "speed": {}}
    # own walks, closed loop: independent pools, one group per row
    for arm, (ag, nm) in agents.items():
        Ha, spa = collect_pool(ag, nm, P, 0.0, n, STEPS, "cpu", ba, RAY_STEPS)
        Hs, sps = collect_pool(ag, nm, P, DRIFT, n, STEPS, "cpu", bs, RAY_STEPS)
        g = np.arange(len(Ha) + len(Hs))
        out["arms"][arm]["own"] = read(Ha, Hs, s, g)
        sp = np.r_[spa, sps]
        out["speed"][f"{arm}_own"] = float(sp.mean()) if sp.size else float("nan")
    target = out["speed"]["survival_own"]
    # arm A: the speed-matched scripted walk
    cal = calibrate(lambda k: pooled_speed(P, (ba, bs), (n, n), scripted_actions(POLICY_SEED_BASE, k)), target)
    out["calibration"] = {**cal, "target_speed": target, "tolerance": SPEED_TOL, "bracket": [SCALE_LO, SCALE_HI]}
    out["thrust_scale"] = cal["thrust_scale"]
    out["speed"]["scripted"] = pooled_speed(P, (ba, bs), (n, n), scripted_actions(POLICY_SEED_BASE, 1.0))
    out["speed"]["speed_matched"] = cal["speed"]
    for walk, k in (("scripted", 1.0), ("speed_matched", cal["thrust_scale"])):
        for arm, (ag, nm) in agents.items():
            Ha, ka = collect_pool_scripted(ag, nm, P, 0.0, n, STEPS, "cpu", ba, RAY_STEPS, thrust_scale=k)
            Hs, ks = collect_pool_scripted(ag, nm, P, DRIFT, n, STEPS, "cpu", bs, RAY_STEPS, thrust_scale=k)
            out["arms"][arm][walk] = read(Ha, Hs, s, np.r_[ka, ks])     # index i shares its action stream
    # arm B: open-loop replay of the survival agent's actions from the registered authentic worlds
    surv, snorm = agents["survival"]
    _, A, K = record_trajectories(surv, snorm, P, 0.0, n, STEPS, "cpu", REGISTERED_AUTH_BASE, RAY_STEPS)
    out["replay_source"] = {"base": REGISTERED_AUTH_BASE, "drift": 0.0, "n_sequences": int(len(A)),
                            "episodes": [int(x) for x in K]}
    out["speed"]["replay"] = pooled_speed(P, (ba, bs), (len(A), len(A)), replay_actions(A))
    for arm, (ag, nm) in agents.items():
        Ha, ka = collect_pool_replayed(ag, nm, P, 0.0, A, ba)
        Hs, ks = collect_pool_replayed(ag, nm, P, DRIFT, A, bs)
        out["arms"][arm]["replay"] = read(Ha, Hs, s, np.r_[ka, ks])     # sequence j in both pools
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(rows: list) -> dict:
    from itasorl.post_review import interval_label, paired_summary, seed_summary
    by = {str(r["seed"]): r for r in rows}
    out = {"seeds": sorted(int(s) for s in by), "arms": {}, "contrasts": {}, "speed": {}}
    for arm in ARMS:
        out["arms"][arm] = {w: seed_summary([r["arms"][arm][w]["target"] for r in rows]) for w in WALKS}
        out["contrasts"][arm] = {}
        for a, b in CONTRASTS:
            d = paired_summary({s: r["arms"][arm][a]["target"] for s, r in by.items()},
                               {s: r["arms"][arm][b]["target"] for s, r in by.items()})
            out["contrasts"][arm][f"{a} - {b}"] = {**d, "reading": interval_label(d)}
    for k in sorted({k for r in rows for k in r["speed"]}):
        out["speed"][k] = seed_summary([r["speed"][k] for r in rows])
    out["thrust_scale"] = seed_summary([r["thrust_scale"] for r in rows])
    out["speed_match"] = {"matched_seeds": [r["seed"] for r in rows if r["calibration"]["matched"]],
                          "unmatched_seeds": [r["seed"] for r in rows if not r["calibration"]["matched"]],
                          "max_abs_speed_gap": float(max((abs(r["speed"]["speed_matched"] - r["speed"]["survival_own"])
                                                          for r in rows), default=float("nan")))}
    return out


def run_provenance(out: str, agents_dir: str, argv: list) -> dict:
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(out), device="cpu")
    return {**p, "agents_dir": repo_rel(agents_dir), "argv": rel_flags(list(argv))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (default: beside --out)")
    ap.add_argument("--dry-run", action="store_true", help="check bases and agents, then exit")
    a = ap.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    ba, bs = b2_bases(a.n_eps)
    missing = [agent_path(a.agents_dir, DRIFT, s, arm) for s in a.seeds for arm in ARMS
               if not os.path.exists(agent_path(a.agents_dir, DRIFT, s, arm))]
    print(f"B2 bases {ba} / {bs}: no collision with the bases in use; seeds {a.seeds}, arms {list(ARMS)}, "
          f"walks {list(WALKS)}; missing agents {len(missing)}", flush=True)
    if missing:
        raise SystemExit(f"missing agents: {missing[:6]}")
    if a.dry_run:
        print("dry run: nothing collected or written")
        return 0
    tasks = [{"agents_dir": a.agents_dir, "seed": s, "n_eps": a.n_eps, "bases": [ba, bs]} for s in a.seeds]
    ckpt = a.ckpt or os.path.join(os.path.dirname(os.path.abspath(a.out)), "ckpt_walk_speed")
    rows = run_tasks(run_task, tasks, ckpt, lambda t: f"s{t['seed']}", a.workers,
                     fingerprint_extra(__file__, a.agents_dir, a.seeds, list(ARMS)))
    rows = sorted([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], key=lambda r: r["seed"])
    finite = all(np.isfinite(r["arms"][arm][w]["target"]) for r in rows for arm in ARMS for w in WALKS)
    from agent_free_common import write_json
    doc = {"provenance": run_provenance(a.out, a.agents_dir, argv), "status": STATUS,
           "bases": {"B2-auth": ba, "B2-surr": bs, "replay_source": "post_review.REGISTERED_AUTH_BASE (800000)",
                     "scripted_policy": "eval_protocols.POLICY_SEED_BASE (600000)"},
           "settings": {"seeds": a.seeds, "n_eps": a.n_eps, "steps": STEPS, "ray_steps": RAY_STEPS, "drift": DRIFT,
                        "walks": list(WALKS), "speed_tolerance": SPEED_TOL, "bracket": [SCALE_LO, SCALE_HI],
                        "probe": "eval_protocols.readout_from_states ([mean h, h_T], scaler and logistic "
                                 "regression, in-fold)",
                        "groups": "own: one group per row; scripted, speed_matched, replay: episode index "
                                  "(both pools share action stream i)"},
           "all_finite": bool(finite), "summary": summarize(rows), "rows": rows}
    write_json(a.out, doc)
    print("wrote", a.out, "all finite" if finite else "NOT all finite", flush=True)
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
