"""Head overlap: does the actor read the decoded direction? (closing item X03, E04)

Post hoc and descriptive; changes no stored value or verdict. Per seed and arm on the registered
pools at drift 0.45 (800000 / 850000, 110 x 24): the decoded direction u
(mechanism_readouts.probe_direction) and the rank-two probe block
(novel_readouts.probe_block_directions), then
(a) the share of u (and of the block) inside the row space of each head, by QR, divided by the
    share a random direction would have: the actor without its dead emit row (4 / 96), the
    critic (1 / 96), and the decoder's state block decoder.0.weight[:, :hidden] (64 / 96, weak
    by construction: a random 64-dimensional subspace of 96 holds two thirds of any direction);
(b) the one-step action sensitivity ||d env_action / d h . u|| through the actor and
    to_env_action (Bernoulli probabilities in place of the thresholded binaries, emit left out)
    at every visited pooled state, over the median of the same for 1000 random unit directions;
(c) the recurrence sensitivity ||d h_{t+1} / d h_t . u|| of the GRU cell with its next input
    held fixed, the same ratio.
Both (b) and (c) use torch.autograd.functional.jvp. A one-step overlap is not a behavioral
test: the recurrence can carry a direction outside the actor's row space into later actions
(scripts/run_nudge_dose_readout.py measures that).

Gate: each fitted probe's pooled target equals the cell's recorded target within 0.002 for every
(arm, seed), else nothing is written. Rule on the survival arm: "policy-blind" if the upper
bounds of r_actor and of the action ratio are below 0.5; "policy-read" if both lower bounds
exceed 2; otherwise "undetermined". The untrained arm is the floor and decides nothing.

    python scripts/run_head_overlap_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --arms survival predictor untrained --workers 2 --ckpt fullruns/closing_2026-10/x03/ckpt \\
        --out fullruns/closing_2026-10/x03/corrected_l3_h8_wm.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from _readout_common import AUTH_BASE, DRIFT, SURR_BASE, fingerprint_extra, load, run_tasks, setup_l3, world_params

GENERATED_BY = "scripts/run_head_overlap_readout.py"
ARMS = ("survival", "predictor", "untrained")
N_RANDOM = 1000
TOL = 0.002
STATUS = "post hoc, descriptive; changes no stored value or verdict"


def row_space(W: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthogonal projector onto the row space of W, from a QR of W^T (columns whose R
    diagonal is negligible are dropped)."""
    Q, R = np.linalg.qr(np.asarray(W, float).T)
    d = np.abs(np.diag(R))
    keep = d > tol * (d.max() if d.size and d.max() > 0 else 1.0)
    Q = Q[:, keep]
    return Q @ Q.T


def heads(agent) -> dict:
    """{name: weight rows} of the actor (emit row dropped), the critic and the decoder's state
    block (absent when the agent has no decoder)."""
    hid = agent.hidden
    W = agent.actor.weight.detach().cpu().numpy().astype(np.float64)
    out = {"actor": np.delete(W, agent.n_cont + agent.n_bin - 1, axis=0),
           "critic": agent.critic.weight.detach().cpu().numpy().astype(np.float64)}
    if getattr(agent, "world_model", False):
        out["decoder_state_block"] = agent.decoder[0].weight.detach().cpu().numpy().astype(np.float64)[:, :hid]
    return out


def head_overlaps(agent, u: np.ndarray, block: np.ndarray | None = None) -> dict:
    """Per head: the share of u inside its row space, the random expectation rank / hidden, and
    their ratio; with `block`, the mean share of the block's orthonormal basis as well."""
    hid = agent.hidden
    u = np.asarray(u, float) / np.linalg.norm(u)
    Bq = None
    if block is not None:
        Bq = np.linalg.qr(np.asarray(block, float).T)[0].T
    out = {}
    for name, W in heads(agent).items():
        P = row_space(W)
        rank = int(round(np.trace(P)))
        expect = rank / hid
        share = float(u @ P @ u)
        row = {"rank": rank, "expectation": expect, "share_u": share, "ratio_u": share / expect}
        if Bq is not None:
            sb = float(np.trace(Bq @ P @ Bq.T) / len(Bq))
            row.update(share_block=sb, ratio_block=sb / expect)
        out[name] = row
    return out


def _unit_dirs(hid: int, n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(2_000_003 + seed).standard_normal((n, hid))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def jvp_ratio(fn, X, u: np.ndarray, dirs: np.ndarray) -> dict:
    """Mean over states of ||J(x) u||, over the median of the same for the random directions."""
    import torch
    from torch.autograd.functional import jvp

    def mean_norm(v):
        tangent = torch.as_tensor(np.asarray(v, np.float32)).expand_as(X).contiguous()
        _, out = jvp(fn, X, tangent)
        return float(out.norm(dim=-1).mean())

    val = mean_norm(u / np.linalg.norm(u))
    rnd = np.array([mean_norm(v) for v in dirs])
    med = float(np.median(rnd))
    return {"u": val, "random_median": med, "ratio": val / med if med > 0 else float("nan")}


def action_fn(agent):
    import torch

    def f(h):
        out = agent.actor(h)
        mu, logits = out[..., :agent.n_cont], out[..., agent.n_cont:]
        return agent.to_env_action(mu, torch.sigmoid(logits))[..., :agent.n_cont + agent.n_bin - 1]
    return f


def recurrence_fn(agent, Ot: np.ndarray, At: np.ndarray):
    """h_t -> h_{t+1} with the next input (encoded observation t+1, action t) held fixed, over the
    states at t = 0 .. T-2."""
    import torch
    o = torch.as_tensor(np.asarray(Ot[:, 1:], np.float32)).reshape(-1, Ot.shape[-1])
    a = torch.as_tensor(np.asarray(At[:, :-1], np.float32)).reshape(-1, At.shape[-1])
    with torch.no_grad():
        x = torch.cat([agent.encoder(o), a], dim=-1)
    return lambda h: agent.cell(x, h)


def run_cell(t: dict) -> dict:
    import torch

    from itasorl import mechanism_readouts as mr
    from itasorl.eval_protocols import readout_from_states
    from itasorl.experiment_b2 import collect_pool
    from itasorl.novel_readouts import probe_block_directions

    setup_l3()
    P = world_params()
    s, g = t["seed"], t["arm"]
    t0 = time.time()
    ag, nm = load(t["agents_dir"], DRIFT, s, g)
    Ha, _, Oa, Aa = collect_pool(ag, nm, P, 0.0, t["pool_n"], t["pool_steps"], "cpu", AUTH_BASE, t["ray_steps"],
                                 return_obs=True, return_actions=True)
    Hs, _, Os, As = collect_pool(ag, nm, P, DRIFT, t["pool_n"], t["pool_steps"], "cpu", SURR_BASE, t["ray_steps"],
                                 return_obs=True, return_actions=True)
    hid = Ha.shape[-1]
    # independent pools (800000 and 850000 are different worlds): one CV group per episode,
    # the partition the cell's pooled target used, written out
    target = float(readout_from_states(Ha, Hs, seed=s, groups=np.arange(len(Ha) + len(Hs)))["target"])
    u, spread = mr.probe_direction(Ha, Hs)
    out = {"seed": s, "arm": g, "n": [int(len(Ha)), int(len(Hs))], "pooled_target": target, "s": spread}
    if not np.isfinite(spread):
        out.update(reason="degenerate probe", seconds=round(time.time() - t0, 1))
        return out
    block = probe_block_directions(Ha, Hs)
    out["overlap"] = head_overlaps(ag, u, block)
    dirs = _unit_dirs(hid, t["n_random"], s)
    H = torch.as_tensor(np.concatenate([Ha, Hs]).reshape(-1, hid).astype(np.float32))
    out["action_sensitivity"] = jvp_ratio(action_fn(ag), H, u, dirs)
    Ht = torch.as_tensor(np.concatenate([Ha[:, :-1], Hs[:, :-1]]).reshape(-1, hid).astype(np.float32))
    out["recurrence_sensitivity"] = jvp_ratio(
        recurrence_fn(ag, np.concatenate([Oa, Os]), np.concatenate([Aa, As])), Ht, u, dirs)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def cell_targets(agents_dir: str, seeds, arms) -> dict:
    cells_dir = os.path.join(os.path.dirname(os.path.normpath(agents_dir)), "cells")
    out = {}
    for s in seeds:
        p = os.path.join(cells_dir, f"cell_d{DRIFT:.2f}_s{s}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        for g in arms:
            if g in c.get("agents", {}):
                out[(g, s)] = float(c["agents"][g]["pool"]["target"])
    return out


def gate(rows: list, targets: dict) -> list[str]:
    fails = []
    for r in rows:
        ref = targets.get((r["arm"], r["seed"]))
        if ref is None:
            fails.append(f"{r['arm']} seed {r['seed']}: no cell target")
        elif not (np.isfinite(r["pooled_target"]) and abs(r["pooled_target"] - ref) <= TOL):
            fails.append(f"{r['arm']} seed {r['seed']}: pooled target {r['pooled_target']!r} vs cell {ref!r}")
    return fails


def summarize(rows: list, arms) -> dict:
    from itasorl.novel_readouts import paired_t90
    out = {}
    for g in arms:
        rr = [r for r in rows if r["arm"] == g and "overlap" in r]
        blk = {"n": len(rr)}
        for head in ("actor", "critic", "decoder_state_block"):
            vals = [r["overlap"][head]["ratio_u"] for r in rr if head in r["overlap"]]
            if vals:
                blk[f"r_{head}"] = paired_t90(vals)
                blk[f"r_{head}_block"] = paired_t90([r["overlap"][head]["ratio_block"] for r in rr
                                                     if head in r["overlap"]])
        blk["action_ratio"] = paired_t90([r["action_sensitivity"]["ratio"] for r in rr])
        blk["recurrence_ratio"] = paired_t90([r["recurrence_sensitivity"]["ratio"] for r in rr])
        if g == "survival" and "r_actor" in blk:
            ra, ac = blk["r_actor"]["t90"], blk["action_ratio"]["t90"]
            blk["label"] = ("policy-blind" if ra[1] < 0.5 and ac[1] < 0.5
                            else "policy-read" if ra[0] > 2 and ac[0] > 2 else "undetermined")
        out[g] = blk
    return out


def run_provenance(a) -> dict:
    from _readout_common import _rel_argv
    from agent_free_common import provenance
    from promote_bv3_gates import repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(a.out), device="cpu")
    return {**p, "agents_dir": repo_rel(a.agents_dir), "argv": _rel_argv(list(sys.argv[1:]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--pool-n", type=int, default=110)
    ap.add_argument("--pool-steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (default OUT.ckpt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args(argv)
    targets = cell_targets(a.agents_dir, a.seeds, a.arms)
    tasks = [{"agents_dir": a.agents_dir, "seed": s, "arm": g, "pool_n": a.pool_n, "pool_steps": a.pool_steps,
              "ray_steps": a.ray_steps, "n_random": a.n_random} for g in a.arms for s in a.seeds]
    ckpt = a.ckpt or a.out + ".ckpt"
    if a.dry_run:
        print(f"{len(tasks)} cell(s); {len(targets)} cell targets found; checkpoints {ckpt}; would write {a.out}")
        return 0
    rows = sorted(run_tasks(run_cell, tasks, ckpt, lambda t: f"{t['arm']}_s{t['seed']}", a.workers,
                            fingerprint_extra(__file__, a.agents_dir, a.seeds, a.arms)),
                  key=lambda r: (r["arm"], r["seed"]))
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    fails = gate(rows, targets)
    if fails:
        print("integrity gate failed; nothing written:\n  " + "\n  ".join(fails), file=sys.stderr)
        return 1
    from agent_free_common import write_json
    doc = {"provenance": run_provenance(a), "status": STATUS,
           "settings": {"drift": DRIFT, "pools": [AUTH_BASE, SURR_BASE], "pool_n": a.pool_n,
                        "pool_steps": a.pool_steps, "n_random": a.n_random, "gate_tolerance": TOL},
           "integrity": {"gate": "PASS", "n_cells": len(rows)},
           "summary": summarize(rows, a.arms), "rows": rows}
    write_json(a.out, doc)
    print("survival:", doc["summary"].get("survival", {}).get("label"), "| wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
