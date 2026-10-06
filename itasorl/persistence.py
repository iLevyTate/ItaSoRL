"""Controlled persistence of prefix condition in the recurrent state (revision step 7).

The common-garden readout (`experiment_b2.common_garden_rollout`) lets each branch keep BOTH
its prefix hidden state AND its prefix physical state (position, velocity, energy, the
pellets it ate). Decoding the prefix world from its tail therefore cannot tell internal
memory from the external footprint the prefix left in the world. This module separates them.

Per pair p (world seed seed_base + p): the frozen agent runs a prefix in the authentic world
(branch A) and in the surrogate world (branch S) from the same seed. A pair is dropped if
either prefix dies, and a condition drops a pair if any of its tails dies (symmetric). Every
tail runs under AUTHENTIC dynamics. Conditions:

  replay        PRIMARY. Both prefix hidden states h_A and h_S read the IDENTICAL recorded
                tail input (observations and previous actions), open loop. The input is the
                closed-loop authentic tail of one branch, alternating A and S by pair parity
                so its source is balanced across labels; the first previous action is that
                source branch's last prefix action for both members. Any decodability is
                retention of the prefix in the state under common input.
  common_state  Interactive. Both hidden states are restored into the SAME physical state (the
                source branch's snapshot) with the same previous action and run closed loop.
                Different states can act differently and recreate sensory differences, so
                this measures memory plus its behavioral consequences.
  factorial     Hidden origin crossed with physical origin: (h_A, world_A), (h_S, world_S),
                and the two swaps (h_A, world_S), (h_S, world_A), all with previous action
                standardized to branch A's. Two labels are decoded on the same four tails:
                the hidden state's prefix world and the physical state's prefix world.
  reset_hidden  Each branch keeps its own physical state but starts the tail with a zero
                hidden state and a zero previous action: the external footprint alone.

Every member of a pair shares one CV group. Readouts: AUROC of the prefix label from h_t
alone at each tail step t (the decay curve), and from [mean h, final h] over the whole tail
and over its last `late_k` steps. All labels are the prefix condition; no world label enters
the agent.
"""

from __future__ import annotations

import numpy as np
import torch

from . import folds
from .experiment_a import grouped_auroc
from .experiment_b2 import _episode_feature, _seeds, make_world

SEED_BASE = 980_000          # disjoint from every other seed base in the pipeline
CONDITIONS = ("replay", "common_state", "factorial", "reset_hidden")


def _authentic_from(snapshot, seeds, params, ray_steps):
    w = make_world(params, 0.0, ray_steps)
    w.reset(seeds)            # populates w._rng so the key filter below is correct
    snap = dict(snapshot)
    snap["rng"] = {k: v for k, v in snapshot["rng"].items() if k in w._rng}
    w.set_state({**snap, "drift_w": 0.0})
    return w


def _prefix(agent, norm, params, drift, seeds, steps, ray_steps, device):
    w = make_world(params, drift, ray_steps)
    w.reset(seeds)
    h = agent.initial_state(1, device)
    prev = torch.zeros(1, agent.act_dim, device=device)
    obs = w.observe().astype(np.float64)
    for _ in range(steps):
        o = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
        _, act, _, _, h = agent.act(o, prev, h, deterministic=True)
        r = w.step(act[0].detach().cpu().numpy().astype(np.float32))
        obs, prev = r.obs.astype(np.float64), act
        if r.terminated:
            return None
    return h.clone(), prev.clone(), w.get_state()


def _closed_tail(agent, norm, world, h, prev, steps, device, record: bool = False):
    """Closed-loop tail. Returns (H (T,hidden) or None on death, obs list, action list)."""
    h, prev = h.clone(), prev.clone()
    obs = world.observe().astype(np.float64)
    Hs, O, A = [], [], []
    for _ in range(steps):
        o = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
        _, act, _, _, h = agent.act(o, prev, h, deterministic=True)
        Hs.append(h[0].detach().cpu().numpy())
        a = act[0].detach().cpu().numpy().astype(np.float32)
        if record:
            O.append(obs.copy())
            A.append(a)
        r = world.step(a)
        obs, prev = r.obs.astype(np.float64), act
        if r.terminated:
            return None, O, A
    return np.asarray(Hs, np.float32), O, A


@torch.no_grad()
def _replay_tail(agent, norm, h, prev0, obs_seq, act_seq, device):
    h, prev = h.clone(), prev0.clone()
    Hs = []
    for o_raw, a in zip(obs_seq, act_seq):
        o = torch.as_tensor(norm(o_raw)[None], dtype=torch.float32, device=device)
        h = agent.step_state(o, prev, h)
        Hs.append(h[0].cpu().numpy())
        prev = torch.as_tensor(a[None], dtype=torch.float32, device=device)
    return np.asarray(Hs, np.float32)


def collect_persistence(agent, norm, params, drift_sigma, *, n_pairs=110, prefix_steps=20,
                        tail_steps=24, seed_base=SEED_BASE, ray_steps=5, device="cpu") -> dict:
    """Tails per condition. Returns {condition: {"H": (n, T, hidden), "labels": {name: (n,)},
    "groups": (n,)}} with every member of a pair in one group."""
    out = {c: {"H": [], "labels": {}, "groups": []} for c in CONDITIONS}
    lab = {c: {} for c in CONDITIONS}
    zero_h = agent.initial_state(1, device)
    zero_a = torch.zeros(1, agent.act_dim, device=device)

    def add(cond, p, H_list, **labels):
        out[cond]["H"].extend(H_list)
        out[cond]["groups"].extend([p] * len(H_list))
        for k, v in labels.items():
            lab[cond].setdefault(k, []).extend(v)

    for p in range(n_pairs):
        seeds = _seeds(seed_base + p)
        pa = _prefix(agent, norm, params, 0.0, seeds, prefix_steps, ray_steps, device)
        ps = _prefix(agent, norm, params, drift_sigma, seeds, prefix_steps, ray_steps, device)
        if pa is None or ps is None:
            continue
        (hA, aA, wA), (hS, aS, wS) = pa, ps
        src_h, src_a, src_w = (hA, aA, wA) if p % 2 == 0 else (hS, aS, wS)
        # replay: record the source branch's closed-loop authentic tail, feed it to both states
        Hsrc, O, A = _closed_tail(agent, norm, _authentic_from(src_w, seeds, params, ray_steps),
                                  src_h, src_a, tail_steps, device, record=True)
        if Hsrc is not None:
            add("replay", p, [_replay_tail(agent, norm, hA, src_a, O, A, device),
                              _replay_tail(agent, norm, hS, src_a, O, A, device)], prefix=[0, 1])
        # common_state: both states from the source branch's physical state, closed loop
        t1, _, _ = _closed_tail(agent, norm, _authentic_from(src_w, seeds, params, ray_steps),
                                hA, src_a, tail_steps, device)
        t2, _, _ = _closed_tail(agent, norm, _authentic_from(src_w, seeds, params, ray_steps),
                                hS, src_a, tail_steps, device)
        if t1 is not None and t2 is not None:
            add("common_state", p, [t1, t2], prefix=[0, 1])
        # factorial: hidden origin x physical origin, previous action standardized to A's
        cells, ok = [], True
        for (h, yh), (w, yp) in (((hA, 0), (wA, 0)), ((hS, 1), (wS, 1)),
                                 ((hA, 0), (wS, 1)), ((hS, 1), (wA, 0))):
            H, _, _ = _closed_tail(agent, norm, _authentic_from(w, seeds, params, ray_steps),
                                   h, aA, tail_steps, device)
            ok &= H is not None
            cells.append((H, yh, yp))
        if ok:
            add("factorial", p, [c[0] for c in cells], hidden=[c[1] for c in cells],
                physical=[c[2] for c in cells])
        # reset_hidden: own physical state, zero hidden state and previous action
        r1, _, _ = _closed_tail(agent, norm, _authentic_from(wA, seeds, params, ray_steps),
                                zero_h, zero_a, tail_steps, device)
        r2, _, _ = _closed_tail(agent, norm, _authentic_from(wS, seeds, params, ray_steps),
                                zero_h, zero_a, tail_steps, device)
        if r1 is not None and r2 is not None:
            add("reset_hidden", p, [r1, r2], prefix=[0, 1])
    for c in CONDITIONS:
        H = out[c]["H"]
        out[c]["H"] = np.stack(H) if H else np.zeros((0, tail_steps, agent.hidden), np.float32)
        out[c]["groups"] = np.asarray(out[c]["groups"], dtype=int)
        out[c]["labels"] = {k: np.asarray(v, dtype=int) for k, v in lab[c].items()}
    return out


def score_condition(H: np.ndarray, y: np.ndarray, groups: np.ndarray, late_k: int = 8) -> dict:
    """AUROC of y from h_t at every tail step, plus window readouts, grouped by pair."""
    n_groups = len(np.unique(groups)) if len(groups) else 0
    res = {"n": int(len(y)), "n_pairs": int(n_groups), "auc_by_t": [], "window": float("nan"),
           "late": float("nan"), "first_t_below_055": None}
    if n_groups < 5:
        return res
    T = H.shape[1]
    res["auc_by_t"] = [grouped_auroc(H[:, t], y, groups) for t in range(T)]
    k = min(late_k, T)
    res["window"] = grouped_auroc(np.stack([_episode_feature(h) for h in H]), y, groups)
    res["late"] = grouped_auroc(np.stack([_episode_feature(h[-k:]) for h in H]), y, groups)
    below = [t + 1 for t, a in enumerate(res["auc_by_t"]) if a < 0.55]
    res["first_t_below_055"] = below[0] if below else None
    res["partition"] = folds.partition_record(groups, y)
    return res


def persistence_readout(agent, norm, params, drift_sigma, **kw) -> dict:
    """Collect every condition and score each of its labels."""
    data = collect_persistence(agent, norm, params, drift_sigma, **kw)
    out = {}
    for c, d in data.items():
        for name, y in d["labels"].items():
            out[f"{c}:{name}"] = score_condition(d["H"], y, d["groups"])
        if not d["labels"]:
            out[f"{c}:prefix"] = score_condition(d["H"], np.zeros(0, int), d["groups"])
    return out
