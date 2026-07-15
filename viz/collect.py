"""Exporter: real recorded runs -> viz/data/scene.json for the outreach film player.

Record-then-render (spec docs/superpowers/specs/2026-07-14-outreach-video-design.md):
rolls a saved survival agent in the authentic world and the L3 learned-surrogate
world on identical seeds, samples the terrain fields the player bakes, and fails
loudly if the film's caption numbers drift from the committed artifacts.
CPU only: fullruns/ is read-only and the GPU may be busy with a live run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT_ROOT))

from itasorl.world import WorldParams  # noqa: E402

# Frozen world params, identical to scripts/run_expB2.py:57.
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RAY_STEPS = 5           # run_expB2.py --ray_steps default; the eval standard
ENERGY_FULL = 1.0       # SURVIVAL_METAB E0; player energy bar expects [0, 1]
# Beat 2 (trick, 10s-25s at step_ms=100) shows a 477px center strip of the
# 960px world; x outside [0.25, 0.75] is clipped. 0.28/0.72 leaves a margin.
TRICK_I0, TRICK_I1 = 100, 250
SPLIT_LO, SPLIT_HI = 0.28, 0.72
LAST_WORLD_MS = 75_000  # the world is on screen through beat 5 (55s-75s)
STEP_MS = 100


def sample_fields(world, n):
    """Sample the reset world's analytic height/wetness on an n*n row-major grid
    (index j*n+i, x=i/(n-1), y=j/(n-1)) to mirror player.js placeholderScene."""
    height = np.empty(n * n)
    wet = np.empty(n * n)
    for j in range(n):
        y = j / (n - 1)
        for i in range(n):
            x = i / (n - 1)
            height[j * n + i] = world._H(x, y)
            wet[j * n + i] = 1.0 if world._wetness(x, y) > 0.5 else 0.0
    height = (height - height.min()) / (height.max() - height.min())
    return height, wet


def rollout(agent, norm, world, steps, device="cpu"):
    """Deterministic policy rollout; per-step [x, y, heading, energy] plus active
    pellet positions. Returns None if the creature dies (caller tries new seeds).
    Mirrors common_garden_rollout (itasorl/experiment_b2.py:869-887)."""
    import torch
    h = agent.initial_state(1, device)
    prev = torch.zeros(1, agent.act_dim, device=device)
    obs = world.observe().astype(np.float64)
    pts, pellets_t = [], []
    for _ in range(steps):
        obs_t = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
        _, env_act, _, _, h = agent.act(obs_t, prev, h, deterministic=True)
        r = world.step(env_act[0].detach().cpu().numpy().astype(np.float32))
        obs = r.obs.astype(np.float64)
        prev = env_act
        pts.append([float(world.pos[0]), float(world.pos[1]), float(world.heading),
                    min(1.0, max(0.0, float(world.E) / ENERGY_FULL))])
        active = world.pellet_amt > 1e-9
        pellets_t.append([[round(float(x), 4), round(float(y), 4)]
                          for x, y in world.pellets[active]])
        if r.terminated:
            return None
    return {"pts": pts, "pellets_t": pellets_t}


def in_split_window(pts, i0=TRICK_I0, i1=TRICK_I1, lo=SPLIT_LO, hi=SPLIT_HI):
    """True if the creature stays visible in the split-beat center strip."""
    return all(lo <= p[0] <= hi for p in pts[i0:i1 + 1])


def _pooled(cells, agent, key):
    vals = [float(c[key]) for c in cells
            if c["agent"] == agent and abs(float(c["drift"]) - 0.45) < 1e-9]
    if not vals:
        raise SystemExit("number honesty: no drift=0.45 cells for agent '" + agent + "'")
    return sum(vals) / len(vals)


def verify_numbers(beats, cells, findings_text):
    """Binding spec rule: the film's three numbers must match the committed
    artifacts. Fails loudly; returns the verified values for scene meta."""
    disp = {b["id"]: b["gauge"]["display"] for b in beats["beats"] if "gauge" in b}
    nums = beats["numbers"]

    surv = _pooled(cells, "survival", "resid_trace")
    if f"{surv:.2f}" != disp["survival"] or disp["survival"] != nums["probe_survival"]["display"]:
        raise SystemExit("number honesty: survival resid_trace pools to "
                         + f"{surv:.4f}, film shows " + disp["survival"])

    untr = _pooled(cells, "untrained", "target")
    if (abs(untr - 0.50) > 0.03 or disp["nocare"] != "0.50"
            or nums["probe_chance"]["display"] != "0.50"):
        raise SystemExit("number honesty: untrained target pools to "
                         + f"{untr:.4f}, film shows " + disp["nocare"] + " as chance")

    if ("0.993" not in findings_text or disp["observer"] != "0.99"
            or nums["oracle_l2"]["display"] != "0.99"):
        raise SystemExit("number honesty: 0.993 missing from FINDINGS or gauge is not 0.99")

    return {"survival_resid_trace_pooled": round(surv, 4),
            "untrained_target_pooled": round(untr, 4),
            "oracle_l2": 0.993}


def _round_traj(pts):
    return [[round(x, 4), round(y, 4), round(a, 3), round(e, 3)] for x, y, a, e in pts]


def build_scene(meta, grid_n, height, wet, ra, rs, step_ms=STEP_MS):
    scene = {
        "meta": meta,
        "grid_n": grid_n,
        "height": [round(float(h), 4) for h in height],
        "wet": [float(w) for w in wet],
        "pellets": ra["pellets_t"][0],
        "trajs": {"auth": _round_traj(ra["pts"]), "surr": _round_traj(rs["pts"])},
        "pellets_t": {"auth": ra["pellets_t"], "surr": rs["pellets_t"]},
        "step_ms": step_ms,
    }
    assert len(scene["trajs"]["auth"]) * step_ms >= LAST_WORLD_MS, \
        "trajectory too short: the world is on screen through beat 5"
    assert len(scene["trajs"]["surr"]) * step_ms >= LAST_WORLD_MS
    return scene
