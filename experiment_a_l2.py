"""
ITASORL - Experiment A for the L2 rollout-drift rung.

L2 alters the *dynamics* (a slowly drifting drag coefficient), unlike L1 which
only quantizes the observation. So the matched pair must run the world twice from
a shared snapshot (authentic vs. drifting), and detection is invariant-based: the
oracle re-applies the EXACT authentic one-step motion law and measures the
residual. With measurement noise on the logged velocity, the per-step residual is
~noise for authentic and grows with the drift for L2 - a chaos-robust ceiling.

Tamed config (high uniform drag, gravity off, no turn, central start) isolates the
drift signal cleanly; the same detector applies to the full world.
"""

from __future__ import annotations

import numpy as np

from experiment_a import grouped_auroc, leak_length, leak_meta, leak_reward
from patch_of_earth import PatchOfEarthV0
from world import SeedBundle, WorldParams

L2_PARAMS = dict(k_land=4.0, k_water=4.0, gravity=0.0, dt=0.05)
L2_POLICY = np.array([0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # forward thrust, no turn, no foraging
DRAG, DT, ASCALE = 4.0, 0.05, 0.2 * 0.6  # nominal drag, dt, thrust*thrust_scale (known to the oracle)


def generate_l2_pairs(n_pairs: int, drift_sigma: float, prefix: int = 10, branch: int = 30,
                      seed0: int = 2000) -> list[dict]:
    params = WorldParams(**L2_PARAMS)
    eps: list[dict] = []
    for i in range(n_pairs):
        sb = SeedBundle(world=seed0 + i, weather=seed0 + 5000 + i, ecology=seed0 + 9000 + i)
        base = PatchOfEarthV0(params)
        base.ray_steps = 2  # observations unused here; keep raymarch cheap
        base.reset(sb)
        base.heading = 0.0  # straight-line motion -> a is a known constant for the oracle
        for _ in range(prefix):
            base.step(L2_POLICY)
        snap = base.get_state()
        for label, ds in ((0, 0.0), (1, drift_sigma)):  # authentic, then surrogate
            w = PatchOfEarthV0(params, drift_sigma=ds)
            w.ray_steps = 2
            w.reset(sb)
            w.set_state(snap)
            vel, rew, ts = [], [], []
            for _ in range(branch):
                r = w.step(L2_POLICY)
                vel.append(w.vel.copy())
                rew.append(r.reward)
                ts.append(r.info["t"])
            eps.append({"pair": i, "label": label, "vel": np.asarray(vel),
                        "rew": np.asarray(rew), "t": np.asarray(ts)})
    return eps


def oracle_features_L2(vel: np.ndarray, sigma_meas: float, rng) -> np.ndarray:
    """Re-apply the authentic one-step law to the (noisy) velocity and summarize the
    residual. Authentic: residual ~ measurement noise. L2: residual = -drift*drag*dt*v,
    growing with the drag wander."""
    v = vel + rng.normal(0.0, sigma_meas, size=vel.shape)
    pred = np.empty_like(v[1:])
    pred[:, 0] = (1.0 - DRAG * DT) * v[:-1, 0] + ASCALE * DT
    pred[:, 1] = (1.0 - DRAG * DT) * v[:-1, 1]
    resid = v[1:] - pred
    # std/mean of the residual; the max order-statistic is dropped (noisy, inflates the control)
    return np.array([resid[:, 0].std(), resid[:, 1].std(), float(np.abs(resid).mean())])


def run_experiment_a_l2(eps: list[dict], sigma_meas: float = 2e-4, contaminate: float = 0.0,
                        seed: int = 0, leak_tol: float = 0.1) -> dict:
    rng = np.random.default_rng(seed)
    Xo, y, g, Xr, Xl, Xm = [], [], [], [], [], []
    for e in eps:
        Xo.append(oracle_features_L2(e["vel"], sigma_meas, rng))
        rew = e["rew"].copy()
        if e["label"] == 1:
            rew = rew + contaminate
        Xr.append(leak_reward(rew))
        Xl.append(leak_length(rew))
        Xm.append(leak_meta(e["t"]))
        y.append(e["label"])
        g.append(e["pair"])
    y = np.asarray(y); g = np.asarray(g)
    leaks = {
        "reward": grouped_auroc(np.asarray(Xr), y, g),
        "length": grouped_auroc(np.asarray(Xl), y, g),
        "metadata": grouped_auroc(np.asarray(Xm), y, g),
    }
    return {
        "oracle_auroc": grouped_auroc(np.asarray(Xo), y, g),
        "leakage": leaks,
        "leakage_pass": all(abs(a - 0.5) < leak_tol for a in leaks.values()),
    }
