"""Surrogate and detector diagnostics (revision step 10), agent free.

For every learned fingerprint instance the project used (hidden 8 G seed 0, hidden 7 G seed 0,
hidden 10 G seed 1, hidden 8 G seed 2) and for the comparators (Gaussian jitter at the
published gate-0 sigma, quadratic drag at a given eps), records on world P:
  * the linear-fit control and G's one-step RMS error on training and held-out authentic
    transitions, and the open-loop rollout divergence (`surrogate_l3.surrogate_diagnostics`);
  * the magnitude and temporal structure of the deviation (`perturbation_profile`);
  * an AGENT-ACCESSIBLE detector score: the velocity law checked from raw observations alone
    (`experiment_a_l3.observation_law_detector`) on scripted-policy pools of 110 + 110
    episodes x 24 steps (world seeds 800000 / 850000), with no detector-side noise.
The privileged gate-0 oracle, by contrast, reads the world's internal transitions with an
added detector-side noise of sigma = 0.02.

Usage:
    python scripts/run_surrogate_diagnostics.py --out artifacts/surrogate_diagnostics.json \\
        [--qd-eps 4.0] [--quick]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np

from itasorl.experiment_a_l3 import observation_law_detector
from itasorl.experiment_b import scripted_policy
from itasorl.experiment_b2 import _seeds
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.surrogate_l3 import surrogate_diagnostics, train_g_motion
from itasorl.surrogate_l3_families import make_g_gn, make_g_qd, perturbation_profile
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def scripted_obs_pool(g, seed_base: int, n: int, steps: int, ray_steps: int = 5) -> np.ndarray:
    out = []
    for i in range(n):
        w = PatchOfEarthV0(P, drift_sigma=0.45 if g is not None else 0.0, drift_mode="l3")
        w.ray_steps = ray_steps
        w._g_motion = g
        w.reset(_seeds(seed_base + i))
        rng = np.random.default_rng(600_000 + i)
        rows = [w.observe().astype(np.float32)]
        for _ in range(steps - 1):
            rows.append(w.step(scripted_policy(rng)).obs.astype(np.float32))
        out.append(np.stack(rows))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--qd-eps", type=float, default=None, help="gate-0 eps of the qd comparator")
    ap.add_argument("--gn-sigma", type=float, default=0.01)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n, steps = (30, 12) if a.quick else (110, 24)
    t0 = time.time()
    surrogates = {}
    for hidden, gseed in ((8, 0), (7, 0), (10, 1), (8, 2)):
        if a.quick and (hidden, gseed) != (8, 0):
            continue
        surrogates[f"gmotion_h{hidden}_s{gseed}"] = train_g_motion(hidden=hidden, seed=gseed, params=P)
    surrogates["gn_sigma%g" % a.gn_sigma] = make_g_gn(sigma_v=a.gn_sigma, params=P, seed=0)
    if a.qd_eps is not None:
        surrogates["qd_eps%g" % a.qd_eps] = make_g_qd(eps=a.qd_eps, params=P)
    auth = scripted_obs_pool(None, 800_000, n, steps)
    rows = {}
    for name, g in surrogates.items():
        diag = surrogate_diagnostics(g, params=P, heldout_eps=20 if a.quick else 60)
        prof = perturbation_profile(g, params=P, n_eps=20 if a.quick else 60)
        if hasattr(g, "reseed"):
            g.reseed(850_000)
        surr = scripted_obs_pool(g, 850_000, n, steps)
        det = observation_law_detector(auth, surr, dt=P.dt)
        rows[name] = {"diagnostics": diag, "profile": prof, "agent_accessible_detector": det}
        print(f"{name:20s} G one-step rms train {diag['g_one_step']['rms_train']:.5f} "
              f"held-out {diag['g_one_step']['rms_heldout']:.5f}  rollout@24 "
              f"{diag['rollout_rms_velocity_gap'].get('24', float('nan')):.4f}  linear-fit held-out "
              f"{diag['linear_fit']['rms_heldout']:.2e}  obs-law detector {det['auroc']:.3f}", flush=True)
    out = {"generated_by": "scripts/run_surrogate_diagnostics.py", "world": "WorldParams(k_land=1.5, "
           "k_water=1.5, gravity=0.4) [P]", "pools": {"n_per_pool": n, "steps": steps,
                                                       "seed_bases": [800_000, 850_000],
                                                       "policy": "scripted, rng 600000 + i"},
           "surrogates": rows, "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
