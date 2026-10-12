"""Agent-accessible ceiling of the L1 rung (2026-10-12 audit).

The registered L1 oracle measures the grid residual of the continuous channels, and the paper
quotes it (0.873 at delta 0.023) as the detector ceiling. A channel that barely moves inside
an episode, Hyd/Hydmax, is snapped to one grid cell for the whole episode, so the surrogate
carries a constant offset on it that a per-episode mean reads and the grid-residual oracle
does not. This script measures both, on the registered scripted pools, with no trained agent.

Reports, at the registered delta and sensor sigma, seed bases 800000 and 850000:
  grid_residual_oracle   the registered detector, on the same pools
  episode_mean_probe     linear probe on the per-episode mean of the continuous channels
  hyd_ablated_probe      the same probe with the Hyd/Hydmax channel removed
  l0_control             the episode-mean probe on two authentic pools (must sit near 0.5)
  same_seed_twins        the episode-mean probe on same-seed authentic vs surrogate twins
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from itasorl.experiment_a import continuous_dims, oracle_features_L1
from itasorl.experiment_b import probe_auroc, scripted_policy
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.world import SeedBundle, WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
DELTA = 0.023               # the registered L1 grid spacing (PREREGISTRATION_L3)
SIGMA = 0.01
N_EPS = 110
STEPS = 24
BASE_AUTH = 800_000
BASE_SURR = 850_000


def _seeds(base: int) -> SeedBundle:
    return SeedBundle(world=base, weather=base + 7000, ecology=base + 13000)


def _rollout(drift_sigma: float, seed_base: int, delta: float, sigma: float,
             shared_noise: bool) -> np.ndarray:
    """Scripted episodes of exactly STEPS, observations stacked (n, STEPS, obs_dim)."""
    rows = []
    for i in range(N_EPS):
        w = PatchOfEarthV0(P, drift_sigma=drift_sigma, drift_mode="l1",
                           l1_delta=delta, sensor_sigma=sigma)
        w._obs_before_drift = shared_noise
        w.reset(_seeds(seed_base + i))
        rng = np.random.default_rng(seed_base + 500_000 + i)
        obs = [w.observe().copy()]
        for _ in range(STEPS - 1):
            w.step(scripted_policy(rng))
            obs.append(w.observe().copy())
        rows.append(np.stack(obs))
    return np.stack(rows)


def _episode_mean_auroc(auth: np.ndarray, surr: np.ndarray, dims: np.ndarray) -> float:
    X = np.concatenate([auth[:, :, dims].mean(1), surr[:, :, dims].mean(1)])
    y = np.r_[np.zeros(len(auth)), np.ones(len(surr))].astype(int)
    return probe_auroc(X, y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/expL1/observation_ceiling.json")
    a = ap.parse_args()
    dims = continuous_dims()
    hyd = int(np.where(dims == _hyd_index())[0][0])
    auth = _rollout(0.0, BASE_AUTH, DELTA, SIGMA, shared_noise=False)
    surr = _rollout(DELTA, BASE_SURR, DELTA, SIGMA, shared_noise=False)
    auth_l0 = _rollout(0.0, BASE_SURR, DELTA, SIGMA, shared_noise=False)
    twins_a = _rollout(0.0, BASE_AUTH, DELTA, SIGMA, shared_noise=True)
    twins_s = _rollout(DELTA, BASE_AUTH, DELTA, SIGMA, shared_noise=True)

    def oracle(pool_a, pool_s):
        fa = np.stack([oracle_features_L1(e, dims, DELTA) for e in pool_a])
        fs = np.stack([oracle_features_L1(e, dims, DELTA) for e in pool_s])
        X = np.concatenate([fa, fs])
        y = np.r_[np.zeros(len(fa)), np.ones(len(fs))].astype(int)
        return probe_auroc(X, y)

    kept = np.delete(np.arange(len(dims)), hyd)
    out = {
        "delta": DELTA, "sensor_sigma": SIGMA, "n_eps": N_EPS, "steps": STEPS,
        "seed_bases": [BASE_AUTH, BASE_SURR], "policy": "scripted",
        "grid_residual_oracle": oracle(auth, surr),
        "episode_mean_probe": _episode_mean_auroc(auth, surr, dims),
        "hyd_ablated_probe": _episode_mean_auroc(auth, surr, dims[kept]),
        "l0_control": _episode_mean_auroc(auth, auth_l0, dims),
        "same_seed_twins": _episode_mean_auroc(twins_a, twins_s, dims),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


def _hyd_index() -> int:
    """Index of the Hyd/Hydmax channel: the sixth interoception entry (patch_of_earth)."""
    from itasorl.patch_of_earth import first_config_obs_spec
    return first_config_obs_spec().slices()["intero"].start + 5


if __name__ == "__main__":
    raise SystemExit(main())
