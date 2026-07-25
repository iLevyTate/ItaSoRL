"""Calibrate an iid observation-noise surrogate for the L1 H2 knockout.

The noise surrogate is authentic observations + extra iid Gaussian noise. The oracle
uses temporal variance / high-frequency features (NOT the L1 grid residual), so the
tell is the noise magnitude itself — matched-band detectability with zero grid
structure.

Usage:
    python scripts/run_expA_l1_noise.py --headline-delta 0.023 --out fullruns/l1_noise_calib.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os

import numpy as np

from itasorl.experiment_a import (
    CONST_POLICY,
    grouped_auroc,
    leak_length,
    leak_meta,
    leak_reward,
)
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.world import SeedBundle, WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
SEED0 = 2000
N_PAIRS = 200
STEPS = 30
RAY_STEPS = 20
SENSOR_SIGMA = 0.01


def generate_clean(params: WorldParams, n_pairs: int, steps: int, seed0: int,
                   ray_steps: int) -> list[dict]:
    data = []
    for i in range(n_pairs):
        w = PatchOfEarthV0(params)
        w.ray_steps = ray_steps
        w.reset(SeedBundle(world=seed0 + i, weather=seed0 + 5000 + i, ecology=seed0 + 9000 + i))
        obs, rew, ts = [], [], []
        for _ in range(steps):
            r = w.step(CONST_POLICY)
            obs.append(r.obs.copy())
            rew.append(r.reward)
            ts.append(r.info["t"])
        data.append({"pair": i, "obs": np.asarray(obs), "rew": np.asarray(rew), "t": np.asarray(ts)})
    return data


def oracle_features_obs_noise(obs_TD: np.ndarray) -> np.ndarray:
    """Variance-inflation / temporal high-frequency tell for iid observation noise."""
    d = np.diff(obs_TD, axis=0)
    return np.array([
        float(np.std(obs_TD)),
        float(np.std(d)),
        float(np.mean(np.abs(d))),
        float(np.percentile(np.abs(d), 90)),
    ])


def evaluate(clean: list[dict], headline_delta: float, sensor_sigma: float,
             noise_sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    Xo, Xrew, Xlen, Xmeta, y, g = [], [], [], [], [], []
    for d in clean:
        for label in (0, 1):  # 0 = authentic (sensor noise only), 1 = noise surrogate
            obs = d["obs"] + rng.normal(0.0, sensor_sigma, size=d["obs"].shape)
            if label == 1:
                obs = obs + rng.normal(0.0, noise_sigma, size=d["obs"].shape)
            Xo.append(oracle_features_obs_noise(obs))
            Xrew.append(leak_reward(d["rew"]))
            Xlen.append(leak_length(d["rew"]))
            Xmeta.append(leak_meta(d["t"]))
            y.append(label)
            g.append(d["pair"])
    y = np.asarray(y)
    g = np.asarray(g)
    leaks = {
        "reward": grouped_auroc(np.asarray(Xrew), y, g),
        "length": grouped_auroc(np.asarray(Xlen), y, g),
        "metadata": grouped_auroc(np.asarray(Xmeta), y, g),
    }
    return {
        "headline_delta": headline_delta,
        "sensor_sigma": sensor_sigma,
        "noise_sigma": noise_sigma,
        "oracle_auroc": grouped_auroc(np.asarray(Xo), y, g),
        "leakage": leaks,
        "leakage_pass": all(abs(a - 0.5) < 0.1 for a in leaks.values()),
    }


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--headline-delta", type=float, required=True)
    ap.add_argument("--sensor-sigma", type=float, default=SENSOR_SIGMA)
    ap.add_argument("--noise-sigmas", type=float, nargs="+",
                    default=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
                             0.06, 0.08, 0.10, 0.12, 0.15])
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--steps", type=int, default=STEPS)
    return ap.parse_args()


def main():
    a = cfg()
    print(f"Generating {a.n_pairs} clean pairs for noise calibration "
          f"(headline delta={a.headline_delta})...")
    clean = generate_clean(P, a.n_pairs, a.steps, SEED0, RAY_STEPS)
    results = []
    for noise_sigma in a.noise_sigmas:
        r = evaluate(clean, a.headline_delta, a.sensor_sigma, noise_sigma, seed=SEED0)
        results.append(r)
        print(f"  noise_sigma={noise_sigma:.4f}  AUROC={r['oracle_auroc']:.3f}  "
              f"leakage={r['leakage']}")
    chosen = None
    for r in results:
        if 0.85 <= r["oracle_auroc"] <= 0.95 and r["leakage_pass"]:
            chosen = r["noise_sigma"]
            print(f"Chosen in-band noise_sigma: {chosen:.4f}")
            break
    payload = {
        "headline_delta": a.headline_delta,
        "sensor_sigma": a.sensor_sigma,
        "noise_sigmas_sweep": results,
        "chosen_sigma": chosen,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
