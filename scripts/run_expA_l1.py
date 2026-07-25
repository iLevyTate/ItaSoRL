"""L1 discretization oracle calibration for the B-v2 world.

Generates agent-free fixed-policy trajectories under the B-v2 world parameters,
adds sensor noise, and measures the oracle detectability of an L1 grid at
varying spacings. The goal is a delta in the [0.85, 0.95] AUROC band for the
organism training run.

Usage:
    python scripts/run_expA_l1.py --out fullruns/l1_calib.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os

import numpy as np

from itasorl.experiment_a import (
    CONST_POLICY,
    continuous_dims,
    grouped_auroc,
    leak_length,
    leak_meta,
    leak_reward,
    oracle_features_L1,
)
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.world import SeedBundle, WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
SIGMA = 0.01
SEED0 = 1000
N_PAIRS = 200
STEPS = 30
RAY_STEPS = 20


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


def evaluate(clean: list[dict], delta: float, sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dims = continuous_dims()
    Xo, Xrew, Xlen, Xmeta, y, g = [], [], [], [], [], []
    for d in clean:
        for label in (0, 1):  # 0 = authentic, 1 = surrogate
            quant = (label == 1)
            sig = np.round(d["obs"] / delta) * delta if quant else d["obs"]
            obs = sig + rng.normal(0.0, sigma, size=d["obs"].shape)
            Xo.append(oracle_features_L1(obs, dims, delta))
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
        "delta": delta,
        "sigma": sigma,
        "oracle_auroc": grouped_auroc(np.asarray(Xo), y, g),
        "leakage": leaks,
        "leakage_pass": all(abs(a - 0.5) < 0.1 for a in leaks.values()),
    }


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSON path to write calibration results")
    ap.add_argument("--sensor-sigma", type=float, default=SIGMA)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015])
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--steps", type=int, default=STEPS)
    return ap.parse_args()


def main():
    a = cfg()
    print(f"Generating {a.n_pairs} clean pairs x {a.steps} steps under B-v2 world params...")
    clean = generate_clean(P, a.n_pairs, a.steps, SEED0, RAY_STEPS)
    results = []
    for delta in a.deltas:
        r = evaluate(clean, delta, a.sensor_sigma, seed=SEED0)
        results.append(r)
        print(f"  delta={delta:.4f}  AUROC={r['oracle_auroc']:.3f}  "
              f"leakage={r['leakage']}")
    chosen = None
    for r in results:
        if 0.85 <= r["oracle_auroc"] <= 0.95:
            chosen = r["delta"]
            print(f"Chosen in-band delta: {chosen:.4f}")
            break
    payload = {
        "sensor_sigma": a.sensor_sigma,
        "deltas_sweep": results,
        "chosen_delta": chosen,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
