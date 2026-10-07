"""Calibrate the scarce rung of the goal-and-stakes runs (frozen rule in
docs/specs/2026-10-07-goal-and-stakes-design.md).

Sweeps n_pellets at a fixed basal burn and reports, per setting and per world (drift 0
authentic, drift 0.45 learned surrogate), the 80-step death rate of the scripted and the
random walker and the rate of deaths inside the first `--window` steps (the pooled readout
length). Chooses the largest n_pellets whose scripted death rate at drift 0 lies in the
band, whose drift-0.45 rate is within the tolerance of it, and with no early deaths.

    python scripts/calibrate_scarcity.py --out artifacts/goal_stakes/calibration.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os

import numpy as np

BAND = (0.40, 0.60)
TOL = 0.10


def death_stats(kind: str, drift: float, n_eps: int, max_steps: int, ray_steps: int,
                seed_base: int, window: int) -> dict:
    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b import scripted_policy
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    rng = np.random.default_rng(seed_base)
    died = early = 0
    lens = []
    for i in range(n_eps):
        w = b2.make_world(P, drift, ray_steps)
        w.reset(b2._seeds(seed_base + i))
        t = 0
        for t in range(1, max_steps + 1):
            if kind == "random":
                a = np.array([rng.uniform(0, 1), rng.uniform(-1, 1), float(rng.random() < 0.5),
                              float(rng.random() < 0.5), 0.0], np.float32)
            else:
                a = scripted_policy(rng)
            r = w.step(a)
            if r.terminated:
                died += 1
                early += int(t <= window)
                break
        lens.append(t)
    return {"death_rate": died / n_eps, "early_death_rate": early / n_eps,
            "mean_len": float(np.mean(lens))}


def choose(rows: list[dict], band=BAND, tol=TOL) -> dict | None:
    """rows ordered by descending n_pellets; the first row meeting the frozen rule wins."""
    for row in rows:
        s0, s1 = row["scripted"]["0.00"], row["scripted"]["0.45"]
        in_band = band[0] <= s0["death_rate"] <= band[1]
        matched = abs(s1["death_rate"] - s0["death_rate"]) <= tol
        no_early = s0["early_death_rate"] == 0.0 and s1["early_death_rate"] == 0.0
        if in_band and matched and no_early:
            return row
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pellets", type=int, nargs="+", default=[24, 16, 12, 8, 6, 4])
    ap.add_argument("--basal", type=float, default=0.4)
    ap.add_argument("--drift", type=float, default=0.45)
    ap.add_argument("--n-eps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=1_300_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import itasorl.experiment_b2 as b2
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=8, device="cpu", seed=0, params=P)
    b2.SURVIVAL_METAB["basal_E"] = a.basal
    rows = []
    for n in sorted(a.pellets, reverse=True):
        b2.SURVIVAL_FOOD["n_pellets"] = n
        row = {"n_pellets": n, "basal_E": a.basal, "scripted": {}, "random": {}}
        for kind in ("scripted", "random"):
            for d in (0.0, a.drift):
                row[kind][f"{d:.2f}"] = death_stats(kind, d, a.n_eps, a.max_steps, a.ray_steps,
                                                    a.seed_base, a.window)
        s = row["scripted"]
        print(f"n_pellets={n:3d}  scripted death 0.00={s['0.00']['death_rate']:.2f} "
              f"0.45={s['0.45']['death_rate']:.2f}  early={s['0.00']['early_death_rate']:.2f}/"
              f"{s['0.45']['early_death_rate']:.2f}", flush=True)
        rows.append(row)
    chosen = choose(rows)
    out = {"rule": {"band": BAND, "tol": TOL, "window": a.window, "basal_E": a.basal},
           "rows": rows, "chosen": chosen}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("chosen:", None if chosen is None else chosen["n_pellets"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
