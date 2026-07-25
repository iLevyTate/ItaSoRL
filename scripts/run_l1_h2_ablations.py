"""L1 H2 substrate-grounding ablations (A1 graded-seam + A2 noise knockout).

READOUT-ONLY: loads saved L1 agents and scores world-identity probes under
a ladder of L1 grid spacings (A1) and a gate-calibrated unstructured observation-
noise surrogate (A2). No training.

Usage:
    python scripts/run_l1_h2_ablations.py \
        --agents-dir fullruns/l1_heldout/agents \
        --states-dir fullruns/l1_heldout/states \
        --noise-json fullruns/l1_noise_calib.json \
        --out-dir fullruns/l1_h2_ablations --device cuda
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl.experiment_b2 import (
    collect_pool,
    default_device,
    format_drift,
    load_agent_bundle,
    pooled_readout,
    transfer_readout,
)
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.stats import rope_test, t_ci90
from itasorl.surrogate_l1_families import LObsNoise
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def parse_agent_filename(name: str) -> tuple[float, int, str]:
    m = AGENT_RE.search(name)
    if not m:
        raise ValueError(f"unrecognized agent filename: {name}")
    return float(m.group(1)), int(m.group(2)), m.group(3)


def spearman_rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def gn_verdict(survival_mean: float, untrained_mean: float) -> dict:
    """Frozen L3-style decision rule applied to the noise channel."""
    clears_bar = survival_mean >= 0.65
    clears_floor = survival_mean > untrained_mean + 0.05
    if not clears_bar and not clears_floor:
        verdict = "H2_SUPPORTED"
    elif not clears_bar and clears_floor:
        verdict = "PARTIAL"
    elif clears_bar and clears_floor and untrained_mean < 0.60:
        verdict = "H2_NEGATIVE"
    else:
        verdict = "UNINFORMATIVE"
    return {
        "gn_rule_pass": bool(clears_bar and clears_floor),
        "gn_rule_margin": round(float(survival_mean - untrained_mean), 4),
        "gn_verdict": verdict,
    }


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--noise-json", default=None,
                    help="gate-0 noise calibration JSON (chosen_sigma). If absent or null, "
                         "noise channel is recorded as dropped.")
    ap.add_argument("--channels", nargs="+", default=["a1", "noise"],
                    choices=("a1", "noise"))
    ap.add_argument("--arms", nargs="+", default=["survival", "untrained", "predictor"],
                    choices=("survival", "untrained", "predictor"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--sensor-sigma", type=float, default=0.01)
    ap.add_argument("--deltas", type=float, nargs="+", default=None,
                    help="A1 graded-seam delta ladder (default: 0.0, 0.25, 0.5, 0.75, 1.0 of headline)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke: seed 0, deltas {0, headline}, tiny pools, no bit compare")
    return ap.parse_args()


def _headline_and_cells(a):
    cells = sorted(f for f in os.listdir(a.agents_dir) if AGENT_RE.search(f))
    cells = [c for c in cells if parse_agent_filename(c)[2] in a.arms]
    if not cells:
        raise SystemExit(f"no agents in {a.agents_dir}")
    drifts = sorted(set(parse_agent_filename(c)[0] for c in cells))
    if len(drifts) != 2 or 0.0 not in drifts:
        raise SystemExit(
            f"expected exactly two drifts (0.0 and headline) in {a.agents_dir}; got {drifts}")
    headline = [d for d in drifts if d != 0.0][0]
    cells = [c for c in cells if parse_agent_filename(c)[0] == headline]
    if a.quick:
        cells = [c for c in cells if "_s0_" in c]
    return headline, cells


def _run_a1(a, dev, headline, cells):
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    deltas = a.deltas if a.deltas is not None else [f * headline for f in fracs]
    if a.quick:
        deltas = [0.0, headline]
    deltas = sorted(set(float(d) for d in deltas))
    if 0.0 not in deltas:
        deltas = [0.0] + list(deltas)
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)
    rows = []
    integrity = {"checked": False, "pools_bit_match": None, "survival_mean_headline": None}
    deltas_ordered = ([headline] + [x for x in deltas if x != headline]) if headline in deltas else list(deltas)
    for delta in deltas_ordered:
        b2.L1_DELTA = delta
        bit_match_all = True
        for name in cells:
            _, seed, arm = parse_agent_filename(name)
            agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
            # drift_sigma triggers L1 mode; actual grid spacing is L1_DELTA.
            # At delta=0, quantization is a no-op (l1_delta>0 guard in PatchOfEarthV0).
            dsig = headline
            out, (Ha, Hs) = pooled_readout(
                agent, norm, P, dsig, n_eps=n_eps, steps=steps,
                device=dev, seed=seed, return_pools=True)
            if not a.quick and abs(delta - headline) < 1e-12:
                dump = os.path.join(
                    a.states_dir, f"states_d{format_drift(headline)}_s{seed}_{arm}.npz")
                if os.path.exists(dump):
                    saved = np.load(dump)
                    if not (np.array_equal(saved["Ha"], Ha) and np.array_equal(saved["Hs"], Hs)):
                        raise SystemExit(
                            f"INTEGRITY GATE FAILED at delta={delta}: pools differ from {dump}")
                else:
                    bit_match_all = False
            rows.append({"delta": delta, "seed": seed, "arm": arm,
                         "target": float(out["target"]),
                         "target_lo": float(out["target_lo"]), "target_hi": float(out["target_hi"]),
                         "pool_leak_clean": bool(out["pool_leak_clean"]),
                         "deaths_auth": int(out["deaths_auth"]), "deaths_surr": int(out["deaths_surr"])})
        done = [r["target"] for r in rows if r["delta"] == delta and r["arm"] == "survival"]
        print(f"  delta={delta:.4f}: survival targets " + " ".join(f"{t:.3f}" for t in done) +
              (f"  mean={np.mean(done):.3f}" if done else ""))
        if not a.quick and abs(delta - headline) < 1e-12 and "survival" in a.arms:
            mean_h = round(float(np.mean(done)), 3) if done else None
            integrity.update({"checked": True, "pools_bit_match": bit_match_all,
                              "survival_mean_headline": mean_h})
            print(f"integrity gate PASSED: headline survival mean {mean_h}")

    agg = {"deltas": [round(float(d), 5) for d in deltas], "arms": a.arms, "quick": a.quick,
           "n_eps": n_eps, "steps": steps, "headline_delta": round(float(headline), 5),
           "sensor_sigma": a.sensor_sigma, "integrity": integrity}
    per_delta = {arm: [] for arm in a.arms}
    for arm in a.arms:
        for delta in deltas:
            vals = [r["target"] for r in rows if abs(r["delta"] - delta) < 1e-12 and r["arm"] == arm]
            v = np.asarray(vals, float)
            m = float(v.mean()) if v.size else float("nan")
            lo, hi = t_ci90(v)
            key = f"{arm}_d{delta:.4f}"
            agg[f"{key}_per_seed"] = [round(float(x), 4) for x in v]
            agg[f"{key}_mean"] = round(m, 4)
            agg[f"{key}_tci90"] = [round(lo, 4), round(hi, 4)]
            agg[f"{key}_n_ge_065"] = int((v >= 0.65).sum())
            per_delta[arm].append(m)
    if "survival" in a.arms:
        sm = per_delta["survival"]
        agg["survival_curve"] = [round(x, 4) for x in sm]
        agg["survival_monotonicity_rho"] = round(spearman_rho(deltas, sm), 4)
        a0 = [r["target"] for r in rows if abs(r["delta"] - 0.0) < 1e-12 and r["arm"] == "survival"]
        rr = rope_test(a0, rope=(0.45, 0.55))
        agg["l0_anchor"] = {"mean": round(rr.mean, 4),
                            "hdi": [round(rr.hdi[0], 4), round(rr.hdi[1], 4)],
                            "p_in_rope": round(rr.p_in_rope, 4),
                            "accept_equiv": bool(rr.accept)}
    return rows, agg


def _run_noise(a, dev, headline, cells, noise_sigma: float):
    """A2: score frozen L1 world-identity probe on matched-band iid observation noise."""
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)
    rows = []
    b2.L1_DELTA = headline
    # Dummy heldout: transfer_readout requires a non-None heldout for the L3
    # G-swap path; under l1 mode make_world never attaches G, and our make_world
    # patch is what actually installs the noise surrogate.
    dummy_heldout = object()
    for name in cells:
        _, seed, arm = parse_agent_filename(name)
        agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
        Ha_train, _ = collect_pool(agent, norm, P, 0.0, n_eps, steps, dev, 800_000 + seed, 5)
        Hs_train, _ = collect_pool(agent, norm, P, headline, n_eps, steps, dev, 850_000 + seed, 5)
        old_make_world = b2.make_world

        def make_noise_world(params, drift_sigma, ray_steps, food_override=None, _seed=seed):
            if drift_sigma == 0.0:
                return old_make_world(params, 0.0, ray_steps, food_override)
            w = PatchOfEarthV0(params or P, drift_sigma=0.0, drift_mode="ar1",
                               l1_delta=headline, sensor_sigma=a.sensor_sigma)
            w.ray_steps = ray_steps
            for k, v in {**b2.SURVIVAL_METAB, **b2.SURVIVAL_FOOD, **(food_override or {})}.items():
                setattr(w, k, v)
            wrap = LObsNoise(w, sigma=noise_sigma, seed=960_000 + _seed)
            wrap.reseed(970_000 + _seed)
            return wrap

        b2.make_world = make_noise_world
        try:
            out = transfer_readout(
                agent, norm, P, headline, Ha_train, Hs_train,
                n_eps=n_eps, steps=steps, ray_steps=5, device=dev,
                seed=seed, dump_path=None, heldout=dummy_heldout,
                seed_base_auth=960_000 + seed, seed_base_surr=970_000 + seed)
        finally:
            b2.make_world = old_make_world
            b2.L1_DELTA = headline
        rows.append({"arm": arm, "seed": seed, "noise_sigma": noise_sigma,
                     "target": float(out["transfer_target"]),
                     "target_lo": float(out["transfer_lo"]),
                     "target_hi": float(out["transfer_hi"])})
        print(f"  noise {name}: target={out['transfer_target']:.3f}")

    agg = {"noise_sigma": round(float(noise_sigma), 5), "quick": a.quick,
           "n_eps": n_eps, "steps": steps, "gn_dropped_at_gate0": False}
    for arm in a.arms:
        vals = [r["target"] for r in rows if r["arm"] == arm]
        v = np.asarray(vals, float)
        m = float(v.mean()) if v.size else float("nan")
        lo, hi = t_ci90(v)
        agg[f"{arm}_mean"] = round(m, 4)
        agg[f"{arm}_per_seed"] = [round(float(x), 4) for x in v]
        agg[f"{arm}_n_ge_065"] = int((v >= 0.65).sum())
        agg[f"{arm}_tci90"] = [round(lo, 4), round(hi, 4)]
    if "survival" in a.arms and "untrained" in a.arms:
        agg.update(gn_verdict(agg["survival_mean"], agg["untrained_mean"]))
    return rows, agg


def main():
    a = cfg()
    dev = default_device() if a.device == "auto" else a.device
    if a.device == "cuda" and dev != "cuda":
        raise SystemExit("--device cuda requested but CUDA unavailable")
    os.makedirs(a.out_dir, exist_ok=True)
    b2.DRIFT_MODE = "l1"
    b2.SENSOR_SIGMA = a.sensor_sigma

    headline, cells = _headline_and_cells(a)
    print(f"L1 headline delta={headline:.5f}  sensor_sigma={a.sensor_sigma:.4f}  "
          f"arms={a.arms}  n_cells={len(cells)}  device={dev}  channels={a.channels}")

    if "a1" in a.channels:
        print("\n--- A1 graded-seam (delta ladder) ---")
        a1_rows, a1_agg = _run_a1(a, dev, headline, cells)
        with open(os.path.join(a.out_dir, "a1_cells.json"), "w") as f:
            json.dump(a1_rows, f, indent=1, default=float)
        with open(os.path.join(a.out_dir, "a1_aggregate.json"), "w") as f:
            json.dump(a1_agg, f, indent=1, default=float)
        print("wrote", os.path.join(a.out_dir, "a1_aggregate.json"))
        if "survival" in a.arms and not a.quick:
            print("survival curve (delta -> mean target):")
            for d, m in zip(a1_agg["deltas"], a1_agg["survival_curve"]):
                print(f"  delta={d:.5f}  target={m:.3f}")
            print(f"monotonicity rho = {a1_agg['survival_monotonicity_rho']}")

    if "noise" in a.channels:
        print("\n--- A2 noise knockout ---")
        noise_sigma = None
        if a.noise_json and os.path.exists(a.noise_json):
            with open(a.noise_json) as f:
                noise_sigma = json.load(f).get("chosen_sigma")
        if noise_sigma is None:
            dropped = {"gn_dropped_at_gate0": True, "gn_verdict": "DROPPED_AT_GATE0",
                       "noise_sigma": None}
            with open(os.path.join(a.out_dir, "noise_aggregate.json"), "w") as f:
                json.dump(dropped, f, indent=1)
            print("noise channel DROPPED at gate 0 (no in-band sigma); wrote",
                  os.path.join(a.out_dir, "noise_aggregate.json"))
        else:
            noise_rows, noise_agg = _run_noise(a, dev, headline, cells, float(noise_sigma))
            with open(os.path.join(a.out_dir, "noise_cells.json"), "w") as f:
                json.dump(noise_rows, f, indent=1, default=float)
            with open(os.path.join(a.out_dir, "noise_aggregate.json"), "w") as f:
                json.dump(noise_agg, f, indent=1, default=float)
            print("wrote", os.path.join(a.out_dir, "noise_aggregate.json"),
                  f"verdict={noise_agg.get('gn_verdict')}")


if __name__ == "__main__":
    main()
