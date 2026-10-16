"""Coherence or integration: the correlated-jitter sweep on saved C1 agents.

Frozen design: docs/specs/2026-10-15-coherence-integration-sweep-design.md. Readout-only.
Runs scripts/run_texture_fresh_probe.py --family ar at eleven settings (sweep A: one-step size
matched; sweep B: accumulated offset matched; rho = 0.9 shared), records each family's
measured profile, and writes summary.json with the margin M = survival minus untrained fresh
probe by point and the three pre-stated readings.

    python scripts/run_ar_sweep.py --workers 2
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import subprocess
import sys

import numpy as np

RHOS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)
SIGMA_A = 0.01
RHO_MATCH = 0.9
OUT = "artifacts/texture/ar_sweep"


def points():
    from itasorl.surrogate_l3_families import ar_accumulated_sd, sigma_for_accumulated_sd
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    lam = 1.0 - P.k_land * P.dt
    target = ar_accumulated_sd(SIGMA_A, RHO_MATCH, lam)
    pts = [{"sweep": "A", "rho": r, "sigma": SIGMA_A} for r in RHOS]
    pts += [{"sweep": "B", "rho": r, "sigma": sigma_for_accumulated_sd(target, r, lam)}
            for r in RHOS if r != RHO_MATCH]
    for p in pts:
        p["file"] = "A_rho0.90.json" if p["rho"] == RHO_MATCH else f"{p['sweep']}_rho{p['rho']:.2f}.json"
        p["accumulated_sd"] = ar_accumulated_sd(p["sigma"], p["rho"], lam)
    return pts, lam, target


def profile(sigma: float, rho: float) -> dict:
    from itasorl.surrogate_l3 import surrogate_diagnostics
    from itasorl.surrogate_l3_families import make_g_ar, perturbation_profile
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    g = make_g_ar(sigma=sigma, rho=rho, params=P, seed=0)
    prof = perturbation_profile(g, params=P)
    g.reseed(0)
    diag = surrogate_diagnostics(g, params=P, horizons=(1, 10, 24))
    return {"rms_one_step": prof["rms_one_step"], "lag1_autocorr": prof["lag1_autocorr"],
            "rollout_gap": diag["rollout_rms_velocity_gap"]}


def margin_by_seed(doc: dict) -> dict:
    by = {}
    for c in doc["cells"]:
        by.setdefault(c["seed"], {})[c["arm"]] = c["fresh"]
    return {s: v["survival"] - v["untrained"] for s, v in by.items() if "survival" in v and "untrained" in v}


def contrast(hi: dict, lo: dict) -> dict:
    from itasorl.stats import t_ci90
    seeds = sorted(set(hi) & set(lo))
    x = np.array([hi[s] - lo[s] for s in seeds], float)
    x = x[np.isfinite(x)]
    lo_, hi_ = t_ci90(x) if len(x) > 1 else (float("nan"), float("nan"))
    return {"mean": float(x.mean()) if len(x) else float("nan"), "t90": [float(lo_), float(hi_)], "n": int(len(x))}


def check_point(path: str, p: dict, agents_dir: str) -> None:
    """An existing point file is reused only when it was written for this point: same family,
    rho, sigma, agents directory and seed bases. A mismatch raises instead of being skipped."""
    from promote_bv3_gates import repo_rel
    from run_texture_fresh_probe import BASES
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    want = {"family": "ar", "rho": p["rho"], "param": p["sigma"], "agents_dir": repo_rel(agents_dir),
            "seed_bases": list(BASES["ar"])}
    bad = [k for k, v in want.items()
           if (abs(doc.get(k, float("nan")) - v) > 1e-12 if isinstance(v, float) else doc.get(k) != v)]
    if bad:
        raise RuntimeError(f"{path} was written for different inputs; differing fields: {bad}")


def main() -> int:
    from promote_bv3_gates import repo_rel

    from itasorl.optional_readouts import provenance
    from itasorl.stats import t_ci90
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", default="fullruns/corrected_l3_h8_wm/agents")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    pts, lam, target = points()
    for p in pts:
        path = os.path.join(a.out_dir, p["file"])
        if os.path.exists(path):
            check_point(path, p, a.agents_dir)
        else:
            print(f"== sweep {p['sweep']} rho {p['rho']} sigma {p['sigma']:.5f}", flush=True)
            subprocess.run([sys.executable, "scripts/run_texture_fresh_probe.py", "--agents-dir", a.agents_dir,
                            "--family", "ar", "--param", repr(p["sigma"]), "--rho", repr(p["rho"]),
                            "--workers", str(a.workers), "--out", path], check=True)
    summ = {"lambda": lam, "accumulated_sd_target": target, "points": []}
    M = {}
    for p in pts:
        doc = json.load(open(os.path.join(a.out_dir, p["file"]), encoding="utf-8"))
        m = margin_by_seed(doc)
        M[(p["sweep"], p["rho"])] = m
        if p["rho"] == RHO_MATCH:
            M[("B", RHO_MATCH)] = m
        agg = doc["aggregate"]
        mv = np.array(list(m.values()), float)
        summ["points"].append({**p, "profile": profile(p["sigma"], p["rho"]),
                               "survival_fresh": agg.get("survival fresh"),
                               "untrained_fresh_mean": (agg.get("untrained fresh") or {}).get("mean"),
                               "survival_transfer_mean": (agg.get("survival transfer") or {}).get("mean"),
                               "M": {"per_seed": mv.tolist(), "mean": float(mv.mean()),
                                     "t90": [float(x) for x in t_ci90(mv)]},
                               "kept": bool((agg.get("survival fresh") or {}).get("mean", 0) >= 0.65
                                            and mv.mean() >= 0.05)})
    cA = contrast(M[("A", 0.99)], M[("A", 0.0)])
    cB = contrast(M[("B", 0.99)], M[("B", 0.0)])
    r1 = cB["t90"][0] > 0
    r2 = (cB["t90"][0] <= 0 <= cB["t90"][1]) and cA["t90"][0] > 0
    r3 = cA["t90"][0] <= 0 <= cA["t90"][1]
    summ["contrasts"] = {"A: M(0.99) - M(0)": cA, "B: M(0.99) - M(0)": cB}
    summ["readings"] = {"1 coherence beyond integration": r1, "2 integration suffices": r2, "3 neither": r3}
    summ["provenance"] = provenance("scripts/run_ar_sweep.py", "python " + " ".join(["scripts/run_ar_sweep.py", *sys.argv[1:]]),
                                    {"agents_dir": repo_rel(a.agents_dir), "rhos": list(RHOS), "sigma_A": SIGMA_A,
                                     "rho_match": RHO_MATCH}, device="cpu")
    summ["provenance"]["status"] = ("readout under docs/specs/2026-10-15-coherence-integration-sweep-design.md; "
                                    "diagnostic, not an adjudication; changes no registered rule or verdict")
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    for p in summ["points"]:
        print(f"{p['sweep']} rho {p['rho']:.2f} sigma {p['sigma']:.5f} lag1 {p['profile']['lag1_autocorr']:.2f} "
              f"surv fresh {p['survival_fresh']['mean']:.3f} M {p['M']['mean']:+.3f} {p['M']['t90']}")
    print(summ["contrasts"], summ["readings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
