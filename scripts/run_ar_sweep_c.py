"""Sweep C: coherence at a matched 24-step gap with independent episodes, on saved C1 agents.

Frozen design: docs/specs/2026-10-15-coherence-finite-horizon-companion-design.md. Readout-only.
Runs scripts/run_texture_fresh_probe.py --family arc at six rho values, with sigma(rho) chosen
so the family's 24-step open-loop RMS velocity gap equals the shared point's (sweep A at
rho = 0.9), each episode drawn afresh from the AR(1) stationary distribution, and writes
summary.json with the margin M = survival minus untrained fresh probe by point, the contrast
M(0.99) - M(0), and the three pre-stated readings.

    python scripts/run_ar_sweep_c.py --workers 2
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from run_ar_sweep import RHOS, SIGMA_A, contrast, margin_by_seed

#: the companion sweep's seed bases (authentic, G_ar); listed in itasorl.post_review.BASES_IN_USE
BASES = (2_500_000, 2_550_000)
#: sweep A's recorded 24-step open-loop RMS velocity gap at each rho (artifacts/texture/ar_sweep/summary.json)
GAP_A = {0.0: 0.0364, 0.5: 0.0598, 0.8: 0.0929, 0.9: 0.1150, 0.95: 0.1302, 0.99: 0.1398}
GAP_TARGET = GAP_A[0.9]
GAP_TOL = 0.03          # a measured gap outside 3 percent of the target is recalibrated once
OUT = "artifacts/texture/ar_sweep_c"
SPEC = "docs/specs/2026-10-15-coherence-finite-horizon-companion-design.md"


def points() -> list:
    """sigma(rho) = sigma_A x gap_target / gap_A(rho): the perturbation is additive and linear in
    sigma, so the 24-step gap scales with it. The rho = 0.9 point is sweep A's shared setting."""
    return [{"sweep": "C", "rho": r, "sigma": SIGMA_A * GAP_TARGET / GAP_A[r],
             "file": f"C_rho{r:.2f}.json"} for r in RHOS]


def profile(sigma: float, rho: float) -> dict:
    """The family's measured profile under the per-episode draw: one-step RMS, lag-1
    autocorrelation, and the open-loop velocity gap at 1, 10 and 24 steps."""
    from run_texture_fresh_probe import per_episode_draw

    from itasorl.surrogate_l3 import surrogate_diagnostics
    from itasorl.surrogate_l3_families import make_g_ar, perturbation_profile
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    g = make_g_ar(sigma=sigma, rho=rho, params=P, seed=0)
    with per_episode_draw(g, base=BASES[1]):
        prof = perturbation_profile(g, params=P)
        g.reseed(0)
        diag = surrogate_diagnostics(g, params=P, horizons=(1, 10, 24))
    return {"rms_one_step": prof["rms_one_step"], "lag1_autocorr": prof["lag1_autocorr"],
            "rollout_gap": diag["rollout_rms_velocity_gap"]}


def calibrate(p: dict) -> dict:
    """Measure the point's 24-step gap; if it misses the target by more than GAP_TOL, take one
    proportional step and measure again. Both measurements are recorded."""
    prof = profile(p["sigma"], p["rho"])
    gap = float(prof["rollout_gap"]["24"] if isinstance(prof["rollout_gap"], dict) else prof["rollout_gap"])
    rec = {"sigma_closed_form": p["sigma"], "gap_closed_form": gap, "corrected": False}
    if abs(gap - GAP_TARGET) > GAP_TOL * GAP_TARGET:
        p["sigma"] = p["sigma"] * GAP_TARGET / gap
        prof = profile(p["sigma"], p["rho"])
        gap = float(prof["rollout_gap"]["24"] if isinstance(prof["rollout_gap"], dict) else prof["rollout_gap"])
        rec.update(corrected=True, sigma_corrected=p["sigma"], gap_corrected=gap)
    rec["within_tolerance"] = bool(abs(gap - GAP_TARGET) <= GAP_TOL * GAP_TARGET)
    return {**rec, "profile": prof}


def readings(c: dict) -> dict:
    lo, hi = c["t90"]
    r1 = bool(lo > 0)
    r2 = bool(-0.05 <= lo and hi <= 0.05)
    return {"1 coherence raises at matched gap": r1, "2 within the margin unit": r2,
            "3 neither": bool(not r1 and not r2)}


def _point_matches(doc: dict, p: dict, agents_dir: str) -> bool:
    """A point file records its inputs at the top level (scripts/run_texture_fresh_probe.py),
    with the agents directory as a repo-relative label."""
    from promote_bv3_gates import repo_rel
    return (doc.get("family") == "arc" and doc.get("rho") == p["rho"]
            and abs(float(doc.get("param", np.nan)) - p["sigma"]) < 1e-12
            and tuple(doc.get("seed_bases", ())) == BASES and doc.get("agents_dir") == repo_rel(agents_dir))


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
    pts = points()
    cal = {}
    for p in pts:
        cal[p["file"]] = calibrate(p)
        print(f"calibrated rho {p['rho']} sigma {p['sigma']:.5f} gap {cal[p['file']]['profile']['rollout_gap']}", flush=True)
    for p in pts:
        path = os.path.join(a.out_dir, p["file"])
        if os.path.exists(path):
            doc = json.load(open(path, encoding="utf-8"))
            if not _point_matches(doc, p, a.agents_dir):
                raise RuntimeError(f"{path} exists but was written for different inputs; move it aside")
            continue
        print(f"== sweep C rho {p['rho']} sigma {p['sigma']:.5f}", flush=True)
        subprocess.run([sys.executable, "scripts/run_texture_fresh_probe.py", "--agents-dir", a.agents_dir,
                        "--family", "arc", "--param", repr(p["sigma"]), "--rho", repr(p["rho"]),
                        "--workers", str(a.workers), "--out", path], check=True)
    summ = {"spec": SPEC, "gap_target": GAP_TARGET, "gap_tolerance": GAP_TOL, "seed_bases": list(BASES),
            "points": []}
    M = {}
    for p in pts:
        doc = json.load(open(os.path.join(a.out_dir, p["file"]), encoding="utf-8"))
        m = margin_by_seed(doc)
        M[p["rho"]] = m
        agg = doc["aggregate"]
        mv = np.array(list(m.values()), float)
        sf = agg.get("survival fresh") or {}
        summ["points"].append({**p, "calibration": cal[p["file"]],
                               "survival_fresh": sf,
                               "untrained_fresh_mean": (agg.get("untrained fresh") or {}).get("mean"),
                               "survival_transfer_mean": (agg.get("survival transfer") or {}).get("mean"),
                               "M": {"per_seed": mv.tolist(), "mean": float(mv.mean()),
                                     "t90": [float(x) for x in t_ci90(mv)]},
                               "kept": bool(sf.get("mean", 0) >= 0.65 and mv.mean() >= 0.05)})
    c = contrast(M[0.99], M[0.0])
    summ["contrast"] = {"C: M(0.99) - M(0)": c}
    summ["readings"] = readings(c)
    summ["shared_point_check"] = {"rho_0.9_M": next(p["M"]["mean"] for p in summ["points"] if p["rho"] == 0.9),
                                  "rho_0.9_survival_fresh": next(p["survival_fresh"].get("mean") for p in summ["points"] if p["rho"] == 0.9),
                                  "companion_sweep_shared_point": {"M": 0.010, "survival_fresh": 0.570}}
    summ["provenance"] = provenance("scripts/run_ar_sweep_c.py", "python " + " ".join(["scripts/run_ar_sweep_c.py", *sys.argv[1:]]),
                                    {"agents_dir": repo_rel(a.agents_dir), "rhos": list(RHOS), "gap_target": GAP_TARGET,
                                     "seed_bases": list(BASES)}, device="cpu")
    summ["provenance"]["status"] = (f"readout under {SPEC}; diagnostic, not an adjudication; "
                                    "changes no registered rule or verdict")
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    for p in summ["points"]:
        print(f"C rho {p['rho']:.2f} sigma {p['sigma']:.5f} gap {p['calibration']['profile']['rollout_gap']} "
              f"surv fresh {p['survival_fresh'].get('mean', float('nan')):.3f} M {p['M']['mean']:+.3f} {p['M']['t90']}")
    print(summ["contrast"], summ["readings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
