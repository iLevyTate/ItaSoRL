"""Sweep C follow-up: rho = 0.95 and 0.99 rerun at the exactly matched 24-step gap.

Readout-only, on the saved C1 agents, beside the frozen sweep C result
(docs/specs/2026-10-15-coherence-finite-horizon-companion-design.md), never in place of it.
Sweep C's calibration (scripts/run_ar_sweep_c.py, profile) measured the 24-step gap with the
per-episode draw inactive, so its "measured" gap reproduced sweep A's numbers by construction
(docs/CORRECTIONS.md, 2026-10-16). For G_ar the open-loop gap is exact in closed form: the
perturbation is additive and the authentic law is linear with decay lam = 1 - drag * dt, so
after H steps from a stationary start the velocity gap is sum_t lam^(H-1-t) eta_t, whose RMS
over both components is sigma * sqrt(2 w'Cw) with w_t = lam^(H-1-t) and C_ij = rho^|i-j|.

The target is the shared point's exact gap (rho = 0.9 at sweep A's sigma, which sweep C ran
unchanged). The four lower-rho points sit within 1.1 percent of it and are not rerun. This
script reruns 0.95 and 0.99 at the sigma that matches it exactly, writes them to their own
directory, and scores the contrast M(0.99) - M(0) with the new rho = 0.99 point and the stored
rho = 0 point, under the frozen spec's three readings.

    python scripts/run_ar_sweep_c_followup.py --workers 2
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from run_ar_sweep import SIGMA_A, contrast, margin_by_seed
from run_ar_sweep_c import BASES, OUT as SWEEP_C, SPEC, _point_matches, readings

OUT = "artifacts/texture/ar_sweep_c_followup"
RERUN = (0.95, 0.99)
H = 24
GAP_TOL = 0.03


def lam() -> float:
    from itasorl.surrogate_l3_families import make_g_ar
    from itasorl.world import WorldParams
    g = make_g_ar(sigma=1.0, rho=0.0, params=WorldParams(k_land=1.5, k_water=1.5, gravity=0.4))
    return 1.0 - g._drag0 * g._dt


def exact_gap(sigma: float, rho: float, horizon: int = H) -> float:
    """RMS open-loop velocity gap after `horizon` steps from a stationary per-episode start."""
    t = np.arange(horizon)
    w = lam() ** (horizon - 1 - t)
    C = rho ** np.abs(t[:, None] - t[None, :])
    return float(sigma * np.sqrt(2.0 * w @ C @ w))


def main() -> int:
    from promote_bv3_gates import repo_rel

    from itasorl.optional_readouts import provenance
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", default="fullruns/corrected_l3_h8_wm/agents")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    stored = json.load(open(os.path.join(SWEEP_C, "summary.json"), encoding="utf-8"))
    target = exact_gap(SIGMA_A, 0.9)
    audit = [{"rho": p["rho"], "sigma": p["sigma"], "exact_gap": exact_gap(p["sigma"], p["rho"]),
              "recorded_gap": p["calibration"]["gap_closed_form"],
              "off_target": exact_gap(p["sigma"], p["rho"]) / target - 1.0} for p in stored["points"]]
    for r in audit:
        r["within_tolerance"] = bool(abs(r["off_target"]) <= GAP_TOL)
        print(f"stored rho {r['rho']:.2f} sigma {r['sigma']:.6f} exact gap {r['exact_gap']:.4f} "
              f"({r['off_target']:+.1%}) recorded {r['recorded_gap']:.4f}", flush=True)
    pts = []
    for rho in RERUN:
        sigma = target / exact_gap(1.0, rho)
        pts.append({"rho": rho, "sigma": sigma, "exact_gap": exact_gap(sigma, rho), "file": f"C_rho{rho:.2f}.json"})
    for p in pts:
        path = os.path.join(a.out_dir, p["file"])
        if os.path.exists(path):
            doc = json.load(open(path, encoding="utf-8"))
            # The resume check of sweep C itself, which also checks the family (2026-10-17).
            if not _point_matches(doc, p, a.agents_dir):
                raise RuntimeError(f"{path} exists but was written for different inputs; move it aside")
            continue
        print(f"== follow-up rho {p['rho']} sigma {p['sigma']:.6f} exact gap {p['exact_gap']:.4f}", flush=True)
        subprocess.run([sys.executable, "scripts/run_texture_fresh_probe.py", "--agents-dir", a.agents_dir,
                        "--family", "arc", "--param", repr(p["sigma"]), "--rho", repr(p["rho"]),
                        "--workers", str(a.workers), "--out", path], check=True)
    M = {}
    summ = {"spec": SPEC, "status": "follow-up readout beside the frozen sweep C result; does not replace it",
            "gap_target_exact": target, "gap_tolerance": GAP_TOL, "seed_bases": list(BASES),
            "stored_points_exact_gaps": audit, "points": []}
    for p in pts:
        doc = json.load(open(os.path.join(a.out_dir, p["file"]), encoding="utf-8"))
        m = margin_by_seed(doc)
        M[p["rho"]] = m
        mv = np.array(list(m.values()), float)
        from itasorl.stats import t_ci90
        sf = doc["aggregate"].get("survival fresh") or {}
        summ["points"].append({**p, "survival_fresh": sf,
                               "untrained_fresh_mean": (doc["aggregate"].get("untrained fresh") or {}).get("mean"),
                               "M": {"per_seed": mv.tolist(), "mean": float(mv.mean()),
                                     "t90": [float(x) for x in t_ci90(mv)]},
                               "kept": bool(sf.get("mean", 0) >= 0.65 and mv.mean() >= 0.05)})
    m0 = margin_by_seed(json.load(open(os.path.join(SWEEP_C, "C_rho0.00.json"), encoding="utf-8")))
    c = contrast(M[0.99], m0)
    # contrast() pairs the seeds the two points share and drops non-finite ones; the readings
    # are defined on all ten seeds, so a shorter pairing stops here instead of being read.
    if c["n"] != 10:
        raise RuntimeError(f"the contrast pairs {c['n']} seeds, not 10; check both point files")
    summ["contrast"] = {"follow-up: M(0.99, matched) - M(0, stored)": c}
    summ["readings"] = readings(c)
    summ["frozen_result"] = {"contrast": stored["contrast"], "readings": stored["readings"]}
    summ["provenance"] = provenance("scripts/run_ar_sweep_c_followup.py",
                                    "python " + " ".join(["scripts/run_ar_sweep_c_followup.py", *sys.argv[1:]]),
                                    {"agents_dir": repo_rel(a.agents_dir), "rhos": list(RERUN),
                                     "gap_target_exact": target, "seed_bases": list(BASES)}, device="cpu")
    summ["provenance"]["status"] = ("follow-up readout after the 2026-10-16 critical review; diagnostic, "
                                    "not an adjudication; changes no registered rule or verdict")
    with open(os.path.join(a.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=1, default=float)
    for p in summ["points"]:
        print(f"follow-up rho {p['rho']:.2f} sigma {p['sigma']:.6f} gap {p['exact_gap']:.4f} "
              f"surv fresh {p['survival_fresh'].get('mean', float('nan')):.3f} M {p['M']['mean']:+.3f} {p['M']['t90']}")
    print(summ["contrast"], summ["readings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
