"""In-configuration L2 oracle ceiling on the organism world P.

Spec: docs/specs/2026-09-27-local-strengthening-probes-design.md (probe A). The
published L2 ceiling (0.993) was measured in Experiment A's tamed diagnostic world;
every L2 organism number lives in world P. This measures the residual oracle on P at
the organism's own drift strength with the L3 gate's detector-side handicap
(sigma_meas = 0.02), for the ar1 (B-v2) and regime (B-v3) artifacts, plus the L0
anchor. Matched pairs, pair-grouped CV, mechanical leakage must sit at chance.

Each row is ONE draw of the detector-side noise (seed 0). --noise-seeds N rescores every cell
at the frozen sigma on noise seeds 0..N-1 (`noise_seed_draws`: per-seed AUROC, mean, sample
sd), so the noise in a single row can be read off the record; the L0 cell's draws are also
written as `l0_anchor_noise_seeds`, the block the record quotes ("the L0 anchor averages
0.475 over eight noise draws"). The output records the fold scheme it was scored under. The
committed artifact was scored on the legacy partition (it predates the explicit folds), so it
regenerates only with ITASORL_FOLDS=legacy on a stack whose scikit-learn GroupKFold splits as
the owner's did.

Usage:
    python scripts/run_expA_l2_inconfig.py --json fullruns/l2_inconfig_oracle.json
    ITASORL_FOLDS=legacy python scripts/run_expA_l2_inconfig.py \
        --sigmas 0.0002 0.001 0.005 0.01 0.02 0.05 --l3-hiddens 8 7 --noise-seeds 8 \
        --json artifacts/expA/l2_inconfig_oracle.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os

import numpy as np

from itasorl import folds
from itasorl.experiment_a_l3 import generate_l2_pairs, run_experiment_a_l2
from itasorl.results_io import git_head
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)   # the run_expB2 organism world
SIGMA_MEAS = 0.02                                        # frozen L3 gate-0 handicap
CELLS = (("l0", "ar1", 0.0), ("ar1", "ar1", 0.45), ("regime", "regime", 0.45))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=60)
    ap.add_argument("--prefix", type=int, default=10)
    ap.add_argument("--branch", type=int, default=30)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[SIGMA_MEAS],
                    help="detector-side noise floors to report (the frozen one first)")
    ap.add_argument("--l3-hiddens", type=int, nargs="*", default=[],
                    help="also score the L3 fingerprint(s) at these capacities (G seed --l3-seed) "
                         "on the SAME sigma list, for a matched-handicap comparison across rungs")
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--noise-seeds", type=int, default=0,
                    help="also rescore every cell at the frozen sigma on detector-noise seeds "
                         "0..N-1 (default 0: single seed-0 rows only)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = []
    draws: dict[str, dict] = {}

    def noise_draws(cell: str, score) -> None:
        if a.noise_seeds <= 0:
            return
        v = [float(score(s)) for s in range(a.noise_seeds)]
        draws[cell] = {"sigma_meas": SIGMA_MEAS, "seeds": list(range(a.noise_seeds)),
                       "oracle_auroc": v, "mean": float(np.mean(v)),
                       "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")}
        print(f"  {cell:12s} noise seeds 0..{a.noise_seeds - 1} at sigma {SIGMA_MEAS}: "
              f"mean {draws[cell]['mean']:.3f} sd {draws[cell]['sd']:.3f}", flush=True)

    for name, mode, ds in CELLS:
        eps = generate_l2_pairs(P, ds, drift_mode=mode, n_pairs=a.n_pairs, prefix=a.prefix,
                                branch=a.branch)
        noise_draws(name, lambda s, eps=eps: run_experiment_a_l2(eps, sigma_meas=SIGMA_MEAS,
                                                                  params=P, seed=s)["oracle_auroc"])
        for sig in a.sigmas:
            r = run_experiment_a_l2(eps, sigma_meas=sig, params=P, seed=0)
            row = {"cell": name, "drift_mode": mode, "drift_sigma": ds, "sigma_meas": sig,
                   "oracle_auroc": float(r["oracle_auroc"]), "leakage": r["leakage"],
                   "leakage_pass": bool(r["leakage_pass"]), "reward_leak": r["reward_leak"]}
            rows.append(row)
            print(f"  {name:7s} mode={mode:6s} drift={ds:.2f} sigma={sig}: oracle={row['oracle_auroc']:.3f} "
                  f"mech_leak_pass={row['leakage_pass']} reward_leak={row['reward_leak']:.3f}", flush=True)
    if a.l3_hiddens:
        from itasorl.experiment_a_l3 import generate_l3_pairs, run_experiment_a_l3
        from itasorl.surrogate_l3 import train_g_motion
        for h in a.l3_hiddens:
            g = train_g_motion(hidden=h, seed=a.l3_seed, params=P)
            eps = generate_l3_pairs(g, n_pairs=a.n_pairs, prefix=a.prefix, branch=a.branch, params=P)
            noise_draws(f"l3_h{h}_seed{a.l3_seed}",
                        lambda s, eps=eps: run_experiment_a_l3(eps, sigma_meas=SIGMA_MEAS,
                                                               seed=s)["oracle_auroc"])
            for sig in a.sigmas:
                r = run_experiment_a_l3(eps, sigma_meas=sig, seed=0)
                row = {"cell": f"l3_h{h}_seed{a.l3_seed}", "drift_mode": "l3", "hidden": h,
                       "g_seed": a.l3_seed, "sigma_meas": sig, "oracle_auroc": float(r["oracle_auroc"]),
                       "leakage": r["leakage"], "leakage_pass": bool(r["leakage_pass"]),
                       "reward_leak": r["reward_leak"]}
                rows.append(row)
                print(f"  {row['cell']:12s} sigma={sig}: oracle={row['oracle_auroc']:.3f} "
                      f"mech_leak_pass={row['leakage_pass']}", flush=True)
    out = {"world": "WorldParams(k_land=1.5, k_water=1.5, gravity=0.4) [P]", "n_pairs": a.n_pairs,
           "prefix": a.prefix, "branch": a.branch, "sigma_frozen": SIGMA_MEAS,
           "spec": "docs/specs/2026-09-27-local-strengthening-probes-design.md",
           "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
           "rows": rows}
    if "l0" in draws:
        d0 = draws["l0"]
        out["l0_anchor_noise_seeds"] = {"seeds": d0["seeds"], "oracle_auroc": d0["oracle_auroc"],
                                        "mean": d0["mean"]}
    if draws:
        out["noise_seed_draws"] = draws
    out["generated_by"] = "scripts/run_expA_l2_inconfig.py"
    out["git_commit"] = git_head()
    if a.json:
        d = os.path.dirname(a.json)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
