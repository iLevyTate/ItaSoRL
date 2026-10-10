"""Surrogate and detector diagnostics (revision step 10), agent free.

For every learned fingerprint instance the project used (hidden 8 G seed 0, hidden 7 G seed 0,
hidden 10 G seed 1, hidden 8 G seed 2) and for the comparators (Gaussian jitter at the
published gate-0 sigma, quadratic drag at a given eps), records on world P:
  * the linear-fit control and G's one-step RMS error on G's own training episodes (each G is
    scored on the seed range it was trained on) and on held-out authentic episodes at seeds
    disjoint from every training range, and the open-loop rollout divergence
    (`surrogate_l3.surrogate_diagnostics`);
  * the magnitude and temporal structure of the deviation (`perturbation_profile`) on the
    seed-77 calibration sample, the sample the quadratic-drag eps was matched on. Its episodes
    77 to 136 lie inside every G's training range, so for a learned G it is an in-sample
    profile, not a held-out one; the untrained comparators have no training data to overlap.
    The same profile on the disjoint held-out sample is recorded beside it (`profile_heldout`);
  * an AGENT-ACCESSIBLE detector score: the velocity law checked from raw observations alone
    (`experiment_a_l3.observation_law_detector`) on scripted-policy pools of 110 + 110
    episodes x 24 steps (world seeds 800000 / 850000), with no detector-side noise.
The privileged gate-0 oracle, by contrast, reads the world's internal transitions with an
added detector-side noise of sigma = 0.02.

Usage:
    python scripts/run_surrogate_diagnostics.py --out <fresh artifact> [--qd-eps 6.0] [--quick]
    python scripts/run_surrogate_diagnostics.py --qd-eps 6.0 \\
        --merge-into artifacts/surrogate_diagnostics.json

`--merge-into` keeps every value already in the committed artifact (its per-surrogate
`diagnostics` were scored before 2026-10-10 on the seed-77 sample, which overlaps G's training
episodes, with the train column at seed 0 for every G) and adds the disjoint-seed diagnostics
under the key `disjoint_heldout`. It refuses to write unless this run reproduces the stored
agent-accessible detector scores exactly and the stored profile values to within
`PROFILE_TOL`, so the added block is known to come from the same surrogates.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import platform
import time

import numpy as np

from itasorl.experiment_a_l3 import observation_law_detector
from itasorl.experiment_b import scripted_policy
from itasorl.experiment_b2 import _seeds
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.surrogate_l3 import HELDOUT_SEED, surrogate_diagnostics, train_g_motion
from itasorl.surrogate_l3_families import make_g_gn, make_g_qd, perturbation_profile
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
PROFILE_TOL = 1e-6
"""Largest absolute difference allowed between a rerun's profile values and the stored ones.
A G retrained on another numpy or torch build differs in its float32 weights, which moves the
profile by about 1e-8 and leaves every quoted digit unchanged."""
PROFILE_KEYS = ("rms_one_step", "linear_explained_share", "lag1_autocorr", "mean_dev")
MERGE_KEY = "disjoint_heldout"


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


def merge_disjoint_block(existing: dict, fresh: dict, *, date: str) -> dict:
    """Return `existing` with the disjoint-seed diagnostics of `fresh` added under MERGE_KEY.

    Every stored value is kept. Raises ValueError unless `fresh` covers the same surrogates,
    reproduces each stored agent-accessible detector AUROC exactly, and reproduces each stored
    profile value to within PROFILE_TOL."""
    old, new = existing["surrogates"], fresh["surrogates"]
    if set(old) != set(new):
        raise ValueError(f"surrogate sets differ: stored {sorted(old)}, rerun {sorted(new)}")
    worst = 0.0
    for name in old:
        a = old[name]["agent_accessible_detector"]["auroc"]
        b = new[name]["agent_accessible_detector"]["auroc"]
        if a != b:
            raise ValueError(f"{name}: detector AUROC {b!r} does not reproduce the stored {a!r}")
        for key in PROFILE_KEYS:
            sa = np.atleast_1d(np.asarray(old[name]["profile"][key], float))
            sb = np.atleast_1d(np.asarray(new[name]["profile"][key], float))
            diff = float(np.max(np.abs(sa - sb)))
            if not diff <= PROFILE_TOL:
                raise ValueError(f"{name}: profile {key} moved by {diff:.3g} (> {PROFILE_TOL})")
            worst = max(worst, diff)
    out = dict(existing)
    out[MERGE_KEY] = {
        "date": date,
        "note": ("The per-surrogate `diagnostics` above were scored with the held-out sample at "
                 "seed 77 (episodes 77 to 136, inside every G's training range) and the train "
                 "column at seed 0 for every G. This block rescores them with the held-out "
                 "sample disjoint from every training range and each G's train column on its "
                 "own training seed, and adds the deviation profile on the same held-out "
                 "sample. The seed-77 profiles and the detector scores are unchanged and stay "
                 "where they are."),
        "heldout_seed": fresh["heldout_seed"],
        "train_seed_by_surrogate": {n: new[n]["diagnostics"]["train_seed"] for n in new},
        "determinism": {"detector_auroc_exact": True,
                        "profile_max_abs_diff": worst, "profile_tol": PROFILE_TOL},
        "environment": fresh["environment"],
        "wall_seconds": fresh["wall_seconds"],
        "surrogates": {n: {"diagnostics": new[n]["diagnostics"],
                           "profile_heldout": new[n]["profile_heldout"]} for n in new},
    }
    return out


def _environment() -> dict:
    env = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:  # torch is required to train G, so this only guards odd installs
        pass
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out", help="write a fresh artifact here")
    dest.add_argument("--merge-into", help="add the disjoint-seed block to this artifact")
    ap.add_argument("--qd-eps", type=float, default=None, help="gate-0 eps of the qd comparator")
    ap.add_argument("--gn-sigma", type=float, default=0.01)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.merge_into and a.quick:
        ap.error("--merge-into needs the full-size run, not --quick")
    n, steps = (30, 12) if a.quick else (110, 24)
    t0 = time.time()
    surrogates, train_seeds = {}, {}
    for hidden, gseed in ((8, 0), (7, 0), (10, 1), (8, 2)):
        if a.quick and (hidden, gseed) != (8, 0):
            continue
        name = f"gmotion_h{hidden}_s{gseed}"
        surrogates[name] = train_g_motion(hidden=hidden, seed=gseed, params=P)
        train_seeds[name] = gseed
    surrogates["gn_sigma%g" % a.gn_sigma] = make_g_gn(sigma_v=a.gn_sigma, params=P, seed=0)
    if a.qd_eps is not None:
        surrogates["qd_eps%g" % a.qd_eps] = make_g_qd(eps=a.qd_eps, params=P)
    auth = scripted_obs_pool(None, 800_000, n, steps)
    rows = {}
    for name, g in surrogates.items():
        diag = surrogate_diagnostics(g, params=P, train_seed=train_seeds.get(name, 0),
                                     heldout_eps=20 if a.quick else 60)
        prof = perturbation_profile(g, params=P, n_eps=20 if a.quick else 60)
        prof_ho = perturbation_profile(g, params=P, n_eps=20 if a.quick else 60,
                                       seed=HELDOUT_SEED)
        if hasattr(g, "reseed"):
            g.reseed(850_000)
        surr = scripted_obs_pool(g, 850_000, n, steps)
        det = observation_law_detector(auth, surr, dt=P.dt)
        rows[name] = {"diagnostics": diag, "profile": prof, "profile_heldout": prof_ho,
                      "agent_accessible_detector": det}
        print(f"{name:20s} G one-step rms train {diag['g_one_step']['rms_train']:.5f} "
              f"held-out {diag['g_one_step']['rms_heldout']:.5f}  rollout@24 "
              f"{diag['rollout_rms_velocity_gap'].get('24', float('nan')):.4f}  linear-fit held-out "
              f"{diag['linear_fit']['rms_heldout']:.2e}  obs-law detector {det['auroc']:.3f}", flush=True)
    out = {"generated_by": "scripts/run_surrogate_diagnostics.py", "world": "WorldParams(k_land=1.5, "
           "k_water=1.5, gravity=0.4) [P]", "pools": {"n_per_pool": n, "steps": steps,
                                                       "seed_bases": [800_000, 850_000],
                                                       "policy": "scripted, rng 600000 + i"},
           "heldout_seed": HELDOUT_SEED, "profile_sample": "seed 77, 60 x 40 (calibration sample)",
           "surrogates": rows, "environment": _environment(),
           "wall_seconds": round(time.time() - t0, 1)}
    path = a.merge_into or a.out
    if a.merge_into:
        with open(a.merge_into, encoding="utf-8") as fh:
            out = merge_disjoint_block(json.load(fh), out, date=time.strftime("%Y-%m-%d"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
