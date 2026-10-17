"""Validate the evolutionary readout on agents known to carry the signal (revision step 12).

Experiment C's panel pools tail states across individuals and fits ONE probe. If individuals
encode the prefix world along different directions, the pooled probe can read chance while
every individual is decodable. This script measures both estimators on populations whose
answer is known:

  positive  the saved survival agents of one run, one per agent seed: independently trained,
            so their state directions are not aligned, and each is individually decodable
            under the common garden (the published per-seed common-garden reading);
  null      the untrained agents of the same run.

It also reports, under the key `value_of_world_information`, the matched assignment minus the
better single policy from the run's cross-evaluation cells
(`experiment_c.value_of_world_information`): whether rearing in a world gives the run's
policies an advantage there. That is not the value of knowing the world (see that function's
docstring).

`--group-by-world-seed` (closing item G21, opt-in) scores the SAME pooled tails a second time
with one CV group per world seed across individuals (experiment_c.group_by_world_seed): every
individual runs on the same pair seeds, so the default pooled-row grouping can put pair p of
two individuals in different folds. The panel then also carries `pooled_probe_world_grouped`,
its `_ci` and `n_world_groups`; `pooled_probe` stays the row-grouped value. Without the flag
the panel keys and values are as before.

Usage:
    python scripts/validate_population_readout.py --run-dir fullruns/corrected_l3_h8_wm \\
        --out artifacts/population_readout/corrected_l3_h8_wm.json [--quick] [--group-by-world-seed]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from itasorl.results_io import git_head  # noqa: E402

import argparse
import glob
import json
import os
import sys

import numpy as np


def pooled_probe_block(auth: list, surr: list, world_index: list | None = None) -> dict:
    """The pooled probe over all individuals' tails, as Experiment C's panel computes it (one
    CV group per pooled pair). With `world_index` (per individual, the surviving pair indices
    from common_garden_rollout(..., return_index=True)) the same tails are also scored with
    one group per world seed, and the block gains the world-grouped keys."""
    from itasorl.experiment_b2 import cg_probe
    out = {"pooled_probe": float(cg_probe(auth, surr)["cg_tail_target"])}
    if world_index is not None:
        from itasorl.experiment_c import group_by_world_seed
        g = group_by_world_seed(world_index)
        r = cg_probe(auth, surr, groups=g)
        out["pooled_probe_world_grouped"] = float(r["cg_tail_target"])
        out["pooled_probe_world_grouped_ci"] = [float(r["cg_tail_lo"]), float(r["cg_tail_hi"])]
        out["n_world_groups"] = int(len(np.unique(g)))
    return out


def provenance_block(out_path: str, run_dir: str, argv=None) -> dict:
    """Closing item G46: commit, tracked changes, stack, agents_dir and argv. A readout of
    saved agents, so the commit key is git_commit_at_promotion; the script's git_commit key
    stays beside it for the stored schema."""
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    prov = provenance("scripts/validate_population_readout.py",
                      commit_key="git_commit_at_promotion", output=out_path, device="cpu")
    prov.pop("generated_by")
    prov["agents_dir"] = repo_rel(os.path.join(run_dir, "agents"))
    prov["argv"] = rel_flags(list(sys.argv[1:] if argv is None else argv))
    return prov


def main(argv=None) -> int:
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b2 import format_drift, load_agent_bundle
    from itasorl.experiment_c import individual_probe_panel, value_of_world_information
    from itasorl.stats import t_ci90
    from itasorl.world import WorldParams

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=110)
    ap.add_argument("--prefix", type=int, default=20)
    ap.add_argument("--tail", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--group-by-world-seed", action="store_true",
                    help="also score the pooled tails with one CV group per world seed "
                         "across individuals (G21); pooled_probe stays row-grouped")
    a = ap.parse_args(argv)
    if a.quick:
        a.n_pairs, a.prefix, a.tail, a.ray_steps = 15, 8, 10, 4
    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=8, device="cpu", seed=0, params=P)
    cells = {}
    for p in glob.glob(os.path.join(a.run_dir, "cells", "cell_d*_s*.json")):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        cells[(float(c["drift"]), int(c["seed"]))] = c
    dmax = max(d for d, _ in cells)
    seeds = sorted(s for d, s in cells if d == dmax)
    out = {"generated_by": "scripts/validate_population_readout.py", "git_commit": git_head(), "run_dir": a.run_dir,
           "drift": dmax, "seeds": seeds, "panels": {}}
    for arm in ("survival", "untrained"):
        pop, norms = [], []
        for s in seeds:
            ag, nm = load_agent_bundle(os.path.join(a.run_dir, "agents",
                                                    f"agent_d{format_drift(dmax)}_s{s}_{arm}.pt"))
            pop.append(ag)
            norms.append(nm)
        # each individual keeps its own frozen normalizer: score them one at a time
        per = [individual_probe_panel([ag], drift_sigma=dmax, n_pairs=a.n_pairs,
                                      prefix_steps=a.prefix, tail_steps=a.tail, seed_base=930_000,
                                      params=P, ray_steps=a.ray_steps, norm=nm)
               for ag, nm in zip(pop, norms)]
        indiv = [p["per_individual"][0] for p in per]
        # pooled probe over all individuals' tails, as Experiment C's panel computes it
        from itasorl.experiment_b2 import common_garden_rollout
        A, S, widx = [], [], []
        for ag, nm in zip(pop, norms):
            # return_index only appends the surviving pair indices; the rollout is unchanged.
            # device="cpu" matches the agents and the surrogate above; the default
            # (default_device()) is CUDA on a GPU machine, where this call used to crash.
            x, y, ix = common_garden_rollout(ag, nm, P, dmax, n_pairs=a.n_pairs, prefix_steps=a.prefix,
                                             tail_steps=a.tail, ray_steps=a.ray_steps, seed_base=930_000,
                                             device="cpu", return_index=True)
            A.extend(x)
            S.extend(y)
            widx.append(ix)
        pooled = pooled_probe_block(A, S, widx if a.group_by_world_seed else None)
        v = np.asarray([x for x in indiv if np.isfinite(x)], float)
        out["panels"][arm] = {"per_individual": indiv, "mean": float(v.mean()),
                              "t90": [float(x) for x in t_ci90(v)] if v.size > 1 else None,
                              "share_ge_065": float((v >= 0.65).mean()),
                              **pooled}
        extra = (f"  world-grouped {pooled['pooled_probe_world_grouped']:.3f} "
                 f"({pooled['n_world_groups']} world groups)" if a.group_by_world_seed else "")
        print(f"{arm:10s} per-individual mean {v.mean():.3f}  share>=0.65 {(v >= 0.65).mean():.2f}  "
              f"pooled probe {pooled['pooled_probe']:.3f}{extra}", flush=True)
    voi = [value_of_world_information(cells[(0.0, s)]["xeval"], cells[(dmax, s)]["xeval"])
           for s in seeds if (0.0, s) in cells]
    vv = [x["value_of_information"] for x in voi]
    out["value_of_world_information"] = {"per_seed": voi, "mean": float(np.mean(vv)),
                                         "t90": [float(x) for x in t_ci90(vv)] if len(vv) > 1 else None}
    if a.group_by_world_seed:
        out["group_by_world_seed"] = True
    out.update(provenance_block(a.out, a.run_dir, argv))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"value of world information {np.mean(vv):+.4f}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
