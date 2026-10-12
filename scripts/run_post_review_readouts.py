"""Post-review readouts R210, R35, R206, R33 and R31 on one run's saved agents.

Spec, frozen before any run: docs/specs/2026-10-11-post-review-readouts-design.md.

Readout-only. Trains nothing that the run trained: each worker refits the run's L3 velocity
law G_motion from the knobs recorded in the cells, on one torch thread, as the run workers
and scripts/regenerate_state_dumps.py do, and the r31 stream decoders are fit per fold as
FINDINGS 17.8's decoder was. One task per (readout, seed):

  r210  probe classes on the registered pools, both drifts, every arm (bit-checked)
  r205  the linear probe on mean h, on h_T and on both, on the same bit-checked pools
  r35   normalizer-matched replay at the strongest drift
  r206  law deviation along the visited states of four walks at the strongest drift
  r33   balanced readout of every agent at both drifts, read at the strongest drift
  r31   stream-decoder capacity curve on the survival agent's streams and the scripted walk

Every finished task is appended to a partial file (default <out>.partial.jsonl), so a
session that is cut off resumes where it stopped; the output JSON holds every finished
task in the partial file, the summaries, the seed bases, the commit, the stack and the
run's cell fingerprints.

    ITASORL_FOLDS=explicit python scripts/run_post_review_readouts.py \\
        --run-dir artifacts/goal_stakes/C1-rerun \\
        --out artifacts/goal_stakes/C1-rerun/post_review.json --workers 4 \\
        [--readouts r210,r35,r206,r33,r31] [--seeds 0 1 2] \\
        [--balanced-reference artifacts/l0_audit/corrected_l3_h8_wm.json]

    # the balanced C1 minus C2 survival contrast from two finished outputs
    python scripts/run_post_review_readouts.py --contrast C1.json C2.json --out contrast.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import platform
import re
import sys
import time
import traceback

import numpy as np

from itasorl import post_review as pr

CELL_RE = re.compile(r"cell_d(\d+\.\d+)_s(\d+)\.json$")
# longest first, so a pool of workers packs the r31 decoders early
TASK_ORDER = ("r31", "r210", "r205", "r33", "r35", "r206")


# ---------------------------------------------------------------------------
# Run discovery and settings
# ---------------------------------------------------------------------------
def load_cells(run_dir: str) -> dict:
    """{(drift_tag, seed): {"fingerprint", "cell"}} for every cell file of the run."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "cells", "cell_d*_s*.json"))):
        m = CELL_RE.search(os.path.basename(p))
        if not m:
            continue
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        out[(m.group(1), int(m.group(2)))] = {"fingerprint": doc.get("fingerprint"), "cell": doc["cell"]}
    if not out:
        raise SystemExit(f"no cells under {run_dir}/cells")
    return out


def run_config(cells: dict) -> dict:
    """The knobs every worker needs, read from the cells; refuses a run this design does not
    cover (another drift mode or surrogate family) and a run whose cells disagree."""
    knobs = {json.dumps(c["cell"].get("knobs") or {}, sort_keys=True) for c in cells.values()}
    if len(knobs) != 1:
        raise SystemExit(f"cells disagree on their knobs: {sorted(knobs)}")
    k = json.loads(knobs.pop())
    if k.get("drift_mode") not in (None, "l3"):
        raise SystemExit(f"run drift_mode={k['drift_mode']!r}; these readouts are defined for l3")
    if k.get("l3_family") not in (None, "gmotion"):
        raise SystemExit(f"run l3_family={k['l3_family']!r}; these readouts rebuild only G_motion")
    any_cell = next(iter(cells.values()))["cell"]
    drifts = sorted({d for d, _ in cells}, key=float)
    return {"knobs": k, "objective": any_cell.get("objective", "survival"),
            "mortal": bool(any_cell.get("mortal", True)), "drifts": drifts, "dmax": drifts[-1]}


def settings_for(quick: bool) -> dict:
    s = {"n_eps": pr.POOL_N, "steps": 24, "ray_steps": 5, "n_perm": pr.N_PERM,
         "rf_trees": pr.RF_TREES, "mlp": {"hidden": list(pr.MLP_HIDDEN), "alpha": pr.MLP_ALPHA,
                                          "lr": pr.MLP_LR, "max_iter": pr.MLP_MAX_ITER},
         "gru_epochs": pr.GRU_EPOCHS, "width_mults": list(pr.WIDTH_MULTS),
         "conv": {"channels": list(pr.CONV_CHANNELS), "kernel": pr.CONV_KERNEL}, "quick": False}
    if quick:
        s.update(n_eps=12, steps=8, gru_epochs=2, width_mults=[1, 2], rf_trees=20, n_perm=2, quick=True)
    return s


def seed_bases() -> dict:
    return {"fresh_authentic": pr.FRESH_AUTH_BASE, "fresh_surrogate": pr.FRESH_SURR_BASE,
            "balanced": dict(pr.R33_BASES), "registered": [pr.REGISTERED_AUTH_BASE, pr.REGISTERED_SURR_BASE],
            "scripted_policy": pr.SCRIPTED_POLICY_BASE,
            "note": "fresh bases are disjoint from every base in itasorl.post_review.BASES_IN_USE "
                    "(seed footprints, all three RNG streams); r210 and the r33 continuity entry "
                    "read the registered pair by design"}


def recorded_pools(cells: dict) -> dict:
    """{drift_tag: {seed: {arm: {"target", "shuffled"}}}} as the run recorded them."""
    out: dict = {}
    for (d, s), c in cells.items():
        out.setdefault(d, {})[str(s)] = {arm: {"target": a["pool"].get("target"),
                                               "shuffled": a["pool"].get("shuffled")}
                                         for arm, a in c["cell"]["agents"].items()}
    return out


def plan_tasks(readouts, seeds, done: dict) -> list[tuple[str, str]]:
    """(readout, seed) pairs still to run, longest readouts first."""
    todo = [(r, str(s)) for r in TASK_ORDER if r in readouts for s in seeds
            if (r, str(s)) not in done]
    return todo


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
def _setup(task: dict):
    """Install the run's world in this process, as regenerate_state_dumps.run_task does."""
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.world import WorldParams

    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    knobs = task["knobs"]
    b2.DRIFT_MODE = "l3"
    b2.OBJECTIVE = task["objective"]
    b2.MORTAL = task["mortal"]
    for key, field in (("n_pellets", "n_pellets"), ("reach", "reach")):
        if knobs.get(key) is not None:
            b2.SURVIVAL_FOOD[field] = knobs[key]
    if knobs.get("basal_e") is not None:
        b2.SURVIVAL_METAB["basal_E"] = knobs["basal_e"]
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=knobs.get("l3_hidden") or 8, device="cpu",
                              seed=knobs.get("l3_seed") or 0, params=P)
    return P


def _agent_path(run_dir: str, drift_tag: str, seed, arm: str) -> str:
    return os.path.join(run_dir, "agents", f"agent_d{drift_tag}_s{int(seed)}_{arm}.pt")


def _load(task: dict, drift_tag: str, arm: str):
    from itasorl.experiment_b2 import load_agent_bundle
    return load_agent_bundle(_agent_path(task["run_dir"], drift_tag, task["seed"], arm))


def task_r210(task: dict, P) -> dict:
    from itasorl.experiment_b2 import pooled_readout
    st, seed = task["settings"], int(task["seed"])
    out = {"cells": {}, "reproduced": {}}
    for d in task["drifts"]:
        out["cells"][d] = {}
        for arm in pr.ARMS:
            agent, norm = _load(task, d, arm)
            pool, (Ha, Hs) = pooled_readout(agent, norm, P, float(d), n_eps=st["n_eps"], steps=st["steps"],
                                            ray_steps=st["ray_steps"], device="cpu", seed=seed,
                                            return_pools=True)
            out["cells"][d][arm] = pr.r210_cell(Ha, Hs, pool=pool, recorded=task["recorded"][d][str(seed)][arm],
                                                seed=seed, n_perm=st["n_perm"], rf_trees=st["rf_trees"])
        out["reproduced"][d] = all(out["cells"][d][a]["reproduced"] for a in pr.ARMS)
    return out


def task_r205(task: dict, P) -> dict:
    """r210's regeneration and bit check, then the probe on each half of the feature."""
    from itasorl.experiment_b2 import pooled_readout
    st, seed = task["settings"], int(task["seed"])
    out = {"cells": {}, "reproduced": {}}
    for d in task["drifts"]:
        out["cells"][d] = {}
        for arm in pr.ARMS:
            agent, norm = _load(task, d, arm)
            pool, (Ha, Hs) = pooled_readout(agent, norm, P, float(d), n_eps=st["n_eps"], steps=st["steps"],
                                            ray_steps=st["ray_steps"], device="cpu", seed=seed,
                                            return_pools=True)
            out["cells"][d][arm] = pr.r205_cell(Ha, Hs, pool=pool, recorded=task["recorded"][d][str(seed)][arm],
                                                seed=seed, n_perm=st["n_perm"])
        out["reproduced"][d] = all(out["cells"][d][a]["reproduced"] for a in pr.ARMS)
    return out


def task_r35(task: dict, P) -> dict:
    from itasorl.eval_protocols import readout_from_states, record_trajectories, replay_states
    st, seed, d = task["settings"], int(task["seed"]), task["dmax"]
    arms = {a: _load(task, d, a) for a in pr.ARMS}
    sa, sn = arms["survival"]
    n, steps, rs = st["n_eps"], st["steps"], st["ray_steps"]
    obs_a, act_a, _ = record_trajectories(sa, sn, P, 0.0, n, steps, "cpu", pr.FRESH_AUTH_BASE, rs)
    obs_s, act_s, _ = record_trajectories(sa, sn, P, float(d), n, steps, "cpu", pr.FRESH_SURR_BASE, rs)
    na = len(obs_a)
    obs = np.concatenate([obs_a, obs_s])
    y = np.r_[np.zeros(na), np.ones(len(obs_s))].astype(int)
    refit = pr.refit_normalizer([obs_a, obs_s])          # label-blind, both pools pooled
    out = {"n_auth": int(na), "n_surr": int(len(obs_s)), "arms": {},
           "refit_norm": {"mean_abs_mean": float(np.abs(refit.mean).mean()),
                          "mean_var": float(np.mean(refit.var))}}
    for g in pr.ARMS:
        ag, nm = arms[g]

        def states(norm, ag=ag):
            # each pool replayed in its own batch, as run_policy_controlled_readouts does
            return np.concatenate([replay_states(ag, norm, obs_a, act_a), replay_states(ag, norm, obs_s, act_s)])

        res = {}
        for v, norm in (("own", nm), ("survival_norm", sn), ("refit", refit)):
            H = states(norm)
            res[v] = readout_from_states(H[:na], H[na:], seed=seed)["target"]
        res["refit_infold"] = pr.infold_normalizer_auroc(obs, y, states) if min(na, len(obs_s)) >= 5 \
            else float("nan")
        out["arms"][g] = res
    out["reference"] = out["arms"]["survival"]["own"]
    return out


def task_r206(task: dict, P) -> dict:
    st, d = task["settings"], task["dmax"]
    kw = dict(n_eps=st["n_eps"], steps=st["steps"], seed_base=pr.FRESH_SURR_BASE, ray_steps=st["ray_steps"])
    walks = {}
    for g in pr.ARMS:
        ag, nm = _load(task, d, g)
        walks[f"{g}_own"] = pr.walk_deviation(own_policy_factory(ag, nm), P, float(d), **kw)
    walks["scripted"] = pr.walk_deviation(scripted_policy_factory(), P, float(d), **kw)
    return {"walks": {w: pr.deviation_summary(walks[w]) for w in pr.R206_WALKS},
            "kept": {w: walks[w]["kept"].tolist() for w in pr.R206_WALKS}}


def own_policy_factory(agent, norm):
    """The agent's deterministic policy, stepped as collect_pool steps it."""
    import torch

    def make(i):
        state = {"h": agent.initial_state(1, "cpu"), "prev": torch.zeros(1, agent.act_dim)}

        def act(obs):
            o = torch.as_tensor(norm(obs)[None], dtype=torch.float32)
            _, env_act, _, _, h = agent.act(o, state["prev"], state["h"], deterministic=True)
            state["h"], state["prev"] = h, env_act
            return env_act[0].detach().cpu().numpy().astype(np.float32)
        return act
    return make


def scripted_policy_factory(policy_seed_base: int = pr.SCRIPTED_POLICY_BASE):
    """The scripted walk of eval_protocols: action stream i is default_rng(base + i)."""
    from itasorl.experiment_b import scripted_policy

    def make(i):
        rng = np.random.default_rng(policy_seed_base + i)
        return lambda obs: scripted_policy(rng)
    return make


def task_r33(task: dict, P) -> dict:
    from itasorl.l0_audit import paired_pooled_readout
    st, seed, dmax = task["settings"], int(task["seed"]), float(task["dmax"])
    cells = []
    for d in task["drifts"]:
        for arm in pr.ARMS:
            agent, norm = _load(task, d, arm)
            for bname, base in pr.R33_BASES.items():
                r = paired_pooled_readout(agent, norm, P, dmax, n_eps=st["n_eps"], steps=st["steps"],
                                          ray_steps=st["ray_steps"], device="cpu", seed_base=base, seed=seed)
                cells.append({"agent_drift": float(d), "readout_drift": dmax, "arm": arm, "base": bname,
                              "seed_base": int(base), "target": float(r["target"]),
                              "n_pairs": int(r["n_pairs"]), "dropped_pairs": int(r["dropped_pairs"]),
                              "cluster_ci95": [float(x) for x in r["cluster_ci95"]],
                              "identical_first_state": r.get("identical_first_state")})
    return {"cells": cells}


def task_r31(task: dict, P) -> dict:
    from itasorl.eval_protocols import collect_pool_scripted
    from itasorl.experiment_b import episode_features, probe_auroc
    from itasorl.experiment_b2 import collect_pool
    from itasorl.mechanism_readouts import scripted_observation_streams

    st, seed, d = task["settings"], int(task["seed"]), task["dmax"]
    n, steps, rs = st["n_eps"], st["steps"], st["ray_steps"]
    agent, nm = _load(task, d, "survival")
    out = {}
    # the survival agent's own streams, exactly as 17.8 built them: normalized observation and
    # the previous action, on the fresh pair
    pools = {}
    for lab, dd, base in (("a", 0.0, pr.FRESH_AUTH_BASE), ("s", float(d), pr.FRESH_SURR_BASE)):
        H, _, Ot, At = collect_pool(agent, nm, P, dd, n, steps, "cpu", base, rs,
                                    return_obs=True, return_actions=True)
        a_prev = np.concatenate([np.zeros_like(At[:, :1]), At[:, :-1]], axis=1)
        pools[lab] = (H, np.concatenate([Ot, a_prev], axis=2))
    # the scripted walk on the same worlds: streams and the survival trunk's states under it
    for lab, dd, base in (("sa", 0.0, pr.FRESH_AUTH_BASE), ("ss", float(d), pr.FRESH_SURR_BASE)):
        O, A = scripted_observation_streams(nm, P, dd, n_eps=n, steps=steps, seed_base=base, ray_steps=rs)
        H, _ = collect_pool_scripted(agent, nm, P, dd, n, steps, "cpu", base, rs)
        pools[lab] = (H, np.concatenate([O, A], axis=2))
    for stream, (ka, ks) in (("survival_own", ("a", "s")), ("scripted", ("sa", "ss"))):
        (Ha, Sa), (Hs, Ss) = pools[ka], pools[ks]
        y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
        row = {"n_auth": int(len(Ha)), "n_surr": int(len(Hs))}
        if min(len(Ha), len(Hs)) < 5:
            row["error"] = "too few survivors"
            out[stream] = row
            continue
        row["state"] = float(probe_auroc(episode_features(np.concatenate([Ha, Hs])), y))
        row["decoders"] = pr.stream_decoder_curve(np.concatenate([Sa, Ss]), y, seed=seed,
                                                  epochs=st["gru_epochs"], mults=st["width_mults"])
        out[stream] = row
    return out


TASKS = {"r210": task_r210, "r205": task_r205, "r35": task_r35, "r206": task_r206, "r33": task_r33,
         "r31": task_r31}


def run_task(task: dict) -> dict:
    """One (readout, seed); never raises, so one failure cannot take down the pool."""
    t0 = time.time()
    try:
        P = _setup(task)
        result = TASKS[task["readout"]](task, P)
        result["seconds"] = round(time.time() - t0, 1)
        return {"readout": task["readout"], "seed": task["seed"], "result": result}
    except Exception:
        return {"readout": task["readout"], "seed": task["seed"], "error": traceback.format_exc(),
                "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------
def stack_versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "numpy", "sklearn", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def summarize(per_seed: dict, cells: dict, dmax: str) -> dict:
    out = {}
    if per_seed.get("r210"):
        out["r210"] = pr.summarize_r210(per_seed["r210"])
    if per_seed.get("r205"):
        out["r205"] = pr.summarize_r205(per_seed["r205"])
    if per_seed.get("r35"):
        out["r35"] = pr.summarize_r35(per_seed["r35"])
    if per_seed.get("r206"):
        rec = {str(s): c["cell"]["agents"]["survival"]["pool"]["target"]
               for (d, s), c in cells.items() if d == dmax}
        out["r206"] = pr.summarize_r206(per_seed["r206"], rec)
    if per_seed.get("r33"):
        out["r33"] = pr.summarize_r33(per_seed["r33"])
    if per_seed.get("r31"):
        out["r31"] = pr.summarize_r31(per_seed["r31"])
    return out


def integrity(per_seed: dict, drifts) -> dict:
    """The agents' identity with the recorded run rests on r210's bit check, or on r205's (the
    same check) in an output without r210; the other readouts are not filtered by it, but every
    seed it could not reproduce is listed here."""
    by = "r210" if per_seed.get("r210") else "r205"
    r = per_seed.get(by)
    if not r:
        return {"checked": False, "note": "neither r210 nor r205 run; the agents' identity with the "
                                          "recorded run is unchecked in this output"}
    return {"checked": True, "checked_by": by, "seeds_checked": sorted(r, key=int),
            "unreproduced": {d: sorted([s for s, v in r.items() if d in v.get("reproduced", {})
                                        and not v["reproduced"][d]], key=int) for d in drifts}}


def write_json(path: str, doc: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=pr._jsonable)
    os.replace(tmp, path)


def contrast_main(a_path: str, b_path: str, out_path: str) -> int:
    with open(a_path, encoding="utf-8") as fh:
        A = json.load(fh)
    with open(b_path, encoding="utf-8") as fh:
        B = json.load(fh)
    try:
        doc = pr.balanced_contrast(A, B)
    except ValueError as e:
        raise SystemExit(str(e))
    doc.update({"generated_by": "scripts/run_post_review_readouts.py --contrast", "spec": pr.SPEC,
                "a_path": a_path, "b_path": b_path, "seed_bases": A.get("seed_bases")})
    write_json(out_path, doc)
    p = doc["primary"]
    if p:
        c = doc["contrasts"][p]
        print(f"balanced survival contrast {p}: {c['mean']:+.3f} t90 [{c['t90'][0]:+.3f}, "
              f"{c['t90'][1]:+.3f}] n={c['n']} -> {c['reading']}")
    print(f"wrote {out_path}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--readouts", default=",".join(pr.READOUTS),
                    help="comma-separated subset of " + ",".join(pr.READOUTS))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--partial", default=None, help="resume file (default <out>.partial.jsonl)")
    ap.add_argument("--balanced-reference", default=None,
                    help="a run_l0_audit.py output of the same agents; the r33 registered-base "
                         "targets at the strongest drift are checked against its paired entries")
    ap.add_argument("--contrast", nargs=2, metavar=("A_JSON", "B_JSON"), default=None,
                    help="write the balanced A minus B survival contrast of two outputs and exit")
    ap.add_argument("--quick", action="store_true",
                    help="tiny pools and decoders for a smoke run; the r210 bit check then fails "
                         "by design and no number from it is a result")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.contrast:
        return contrast_main(a.contrast[0], a.contrast[1], a.out)
    if not a.run_dir:
        raise SystemExit("--run-dir is required unless --contrast is given")
    if os.environ.get("ITASORL_FOLDS") != "explicit":
        raise SystemExit("set ITASORL_FOLDS=explicit, the partition every recorded target uses")
    readouts = [r.strip() for r in a.readouts.split(",") if r.strip()]
    bad = [r for r in readouts if r not in pr.READOUTS]
    if bad:
        raise SystemExit(f"unknown readout(s) {bad}; choose from {pr.READOUTS}")
    hits = pr.base_collisions()
    if hits:
        raise SystemExit(f"post-review seed bases collide with bases in use: {hits}")
    cells = load_cells(a.run_dir)
    cfg = run_config(cells)
    have = sorted({s for _, s in cells})
    complete = [s for s in have if all((d, s) in cells for d in cfg["drifts"])]
    seeds = complete if a.seeds is None else list(a.seeds)
    missing = [s for s in seeds if s not in complete]
    if missing:
        raise SystemExit(f"seed(s) {missing} lack a cell at some drift; complete seeds: {complete}")
    for d in cfg["drifts"]:
        for s in seeds:
            for arm in pr.ARMS:
                if not os.path.exists(_agent_path(a.run_dir, d, s, arm)):
                    raise SystemExit(f"missing saved agent {_agent_path(a.run_dir, d, s, arm)}")
    settings = settings_for(a.quick)
    fingerprints = {f"d{d}_s{s}": c["fingerprint"] for (d, s), c in sorted(cells.items())}
    header = {"spec": pr.SPEC, "fingerprints": fingerprints, "settings": settings,
              "seed_bases": seed_bases(), "knobs": cfg["knobs"], "drifts": cfg["drifts"]}
    store = pr.PartialStore(a.partial or a.out + ".partial.jsonl", header)
    try:
        done = store.load()
    except ValueError as e:
        raise SystemExit(str(e))
    todo = plan_tasks(readouts, seeds, done)
    base = {"run_dir": a.run_dir, "knobs": cfg["knobs"], "objective": cfg["objective"],
            "mortal": cfg["mortal"], "drifts": cfg["drifts"], "dmax": cfg["dmax"], "settings": settings,
            "recorded": recorded_pools(cells)}
    tasks = [{**base, "readout": r, "seed": s} for r, s in todo]
    print(f"post-review readouts on {a.run_dir}: {len(done)} task(s) already done, {len(tasks)} to run "
          f"({', '.join(readouts)}; seeds {seeds})", flush=True)
    errors = []

    def record(res):
        if "error" in res:
            store.append(res["readout"], res["seed"], error=res["error"])
            errors.append({"readout": res["readout"], "seed": res["seed"], "error": res["error"]})
            print(f"  {res['readout']} seed {res['seed']}: ERROR ({res['seconds']} s)", flush=True)
        else:
            store.append(res["readout"], res["seed"], result=res["result"])
            done[(res["readout"], str(res["seed"]))] = res["result"]
            print(f"  {res['readout']} seed {res['seed']}: done ({res['result']['seconds']} s)", flush=True)

    t0 = time.time()
    if a.workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:   # spawn: safe with torch
            for res in pool.imap_unordered(run_task, tasks):
                record(res)
    else:
        for t in tasks:
            record(run_task(t))
    per_seed: dict = {}
    for (r, s), res in done.items():
        per_seed.setdefault(r, {})[s] = res
    from itasorl import folds
    from itasorl.results_io import git_head
    doc = {"generated_by": "scripts/run_post_review_readouts.py", "spec": pr.SPEC,
           "git_commit": git_head(), "stack": stack_versions(), "run_dir": a.run_dir,
           "knobs": cfg["knobs"], "objective": cfg["objective"], "mortal": cfg["mortal"],
           "drifts": cfg["drifts"], "dmax": cfg["dmax"], "cell_fingerprints": fingerprints,
           "fingerprint_set": sorted({f for f in fingerprints.values() if f}),
           "gae_bootstrap": sorted({str(c["cell"].get("gae_bootstrap")) for c in cells.values()}),
           "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
           "seed_bases": seed_bases(), "settings": settings, "requested": {"readouts": readouts, "seeds": seeds},
           "per_seed": {r: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for r, v in per_seed.items()},
           "summary": summarize(per_seed, cells, cfg["dmax"]), "errors": errors,
           "wall_seconds_this_session": round(time.time() - t0, 1)}
    doc["integrity"] = integrity(per_seed, cfg["drifts"])
    if a.balanced_reference and per_seed.get("r33"):
        with open(a.balanced_reference, encoding="utf-8") as fh:
            ref = json.load(fh)
        doc["balanced_reference_check"] = {"reference": a.balanced_reference,
                                           **pr.balanced_reference_check(per_seed["r33"], ref,
                                                                         drift=float(cfg["dmax"]))}
    want = {(r, str(s)) for r in readouts for s in seeds}
    doc["complete"] = bool(want <= set(done)) and not errors
    doc["complete_all_readouts"] = bool({(r, str(s)) for r in pr.READOUTS for s in complete} <= set(done))
    write_json(a.out, doc)
    print(f"wrote {a.out} (complete for this request: {doc['complete']})", flush=True)
    return 0 if doc["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
