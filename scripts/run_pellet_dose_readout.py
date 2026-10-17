"""Evaluation-time pellet-density dose on saved agents (closing item X07, E10).

Post hoc and descriptive; changes no stored value or verdict. Built from
scripts/regenerate_state_dumps.py run_task (the run's own knobs, load_agent_bundle, then the
registered pooled readout: 110 x 24 episodes, pools 800000 and 850000, explicit folds), with one
change: `b2.SURVIVAL_FOOD["n_pellets"]` is set to each value of --n-pellets for the evaluation
worlds. The agents stay frozen; only the worlds they are read in change. Zero is refused, since
a world with no pellet cannot be reset (itasorl/patch_of_earth.py stacks one pellet per count).
At each density the untrained trunk replays the survival arm's own episodes (record_trajectories
then replay_states): the input-reference row.

Gate: the 24-pellet row (the training density) reproduces every cell's recorded target within
0.002 for every (arm, seed) read, else the step is VOID and only the gate is written. Rule on the
survival arm: (a) "pellet-directed" iff at 2 pellets its mean is under 0.65, the seed-paired
24-minus-2 contrast is at least +0.05 with an interval excluding zero, and the untrained arm stays
within 0.1 of 0.5 at every density; (b) "locomotion-carried" iff at 2 pellets its mean is at least
0.65 and its margin over the untrained arm there is at least +0.05 with an interval excluding
zero; (c) otherwise "not separated". The dose curve is a table, not thresholded, and every row
notes that the gate-0 oracle was not recalibrated at its density.

    python scripts/run_pellet_dose_readout.py --run-dir fullruns/corrected_l3_h8_wm \\
        --n-pellets 2 6 12 24 48 96 --arms survival predictor untrained --workers 4 \\
        --ckpt fullruns/closing_2026-10/x07/ckpt --out fullruns/closing_2026-10/x07/corrected_l3_h8_wm.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

GENERATED_BY = "scripts/run_pellet_dose_readout.py"
ARMS = ("survival", "predictor", "untrained")
TRAINING_DENSITY = 24
TOL = 0.002
BAR = 0.65
NOTE = "gate-0 oracle not recalibrated at this density"
STATUS = "post hoc, descriptive; changes no stored value or verdict"


def run_density(t: dict) -> dict:
    """One (arm, seed, density): regenerate_state_dumps.run_task with the density override (its
    dump goes to a temporary directory and is deleted), or, for kind "reference", the untrained
    trunk on the survival arm's episodes at that density."""
    import itasorl.experiment_b2 as b2
    from regenerate_state_dumps import run_task

    t0 = time.time()
    saved = dict(b2.SURVIVAL_FOOD)
    tmp = tempfile.mkdtemp(prefix="pellet_dose_")
    try:
        task = {**t["regen"], "knobs": {**(t["regen"].get("knobs") or {}), "n_pellets": t["n_pellets"]},
                "out_dir": tmp}
        if t["kind"] == "arm":
            r = run_task(task)
            applied = b2.SURVIVAL_FOOD["n_pellets"]
            out = {"kind": "arm", "arm": t["arm"], "seed": t["seed"], "n_pellets": t["n_pellets"],
                   "applied_n_pellets": applied, "target": r["target"], "recorded_target": r["recorded_target"]}
        else:
            out = {"kind": "reference", "arm": "untrained_on_survival_episodes", "seed": t["seed"],
                   "n_pellets": t["n_pellets"], **input_reference(task, t["untrained_agent"])}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        b2.SURVIVAL_FOOD.clear()
        b2.SURVIVAL_FOOD.update(saved)
    out["note"] = NOTE
    out["seconds"] = round(time.time() - t0, 1)
    return out


def input_reference(task: dict, untrained_path: str) -> dict:
    """The untrained trunk replaying the survival arm's own registered-pool episodes, with the
    world knobs (density included) applied as run_task applies them."""
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.eval_protocols import readout_from_states, record_trajectories, replay_states
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
        b2.setup_l3_surrogate(hidden=knobs.get("l3_hidden", 8), device="cpu", seed=knobs.get("l3_seed", 0), params=P)
    surv, snorm = b2.load_agent_bundle(task["agent"])
    untr, unorm = b2.load_agent_bundle(untrained_path)
    Oa, Aa, _ = record_trajectories(surv, snorm, P, 0.0, 110, 24, "cpu", 800_000, 5)
    Os, As, _ = record_trajectories(surv, snorm, P, task["drift"], 110, 24, "cpu", 850_000, 5)
    # independent pools (different worlds): one CV group per episode, written out
    r = readout_from_states(replay_states(untr, unorm, Oa, Aa), replay_states(untr, unorm, Os, As),
                            seed=task["seed"], groups=np.arange(len(Oa) + len(Os)))
    return {"applied_n_pellets": b2.SURVIVAL_FOOD["n_pellets"], "target": float(r["target"]),
            "n": [int(len(Oa)), int(len(Os))]}


def gate(rows: list) -> list[str]:
    fails = []
    for r in rows:
        if r["kind"] != "arm" or r["n_pellets"] != TRAINING_DENSITY:
            continue
        d = abs(float(r["target"]) - float(r["recorded_target"]))
        if not (np.isfinite(d) and d <= TOL):
            fails.append(f"{r['arm']} seed {r['seed']}: 24-pellet target {r['target']!r} vs cell {r['recorded_target']!r}")
    return fails


def summarize(rows: list, densities, arms) -> dict:
    from itasorl.novel_readouts import paired_t90

    def vals(kind, arm, n):
        rr = sorted([r for r in rows if r["kind"] == kind and r["arm"] == arm and r["n_pellets"] == n],
                    key=lambda r: r["seed"])
        return {r["seed"]: float(r["target"]) for r in rr}

    curve = {}
    for n in densities:
        row = {a: paired_t90(list(vals("arm", a, n).values())) for a in arms}
        ref = vals("reference", "untrained_on_survival_episodes", n)
        if ref:
            row["input_reference"] = paired_t90(list(ref.values()))
        row["note"] = NOTE
        curve[str(n)] = row
    out = {"dose_curve": curve}
    if {2, TRAINING_DENSITY} <= set(densities) and {"survival", "untrained"} <= set(arms):
        s2, s24, u2 = vals("arm", "survival", 2), vals("arm", "survival", TRAINING_DENSITY), vals("arm", "untrained", 2)
        seeds = sorted(set(s2) & set(s24) & set(u2))
        c24 = paired_t90([s24[s] for s in seeds], [s2[s] for s in seeds])
        m2 = paired_t90([s2[s] for s in seeds], [u2[s] for s in seeds])
        surv2 = curve["2"]["survival"]["mean"]
        untr_ok = all(abs(curve[str(n)]["untrained"]["mean"] - 0.5) <= 0.1 for n in densities)
        a = bool(surv2 < BAR and c24["mean"] >= 0.05 and c24["t90"][0] > 0 and untr_ok)
        b = bool(surv2 >= BAR and m2["mean"] >= 0.05 and m2["t90"][0] > 0)
        out["readings"] = {"survival 24 minus 2": c24, "survival minus untrained at 2": m2,
                           "untrained within 0.1 of 0.5 at every density": untr_ok,
                           "label": "pellet-directed" if a else "locomotion-carried" if b else "not separated"}
    else:
        out["readings"] = {"label": "not computable (needs densities 2 and 24 and the survival and untrained arms)"}
    return out


def run_provenance(a) -> dict:
    from _readout_common import _rel_argv
    from agent_free_common import provenance
    from promote_bv3_gates import repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(a.out), device="cpu")
    return {**p, "agents_dir": repo_rel(os.path.join(a.run_dir, "agents")), "argv": _rel_argv(list(sys.argv[1:]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-pellets", type=int, nargs="+", required=True, metavar="N")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (default OUT.ckpt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args(argv)
    bad = [n for n in a.n_pellets if n < 1]
    if bad:
        ap.error(f"--n-pellets {bad}: a world needs at least one pellet (itasorl/patch_of_earth.py stacks one "
                 "pellet per count and cannot reset an empty list)")
    if TRAINING_DENSITY not in a.n_pellets:
        ap.error(f"--n-pellets must include {TRAINING_DENSITY}, the training density the integrity gate reads")
    if os.environ.get("ITASORL_FOLDS", "explicit") != "explicit":
        ap.error("the recorded targets use the explicit fold partition; set ITASORL_FOLDS=explicit")
    from regenerate_state_dumps import tasks_for
    regen = tasks_for(a.run_dir)
    if not regen:
        raise SystemExit(f"no saved agents under {a.run_dir}/agents")
    dmax = max(t["drift"] for t in regen)
    regen = [t for t in regen if t["drift"] == dmax and t["arm"] in a.arms
             and (a.seeds is None or t["seed"] in a.seeds)]
    seeds = sorted({t["seed"] for t in regen})
    by = {(t["arm"], t["seed"]): t for t in regen}
    densities = sorted(set(a.n_pellets))
    tasks = [{"kind": "arm", "arm": t["arm"], "seed": t["seed"], "n_pellets": n, "regen": t}
             for n in densities for t in regen]
    if "survival" in a.arms:
        for s in seeds:
            if ("survival", s) in by:
                upath = by[("survival", s)]["agent"].replace("_survival.pt", "_untrained.pt")
                tasks += [{"kind": "reference", "arm": "untrained_on_survival_episodes", "seed": s, "n_pellets": n,
                           "regen": by[("survival", s)], "untrained_agent": upath} for n in densities]
    ckpt = a.ckpt or a.out + ".ckpt"
    if a.dry_run:
        print(f"{len(tasks)} task(s): arms {a.arms}, seeds {seeds}, drift {dmax}, densities {densities}; "
              f"checkpoints {ckpt}; would write {a.out}")
        return 0
    from _readout_common import fingerprint_extra, run_tasks
    fpx = fingerprint_extra(__file__, os.path.join(a.run_dir, "agents"), seeds, [(g, dmax) for g in a.arms])
    rows = run_tasks(run_density, tasks, ckpt, lambda t: f"{t['kind']}_{t['arm']}_s{t['seed']}_n{t['n_pellets']}",
                     a.workers, fpx)
    rows = sorted([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
                  key=lambda r: (r["kind"], r["arm"], r["n_pellets"], r["seed"]))
    for r in rows:
        if r["applied_n_pellets"] != r["n_pellets"]:
            raise SystemExit(f"{r['arm']} seed {r['seed']}: density {r['n_pellets']} was not applied "
                             f"(SURVIVAL_FOOD held {r['applied_n_pellets']})")
    fails = gate(rows)
    from agent_free_common import write_json
    from promote_bv3_gates import repo_rel
    doc = {"provenance": run_provenance(a), "status": STATUS, "run_dir": repo_rel(a.run_dir), "drift": dmax,
           "settings": {"n_pellets": densities, "override": "b2.SURVIVAL_FOOD['n_pellets'] in the evaluation worlds; "
                        "the agents are frozen", "pools": [800_000, 850_000], "n_eps": 110, "steps": 24,
                        "gate_tolerance": TOL, "training_density": TRAINING_DENSITY}}
    if fails:
        doc["integrity"] = {"gate": "VOID", "failures": fails}
        write_json(a.out, doc)
        print("VOID: the 24-pellet row does not reproduce the cells:\n  " + "\n  ".join(fails), file=sys.stderr)
        return 1
    doc["integrity"] = {"gate": "PASS", "n_checked": sum(1 for r in rows if r["kind"] == "arm"
                                                         and r["n_pellets"] == TRAINING_DENSITY)}
    doc["summary"] = summarize(rows, densities, a.arms)
    doc["rows"] = rows
    write_json(a.out, doc)
    print("label:", doc["summary"]["readings"]["label"], "| wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
