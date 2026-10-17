"""Nonlinear state probes on the corrected L1 dumps (closing item H53, runbook 1.6 and R17).

The L1 negative is read by the registered linear probe alone, while the L2 and L3 states were
also read by a random forest and a small MLP. This scores the stored L1 state dumps
(`<run>/states/states_d<tag>_s<seed>_<arm>.npz`, the name pattern of
scripts/rf_probe_historical_h8.py) with three probe classes on the registered feature
(itasorl.experiment_b.episode_features, mean and final state):

  linear  the registered probe (scaler and logistic regression, C = 1, max_iter 2000)
  rf      a random forest of 200 trees, random_state 0, one job (as rf_probe_historical_h8.py
          and scripts/run_expB_nonlinear.py fit it)
  mlp     the 64-unit MLP of the r210 battery (itasorl.post_review.mlp_probe_auroc: scaler,
          one hidden layer of 64 ReLU units, L2 1e-3, Adam 1e-3, 500 iterations,
          random_state = cell seed)

Every class uses the explicit-v1 folds (one group per episode, 5 folds; the pools are drawn at
different world-seed bases, so no world seed repeats across rows) and the same eight label
permutations as pooled_readout's shuffled control (post_review.registered_permutations).

Integrity gate, before anything is scored: the linear C = 1 target recomputed from the dumps
must equal each cell's recorded pool target (`<run>/cells/cell_d<tag>_s<seed>.json`) within
1e-9, and the survival mean at drift 0.023 (recorded 0.5506) must reproduce within 1e-9. If the
gate fails the script reports nothing: it prints the failing cells, writes no output and exits
1. A cell that misses alone is listed and not scored. `--dry-run` checks the inputs and runs the
gate (linear fits only, seconds) and writes nothing.

Drift values become file tags through itasorl.experiment_b2.format_drift (0.023 -> "0.0230");
a listed drift with no dump in the states dir is a hard error naming the dir and the drift.

Post hoc and descriptive: it changes no stored value or verdict.

    python scripts/run_l1_nonlinear_probe.py --states-dir fullruns/corrected_l1_heldout/states \\
        --arms survival predictor untrained --drifts 0.00 0.023 \\
        --out fullruns/closing_2026-10/h53/l1_nonlinear_corrected.json [--dry-run]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from rf_probe_historical_h8 import FNAME

GENERATED_BY = "scripts/run_l1_nonlinear_probe.py"
ARMS = ("survival", "predictor", "untrained")
CLASSES = ("linear", "rf", "mlp")
RF_TREES, RF_STATE = 200, 0
GATE_DRIFT, GATE_ARM, GATE_TOL = 0.023, "survival", 1e-9
STATUS = "post hoc, descriptive; changes no stored value or verdict (closing item H53)"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def dump_name(tag: str, seed: int, arm: str) -> str:
    return f"states_d{tag}_s{seed}_{arm}.npz"


def discover(states_dir: str, drifts, arms) -> dict:
    """{tag: [seed, ...]} for every listed drift, each seed carrying a dump for every arm.
    A drift with no dump, or a seed missing an arm, is a hard error that names the dir."""
    from itasorl.experiment_b2 import format_drift
    if not os.path.isdir(states_dir):
        raise SystemExit(f"states dir {states_dir} does not exist")
    found: dict = {}
    for p in glob.glob(os.path.join(states_dir, "states_d*_s*_*.npz")):
        m = FNAME.search(os.path.basename(p))
        if m:
            found.setdefault(m["drift"], {}).setdefault(int(m["seed"]), set()).add(m["agent"])
    out = {}
    for d in drifts:
        tag = format_drift(d)
        if tag not in found:
            raise SystemExit(f"no state dump for drift {d} (file tag d{tag}) under {states_dir}")
        seeds = sorted(found[tag])
        missing = [(s, arm) for s in seeds for arm in arms if arm not in found[tag][s]]
        if missing:
            raise SystemExit(f"{states_dir}: drift {tag} lacks dumps for (seed, arm) {missing}")
        out[tag] = seeds
    return out


def cells_dir_for(states_dir: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(states_dir)), "cells")


def recorded_pool(cells_dir: str, tag: str, seed: int, arm: str) -> dict:
    path = os.path.join(cells_dir, f"cell_d{tag}_s{seed}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no recorded cell {path}; the gate needs it")
    with open(path, encoding="utf-8") as fh:
        pool = json.load(fh)["cell"]["agents"][arm]["pool"]
    return {"target": pool.get("target"), "shuffled": pool.get("shuffled")}


def load_xy(states_dir: str, tag: str, seed: int, arm: str):
    from itasorl.experiment_b import episode_features
    with np.load(os.path.join(states_dir, dump_name(tag, seed, arm))) as z:
        Ha, Hs = np.asarray(z["Ha"]), np.asarray(z["Hs"])
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Probe classes (explicit folds, one group per episode)
# ---------------------------------------------------------------------------
def linear_auroc(X, y, groups=None, *, seed: int = 0, **_) -> dict:
    """The registered probe: experiment_a.grouped_auroc (scaler and logistic regression)."""
    from itasorl.experiment_a import grouped_auroc
    g = np.arange(len(y)) if groups is None else groups
    return {"auroc": float(grouped_auroc(X, y, g))}


def rf_auroc(X, y, groups=None, *, seed: int = 0, **_) -> dict:
    """200 trees, random_state 0 whatever the cell seed, one job, in-fold; mean fold AUROC."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    from itasorl import folds
    g = np.arange(len(y)) if groups is None else groups
    aucs = []
    for tr, te in folds.split(g, 5):
        if len(np.unique(y[te])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=RF_TREES, random_state=RF_STATE, n_jobs=1)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return {"auroc": float(np.mean(aucs)) if aucs else float("nan")}


def mlp_auroc(X, y, groups=None, *, seed: int = 0, **_) -> dict:
    from itasorl.post_review import mlp_probe_auroc
    g = np.arange(len(y)) if groups is None else groups
    return mlp_probe_auroc(X, y, groups=g, seed=seed)


SCORERS = {"linear": linear_auroc, "rf": rf_auroc, "mlp": mlp_auroc}


# ---------------------------------------------------------------------------
# Gate and scoring
# ---------------------------------------------------------------------------
def gate(states_dir: str, plan: dict, arms, cells_dir: str | None = None) -> dict:
    """Recompute the linear C = 1 target of every cell under the explicit folds and compare it
    with the recorded pool target. Passes when every cell is within 1e-9 and the survival mean
    at drift 0.023 is within 1e-9 of the recorded mean."""
    from itasorl import folds
    from itasorl.experiment_b2 import format_drift
    cells_dir = cells_dir or cells_dir_for(states_dir)
    gtag = format_drift(GATE_DRIFT)
    if gtag not in plan or GATE_ARM not in arms:
        raise SystemExit(f"the gate reads {GATE_ARM} at drift {GATE_DRIFT}; request both")
    rows = []
    with folds.fold_scheme("explicit"):
        for tag, seeds in plan.items():
            for s in seeds:
                for arm in arms:
                    X, y = load_xy(states_dir, tag, s, arm)
                    rec = recorded_pool(cells_dir, tag, s, arm)
                    t = linear_auroc(X, y, np.arange(len(y)))["auroc"]
                    diff = abs(t - float(rec["target"])) if rec["target"] is not None else float("nan")
                    rows.append({"drift": tag, "seed": s, "arm": arm, "linear_c1": t,
                                 "recorded_target": rec["target"], "abs_diff": diff,
                                 "reproduced": bool(np.isfinite(diff) and diff <= GATE_TOL)})
    g = [r for r in rows if r["drift"] == gtag and r["arm"] == GATE_ARM]
    mean_new = float(np.mean([r["linear_c1"] for r in g]))
    mean_rec = float(np.mean([r["recorded_target"] for r in g]))
    mean_diff = abs(mean_new - mean_rec)
    bad = [(r["drift"], r["seed"], r["arm"]) for r in rows if not r["reproduced"]]
    gate_cells_ok = all(r["reproduced"] for r in g)
    return {"rule": f"linear C = 1 (explicit folds) reproduces every recorded cell target and the "
                    f"{GATE_ARM} mean at drift {gtag} within {GATE_TOL:g}",
            "cells_dir": cells_dir, "fold_scheme": "explicit", "fold_version": folds.scheme_version("explicit"),
            "gate_drift": gtag, "gate_arm": GATE_ARM, "n_gate_seeds": len(g),
            "reproduced_mean": mean_new, "recorded_mean": mean_rec, "mean_abs_diff": mean_diff,
            "unreproduced_cells": bad,
            "pass": bool(gate_cells_ok and np.isfinite(mean_diff) and mean_diff <= GATE_TOL),
            "cells": rows}


def score_cell(task: dict) -> dict:
    """Every class's target, shuffled control over the eight registered permutations, and
    selectivity on one dump."""
    from threadpoolctl import threadpool_limits

    from itasorl import folds
    from itasorl.post_review import N_PERM, registered_permutations, score_probe_classes
    t0 = time.time()
    with folds.fold_scheme("explicit"), threadpool_limits(1):
        X, y = load_xy(task["states_dir"], task["drift"], task["seed"], task["arm"])
        perms = registered_permutations(y, task["seed"], N_PERM)
        classes = score_probe_classes(X, y, seed=task["seed"], perms=perms, classes=CLASSES,
                                      scorers=SCORERS)
    rec = task["recorded"]
    out = {k: task[k] for k in ("drift", "seed", "arm")}
    out.update(n_auth=int((y == 0).sum()), n_surr=int((y == 1).sum()), classes=classes,
               recorded_target=rec["target"], recorded_shuffled=rec["shuffled"],
               perm_check=bool(rec["shuffled"] is not None
                               and classes["linear"]["shuffled"] == float(rec["shuffled"])),
               seconds=round(time.time() - t0, 1))
    return out


def summarize(cells: list, arms) -> dict:
    from itasorl.post_review import interval_label, paired_summary, seed_summary
    out = {}
    for tag in sorted({c["drift"] for c in cells}, key=float):
        by = {(c["seed"], c["arm"]): c for c in cells if c["drift"] == tag}
        seeds = sorted({s for s, _ in by})

        def val(s, arm, cls, key="target"):
            c = by.get((s, arm))
            return c["classes"][cls][key] if c else float("nan")

        block = {"seeds": seeds, "arms": {}, "class_minus_linear": {}, "survival_minus": {}}
        for arm in arms:
            block["arms"][arm] = {cls: {k: seed_summary([val(s, arm, cls, k) for s in seeds])
                                        for k in ("target", "shuffled", "selectivity")} for cls in CLASSES}
            block["class_minus_linear"][arm] = {}
            for cls in CLASSES[1:]:
                d = paired_summary({str(s): val(s, arm, cls) for s in seeds},
                                   {str(s): val(s, arm, "linear") for s in seeds})
                block["class_minus_linear"][arm][cls] = {**d, "reading": interval_label(d)}
        if "survival" in arms:
            for other in [a for a in arms if a != "survival"]:
                block["survival_minus"][other] = {}
                for cls in CLASSES:
                    d = paired_summary({str(s): val(s, "survival", cls) for s in seeds},
                                       {str(s): val(s, other, cls) for s in seeds})
                    block["survival_minus"][other][cls] = {**d, "reading": interval_label(d)}
        out[tag] = block
    return out


def run_provenance(out: str, states_dir: str, argv: list) -> dict:
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(out), device="cpu")
    return {**p, "states_dir": repo_rel(states_dir), "argv": rel_flags(list(argv))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--drifts", type=float, nargs="+", default=[0.0, 0.023])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="check the inputs and run the linear gate; write nothing")
    a = ap.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_free_common import pin_one_thread, write_json
    pin_one_thread()
    plan = discover(a.states_dir, a.drifts, a.arms)
    print(f"{a.states_dir}: " + ", ".join(f"d{t} seeds {s}" for t, s in plan.items())
          + f", arms {a.arms}", flush=True)
    g = gate(a.states_dir, plan, a.arms)
    print(f"gate: {GATE_ARM} d{g['gate_drift']} linear C=1 mean {g['reproduced_mean']!r} against "
          f"recorded {g['recorded_mean']!r} (|diff| {g['mean_abs_diff']:.3g}); "
          f"unreproduced cells {g['unreproduced_cells']}; pass {g['pass']}", flush=True)
    if not g["pass"]:
        print("GATE FAILED: nothing is reported", file=sys.stderr)
        return 1
    if a.dry_run:
        print("dry run: inputs and gate checked; nothing scored or written")
        return 0
    cells_dir = cells_dir_for(a.states_dir)
    tasks = [{"states_dir": a.states_dir, "drift": r["drift"], "seed": r["seed"], "arm": r["arm"],
              "recorded": recorded_pool(cells_dir, r["drift"], r["seed"], r["arm"])}
             for r in g["cells"] if r["reproduced"]]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            cells = list(pool.imap(score_cell, tasks))
    else:
        cells = [score_cell(t) for t in tasks]
    for c in cells:
        cl = c["classes"]
        print(f"d{c['drift']} s{c['seed']} {c['arm']:9s} linear {cl['linear']['target']:.3f} "
              f"rf {cl['rf']['target']:.3f} mlp {cl['mlp']['target']:.3f}", flush=True)
    from promote_bv3_gates import repo_rel
    doc = {"provenance": run_provenance(a.out, a.states_dir, argv), "status": STATUS,
           "states_dir": repo_rel(a.states_dir), "arms": list(a.arms),
           "drifts": {t: s for t, s in plan.items()},
           "features": "itasorl.experiment_b.episode_features (mean and final state)",
           "folds": "explicit-v1, one group per episode, 5 folds (independent pools at different bases)",
           "classes": {"linear": "experiment_a.grouped_auroc, scaler and logistic regression, C = 1",
                       "rf": {"n_estimators": RF_TREES, "random_state": RF_STATE, "n_jobs": 1},
                       "mlp": "itasorl.post_review.mlp_probe_auroc (64 units, alpha 1e-3, lr 1e-3, "
                              "max_iter 500, random_state = cell seed)"},
           "shuffled": "mean over post_review.registered_permutations(y, cell seed, 8)",
           "gate": {k: v for k, v in g.items() if k != "cells"}, "gate_cells": g["cells"],
           "cells": cells, "summary": summarize(cells, a.arms),
           "seconds": round(time.time() - t0, 1)}
    write_json(a.out, doc)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
