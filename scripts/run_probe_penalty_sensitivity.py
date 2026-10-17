"""Probe-penalty sensitivity of the registered linear probe (closing item G01, runbook 1.6, R11).

The registered probe is a scaler and an L2 logistic regression at C = 1 on the 192-column
feature [mean h, h_T], with about 176 training episodes per fold, so the penalty, not the
linear form, bounds what it can read. This re-scores stored state dumps with the same probe at
C in {0.001, 0.01, 0.1, 1, 10}, and with C chosen inside each outer fold by a grouped inner
cross-validation on the outer-training rows only (nested; the test fold never informs C).

Per dump (one (drift, seed, arm) cell) and per C: the target AUROC, a shuffled-label baseline
(mean over the eight label permutations of pooled_readout's control,
post_review.registered_permutations) and the selectivity, with the convergence warnings of
every fit counted. Folds are the explicit-v1 partition whatever ITASORL_FOLDS says, one group
per episode (the pools are drawn at different world-seed bases), max_iter 2000.

Integrity, per states dir, before anything is reported: C = 1 must reproduce the dir's
reference within 1e-9 in every cell (`resolve_reference`):
  l3_h8_heldout, l1_heldout          explicit target of artifacts/fold_rescore/<run>.json
                                     (their cells were scored under the legacy partition)
  corrected_l1_heldout,              the cell's recorded pool target, <run>/cells
  balanced_confirmation_wm/_touch
  any other dir (C1, the G57 dumps)  the regeneration report of scripts/regenerate_state_dumps.py
                                     beside it: C = 1 must equal the rebuilt target within 1e-9,
                                     and the rows carry the G57 label "cross-stack regenerated,
                                     max abs target delta = x" against the recorded target
A dir whose reference is missing or not reproduced is reported as VOID: nothing is written for
it and the script exits 1. Each passing dir is written to OUT_DIR/<label>.json, and the
survival mean per drift is printed at full precision.

Drift values given with --drifts become file tags through experiment_b2.format_drift
(0.023 -> "0.0230"); a listed drift with no dump is a hard error naming the dir and the drift.
Without --drifts every drift found in the dir is read.

No TOST or gate is read at any C other than 1. Post hoc and descriptive: it changes no stored
value or verdict.

    python scripts/run_probe_penalty_sensitivity.py --states fullruns/l3_h8_heldout/states \\
        fullruns/corrected_l1_heldout/states ... --out-dir fullruns/closing_2026-10/g01
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

from rf_probe_historical_h8 import FNAME

GENERATED_BY = "scripts/run_probe_penalty_sensitivity.py"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
MAX_ITER = 2000
N_SPLITS = 5
REF_TOL = 1e-9
ARMS = ("survival", "predictor", "untrained")
#: runs whose cells were scored under the legacy partition: the explicit fold re-score is the reference
FOLD_RESCORE_RUNS = ("l3_h8_heldout", "l1_heldout")
#: runs scored under the explicit partition: the recorded cell target is the reference
CELL_TARGET_RUNS = ("corrected_l1_heldout", "balanced_confirmation_wm", "balanced_confirmation_touch")
STATUS = "post hoc, descriptive; no TOST or gate at any C other than 1; changes no stored value or verdict"


# ---------------------------------------------------------------------------
# Dumps and references (shared with scripts/run_step_resolved_readout.py)
# ---------------------------------------------------------------------------
def discover(states_dir: str, drifts=None, arms=ARMS) -> dict:
    """{tag: [seed, ...]} of the dumps under states_dir (name pattern of
    rf_probe_historical_h8.py), each seed carrying every arm. Listed drifts become tags through
    format_drift; one with no dump is a hard error that names the dir and the drift."""
    from itasorl.experiment_b2 import format_drift
    if not os.path.isdir(states_dir):
        raise SystemExit(f"states dir {states_dir} does not exist")
    found: dict = {}
    for p in glob.glob(os.path.join(states_dir, "states_d*_s*_*.npz")):
        m = FNAME.search(os.path.basename(p))
        if m:
            found.setdefault(m["drift"], {}).setdefault(int(m["seed"]), set()).add(m["agent"])
    if drifts is None:
        tags = sorted(found, key=float)
        if not tags:
            raise SystemExit(f"no state dumps under {states_dir}")
    else:
        tags = []
        for d in drifts:
            tag = format_drift(d)
            if tag not in found:
                raise SystemExit(f"no state dump for drift {d} (file tag d{tag}) under {states_dir}")
            tags.append(tag)
    out = {}
    for tag in tags:
        seeds = sorted(found[tag])
        missing = [(s, a) for s in seeds for a in arms if a not in found[tag][s]]
        if missing:
            raise SystemExit(f"{states_dir}: drift {tag} lacks dumps for (seed, arm) {missing}")
        out[tag] = seeds
    return out


def load_states(states_dir: str, tag: str, seed: int, arm: str):
    """(H (n, T, hidden), y) of one dump, authentic rows first."""
    with np.load(os.path.join(states_dir, f"states_d{tag}_s{seed}_{arm}.npz")) as z:
        Ha, Hs = np.asarray(z["Ha"]), np.asarray(z["Hs"])
    return np.concatenate([Ha, Hs]), np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)


def _regen_report(run_dir: str) -> tuple[str, dict]:
    hits = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        try:
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and "run_dir" in doc and isinstance(doc.get("cells"), list) \
                and all({"drift", "seed", "arm", "target", "recorded_target"} <= set(c) for c in doc["cells"]):
            hits.append((p, doc))
    if len(hits) != 1:
        raise SystemExit(f"{run_dir}: expected one regenerate_state_dumps report beside the dumps, "
                         f"found {len(hits)}")
    return hits[0]


def resolve_reference(states_dir: str) -> dict:
    """The C = 1 reference of every cell of a states dir: {"kind", "label", "source",
    "targets": {(tag, seed, arm): target}} plus, for regenerated dumps, the recorded targets,
    the tolerance and the G57 label. Fails closed on a dir it cannot place."""
    from itasorl.experiment_b2 import format_drift
    from promote_bv3_gates import repo_rel
    run_dir = os.path.dirname(os.path.normpath(os.path.abspath(states_dir)))
    name = os.path.basename(run_dir)
    if name in FOLD_RESCORE_RUNS:
        path = os.path.join(ROOT, "artifacts", "fold_rescore", f"{name}.json")
        if not os.path.exists(path):
            raise SystemExit(f"{states_dir}: reference {path} is missing")
        with open(path, encoding="utf-8") as fh:
            cells = json.load(fh)["cells"]
        return {"kind": "fold_rescore_explicit", "label": name,
                "source": f"artifacts/fold_rescore/{name}.json (explicit target)",
                "targets": {(c["drift"], int(c["seed"]), c["agent"]): float(c["explicit"]["target"])
                            for c in cells}}
    if name in CELL_TARGET_RUNS:
        targets = {}
        for p in sorted(glob.glob(os.path.join(run_dir, "cells", "cell_d*_s*.json"))):
            tag, seed = os.path.basename(p)[len("cell_d"):-len(".json")].split("_s")
            with open(p, encoding="utf-8") as fh:
                agents = json.load(fh)["cell"]["agents"]
            for arm, rec in agents.items():
                t = rec.get("pool", {}).get("target")
                if t is not None:
                    targets[(tag, int(seed), arm)] = float(t)
        if not targets:
            raise SystemExit(f"{states_dir}: no recorded cell targets under {run_dir}/cells")
        return {"kind": "cell_target", "label": name, "source": f"{repo_rel(run_dir)}/cells (pool.target)",
                "targets": targets}
    path, rep = _regen_report(run_dir)
    tol = rep.get("tolerance")
    tol = 0.0 if tol is None else float(tol)
    run_label = os.path.basename(os.path.normpath(str(rep["run_dir"])))
    key = lambda c: (format_drift(float(c["drift"])), int(c["seed"]), c["arm"])  # noqa: E731
    diffs = [abs(float(c["target"]) - float(c["recorded_target"])) for c in rep["cells"]]
    max_d = float(max(diffs)) if diffs else float("nan")
    return {"kind": "regenerated", "label": f"{run_label}_regenerated",
            "source": f"{repo_rel(path)} (rebuilt target; recorded target of {repo_rel(str(rep['run_dir']))})",
            "targets": {key(c): float(c["target"]) for c in rep["cells"]},
            "recorded": {key(c): float(c["recorded_target"]) for c in rep["cells"]},
            "tolerance": tol, "max_abs_delta": max_d,
            "within_tolerance": bool(np.isfinite(max_d) and max_d <= tol),
            "tolerance_label": f"cross-stack regenerated, max abs target delta = {max_d:.3g}"}


def check_reference(ref: dict, c1: dict) -> dict:
    """Integrity of one dir: every recomputed C = 1 target {(tag, seed, arm): value} against
    the reference within 1e-9 (and, for regenerated dumps, the rebuild within its tolerance)."""
    rows, bad = [], []
    for k in sorted(c1):
        r = ref["targets"].get(k)
        diff = abs(c1[k] - r) if r is not None else float("nan")
        ok = bool(np.isfinite(diff) and diff <= REF_TOL)
        rows.append({"drift": k[0], "seed": k[1], "arm": k[2], "c1": c1[k], "reference": r,
                     "abs_diff": diff, "ok": ok})
        if not ok:
            bad.append(list(k))
    out = {"kind": ref["kind"], "source": ref["source"], "tolerance": REF_TOL, "n_checked": len(rows),
           "failed": bad, "worst_abs_diff": float(max((r["abs_diff"] for r in rows), default=float("nan"))),
           "cells": rows}
    survival = {}
    for tag in sorted({k[0] for k in c1}, key=float):
        ks = [k for k in c1 if k[0] == tag and k[2] == "survival"]
        if ks:
            survival[tag] = {"c1_mean": float(np.mean([c1[k] for k in ks])),
                             "reference_mean": float(np.mean([ref["targets"].get(k, np.nan) for k in ks])),
                             "n": len(ks)}
    out["survival_means"] = survival
    out["pass"] = bool(rows and not bad)
    if ref["kind"] == "regenerated":
        out.update(regen_tolerance=ref["tolerance"], regen_max_abs_delta=ref["max_abs_delta"],
                   regen_within_tolerance=ref["within_tolerance"], tolerance_label=ref["tolerance_label"])
        out["pass"] = bool(out["pass"] and ref["within_tolerance"])
    return out


# ---------------------------------------------------------------------------
# The probe at a given C
# ---------------------------------------------------------------------------
def cv_auroc(X, y, groups, C: float, splits=None) -> tuple[float, int]:
    """experiment_a.grouped_auroc with the logistic regression at penalty C: scaler and
    LogisticRegression(C, max_iter 2000) fit on the training folds, mean fold AUROC (folds whose
    test rows hold one class are skipped). Returns (AUROC, convergence warnings). At C = 1 this
    is the registered probe, bit for bit."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from itasorl import folds
    splits = list(folds.split(groups, N_SPLITS, scheme="explicit")) if splits is None else splits
    aucs, n_warn = [], 0
    for tr, te in splits:
        if len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X[tr], y[tr])
        n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return (float(np.mean(aucs)) if aucs else float("nan")), n_warn


def nested_auroc(X, y, groups, grid=C_GRID) -> dict:
    """C chosen inside each outer fold by a grouped inner 5-fold CV on the outer-training rows
    only (the explicit partition of their groups), the largest mean inner AUROC winning (ties go
    to the smaller C); the probe is then refit on the outer-training rows at that C and scored on
    the outer test fold."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from itasorl import folds
    aucs, chosen, inner_all, n_warn = [], [], [], 0
    for tr, te in folds.split(groups, N_SPLITS, scheme="explicit"):
        if len(np.unique(y[te])) < 2:
            continue
        Xt, yt, gt = X[tr], y[tr], np.asarray(groups)[tr]
        inner_splits = list(folds.split(gt, N_SPLITS, scheme="explicit"))
        inner = []
        for C in grid:
            a, k = cv_auroc(Xt, yt, gt, C, splits=inner_splits)
            inner.append(a)
            n_warn += k
        finite = [a if np.isfinite(a) else -np.inf for a in inner]
        best = grid[int(np.argmax(finite))]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=MAX_ITER))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(Xt, yt)
        n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        chosen.append(best)
        inner_all.append(inner)
    return {"target": float(np.mean(aucs)) if aucs else float("nan"), "chosen_C": chosen,
            "inner_auroc": inner_all, "convergence_warnings": n_warn}


def score_cell(task: dict) -> dict:
    """The C grid, its shuffled baselines and the nested choice on one dump."""
    from threadpoolctl import threadpool_limits

    from itasorl.experiment_b import episode_features
    from itasorl.post_review import N_PERM, registered_permutations
    t0 = time.time()
    H, y = load_states(task["states_dir"], task["drift"], task["seed"], task["arm"])
    X = episode_features(H)
    g = np.arange(len(y))           # independent pools: one group per episode
    perms = registered_permutations(y, task["seed"], N_PERM)
    grid = {}
    with threadpool_limits(1):
        for C in task["grid"]:
            t, wt = cv_auroc(X, y, g, C)
            sh = [cv_auroc(X, yp, g, C) for yp in perms]
            s = float(np.mean([a for a, _ in sh]))
            grid[f"{C:g}"] = {"target": t, "shuffled": s, "selectivity": t - s,
                              "shuffled_draws": [a for a, _ in sh],
                              "convergence_warnings_target": wt,
                              "convergence_warnings_shuffled": int(sum(k for _, k in sh))}
        nested = nested_auroc(X, y, g, tuple(task["grid"]))
    return {"drift": task["drift"], "seed": task["seed"], "arm": task["arm"],
            "n_auth": int((y == 0).sum()), "n_surr": int((y == 1).sum()), "n_features": int(X.shape[1]),
            "grid": grid, "nested": nested, "seconds": round(time.time() - t0, 1)}


def summarize(cells: list, grid) -> dict:
    from itasorl.post_review import seed_summary
    out = {}
    for tag in sorted({c["drift"] for c in cells}, key=float):
        for arm in ARMS:
            rows = sorted([c for c in cells if c["drift"] == tag and c["arm"] == arm], key=lambda c: c["seed"])
            if not rows:
                continue
            block = {"seeds": [c["seed"] for c in rows], "by_C": {}}
            for C in grid:
                k = f"{C:g}"
                block["by_C"][k] = {m: seed_summary([c["grid"][k][m] for c in rows])
                                    for m in ("target", "shuffled", "selectivity")}
                block["by_C"][k]["convergence_warnings"] = int(sum(
                    c["grid"][k]["convergence_warnings_target"] + c["grid"][k]["convergence_warnings_shuffled"]
                    for c in rows))
            block["nested"] = {"target": seed_summary([c["nested"]["target"] for c in rows]),
                               "chosen_C_counts": {f"{C:g}": int(sum(x == C for c in rows
                                                                     for x in c["nested"]["chosen_C"]))
                                                   for C in grid},
                               "convergence_warnings": int(sum(c["nested"]["convergence_warnings"] for c in rows))}
            out[f"d={tag} {arm}"] = block
    return out


def c1_targets(states_dir: str, plan: dict) -> dict:
    """The registered probe (C = 1, explicit folds) on every dump of the plan."""
    from itasorl.experiment_b import episode_features
    out = {}
    for tag, seeds in plan.items():
        for s in seeds:
            for arm in ARMS:
                H, y = load_states(states_dir, tag, s, arm)
                out[(tag, s, arm)] = cv_auroc(episode_features(H), y, np.arange(len(y)), 1.0)[0]
    return out


def run_provenance(out: str, states_dir: str, argv: list) -> dict:
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(out), device="cpu")
    return {**p, "states_dir": repo_rel(states_dir), "argv": rel_flags(list(argv))}


def print_integrity(states_dir: str, integ: dict) -> None:
    for tag, v in integ["survival_means"].items():
        print(f"  {states_dir} d{tag} survival C=1 mean {v['c1_mean']!r} reference {v['reference_mean']!r} "
              f"(n={v['n']})", flush=True)
    extra = f"; {integ['tolerance_label']}" if "tolerance_label" in integ else ""
    print(f"  integrity {'PASS' if integ['pass'] else 'FAIL'}: {integ['n_checked']} cells, worst |diff| "
          f"{integ['worst_abs_diff']:.3g}, failed {integ['failed']}{extra}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", nargs="+", required=True, help="states dirs (each <run>/states or a G57 dir)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drifts", type=float, nargs="+", default=None,
                    help="drifts to read (default: every drift found in each dir)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the references and run the C = 1 integrity check; write nothing")
    a = ap.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_free_common import pin_one_thread, write_json
    from promote_bv3_gates import repo_rel
    pin_one_thread()
    plans = {d: discover(d, a.drifts) for d in a.states}          # every input checked before any fit
    refs = {d: resolve_reference(d) for d in a.states}
    labels = [refs[d]["label"] for d in a.states]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"two states dirs share an output label: {labels}")
    void = []
    for d in a.states:
        plan, ref = plans[d], refs[d]
        print(f"{d}: {ref['kind']} reference, " + ", ".join(f"d{t} seeds {s}" for t, s in plan.items()), flush=True)
        integ = check_reference(ref, c1_targets(d, plan))
        print_integrity(d, integ)
        if not integ["pass"]:
            print(f"VOID: {d} does not reproduce its reference; nothing is reported for it", file=sys.stderr)
            void.append(d)
            continue
        if a.dry_run:
            continue
        tasks = [{"states_dir": d, "drift": t, "seed": s, "arm": arm, "grid": list(C_GRID)}
                 for t, seeds in plan.items() for s in seeds for arm in ARMS]
        t0 = time.time()
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers) as pool:
                cells = list(pool.imap(score_cell, tasks))
        else:
            cells = [score_cell(t) for t in tasks]
        checked = {(r["drift"], r["seed"], r["arm"]): r["c1"] for r in integ["cells"]}
        for c in cells:
            if c["grid"]["1"]["target"] != checked[(c["drift"], c["seed"], c["arm"])]:
                raise SystemExit(f"{d}: the C = 1 grid value of {c['drift'], c['seed'], c['arm']} moved")
        if ref["kind"] == "regenerated":
            for c in cells:
                c["label"] = ref["tolerance_label"]
        out = os.path.join(a.out_dir, f"{ref['label']}.json")
        doc = {"provenance": run_provenance(out, d, argv), "status": STATUS,
               "states_dir": repo_rel(d), "reference": {k: v for k, v in ref.items()
                                                        if k not in ("targets", "recorded")},
               "integrity": integ,
               "settings": {"C_grid": list(C_GRID), "max_iter": MAX_ITER, "n_splits": N_SPLITS,
                            "folds": "explicit-v1, one group per episode", "n_perm": 8,
                            "nested": "inner grouped 5-fold CV on the outer-training rows; ties to the smaller C",
                            "features": "itasorl.experiment_b.episode_features (mean and final state)"},
               "cells": cells, "summary": summarize(cells, C_GRID), "seconds": round(time.time() - t0, 1)}
        write_json(out, doc)
        print("wrote", out, flush=True)
    if void:
        print(f"VOID dirs: {void}", file=sys.stderr)
        return 1
    if a.dry_run:
        print("dry run: references resolved and C = 1 checked; nothing scored or written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
