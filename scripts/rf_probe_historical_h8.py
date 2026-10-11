"""Random-forest and linear probes on the historical hidden-8 state dumps (HISTORICAL).

FINDINGS 10.3 (and PREREGISTRATION_L3 section 12, 2026-07-12) reports a random-forest probe
on the untrained agent at 0.517 (shuffled baseline 0.506; L0 untrained 0.483) as a post-hoc
check on the dumped states, with no committed record. The audit of 2026-07-12 says the
behavior figures of that time came from an uncommitted ad hoc analysis, and the committed
random-forest path (`itasorl.behavior_audit.behavior_only_auroc(nonlinear=True)`) fits
behavior features, not recurrent states, so the 0.517 cannot have come from it. The closest
committed code is `scripts/run_expB_nonlinear.py` (Experiment B): RandomForestClassifier
with 200 trees and random_state 0 on `episode_features` of the recurrent states, episode-level
grouped folds, one label permutation from default_rng(seed) as the shuffled control.

This scores the 60 dumps of `fullruns/l3_n10/states` (the historical hidden-8 n = 10 run,
pre-correction trainer: drift 0 and 0.45, seeds 0 to 9, arms survival, predictor, untrained)
with that random forest and with the registered linear probe (`probe_auroc` on the same
`episode_features`, the probe `scripts/rescore_fold_split.py` re-scores), plus their shuffled
controls, under both fold schemes. The linear shuffled control is reported two ways: the
first permutation of default_rng(seed), which is the historical single-draw control, and the
mean of eight permutations from that stream, which is the control `pooled_readout` has used
since 2026-07-18. The random forest's shuffled control uses the first permutation.

Everything here is labeled historical: the agents were trained by the pre-correction
(pre-transition value) trainer.

Usage (each drift fits in one short process; the second call merges into the first):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/rf_probe_historical_h8.py --drift 0.00
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/rf_probe_historical_h8.py --drift 0.45
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from agent_free_common import pin_one_thread, provenance, write_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_FULLRUNS = os.environ.get("ITASORL_FULLRUNS",
                                  os.path.join(os.path.dirname(os.path.dirname(ROOT)), "fullruns")
                                  if os.path.basename(os.path.dirname(ROOT)) == ".worktrees"
                                  else os.path.join(ROOT, "fullruns"))
STATES = "fullruns/l3_n10/states"
RESULTS = "fullruns/l3_n10/expB2_results.json"
FNAME = re.compile(r"states_d(?P<drift>[0-9.]+)_s(?P<seed>\d+)_(?P<agent>[a-z]+)\.npz$")
SCHEMES = ("legacy", "explicit")
ARMS = ("survival", "predictor", "untrained")
N_TREES, RF_STATE, N_PERM = 200, 0, 8
BAR = 0.65
RECORDED_RF = {"untrained d=0.45": 0.517, "untrained d=0.45 shuffled": 0.506,
               "untrained d=0.00": 0.483}


def rf_auroc(X: np.ndarray, y: np.ndarray, seed: int = RF_STATE) -> float:
    """scripts/run_expB_nonlinear.py's rf_auroc on one thread (n_jobs does not change a
    seeded forest): episode-level grouped folds from itasorl.folds, mean of fold AUROCs."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    from itasorl import folds
    g = np.arange(len(y))
    aucs = []
    for tr, te in folds.split(g, 5):
        if len(np.unique(y[te])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=1)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def score_cell(path: str, seed: int) -> dict:
    from itasorl import folds
    from itasorl.experiment_b import episode_features, probe_auroc
    with np.load(path) as z:
        Ha, Hs = np.asarray(z["Ha"]), np.asarray(z["Hs"])
    H = np.concatenate([Ha, Hs])
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    X = episode_features(H)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(N_PERM)]
    out = {"n_auth": int(len(Ha)), "n_surr": int(len(Hs)), "state_dim": int(H.shape[2]),
           "steps": int(H.shape[1])}
    for scheme in SCHEMES:
        with folds.fold_scheme(scheme):
            lin_sh = [float(probe_auroc(X, p)) for p in perms]
            out[scheme] = {"linear": float(probe_auroc(X, y)),
                           "linear_shuffled_first": lin_sh[0],
                           "linear_shuffled_mean8": float(np.mean(lin_sh)),
                           "rf": rf_auroc(X, y),
                           "rf_shuffled_first": rf_auroc(X, perms[0])}
    return out


METRICS = ("linear", "linear_shuffled_first", "linear_shuffled_mean8", "rf", "rf_shuffled_first")


def aggregate(cells: list[dict]) -> dict:
    from itasorl.stats import t_ci90
    out = {}
    for drift in sorted({c["drift"] for c in cells}):
        for arm in ARMS:
            rows = sorted((c for c in cells if c["drift"] == drift and c["agent"] == arm),
                          key=lambda c: c["seed"])
            if not rows:
                continue
            block = {"seeds": [c["seed"] for c in rows]}
            for scheme in SCHEMES:
                for m in METRICS:
                    v = [c[scheme][m] for c in rows]
                    entry = {"per_seed": v, "mean": float(np.mean(v)), "n": len(v),
                             "n_ge_065": int(sum(x >= BAR for x in v))}
                    if len(v) > 1:
                        entry["t90"] = [float(x) for x in t_ci90(v)]
                    block.setdefault(scheme, {})[m] = entry
            out[f"d={drift} {arm}"] = block
    return out


def recorded_run() -> dict | None:
    """The run's own pooled linear target and single-draw shuffled control (legacy split),
    for the reproduction check."""
    path = os.path.join(PRIMARY_FULLRUNS, RESULTS[len("fullruns/"):])
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    return {f"{float(d):.2f}": {arm: {"pool_target": run[d][arm]["pool_target"],
                                      "pool_shuffled": run[d][arm]["pool_shuffled"]}
                                for arm in ARMS if arm in run[d]} for d in run}


def reproduction(cells: list[dict], rec: dict | None) -> dict:
    """Max |difference| between this re-score (legacy) and the run's recorded per-seed values,
    and against the committed fold re-score (both schemes)."""
    out = {}
    if rec is not None:
        dt, ds = [], []
        for c in cells:
            r = rec.get(c["drift"], {}).get(c["agent"])
            if r is None:
                continue
            dt.append(abs(c["legacy"]["linear"] - r["pool_target"][c["seed"]]))
            ds.append(abs(c["legacy"]["linear_shuffled_first"] - r["pool_shuffled"][c["seed"]]))
        out["vs_run_record"] = {"source": RESULTS, "n": len(dt),
                                "max_abs_diff_linear_target": max(dt) if dt else None,
                                "max_abs_diff_linear_shuffled_first": max(ds) if ds else None}
    fr = os.path.join(ROOT, "artifacts", "fold_rescore", "l3_n10.json")
    with open(fr, encoding="utf-8") as fh:
        frd = {(c["drift"], c["seed"], c["agent"]): c for c in json.load(fh)["cells"]}
    diffs = {s: [] for s in SCHEMES}
    for c in cells:
        r = frd.get((c["drift"], c["seed"], c["agent"]))
        if r is None:
            continue
        for s in SCHEMES:
            diffs[s].append(abs(c[s]["linear"] - r[s]["target"]))
    out["vs_fold_rescore"] = {"source": "artifacts/fold_rescore/l3_n10.json",
                              **{f"max_abs_diff_{s}": (max(v) if v else None)
                                 for s, v in diffs.items()}, "n": len(diffs["legacy"])}
    return out


def attempt_cell(path: str, seed: int) -> dict:
    """Reproduction attempt for the recorded 0.517 / 0.506 / 0.483: the same forest with
    random_state set to the cell seed (the convention of itasorl.behavior_audit, whose RF takes
    a seed argument), legacy split, target and first-permutation shuffled control."""
    from itasorl import folds
    from itasorl.experiment_b import episode_features
    with np.load(path) as z:
        Ha, Hs = np.asarray(z["Ha"]), np.asarray(z["Hs"])
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    X = episode_features(np.concatenate([Ha, Hs]))
    perm = np.random.default_rng(seed).permutation(y)
    with folds.fold_scheme("legacy"):
        return {"rf_rs_seed": rf_auroc(X, y, seed=seed),
                "rf_rs_seed_shuffled_first": rf_auroc(X, perm, seed=seed)}


def attempts_summary(cells: list[dict]) -> dict:
    out = {}
    for drift in ("0.00", "0.45"):
        rows = [c for c in cells if c["drift"] == drift and c["agent"] == "untrained"
                and "attempt" in c]
        if not rows:
            continue
        out[f"d={drift} untrained"] = {
            "n": len(rows),
            "rf_random_state_0": float(np.mean([c["legacy"]["rf"] for c in rows])),
            "rf_random_state_0_shuffled": float(np.mean([c["legacy"]["rf_shuffled_first"]
                                                         for c in rows])),
            "rf_random_state_seed": float(np.mean([c["attempt"]["rf_rs_seed"] for c in rows])),
            "rf_random_state_seed_shuffled": float(np.mean(
                [c["attempt"]["rf_rs_seed_shuffled_first"] for c in rows]))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/agent_free/rf_probe_l3_n10_historical.json")
    ap.add_argument("--drift", default=None, help="score only this drift (0.00 or 0.45); "
                    "cells already in --out for the other drift are kept")
    ap.add_argument("--attempts", action="store_true",
                    help="add the random_state = seed reproduction attempt to the untrained "
                         "cells already in --out (legacy split)")
    a = ap.parse_args()
    pin_one_thread()
    out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    states = os.path.join(PRIMARY_FULLRUNS, STATES[len("fullruns/"):])
    files = sorted(glob.glob(os.path.join(states, "states_*.npz")))
    if not files:
        raise SystemExit(f"no dumps under {STATES}")
    if a.attempts:
        with open(out_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        t0 = time.time()
        for c in doc["cells"]:
            if c["agent"] == "untrained":
                c["attempt"] = attempt_cell(os.path.join(states, os.path.basename(c["dump"])),
                                            c["seed"])
                print(f"d={c['drift']} s{c['seed']} untrained rf(rs=seed) "
                      f"{c['attempt']['rf_rs_seed']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
        doc["reproduction_attempts"] = {
            "recorded": RECORDED_RF,
            "note": "RF on states, legacy split (the split of July 2026); random_state 0 is the "
                    "scripts/run_expB_nonlinear.py convention, random_state = cell seed the "
                    "itasorl.behavior_audit convention",
            "provenance": provenance("scripts/rf_probe_historical_h8.py --attempts", "git_commit_at_promotion"),
            "summary": attempts_summary(doc["cells"])}
        write_json(out_path, doc)
        print(json.dumps(doc["reproduction_attempts"]["summary"], indent=1))
        return 0
    kept = []
    if a.drift is not None and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            kept = [c for c in json.load(fh)["cells"] if c["drift"] != a.drift]
    t0 = time.time()
    cells = []
    for f in files:
        m = FNAME.search(os.path.basename(f))
        if not m or (a.drift is not None and m["drift"] != a.drift):
            continue
        seed = int(m["seed"])
        cell = {"drift": m["drift"], "seed": seed, "agent": m["agent"],
                "dump": f"{STATES}/{os.path.basename(f)}", **score_cell(f, seed)}
        cells.append(cell)
        print(f"d={cell['drift']} s{seed} {cell['agent']:9s} linear {cell['legacy']['linear']:.3f}"
              f" rf {cell['legacy']['rf']:.3f} rf-shuf {cell['legacy']['rf_shuffled_first']:.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
    cells = sorted(kept + cells, key=lambda c: (c["drift"], c["agent"], c["seed"]))
    doc = {
        **provenance("scripts/rf_probe_historical_h8.py", "git_commit_at_promotion"),
        "label": "HISTORICAL: hidden-8 L3 run fullruns/l3_n10, pre-correction (pre-transition "
                 "value) trainer; post-hoc probes on its dumped recurrent states",
        "states_dir": STATES, "n_cells": len(cells), "complete": len(cells) == 60,
        "features": "itasorl.experiment_b.episode_features (mean and final state, 192 dims)",
        "folds": "itasorl.folds, one group per episode (220 episodes), 5 folds; legacy = "
                 "installed scikit-learn GroupKFold, explicit = itasorl.folds rule",
        "linear_probe": "itasorl.experiment_b.probe_auroc (scaled logistic regression), the "
                        "registered pooled probe",
        "random_forest": {"n_estimators": N_TREES, "random_state": RF_STATE, "n_jobs": 1,
                          "as_in": "scripts/run_expB_nonlinear.py rf_auroc"},
        "shuffled": {"rng": "numpy default_rng(cell seed)", "linear_first": "first permutation "
                     "(the historical single-draw control)", "linear_mean8": "mean over 8 "
                     "permutations (pooled_readout since 2026-07-18)",
                     "rf": "first permutation"},
        "recorded_rf_values_findings_10_3": RECORDED_RF,
        "rf_path_note": "itasorl.behavior_audit.behavior_only_auroc(nonlinear=True) fits a "
                        "random forest on BEHAVIOR features, not on states; it is not the source "
                        "of the 0.517. No committed code produced the 0.517 (FINDINGS 10.3, "
                        "PREREGISTRATION_L3 sec. 12 entry of 2026-07-12: post-hoc, on the dumped "
                        "states).",
        "reproduction": reproduction(cells, recorded_run()),
        "aggregate": aggregate(cells),
        "cells": cells,
        "seconds_this_call": round(time.time() - t0, 1),
    }
    write_json(out_path, doc)
    print(f"wrote {os.path.relpath(out_path, ROOT)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
