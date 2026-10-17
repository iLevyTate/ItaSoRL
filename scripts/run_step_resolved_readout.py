"""Step-resolved decoding of the primary readout (closing item G04, runbook 1.6, R12).

The registered probe reads [mean h, h_T] at the end of a 24-step episode. This reads the same
probe (scaler and logistic regression at C = 1, explicit-v1 folds, one group per episode) on
the recurrent state h_t alone at steps t = 1, 2, 4, 8, 12, 16 and 24, for every arm at every
drift of each stored dump set (d0.45 and d0 for L3, d0.0230 and d0 for L1), with the shuffled
control over pooled_readout's eight label permutations at each step.

h_t is taken as the h_T half of the registered feature on the episode cut at step t
(episode_features(H[:, :t])[:, hidden:]), so at t = 24 the step column is r205's "final" set,
array for array. Two checks run before anything is written for a states dir:
  1. the dump reproduces its reference: the registered target (C = 1 on the concatenation)
     within 1e-9 of the dir's reference, resolved as in scripts/run_probe_penalty_sensitivity.py
     (fold re-score, recorded cell target, or for the C1 dumps the G57 regeneration report,
     whose rows carry the G57 tolerance label);
  2. the step-24 value equals, to the bit, the r205 "final" reading (post_review.linear_probe_auroc
     on post_review.feature_halves(H)["final"]), the h_T component of the recorded target.
A dir that fails either is VOID: nothing is written for it and the script exits 1.

Drift values given with --drifts become file tags through experiment_b2.format_drift; a listed
drift with no dump is a hard error naming the dir and the drift. Without --drifts every drift
in the dir is read. The curve alone does not separate a coherent signal from an integrated
deviation; post hoc and descriptive, it changes no stored value or verdict.

    python scripts/run_step_resolved_readout.py --states fullruns/l3_h8_heldout/states ... \\
        --out-dir fullruns/closing_2026-10/g04
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import sys
import time

import numpy as np

from run_probe_penalty_sensitivity import (ARMS, check_reference, discover, load_states,
                                           print_integrity, resolve_reference)

GENERATED_BY = "scripts/run_step_resolved_readout.py"
STEPS = (1, 2, 4, 8, 12, 16, 24)
STATUS = ("post hoc, descriptive; the curve does not by itself separate coherence from an "
          "integrated deviation; changes no stored value or verdict")


def step_features(H: np.ndarray, t: int) -> np.ndarray:
    """h_t for every episode: the h_T half of episode_features on the episode cut at step t."""
    from itasorl.experiment_b import episode_features
    return episode_features(H[:, :t])[:, H.shape[-1]:]


def score_cell(task: dict) -> dict:
    """The registered target on the dump, the step curve with its shuffled controls, and the
    r205 "final" reading the step-24 column must equal."""
    from threadpoolctl import threadpool_limits

    from itasorl import folds
    from itasorl import post_review as pr
    from itasorl.experiment_b import episode_features
    from run_probe_penalty_sensitivity import cv_auroc
    t0 = time.time()
    H, y = load_states(task["states_dir"], task["drift"], task["seed"], task["arm"])
    if H.shape[1] < max(task["steps"]):
        raise SystemExit(f"{task['states_dir']}: dumps hold {H.shape[1]} steps, fewer than {max(task['steps'])}")
    g = np.arange(len(y))           # independent pools: one group per episode
    perms = pr.registered_permutations(y, task["seed"], pr.N_PERM)
    steps = {}
    with threadpool_limits(1), folds.fold_scheme("explicit"):
        target = cv_auroc(episode_features(H), y, g, 1.0)[0]
        for t in task["steps"]:
            X = step_features(H, t)
            v = cv_auroc(X, y, g, 1.0)[0]
            draws = [cv_auroc(X, yp, g, 1.0)[0] for yp in perms]
            sh = float(np.mean(draws))
            steps[str(t)] = {"target": v, "shuffled": sh, "selectivity": v - sh, "shuffled_draws": draws}
        r205_final = float(pr.linear_probe_auroc(pr.feature_halves(H)["final"], y, g, seed=task["seed"])["auroc"])
    last = str(H.shape[1])
    return {"drift": task["drift"], "seed": task["seed"], "arm": task["arm"],
            "n_auth": int((y == 0).sum()), "n_surr": int((y == 1).sum()), "registered_target": target,
            "steps": steps, "r205_final": r205_final,
            "step_final_bit_equal": bool(last in steps and steps[last]["target"] == r205_final),
            "seconds": round(time.time() - t0, 1)}


def summarize(cells: list, step_list) -> dict:
    from itasorl.post_review import interval_label, paired_summary, seed_summary
    out = {}
    for tag in sorted({c["drift"] for c in cells}, key=float):
        by = {(c["seed"], c["arm"]): c for c in cells if c["drift"] == tag}
        seeds = sorted({s for s, _ in by})
        block = {"seeds": seeds, "arms": {}, "survival_minus_untrained": {}}
        for arm in ARMS:
            block["arms"][arm] = {str(t): {k: seed_summary([by[(s, arm)]["steps"][str(t)][k] for s in seeds])
                                           for k in ("target", "shuffled", "selectivity")} for t in step_list}
        for t in step_list:
            d = paired_summary({str(s): by[(s, "survival")]["steps"][str(t)]["target"] for s in seeds},
                               {str(s): by[(s, "untrained")]["steps"][str(t)]["target"] for s in seeds})
            block["survival_minus_untrained"][str(t)] = {**d, "reading": interval_label(d)}
        out[tag] = block
    return out


def run_provenance(out: str, states_dir: str, argv: list) -> dict:
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(out), device="cpu")
    return {**p, "states_dir": repo_rel(states_dir), "argv": rel_flags(list(argv))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", nargs="+", required=True, help="states dirs (each <run>/states or a G57 dir)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drifts", type=float, nargs="+", default=None,
                    help="drifts to read (default: every drift found in each dir)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the references and check every dump against its reference; write nothing")
    a = ap.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_free_common import pin_one_thread, write_json
    from promote_bv3_gates import repo_rel
    from run_probe_penalty_sensitivity import c1_targets
    pin_one_thread()
    plans = {d: discover(d, a.drifts) for d in a.states}
    refs = {d: resolve_reference(d) for d in a.states}
    labels = [refs[d]["label"] for d in a.states]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"two states dirs share an output label: {labels}")
    void = []
    for d in a.states:
        plan, ref = plans[d], refs[d]
        print(f"{d}: {ref['kind']} reference, " + ", ".join(f"d{t} seeds {s}" for t, s in plan.items()), flush=True)
        if a.dry_run:
            integ = check_reference(ref, c1_targets(d, plan))
            print_integrity(d, integ)
            if not integ["pass"]:
                void.append(d)
            continue
        tasks = [{"states_dir": d, "drift": t, "seed": s, "arm": arm, "steps": list(STEPS)}
                 for t, seeds in plan.items() for s in seeds for arm in ARMS]
        t0 = time.time()
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers) as pool:
                cells = list(pool.imap(score_cell, tasks))
        else:
            cells = [score_cell(t) for t in tasks]
        integ = check_reference(ref, {(c["drift"], c["seed"], c["arm"]): c["registered_target"] for c in cells})
        print_integrity(d, integ)
        not_equal = [(c["drift"], c["seed"], c["arm"]) for c in cells if not c["step_final_bit_equal"]]
        print(f"  step-{STEPS[-1]} bit-check against r205 final: {len(cells) - len(not_equal)}/{len(cells)} equal",
              flush=True)
        if not integ["pass"] or not_equal:
            print(f"VOID: {d} fails its integrity or step-{STEPS[-1]} check {not_equal}; nothing is reported",
                  file=sys.stderr)
            void.append(d)
            continue
        if ref["kind"] == "regenerated":
            for c in cells:
                c["label"] = ref["tolerance_label"]
        out = os.path.join(a.out_dir, f"{ref['label']}.json")
        doc = {"provenance": run_provenance(out, d, argv), "status": STATUS, "states_dir": repo_rel(d),
               "reference": {k: v for k, v in ref.items() if k not in ("targets", "recorded")},
               "integrity": integ,
               "step_check": {"rule": f"step {STEPS[-1]} equals the r205 final reading to the bit",
                              "n_checked": len(cells), "pass": True},
               "settings": {"steps": list(STEPS), "probe": "scaler and logistic regression, C = 1, max_iter 2000",
                            "folds": "explicit-v1, one group per episode", "n_perm": 8,
                            "feature": "h_t = episode_features(H[:, :t])[:, hidden:]"},
               "cells": cells, "summary": summarize(cells, STEPS), "seconds": round(time.time() - t0, 1)}
        write_json(out, doc)
        print("wrote", out, flush=True)
    if void:
        print(f"VOID dirs: {void}", file=sys.stderr)
        return 1
    if a.dry_run:
        print("dry run: references resolved and every dump checked; nothing scored or written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
