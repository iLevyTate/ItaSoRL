"""Prequential MDL and sparse-support probes on stored pooled-state dumps (closing items X05, X06).

Post hoc and descriptive: changes no stored value or verdict. Readout-only on the
states_d<drift>_s<seed>_<arm>.npz dumps of run_expB2.py --dump-states (or
scripts/regenerate_state_dumps.py), with the registered probe input (experiment_b
episode_features: mean and final recurrent state) and the explicit fold partition
(itasorl.folds).

--mode mdl (X05, E07): the online code length L in bits of the 220 world labels in a fixed
coding order, the explicit fold order taken in whole twin groups. Episode i of the authentic
pool and episode i of the surrogate pool form one group (the pooled probe's pair rule,
experiment_b2.probe_world_identity), the groups are ordered by (explicit fold, group id), and
both members of a group sit in the same block, so no prefix boundary splits a pair. With the
prefixes 12, 22, 44, 88 and 176 episodes (6, 11, 22, 44 and 88 pairs) the first block is coded
under the uniform model (one bit per episode) and every later block by the registered probe
(StandardScaler then LogisticRegression(max_iter=2000)) fit on everything before it.
Compression is n / L. The AUROC-versus-prefix curve fits the same probe on the first P
training episodes of each explicit fold (in the coding order) and scores the held-out fold; at
the full training fold it is the registered estimate.

--mode sparse (X06, E14): the explicit-fold AUROC of the L1-penalized probe (saga,
max_iter 5000, scaler fit in fold) at eight penalties C from 0.001 to 10, with the mean
nonzero coefficient count per fold; k65, the smallest mean support whose AUROC reaches 0.65;
and the AUROC at a support of at most 8. Convergence warnings are counted, and a penalty with a
non-converged fold is not read. k65 is a property of the standardized L1-penalized probe family
under collinear units, not an anatomical count.

Integrity gate (both modes): the full-n explicit-fold AUROC of the registered probe on the dump
(experiment_a.grouped_auroc, one group per episode, which is how the pooled target was scored)
reproduces the recorded target within 1e-9, else the cell is VOID and nothing else is reported
for it. Reference targets, per states dir, in this order: --targets DIR=PATH; the explicit
re-score artifacts/fold_rescore/<run>.json; the run's own cells/ beside the states dir; a
regenerate_state_dumps.py report c1_regen.json beside it (the R10 report; tolerance 0.002,
read against its regenerated targets). A dir with no reference stops the script.

Drift tags come from itasorl.experiment_b2.format_drift (0.023 is "0.0230"). Without --drifts
every drift a dir holds is read; with --drifts each dir is read at the listed drifts it holds,
and a dir that holds none of them, or holds a listed drift under another spelling, is a hard
error naming the dir and the drift.

    python scripts/run_mdl_probe.py --mode mdl --states fullruns/l3_h8_heldout/states \\
        fullruns/corrected_l1_heldout/states --prefixes 12 22 44 88 176 --workers 2 \\
        --out-dir fullruns/closing_2026-10/x05
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings

import numpy as np

GENERATED_BY = "scripts/run_mdl_probe.py"
FNAME = re.compile(r"^states_d(?P<drift>\d+\.\d+)_s(?P<seed>\d+)_(?P<arm>[a-z]+)\.npz$")
ARMS = ("survival", "predictor", "untrained")
PREFIXES = (12, 22, 44, 88, 176)
PENALTIES = tuple(float(c) for c in np.logspace(-3, 1, 8))
BAR = 0.65
SUPPORT_FIXED = 8
CONCENTRATED_MAX = 16
TOL_EXACT = 1e-9
TOL_REGEN = 0.002            # the R10 cross-stack tolerance
STATUS = "post hoc, descriptive; changes no stored value or verdict"


# ---------------------------------------------------------------------------
# coding order, prequential code, curve
# ---------------------------------------------------------------------------

def twin_groups(n_auth: int, n_surr: int) -> np.ndarray:
    """Episode i of each pool shares group i (the pooled probe's pair rule)."""
    return np.r_[np.arange(n_auth), np.arange(n_surr)]


def coding_order(groups: np.ndarray, n_splits: int = 5) -> tuple[np.ndarray, list[int]]:
    """Sample indices in the explicit fold order, whole groups at a time ((fold, group id)
    order, members in index order), and the cumulative sample count at each group boundary."""
    from itasorl import folds
    groups = np.asarray(groups)
    fos = folds.fold_of_sample(groups, n_splits)
    ug = np.unique(groups)
    gfold = {g: int(fos[np.flatnonzero(groups == g)[0]]) for g in ug}
    order, bounds = [], [0]
    for g in sorted(ug, key=lambda g: (gfold[g], g)):
        order.extend(np.flatnonzero(groups == g).tolist())
        bounds.append(len(order))
    return np.asarray(order, int), bounds


def snap(prefixes, bounds: list[int]) -> list[int]:
    """Each prefix moved down to the nearest group boundary (a no-op when the pools are equal)."""
    out = []
    for p in prefixes:
        b = max(x for x in bounds if x <= p)
        if b > 0 and b not in out:
            out.append(b)
    return out


def _probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def prequential(X: np.ndarray, y: np.ndarray, groups: np.ndarray, prefixes=PREFIXES) -> dict:
    """Online code length of y in the coding order; first block uniform, each later block coded
    by the probe fit on every earlier episode (rows fit in index order)."""
    order, bounds = coding_order(groups)
    cuts = [0, *snap(prefixes, bounds)]
    if cuts[-1] < len(order):
        cuts.append(len(order))
    block_bits, blocks = [], []
    for k in range(len(cuts) - 1):
        lo, hi = cuts[k], cuts[k + 1]
        idx = order[lo:hi]
        train = np.sort(order[:lo])
        if lo == 0 or len(np.unique(y[train])) < 2:
            bits = float(len(idx))
            coder = "uniform"
        else:
            p = _probe().fit(X[train], y[train]).predict_proba(X[idx])[:, 1]
            p = np.clip(p, 1e-15, 1 - 1e-15)
            bits = float(-np.sum(np.where(y[idx] == 1, np.log2(p), np.log2(1 - p))))
            coder = "probe"
        block_bits.append(bits)
        blocks.append({"start": int(lo), "end": int(hi), "bits": bits, "coder": coder})
    L = float(sum(block_bits))
    return {"code_bits": L, "uniform_bits": float(len(y)), "compression": float(len(y) / L) if L > 0 else float("nan"),
            "prefixes_used": cuts[1:-1], "blocks": blocks}


def prefix_curve(X: np.ndarray, y: np.ndarray, groups: np.ndarray, prefixes=PREFIXES) -> dict:
    """{P: mean over explicit folds of the AUROC of the probe fit on the first P training
    episodes in the coding order}; P larger than a fold's training set is not scored."""
    from sklearn.metrics import roc_auc_score

    from itasorl import folds
    order, bounds = coding_order(groups)
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    out = {}
    for P in snap(prefixes, bounds):
        aucs = []
        for tr, te in folds.split(groups, 5):
            if len(np.unique(y[te])) < 2 or P > len(tr):
                continue
            sub = np.sort(tr[np.argsort(rank[tr], kind="stable")][:P])
            if len(np.unique(y[sub])) < 2:
                continue
            p = _probe().fit(X[sub], y[sub]).predict_proba(X[te])[:, 1]
            aucs.append(float(roc_auc_score(y[te], p)))
        out[str(P)] = float(np.mean(aucs)) if aucs else float("nan")
    return out


# ---------------------------------------------------------------------------
# sparse support
# ---------------------------------------------------------------------------

def sparse_path(X: np.ndarray, y: np.ndarray, groups: np.ndarray, penalties=None) -> list[dict]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from itasorl import folds
    rows = []
    for C in (PENALTIES if penalties is None else penalties):
        aucs, supp, n_warn, converged = [], [], 0, True
        for tr, te in folds.split(groups, 5):
            if len(np.unique(y[te])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(
                penalty="l1", solver="saga", C=C, max_iter=5000, random_state=0))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X[tr], y[tr])
            w = sum(issubclass(c.category, ConvergenceWarning) for c in caught)
            n_warn += w
            if w or int(np.max(clf[-1].n_iter_)) >= 5000:
                converged = False
            aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
            supp.append(int(np.count_nonzero(clf[-1].coef_)))
        rows.append({"C": C, "auroc": float(np.mean(aucs)) if aucs and converged else None,
                     "auroc_unread": float(np.mean(aucs)) if aucs else float("nan"),
                     "mean_support": float(np.mean(supp)) if supp else float("nan"),
                     "convergence_warnings": int(n_warn), "converged": bool(converged)})
    return rows


def k65(path: list[dict], bar: float = BAR):
    """Smallest mean support among read penalties whose AUROC reaches the bar; None if none."""
    hit = [r["mean_support"] for r in path if r["auroc"] is not None and r["auroc"] >= bar]
    return float(min(hit)) if hit else None


def at_support(path: list[dict], k: int = SUPPORT_FIXED) -> dict | None:
    """The read penalty with the largest mean support not above k."""
    ok = [r for r in path if r["auroc"] is not None and r["mean_support"] <= k]
    if not ok:
        return None
    r = max(ok, key=lambda r: (r["mean_support"], r["C"]))
    return {"C": r["C"], "mean_support": r["mean_support"], "auroc": r["auroc"]}


# ---------------------------------------------------------------------------
# dumps and references
# ---------------------------------------------------------------------------

def dir_name(states_dir: str) -> str:
    d = os.path.normpath(os.path.abspath(states_dir))
    base, parent = os.path.basename(d), os.path.basename(os.path.dirname(d))
    return parent if base == "states" else f"{parent}_{base}"


def list_dumps(states_dir: str, arms) -> dict:
    """{drift tag: {(seed, arm): path}} for the dumps whose arm is in arms."""
    out: dict = {}
    for p in sorted(glob.glob(os.path.join(states_dir, "states_d*_s*_*.npz"))):
        m = FNAME.match(os.path.basename(p))
        if m and m.group("arm") in arms:
            out.setdefault(m.group("drift"), {})[(int(m.group("seed")), m.group("arm"))] = p
    return out


def select_drifts(states_dir: str, found: dict, drifts) -> tuple[list[str], list[str]]:
    """(tags to read, listed tags this dir does not hold). Raises SystemExit when the dir holds
    none of the listed drifts or holds one under another spelling."""
    from itasorl.experiment_b2 import format_drift
    if not found:
        raise SystemExit(f"{states_dir}: no states_d*_s*_<arm>.npz dumps")
    if drifts is None:
        return sorted(found, key=float), []
    want = [format_drift(d) for d in drifts]
    for d, tag in zip(drifts, want):
        other = [t for t in found if t != tag and abs(float(t) - float(d)) < 1e-12]
        if other:
            raise SystemExit(f"{states_dir}: drift {d} is spelled {other[0]!r} there, not {tag!r}")
    have = [t for t in want if t in found]
    if not have:
        raise SystemExit(f"{states_dir}: no dump for any listed drift {want} (tags held: {sorted(found)})")
    return have, [t for t in want if t not in found]


def _from_fold_rescore(doc: dict) -> dict:
    return {(c["drift"], int(c["seed"]), c["agent"]): float(c["explicit"]["target"]) for c in doc["cells"]}


def _from_report(doc: dict) -> dict:
    from itasorl.experiment_b2 import format_drift
    return {(format_drift(c["drift"]), int(c["seed"]), c["arm"]): float(c["target"]) for c in doc["cells"]}


def _from_cells(cells_dir: str) -> dict:
    from itasorl.experiment_b2 import format_drift
    out = {}
    for p in glob.glob(os.path.join(cells_dir, "cell_d*_s*.json")):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        for arm, blk in c["agents"].items():
            if "pool" in blk and "target" in blk["pool"]:
                out[(format_drift(c["drift"]), int(c["seed"]), arm)] = float(blk["pool"]["target"])
    return out


def references(states_dir: str, override: str | None = None, root: str | None = None) -> dict:
    """{"source", "tolerance", "targets": {(tag, seed, arm): target}} for a states dir."""
    root = root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent = os.path.dirname(os.path.normpath(os.path.abspath(states_dir)))

    def load_json(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    if override:
        if os.path.isdir(override):
            return {"source": override, "tolerance": TOL_EXACT, "targets": _from_cells(override)}
        doc = load_json(override)
        if doc.get("cells") and "explicit" in doc["cells"][0]:
            return {"source": override, "tolerance": TOL_EXACT, "targets": _from_fold_rescore(doc)}
        return {"source": override, "tolerance": TOL_REGEN, "targets": _from_report(doc)}
    rescore = os.path.join(root, "artifacts", "fold_rescore", os.path.basename(parent) + ".json")
    if os.path.exists(rescore):
        return {"source": rescore, "tolerance": TOL_EXACT, "targets": _from_fold_rescore(load_json(rescore))}
    cells = os.path.join(parent, "cells")
    if os.path.isdir(cells):
        return {"source": cells, "tolerance": TOL_EXACT, "targets": _from_cells(cells)}
    report = os.path.join(parent, "c1_regen.json")
    if os.path.exists(report):
        return {"source": report, "tolerance": TOL_REGEN, "targets": _from_report(load_json(report))}
    raise SystemExit(f"{states_dir}: no reference target (no --targets, no artifacts/fold_rescore/"
                     f"{os.path.basename(parent)}.json, no cells/ or c1_regen.json beside it)")


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def features(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    from itasorl.experiment_b import episode_features
    with np.load(path) as z:
        Ha, Hs = np.asarray(z["Ha"]), np.asarray(z["Hs"])
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    return X, y, twin_groups(len(Ha), len(Hs)), len(Ha), len(Hs)


def run_cell(t: dict) -> dict:
    from itasorl.experiment_a import grouped_auroc
    t0 = time.time()
    X, y, g, na, ns = features(t["path"])
    full = float(grouped_auroc(X, y, np.arange(len(y))))
    ref = t["reference"]
    out = {"drift": t["drift"], "seed": t["seed"], "arm": t["arm"], "n_auth": na, "n_surr": ns,
           "full_n_auroc": full, "reference_target": ref, "tolerance": t["tolerance"]}
    ok = ref is not None and np.isfinite(full) and abs(full - ref) <= t["tolerance"]
    out["gate"] = "PASS" if ok else "VOID"
    if not ok:
        out["void_reason"] = "no reference target" if ref is None else "full-n AUROC does not reproduce the reference"
        out["seconds"] = round(time.time() - t0, 2)
        return out
    if t["mode"] == "mdl":
        out.update(prequential(X, y, g, t["prefixes"]))
        out["curve"] = prefix_curve(X, y, g, t["prefixes"])
    else:
        path = sparse_path(X, y, g)
        out.update({"path": path, "k65": k65(path), "at_support_8": at_support(path),
                    "reached_065": any(r["auroc"] is not None and r["auroc"] >= BAR for r in path)})
    out["seconds"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarize(rows: list, mode: str) -> dict:
    from itasorl.novel_readouts import paired_t90
    out = {}
    for tag in sorted({r["drift"] for r in rows}, key=float):
        live = [r for r in rows if r["drift"] == tag and r["gate"] == "PASS"]
        by = {a: {r["seed"]: r for r in live if r["arm"] == a} for a in ARMS}
        blk: dict = {"n_void": sum(1 for r in rows if r["drift"] == tag and r["gate"] != "PASS")}
        both = sorted(set(by["survival"]) & set(by["untrained"]))
        if mode == "mdl":
            for a in ARMS:
                rr = [by[a][s] for s in sorted(by[a])]
                if not rr:
                    continue
                prefixes = sorted({p for r in rr for p in r["curve"]}, key=int)
                blk[a] = {"compression": paired_t90([r["compression"] for r in rr]),
                          "code_bits": paired_t90([r["code_bits"] for r in rr]),
                          "full_n_auroc": paired_t90([r["full_n_auroc"] for r in rr]),
                          "curve": {p: paired_t90([r["curve"].get(p, np.nan) for r in rr]) for p in prefixes}}
                blk[a]["reaches_065_at_88_or_fewer"] = any(
                    int(p) <= 88 and v["mean"] >= BAR for p, v in blk[a]["curve"].items())
                blk[a]["under_065_at_every_prefix"] = all(v["mean"] < BAR for v in blk[a]["curve"].values())
            if both:
                blk["survival_minus_untrained"] = {
                    "compression": paired_t90([by["survival"][s]["compression"] for s in both],
                                              [by["untrained"][s]["compression"] for s in both]),
                    "code_bits_untrained_minus_survival": paired_t90(
                        [by["untrained"][s]["code_bits"] for s in both],
                        [by["survival"][s]["code_bits"] for s in both]),
                    "seeds": both}
        else:
            for a in ARMS:
                rr = [by[a][s] for s in sorted(by[a])]
                if not rr:
                    continue
                ks = [r["k65"] for r in rr]
                blk[a] = {"k65_per_seed": ks,
                          "n_seeds_k65_at_most_16": sum(k is not None and k <= CONCENTRATED_MAX for k in ks),
                          "n_seeds": len(rr),
                          "reached_065_in_any_seed": any(r["reached_065"] for r in rr),
                          "auroc_at_support_8": paired_t90([(r["at_support_8"] or {}).get("auroc", np.nan) for r in rr]),
                          "convergence_warnings": sum(p["convergence_warnings"] for r in rr for p in r["path"])}
            if "survival" in blk:
                s = blk["survival"]
                s["label"] = "concentrated" if s["n_seeds_k65_at_most_16"] >= 8 else "distributed"
                s["label_rule"] = "concentrated iff k65 <= 16 of 192 in at least 8 seeds"
            if both:
                blk["survival_minus_untrained_at_support_8"] = paired_t90(
                    [(by["survival"][s]["at_support_8"] or {}).get("auroc", np.nan) for s in both],
                    [(by["untrained"][s]["at_support_8"] or {}).get("auroc", np.nan) for s in both])
        out[tag] = blk
    return out


def run_provenance(out: str, states_dirs) -> dict:
    from _readout_common import _rel_argv
    from agent_free_common import provenance
    from promote_bv3_gates import repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(out))
    return {**p, "states_dirs": [repo_rel(d) for d in states_dirs], "argv": _rel_argv(list(sys.argv[1:]))}


def main(argv: list[str] | None = None) -> int:
    from itasorl import folds
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("mdl", "sparse"), required=True)
    ap.add_argument("--states", nargs="+", required=True, metavar="DIR")
    ap.add_argument("--drifts", type=float, nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES), help="mdl only")
    ap.add_argument("--targets", action="append", default=[], metavar="DIR=PATH",
                    help="reference targets for a states dir: a cells dir, a fold_rescore JSON or an R10 report")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="resolve dumps and references; compute nothing")
    a = ap.parse_args(argv)
    if folds.current_scheme() != "explicit":
        raise SystemExit("the reference targets use the explicit fold partition; unset ITASORL_FOLDS or set it to explicit")
    if any(p <= 0 or p % 2 for p in a.prefixes):
        raise SystemExit("--prefixes must be positive and even (whole twin pairs)")
    over = {}
    for spec in a.targets:
        d, sep, p = spec.partition("=")
        if not sep:
            ap.error(f"--targets expects DIR=PATH, got {spec!r}")
        over[os.path.normpath(os.path.abspath(d))] = p
    plan = []
    for d in a.states:
        found = list_dumps(d, a.arms)
        tags, absent = select_drifts(d, found, a.drifts)
        ref = references(d, over.get(os.path.normpath(os.path.abspath(d))))
        tasks = [{"path": p, "drift": tag, "seed": s, "arm": arm, "mode": a.mode, "prefixes": list(a.prefixes),
                  "reference": ref["targets"].get((tag, s, arm)), "tolerance": ref["tolerance"]}
                 for tag in tags for (s, arm), p in sorted(found[tag].items())]
        plan.append((d, tags, absent, ref, tasks))
        print(f"{d}: drifts {tags} (absent {absent}); {len(tasks)} cells; reference {ref['source']} "
              f"(tolerance {ref['tolerance']:g}); {sum(t['reference'] is None for t in tasks)} cells without one",
              flush=True)
    if a.dry_run:
        return 0
    from promote_bv3_gates import repo_rel

    from agent_free_common import write_json
    for d, tags, absent, ref, tasks in plan:
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers) as pool:
                rows = list(pool.imap(run_cell, tasks))
        else:
            rows = [run_cell(t) for t in tasks]
        rows.sort(key=lambda r: (float(r["drift"]), r["arm"], r["seed"]))
        out = os.path.join(a.out_dir, f"{a.mode}_{dir_name(d)}.json")
        body = {"provenance": run_provenance(out, [d]), "status": STATUS, "mode": a.mode,
                "states_dir": repo_rel(d), "drifts": tags, "absent_drifts": absent,
                "reference": {"source": repo_rel(ref["source"]), "tolerance": ref["tolerance"]},
                "settings": {"prefixes": list(a.prefixes), "penalties": list(PENALTIES), "bar": BAR,
                             "support_fixed": SUPPORT_FIXED, "fold_scheme": folds.current_scheme(),
                             "coding_order": "explicit fold order in whole twin groups (episode i of each pool)"},
                "summary": summarize(rows, a.mode), "rows": rows}
        write_json(out, body)
        n_void = sum(r["gate"] != "PASS" for r in rows)
        print(f"wrote {out}: {len(rows)} cells, {n_void} VOID", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
