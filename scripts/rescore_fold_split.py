"""Re-score saved pooled-state dumps under both cross-validation fold schemes.

Readout-only, no training, no GPU. For every states_d<drift>_s<seed>_<agent>.npz
written by `run_expB2.py --dump-states`, this scores the headline pooled target
and (when the dump carries per-timestep traces) the per-timestep behavior control
`resid_trace`, once under the legacy scheme (the installed scikit-learn
GroupKFold, which on the stack that produced the published GPU numbers reproduces
them) and once under the explicit scheme of itasorl.folds (the same partition on
every stack; what the 2026-09 cloud runs used). Spec:
docs/specs/2026-09-28-explicit-cv-folds.md.

The output records the stack (python, numpy, scikit-learn, platform), the fold
partition each scheme produced for a 110 + 110 pool, per-cell values under both
schemes, and per (drift, agent) the two across-seed means, their t-based 90% CIs,
the seeds at or above the bar, and the mean shift.

Usage:
    python scripts/rescore_fold_split.py fullruns/l3_h8_traces/states \
        --json fullruns/fold_rescore/l3_h8_traces.json [--label "L3 hidden 8 headline"]
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
from collections import defaultdict

import numpy as np

from itasorl import folds
from itasorl.behavior_audit import trace_residual_probe_auroc
from itasorl.experiment_b import episode_features, probe_auroc
from itasorl.stats import t_ci90

FNAME = re.compile(r"states_d(?P<drift>[-\d.]+)_s(?P<seed>\d+)_(?P<agent>\w+)\.npz$")
BAR = 0.65
SCHEMES = ("legacy", "explicit")


def score_cell(npz: dict) -> dict:
    """{scheme: {"target": .., "resid_trace": ..}} for one dump; {} if a pool is too small."""
    Ha, Hs = np.asarray(npz["Ha"]), np.asarray(npz["Hs"])
    if len(Ha) < 5 or len(Hs) < 5:
        return {}
    H = np.concatenate([Ha, Hs])
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    X = episode_features(H)
    Bt = (np.concatenate([np.asarray(npz["bta"]), np.asarray(npz["bts"])])
          if "bta" in npz and "bts" in npz else None)
    out = {}
    for scheme in SCHEMES:
        with folds.fold_scheme(scheme):
            row = {"target": float(probe_auroc(X, y))}
            if Bt is not None:
                row["resid_trace"] = float(trace_residual_probe_auroc(H, Bt, y))
        out[scheme] = row
    return out


def _summ(vals: list[float]) -> dict:
    v = [float(x) for x in vals if np.isfinite(x)]
    lo, hi = t_ci90(v) if len(v) > 1 else (float("nan"), float("nan"))
    return {"mean": float(np.mean(v)) if v else float("nan"), "t90": [float(lo), float(hi)],
            "n_seeds": len(v), "n_ge_065": int(sum(x >= BAR for x in v))}


def rescore(states_dir: str, label: str = "") -> dict:
    files = sorted(glob.glob(os.path.join(states_dir, "states_*.npz")))
    cells, by = [], defaultdict(list)
    for f in files:
        m = FNAME.search(os.path.basename(f))
        if not m:
            continue
        with np.load(f) as npz:
            if "Ha" not in npz:        # held-out sibling dumps carry no pooled states
                continue
            res = score_cell(dict(npz))
        if not res:
            continue
        cell = {"drift": m["drift"], "seed": int(m["seed"]), "agent": m["agent"], **res}
        cells.append(cell)
        by[(m["drift"], m["agent"])].append(cell)
    y220 = np.r_[np.zeros(110), np.ones(110)].astype(int)
    aggregate = {}
    for (drift, agent), rows in sorted(by.items()):
        rows = sorted(rows, key=lambda r: r["seed"])
        block = {"seeds": [r["seed"] for r in rows]}
        for metric in ("target", "resid_trace"):
            if not all(metric in r["legacy"] for r in rows):
                continue
            leg = [r["legacy"][metric] for r in rows]
            exp = [r["explicit"][metric] for r in rows]
            block[metric] = {"legacy": _summ(leg), "explicit": _summ(exp),
                             "mean_shift": float(np.mean(exp) - np.mean(leg)),
                             "max_abs_seed_shift": float(np.max(np.abs(np.subtract(exp, leg))))}
        aggregate[f"d={drift} {agent}"] = block
    import sklearn
    return {
        "states_dir": states_dir.replace("\\", "/"), "label": label,
        "spec": "docs/specs/2026-09-28-explicit-cv-folds.md",
        "generated_by": "scripts/rescore_fold_split.py",
        "stack": {"python": sys.version.split()[0], "numpy": np.__version__,
                  "sklearn": sklearn.__version__, "platform": platform.platform(),
                  "machine": platform.machine(), "processor": platform.processor()},
        "partition_110_110": {s: folds.class_counts(y220, np.arange(220), scheme=s) for s in SCHEMES},
        "n_cells": len(cells), "cells": cells, "aggregate": aggregate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states_dirs", nargs="+")
    ap.add_argument("--json", required=True, help="output path (one dir) or directory (several)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    multi = len(a.states_dirs) > 1
    for d in a.states_dirs:
        out = rescore(d, a.label)
        path = (os.path.join(a.json, os.path.basename(os.path.dirname(os.path.normpath(d))) + ".json")
                if multi else a.json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n{d}: {out['n_cells']} cells; partition legacy {out['partition_110_110']['legacy']}, "
              f"explicit {out['partition_110_110']['explicit']}")
        for key, blk in out["aggregate"].items():
            for metric in ("target", "resid_trace"):
                if metric in blk:
                    b = blk[metric]
                    print(f"  {key:22s} {metric:12s} legacy {b['legacy']['mean']:.3f} "
                          f"({b['legacy']['n_ge_065']}/{b['legacy']['n_seeds']})  explicit "
                          f"{b['explicit']['mean']:.3f} ({b['explicit']['n_ge_065']}/{b['explicit']['n_seeds']})  "
                          f"shift {b['mean_shift']:+.4f}")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
