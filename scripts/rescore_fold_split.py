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
the seeds at or above the bar, and the mean shift. Each cell also records the shape of
its dumped pools: `state_dim` (the last axis of Ha / Hs, which is the agent's GRU trunk
width), `steps`, `n_auth` and `n_surr`. The trunk width is otherwise attested only by the
gitignored dumps and agents, and two comparator runs once ran an 8-unit trunk by mistake.

--annotate-shapes adds those shape fields to an artifact written before they existed. It
reads only the array headers of the run's dumps (no probe is refit, no number moves) and
rewrites the artifact with every other field unchanged.

Usage:
    python scripts/rescore_fold_split.py fullruns/l3_h8_traces/states \
        --json fullruns/fold_rescore/l3_h8_traces.json [--label "L3 hidden 8 headline"]
    python scripts/rescore_fold_split.py --annotate-shapes artifacts/fold_rescore/l3_n10.json \
        [--runs-root <checkout>/fullruns]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from promote_bv3_gates import repo_rel  # noqa: E402

import argparse
import glob
import json
import os
import platform
import re
import sys
import zipfile
from collections import defaultdict

import numpy as np

from itasorl import folds
from itasorl.behavior_audit import trace_residual_probe_auroc
from itasorl.experiment_b import episode_features, probe_auroc
from itasorl.stats import t_ci90

FNAME = re.compile(r"states_d(?P<drift>[-\d.]+)_s(?P<seed>\d+)_(?P<agent>\w+)\.npz$")
BAR = 0.65
SCHEMES = ("legacy", "explicit")
SHAPE_KEYS = ("state_dim", "steps", "n_auth", "n_surr")


def npz_shapes(path: str) -> dict:
    """{array name: shape} from the .npy headers inside an .npz, without loading any array."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape = np.lib.format.read_array_header_1_0(fh)[0]
                elif version == (2, 0):
                    shape = np.lib.format.read_array_header_2_0(fh)[0]
                else:
                    # Format 3.0 (a UTF-8 header) has no public reader; the private one reads
                    # every version numpy knows and refuses the rest.
                    shape = np.lib.format._read_array_header(fh, version)[0]
            out[name[:-4] if name.endswith(".npy") else name] = tuple(int(x) for x in shape)
    return out


def pool_shape(shapes: dict) -> dict:
    """The cell's shape fields from its pooled-state arrays Ha (n_auth, steps, dim) and Hs."""
    (na, steps, dim), (ns, steps_s, dim_s) = shapes["Ha"], shapes["Hs"]
    if (steps, dim) != (steps_s, dim_s):
        raise ValueError(f"authentic and surrogate pools disagree: {shapes['Ha']} vs {shapes['Hs']}")
    return {"state_dim": dim, "steps": steps, "n_auth": na, "n_surr": ns}


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
        cell = {"drift": m["drift"], "seed": int(m["seed"]), "agent": m["agent"],
                **pool_shape(npz_shapes(f)), **res}
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
        "states_dir": repo_rel(states_dir), "label": label,
        "spec": "docs/specs/2026-09-28-explicit-cv-folds.md",
        "generated_by": "scripts/rescore_fold_split.py",
        "stack": {"python": sys.version.split()[0], "numpy": np.__version__,
                  "sklearn": sklearn.__version__, "platform": platform.platform(),
                  "machine": platform.machine(), "processor": platform.processor()},
        "partition_110_110": {s: folds.class_counts(y220, np.arange(220), scheme=s) for s in SCHEMES},
        "n_cells": len(cells), "cells": cells, "aggregate": aggregate,
    }


def annotate_shapes(art_path: str, runs_root: str | None = None) -> dict:
    """Add the per-cell shape fields to a rescore artifact written before they existed.

    The dumps are found at the artifact's recorded states_dir (fullruns/<run>/states) under
    `runs_root` (default: this checkout's fullruns/). Only array headers are read. Every field
    already in the artifact is kept as it was, in order; the shape fields go after "agent".
    Refuses when a dump is missing or a cell already carries a different shape."""
    with open(art_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("generated_by") != "scripts/rescore_fold_split.py":
        raise SystemExit(f"{art_path} was not written by this script")
    rel = doc["states_dir"].replace("\\", "/")
    head, _, sub = rel.partition("/")
    if head != "fullruns" or not sub:
        raise SystemExit(f"{art_path}: states_dir {rel!r} is not under fullruns/")
    root = runs_root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "fullruns")
    states = os.path.join(root, *sub.split("/"))
    cells = []
    for c in doc["cells"]:
        dump = os.path.join(states, f"states_d{c['drift']}_s{c['seed']}_{c['agent']}.npz")
        if not os.path.exists(dump):
            raise SystemExit(f"{art_path}: missing dump {dump}")
        shape = pool_shape(npz_shapes(dump))
        known = {k: c[k] for k in SHAPE_KEYS if k in c}
        if known and known != {k: shape[k] for k in known}:
            raise SystemExit(f"{art_path}: cell {c['drift']} s{c['seed']} {c['agent']} records "
                             f"{known}, the dump says {shape}")
        new = {}
        for k, v in c.items():
            if k in SHAPE_KEYS:
                continue
            new[k] = v
            if k == "agent":
                new.update(shape)
        cells.append(new)
    doc["cells"] = cells
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("states_dirs", nargs="*")
    ap.add_argument("--json", default=None, help="output path (one dir) or directory (several)")
    ap.add_argument("--label", default="")
    ap.add_argument("--annotate-shapes", nargs="+", default=None, metavar="ARTIFACT",
                    help="add the per-cell shape fields to existing rescore artifacts in place")
    ap.add_argument("--runs-root", default=None,
                    help="with --annotate-shapes: directory holding the run bundles (read only)")
    a = ap.parse_args()
    if a.annotate_shapes:
        for path in a.annotate_shapes:
            doc = annotate_shapes(path, a.runs_root)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=1)
            dims = sorted({c["state_dim"] for c in doc["cells"]})
            print(f"annotated {path}: {len(doc['cells'])} cells, state_dim {dims}")
        return 0
    if not a.states_dirs or not a.json:
        ap.error("states_dirs and --json are required unless --annotate-shapes is given")
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
