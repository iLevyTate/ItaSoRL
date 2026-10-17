"""Write an integrity reference for a readout from a run's cell records (G45, G56, G02).

Readouts that replay a run's saved agents (scripts/reverse_direction.py,
scripts/run_world_sample_sensitivity.py and the closing-pass readouts) check that their
registered draw reproduces the run's recorded per-seed targets before they report anything.
They read that reference in the artifacts/fold_rescore schema: a list of cells, each with the
drift as a string, the seed, the arm under "agent" and the target under the fold scheme's
name. The corrected C1 run had no such file, so this script copies the pooled targets the
runner wrote into each cell (`cell.agents.<arm>.pool.target`, explicit folds) into one.

Values are copied, not recomputed. Every cell records the file it came from, repo-relative.

Usage:
    python scripts/build_integrity_ref.py --cells-dir fullruns/corrected_l3_h8_wm/cells \\
        --drifts 0.00 0.45 --arms survival predictor untrained \\
        --out artifacts/fold_rescore/corrected_l3_h8_wm_integrity_ref.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import os
import sys

from agent_free_common import dumps_compact, provenance
from promote_bv3_gates import repo_rel

from itasorl.experiment_b2 import format_drift

GENERATED_BY = "scripts/build_integrity_ref.py"
SCHEME = "explicit"


def build(cells_dir: str, drifts, arms, seeds=None) -> list[dict]:
    """One entry per (drift, seed, arm), in drift, seed, arm order. A drift with no cell file
    is a hard error naming the directory and the drift, never a silent skip; so is a
    non-finite target or a cell whose recorded drift or seed disagrees with its file name."""
    out = []
    src_dir = repo_rel(cells_dir).rstrip("/")
    for d in drifts:
        tag = format_drift(float(d))
        names = sorted(n for n in os.listdir(cells_dir)
                       if n.startswith(f"cell_d{tag}_s") and n.endswith(".json"))
        by_seed = {}
        for n in names:
            seed_txt = n[len(f"cell_d{tag}_s"):-len(".json")]
            if not seed_txt.isdigit():
                continue
            by_seed[int(seed_txt)] = n
        if seeds is not None:
            missing = sorted(set(seeds) - set(by_seed))
            if missing:
                raise ValueError(f"{cells_dir}: no cell file for drift {tag} seed(s) {missing}")
            by_seed = {s: by_seed[s] for s in seeds}
        if not by_seed:
            raise ValueError(f"{cells_dir}: no cell file for drift {tag}")
        for s in sorted(by_seed):
            with open(os.path.join(cells_dir, by_seed[s]), encoding="utf-8") as fh:
                c = json.load(fh)["cell"]
            if format_drift(float(c["drift"])) != tag or int(c["seed"]) != s:
                raise ValueError(f"{by_seed[s]}: records drift {c['drift']} seed {c['seed']}")
            for arm in arms:
                t = float(c["agents"][arm]["pool"]["target"])
                if not math.isfinite(t):
                    raise ValueError(f"{by_seed[s]}: {arm} pooled target is not finite")
                out.append({"drift": tag, "seed": s, "agent": arm, SCHEME: {"target": t},
                            "source": f"{src_dir}/{by_seed[s]}"})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--drifts", nargs="+", required=True, help="e.g. 0.00 0.45")
    ap.add_argument("--arms", nargs="+", required=True, help="e.g. survival predictor untrained")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="require exactly these seeds at every drift (default: every seed found)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the counts and write nothing")
    a = ap.parse_args(argv)
    cells = build(a.cells_dir, a.drifts, a.arms, a.seeds)
    doc = {**provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(a.out)),
           "cells_dir": repo_rel(a.cells_dir), "fold_scheme": SCHEME,
           "note": "pooled targets copied from the run's cell records; integrity reference only",
           "n_cells": len(cells), "cells": cells}
    by = {}
    for c in cells:
        by.setdefault((c["drift"], c["agent"]), []).append(c[SCHEME]["target"])
    for (d, arm), v in sorted(by.items()):
        print(f"d{d} {arm:9s} n {len(v)}  mean {sum(v) / len(v):.4f}")
    if a.dry_run:
        print("dry run: nothing written")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps_compact(doc) + "\n")
    print(f"wrote {repo_rel(a.out)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
