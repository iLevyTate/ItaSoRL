"""Promote the B1 and B2 pooled readings on the registered world pair from their cell records.

FINDINGS 21, CORRECTIONS (2026-10-14) and the manuscript quote the pooled survival targets of
the balanced runs at drift 0.45 (0.758 for B1, 0.835 for B2) and their drift-zero companions
(0.561 and 0.536), beside the predictor and untrained means, as recomputed from the cell
records in fullruns/. No committed artifact held them. This script copies
`cell.agents.<arm>.pool.target` for every arm, drift and seed out of the B1 and B2 cells into
artifacts/balanced_pooled/registered_pair.json, with per-seed values, means, t-based 90%
intervals, the drift-zero survival TOST (against 0.5, margin 0.05), the seed-paired survival
margins at drift 0.45, and the SHA-256 of every cell file read.

The pooled readout uses the registered evaluation-world pair (800000 and 850000), not the
balanced pairs at 910000 that decide the run (scripts/promote_balanced_confirmation.py). It is
descriptive and decides nothing. No value is recomputed from a model: each per-seed number is
the one the runner wrote into the cell.

Usage:
    python scripts/promote_balanced_pooled.py                # write the artifact
    python scripts/promote_balanced_pooled.py --dry-run      # print the readings, write nothing
    python scripts/promote_balanced_pooled.py --check        # exit 1 if the artifact is stale
    python scripts/promote_balanced_pooled.py --run B1=<dir> --run B2=<dir> --out <path>
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

from agent_free_common import dumps_compact, provenance
from promote_bv3_gates import repo_rel

from itasorl.stats import equivalence_test, paired_contrast, t_ci90

GENERATED_BY = "scripts/promote_balanced_pooled.py"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "artifacts", "balanced_pooled", "registered_pair.json")
SPEC = "docs/specs/2026-10-12-balanced-confirmation-design.md"
RUNS = {
    "B1": "fullruns/balanced_confirmation_wm",
    "B2": "fullruns/balanced_confirmation_touch",
}
ARMS = ("survival", "predictor", "untrained")
DRIFTS = (0.0, 0.45)
SEEDS = tuple(range(10, 20))
BAR = 0.65
TOST_MARGIN = 0.05
SCOPE = ("descriptive: pooled readout on the registered evaluation-world pair (800000 / 850000), "
         "not the balanced pairs at 910000 that decide B1 and B2; decides nothing")
# Provenance keys that change with the checkout, not with the cells; --check ignores them.
VOLATILE = ("git_commit_at_promotion", "tracked_changes_at_generation", "stack")


def drift_tag(d: float) -> str:
    return f"{d:.2f}"


def load_run(run_id: str, run_dir: str) -> dict:
    """The run's cells keyed by (drift, seed), its single recorded commit and fingerprint, and
    the SHA-256 of each cell file. Raises ValueError naming the file or the gap when a file
    lacks provenance, when the run records more than one commit or fingerprint, or when the
    cell set is not exactly the registered seeds at both drifts."""
    path = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)
    files = sorted(glob.glob(os.path.join(path, "cells", "cell_d*_s*.json")))
    cells, sha, prov = {}, {}, {"git_commit": {}, "fingerprint": {}}
    for f in files:
        with open(f, "rb") as fh:
            raw = fh.read()
        doc = json.loads(raw.decode("utf-8"))
        for key in prov:
            if not doc.get(key):
                raise ValueError(f"{f}: cell file has no top-level {key}")
            prov[key].setdefault(doc[key], os.path.basename(f))
        c = doc["cell"]
        key = (round(float(c["drift"]), 2), int(c["seed"]))
        if key in cells:
            raise ValueError(f"{run_id}: duplicate cell for drift {key[0]} seed {key[1]} ({f})")
        cells[key] = c
        sha[os.path.basename(f)] = hashlib.sha256(raw).hexdigest()
    for key, seen in prov.items():
        if len(seen) != 1:
            raise ValueError(f"{run_id}: cells record more than one {key}: "
                             + ", ".join(f"{v} ({n})" for v, n in sorted(seen.items())))
    want = {(d, s) for d in DRIFTS for s in SEEDS}
    if set(cells) != want:
        raise ValueError(f"{run_id}: cell set in {path} differs from the registered seeds and drifts; "
                         f"missing {sorted(want - set(cells))}, unexpected {sorted(set(cells) - want)}")
    return {"cells": cells, "sha256": sha,
            "git_commit": next(iter(prov["git_commit"])), "fingerprint": next(iter(prov["fingerprint"]))}


def summ(vals) -> dict:
    a = np.asarray(vals, float)
    lo, hi = t_ci90(a)
    return {"per_seed": [float(x) for x in a], "mean": float(a.mean()), "t90": [float(lo), float(hi)],
            "n": int(a.size), "n_ge_065": int((a >= BAR).sum())}


def run_block(run_id: str, run_dir: str) -> dict:
    r = load_run(run_id, run_dir)
    cells = r["cells"]
    vals = {d: {a: [float(cells[(d, s)]["agents"][a]["pool"]["target"]) for s in SEEDS] for a in ARMS}
            for d in DRIFTS}
    if not all(np.isfinite(v).all() for by_arm in vals.values() for v in by_arm.values()):
        raise ValueError(f"{run_id}: a pooled target is not finite")
    l0 = vals[0.0]["survival"]
    tost = equivalence_test(l0, 0.5, margin=TOST_MARGIN)
    surv = vals[0.45]["survival"]
    first = cells[(0.45, SEEDS[0])]
    return {
        "source_run": repo_rel(run_dir),
        "git_commit_at_run": r["git_commit"],
        "fingerprint": r["fingerprint"],
        "objective": first.get("objective"),
        "mortal": first.get("mortal"),
        "run_stack": first.get("stack"),
        "seeds": list(SEEDS),
        "pool_target": {drift_tag(d): {a: summ(vals[d][a]) for a in ARMS} for d in DRIFTS},
        "l0_tost": {"drift": drift_tag(0.0), "arm": "survival", "h0": 0.5, "margin": TOST_MARGIN,
                    "mean": float(np.mean(l0)), "p_value": float(tost.p_value),
                    "equivalent": bool(tost.equivalent)},
        "margins_0.45": {"over_predictor": paired_contrast(surv, vals[0.45]["predictor"]),
                         "over_untrained": paired_contrast(surv, vals[0.45]["untrained"])},
        "cells_sha256": r["sha256"],
    }


def build(runs: dict | None = None, prov: dict | None = None) -> dict:
    runs = RUNS if runs is None else runs
    doc = dict(prov if prov is not None else
               provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(OUT)))
    doc.update({"spec": SPEC, "scope": SCOPE, "bar": BAR,
                "runs": {rid: run_block(rid, d) for rid, d in runs.items()}})
    blocks = doc["runs"]
    if len(blocks) > 1:
        # The predictor and untrained arms train no survival actor-critic; both runs read the
        # same ten cells for them when their values agree, which is recorded, not assumed.
        first = next(iter(blocks.values()))
        doc["baselines_identical_across_runs"] = {
            a: all(b["pool_target"][t][a]["per_seed"] == first["pool_target"][t][a]["per_seed"]
                   for b in blocks.values() for t in first["pool_target"])
            for a in ("predictor", "untrained")}
    return doc


def dumps(doc: dict) -> str:
    return dumps_compact(doc) + "\n"


def _stable(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in VOLATILE}


def readings(doc: dict) -> list[str]:
    out = []
    for rid, b in doc["runs"].items():
        for t, by_arm in b["pool_target"].items():
            for a, s in by_arm.items():
                out.append(f"{rid} d{t} {a:9s} mean {s['mean']:.4f} t90 [{s['t90'][0]:.3f}, {s['t90'][1]:.3f}] "
                           f"n_ge_065 {s['n_ge_065']}/{s['n']}")
        lt = b["l0_tost"]
        out.append(f"{rid} drift-0 survival TOST p {lt['p_value']:.3f} equivalent={lt['equivalent']}")
        for k, m in b["margins_0.45"].items():
            out.append(f"{rid} d0.45 survival {k} {m['mean']:+.3f} t90 [{m['t90'][0]:+.3f}, {m['t90'][1]:+.3f}]")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", default=[], metavar="ID=DIR",
                    help="override a run's directory (B1 or B2); default fullruns/balanced_confirmation_{wm,touch}")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry-run", action="store_true", help="print the readings and write nothing")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if --out differs from a rebuild, ignoring the checkout provenance keys")
    a = ap.parse_args(argv)
    runs = dict(RUNS)
    for spec in a.run:
        rid, sep, d = spec.partition("=")
        if not sep or rid not in RUNS:
            ap.error(f"--run expects B1=DIR or B2=DIR, got {spec!r}")
        runs[rid] = d
    doc = build(runs, provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(a.out)))
    if a.check:
        try:
            with open(a.out, encoding="utf-8") as fh:
                cur = json.load(fh)
        except (OSError, ValueError):
            cur = None
        ok = cur is not None and _stable(cur) == json.loads(json.dumps(_stable(doc)))
        print("balanced pooled pair: " + ("OK" if ok else "stale; run python scripts/promote_balanced_pooled.py"))
        return 0 if ok else 1
    for line in readings(doc):
        print(line)
    if a.dry_run:
        print("dry run: nothing written")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps(doc))
    print(f"wrote {repo_rel(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
