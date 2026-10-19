"""Promote the closing-pass L1 grid readouts (B1, B7) from their out-dirs into single artifacts.

scripts/run_l1_h2_ablations.py writes three files per out-dir: `<channel>_aggregate.json` (the
per-seed values, means and intervals), `<channel>_cells.json` (one record per drift, seed and
arm) and `provenance.json` (commit, tracked changes, stack, agents and states folders, argv).
This script joins them into one JSON per readout, in this order: the aggregate's keys in their
own order, then "provenance" (provenance.json), then "cells" (the cells list). Nothing is
recomputed and no value changes; the only edit is that any absolute path string is made
repo-relative through promote_bv3_gates.repo_rel. The output is deterministic: the same inputs
give the same bytes.

Readouts (docs/specs/2026-10-17-closing-readouts-design.md, sections B1 and B7):
    B1 corrected   fullruns/closing_2026-10/b1_corrected   -> artifacts/expL1/l1_coarse_grid_corrected.json
    B1 historical  fullruns/closing_2026-10/b1_historical  -> artifacts/expL1/l1_coarse_grid_historical.json
    B7             fullruns/closing_2026-10/b7             -> artifacts/expL1/l1_h2_noise_per_episode.json

Usage:
    python scripts/promote_l1_grid_readouts.py              # write all three artifacts
    python scripts/promote_l1_grid_readouts.py --check      # exit 1 if any artifact is stale
    python scripts/promote_l1_grid_readouts.py --src DIR --channel a1 --out PATH
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys

from agent_free_common import dumps_compact
from promote_bv3_gates import _ABS_PATH, repo_rel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLOSE = "fullruns/closing_2026-10"
READOUTS = (
    (f"{CLOSE}/b1_corrected", "a1", "artifacts/expL1/l1_coarse_grid_corrected.json"),
    (f"{CLOSE}/b1_historical", "a1", "artifacts/expL1/l1_coarse_grid_historical.json"),
    (f"{CLOSE}/b7", "noise", "artifacts/expL1/l1_h2_noise_per_episode.json"),
)


def _rel_paths(x):
    """The same structure with every absolute path string made repo-relative."""
    if isinstance(x, dict):
        return {k: _rel_paths(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rel_paths(v) for v in x]
    if isinstance(x, str) and _ABS_PATH.match(x):
        return repo_rel(x)
    return x


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sources(src: str, channel: str) -> list[str]:
    """The three files one readout is joined from: aggregate, provenance, cells."""
    return [os.path.join(src, f"{channel}_aggregate.json"), os.path.join(src, "provenance.json"),
            os.path.join(src, f"{channel}_cells.json")]


def build(src: str, channel: str) -> dict:
    """The joined document for one out-dir: aggregate keys, then provenance, then cells."""
    agg = _load(os.path.join(src, f"{channel}_aggregate.json"))
    prov = _load(os.path.join(src, "provenance.json"))
    cells = _load(os.path.join(src, f"{channel}_cells.json"))
    if not isinstance(agg, dict) or not isinstance(prov, dict) or not isinstance(cells, list):
        raise ValueError(f"{src}: unexpected file shapes")
    clash = {"provenance", "cells"} & set(agg)
    if clash:
        raise ValueError(f"{src}: the aggregate already has {sorted(clash)}")
    doc = dict(agg)
    doc["provenance"] = prov
    doc["cells"] = cells
    return _rel_paths(doc)


def dumps(doc: dict) -> str:
    return dumps_compact(doc) + "\n"


def write(text: str, out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", help="one out-dir (with --channel and --out); default: the three readouts")
    ap.add_argument("--channel", help="file prefix in the out-dir, e.g. a1 or noise")
    ap.add_argument("--out", help="output path for --src")
    ap.add_argument("--check", action="store_true", help="exit 1 if an artifact differs from a rebuild")
    a = ap.parse_args(argv)
    if a.src:
        if not (a.channel and a.out):
            ap.error("--src needs --channel and --out")
        jobs = [(a.src, a.channel, a.out)]
    else:
        jobs = [(os.path.join(ROOT, s), c, os.path.join(ROOT, o)) for s, c, o in READOUTS]
    stale = 0
    for src, channel, out in jobs:
        missing = [p for p in sources(src, channel) if not os.path.isfile(p)]
        if missing:
            for p in missing:
                print(f"source missing: {repo_rel(p)}")
            stale += 1
            continue
        text = dumps(build(src, channel))
        if a.check:
            try:
                with open(out, encoding="utf-8") as fh:
                    have = fh.read()
            except OSError:
                have = None
            if have != text:
                stale += 1
                print(f"stale: {repo_rel(out)}")
            continue
        write(text, out)
        print(f"wrote {repo_rel(out)}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
