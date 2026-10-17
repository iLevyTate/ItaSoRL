"""Shared command line and output wiring for the five scripts/run_expB_*.py scripts.

Each script keeps its registered constants as the defaults, so a bare invocation reproduces
the stored run and writes its record figure under docs/figures exactly as before. Two flags
are shared:

  --align    the episode collector's observation-action pairing (experiment_b.collect_episodes);
             the default `legacy` is the stored record, `obs_then_action` is the corrected
             pairing of CORRECTIONS 2026-10-16.
  --out-dir  send every figure into this scratch directory instead of docs/figures and write
             `<out-dir>/<script>.json` with the per-drift and per-seed metrics plus provenance
             (agent_free_common.provenance, commit key `git_commit`: the computation is the run).
             Without it no JSON is written and docs/figures is the figure target, as before.

The size knobs (--pool, --sub, --epochs, --seeds) default to the registered constants and exist
for smoke runs; the JSON records whatever was used, so a reduced run is self-describing.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import sys

from agent_free_common import provenance, write_json
from itasorl.experiment_b import ALIGNMENTS
from itasorl.world import WorldParams

PARAMS = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
WORLD_LABEL = "P(k_land=1.5, k_water=1.5, gravity=0.4)"
FIGURE_DIR = "docs/figures"


def parser(description: str, *, pool: int, sub: int, epochs: int,
           seeds: list[int]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--align", choices=ALIGNMENTS, default="legacy",
                    help="observation-action pairing of the collector; legacy is the stored "
                         "record, obs_then_action gives the model each action with the "
                         "observation it acts on")
    ap.add_argument("--out-dir", default=None,
                    help="scratch directory for the figure and <script>.json; without it the "
                         "figure goes to docs/figures and no JSON is written")
    ap.add_argument("--pool", type=int, default=pool,
                    help="episodes per collected pool (registered %(default)s)")
    ap.add_argument("--sub", type=int, default=sub,
                    help="episodes subsampled per arm for each seed (registered %(default)s)")
    ap.add_argument("--epochs", type=int, default=epochs,
                    help="world-model training epochs (registered %(default)s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=seeds,
                    help="training seeds (registered %(default)s)")
    return ap


def figure_path(out_dir: str | None, name: str) -> str:
    """`<out-dir>/<name>` when a scratch directory is set (created if absent), else the
    record path."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, name)
    return f"{FIGURE_DIR}/{name}"


def subsample(pool, k, rng):
    return [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]


def json_path(out_dir: str, script: str) -> str:
    return os.path.join(out_dir, f"{script}.json")


def write_out_json(out_dir: str, script: str, align: str, config: dict, results: dict,
                   argv=None) -> str:
    """Write `<out-dir>/<script>.json`: config, results and the provenance block (with the
    command line, `sys.argv[1:]` when `argv` is None)."""
    path = json_path(out_dir, script)
    generated_by = f"scripts/run_{script}.py"
    prov = provenance(generated_by, commit_key="git_commit", output=path, device="cpu")
    prov["argv"] = list(sys.argv[1:] if argv is None else argv)
    from itasorl import folds
    payload = {"script": generated_by, "align": align,
               "config": {**config, "fold_scheme": folds.current_scheme()}, **results,
               "provenance": prov}
    write_json(path, payload)
    return path


def seed_stats(values: dict[str, list[float]]) -> dict:
    """Mean and population std per metric (the printed numbers) plus the per-seed lists."""
    import numpy as np
    out = {}
    for k, v in values.items():
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
    out["per_seed"] = {k: [float(x) for x in v] for k, v in values.items()}
    return out
