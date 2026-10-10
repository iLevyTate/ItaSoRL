"""Provenance recorded by the fold re-score scripts.

* rescore_fold_split.py records the shape of every dumped pool, so the agent's GRU trunk width
  (registered 96) is attested by a committed artifact rather than only by gitignored dumps.
* reanalyze_mp_readout.py scores its determinism gate on the legacy partition, the one the
  stored pre-fix values used, whatever scheme is active.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_results_manifest as brm  # noqa: E402
import rescore_fold_split as rfs  # noqa: E402

REGISTERED_TRUNK = 96


def _rescore_artifacts():
    out = []
    for p in sorted(glob.glob(str(ROOT / "artifacts" / "fold_rescore" / "*.json"))):
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("generated_by") == "scripts/rescore_fold_split.py":
            out.append((os.path.relpath(p, ROOT / "artifacts").replace(os.sep, "/"), doc))
    return out


def test_every_rescore_artifact_records_the_trunk_width_of_every_cell():
    arts = _rescore_artifacts()
    assert len(arts) >= 7
    for rel, doc in arts:
        for c in doc["cells"]:
            assert {"state_dim", "steps", "n_auth", "n_surr"} <= set(c), (rel, c["drift"], c["seed"])


def test_every_run_labelled_with_the_registered_trunk_has_state_dim_96():
    runs = {r["id"]: r for r in brm.RUNS}
    for rel, doc in _rescore_artifacts():
        owners = brm._owner(rel)
        assert len(owners) == 1, rel
        hidden = runs[owners[0]]["config"].get("hidden")
        assert hidden == REGISTERED_TRUNK, (rel, owners[0], hidden)
        dims = {c["state_dim"] for c in doc["cells"]}
        assert dims == {REGISTERED_TRUNK}, (rel, dims)


def _dump(path, n_auth=5, n_surr=4, steps=3, dim=6):
    np.savez_compressed(path, Ha=np.zeros((n_auth, steps, dim), np.float32),
                        Hs=np.zeros((n_surr, steps, dim), np.float32), steps=np.int64(steps))


def test_npz_shapes_reads_headers_and_pool_shape_names_the_fields(tmp_path):
    _dump(tmp_path / "s.npz")
    shapes = rfs.npz_shapes(str(tmp_path / "s.npz"))
    assert shapes["Ha"] == (5, 3, 6) and shapes["Hs"] == (4, 3, 6) and shapes["steps"] == ()
    assert rfs.pool_shape(shapes) == {"state_dim": 6, "steps": 3, "n_auth": 5, "n_surr": 4}


def test_annotate_inserts_shapes_after_agent_and_changes_nothing_else(tmp_path):
    states = tmp_path / "runs" / "r" / "states"
    states.mkdir(parents=True)
    _dump(states / "states_d0.45_s0_survival.npz")
    doc = {"states_dir": "fullruns/r/states", "generated_by": "scripts/rescore_fold_split.py",
           "cells": [{"drift": "0.45", "seed": 0, "agent": "survival",
                      "legacy": {"target": 0.7}, "explicit": {"target": 0.71}}],
           "aggregate": {"x": 1}}
    art = tmp_path / "a.json"
    art.write_text(json.dumps(doc), encoding="utf-8")
    out = rfs.annotate_shapes(str(art), str(tmp_path / "runs"))
    c = out["cells"][0]
    assert list(c) == ["drift", "seed", "agent", "state_dim", "steps", "n_auth", "n_surr",
                       "legacy", "explicit"]
    assert c["state_dim"] == 6 and out["aggregate"] == {"x": 1}
    # idempotent, and a recorded shape that disagrees with the dump is refused
    art.write_text(json.dumps(out), encoding="utf-8")
    assert rfs.annotate_shapes(str(art), str(tmp_path / "runs")) == out
    out["cells"][0]["state_dim"] = 8
    art.write_text(json.dumps(out), encoding="utf-8")
    with pytest.raises(SystemExit):
        rfs.annotate_shapes(str(art), str(tmp_path / "runs"))


def test_the_mp_determinism_gate_is_scored_on_the_legacy_partition(monkeypatch):
    pytest.importorskip("torch")
    import reanalyze_mp_readout as rmr
    from itasorl import folds
    seen = []
    eps = [{"H": np.full((4, 3), float(i)), "label": i % 2} for i in range(8)]
    monkeypatch.setattr(rmr, "load_agent_bundle", lambda path, device: (None, None))
    monkeypatch.setattr(rmr, "matched_pair_recurrent_rollout",
                        lambda *a, **k: (eps[::2], eps[1::2]))
    monkeypatch.setattr(rmr, "grouped_auroc",
                        lambda X, y, g: seen.append(folds.current_scheme()) or 0.5)
    monkeypatch.setattr(rmr, "probe_world_identity",
                        lambda a, s, seed: seen.append(folds.current_scheme()) or {"target": 0.6})
    with folds.fold_scheme("explicit"):
        r = rmr.rescore_cell("agent_d0.45_s0_survival.pt", 0.45, 0, "cpu")
    assert seen == ["legacy", "explicit"]
    assert r == {"mp_old": 0.5, "mp_fixed": 0.6, "n_pairs": 4}


def test_npz_shapes_reads_every_npy_header_version(tmp_path):
    import zipfile
    arr = np.zeros((2, 3, 4), np.float32)
    path = tmp_path / "v.npz"
    with zipfile.ZipFile(path, "w") as z:
        for name, version in (("a.npy", (1, 0)), ("b.npy", (2, 0)), ("c.npy", (3, 0))):
            with z.open(name, "w") as fh:
                np.lib.format.write_array(fh, arr, version=version)
    assert rfs.npz_shapes(str(path)) == {"a": (2, 3, 4), "b": (2, 3, 4), "c": (2, 3, 4)}
