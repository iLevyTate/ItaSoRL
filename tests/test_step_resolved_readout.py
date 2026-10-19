"""Closing item G04 (runbook 1.6, R12): scripts/run_step_resolved_readout.py.

Pins the step-24 column: on a dump it is r205's "final" reading (post_review.r205_cell, the h_T
component of the registered target) to the bit, and the earlier steps read h_t alone. Smoke on a
tmp run: integrity against the recorded cells, provenance, VOID on a dump off its reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from _git_checkout import assert_commit_recorded

pytest.importorskip("sklearn")
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_step_resolved_readout as sr  # noqa: E402
from itasorl import folds  # noqa: E402
from itasorl import post_review as pr  # noqa: E402
from itasorl.experiment_b import episode_features, probe_auroc  # noqa: E402

ARMS = ("survival", "predictor", "untrained")


def _pools(seed: int, arm: str, T: int = 24):
    rng = np.random.default_rng(seed + 7 * ARMS.index(arm))
    Ha = rng.normal(size=(20, T, 4)).astype(np.float32)
    Hs = rng.normal(size=(20, T, 4)).astype(np.float32)
    Hs[:, T // 2:] += np.float32(0.5 if arm == "survival" else 0.1)
    return Ha, Hs


def _run(tmp_path, shift=0.0) -> Path:
    run = tmp_path / "balanced_confirmation_touch"
    (run / "states").mkdir(parents=True)
    (run / "cells").mkdir()
    for tag in ("0.00", "0.45"):
        agents = {}
        for arm in ARMS:
            Ha, Hs = _pools(int(float(tag) * 100), arm)
            np.savez(run / "states" / f"states_d{tag}_s10_{arm}.npz", Ha=Ha, Hs=Hs)
            with folds.fold_scheme("explicit"):
                t = probe_auroc(episode_features(np.concatenate([Ha, Hs])), np.r_[np.zeros(20), np.ones(20)].astype(int))
            agents[arm] = {"pool": {"target": float(t) + shift}}
        (run / "cells" / f"cell_d{tag}_s10.json").write_text(json.dumps({"cell": {"agents": agents}}))
    return run


def test_step_24_column_is_r205_final_to_the_bit(tmp_path):
    run = _run(tmp_path)
    for tag in ("0.00", "0.45"):
        for arm in ARMS:
            c = sr.score_cell({"states_dir": str(run / "states"), "drift": tag, "seed": 10, "arm": arm,
                               "steps": list(sr.STEPS)})
            assert list(c["steps"]) == ["1", "2", "4", "8", "12", "16", "24"]
            Ha, Hs = _pools(int(float(tag) * 100), arm)
            with folds.fold_scheme("explicit"):
                r205 = pr.r205_cell(Ha, Hs, pool={"target": c["registered_target"], "shuffled": 0.5},
                                    recorded={"target": c["registered_target"]}, seed=10)
            assert r205["reproduced"]
            assert c["steps"]["24"]["target"] == r205["sets"]["final"]["target"] == c["r205_final"]
            assert c["steps"]["24"]["shuffled"] == r205["sets"]["final"]["shuffled"]
            assert c["step_final_bit_equal"]
            # an earlier step reads h_t alone (a contiguous copy, so equal up to summation order)
            H = np.concatenate([Ha, Hs])
            y = np.r_[np.zeros(20), np.ones(20)].astype(int)
            with folds.fold_scheme("explicit"):
                assert abs(c["steps"]["8"]["target"] - probe_auroc(np.ascontiguousarray(H[:, 7]), y)) < 1e-12


def test_step_features_are_h_t():
    H = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    for t in (1, 3, 5):
        assert np.array_equal(sr.step_features(H, t), H[:, t - 1])


def test_smoke_and_void(tmp_path):
    run = _run(tmp_path)
    out = tmp_path / "g04"
    argv = ["--states", str(run / "states"), "--out-dir", str(out)]
    assert sr.main([*argv, "--dry-run"]) == 0 and not out.exists()
    assert sr.main(argv) == 0
    doc = json.loads((out / "balanced_confirmation_touch.json").read_text())
    assert_commit_recorded(doc["provenance"])
    assert {"tracked_changes_at_generation", "stack", "argv", "states_dir"} <= set(doc["provenance"])
    assert doc["integrity"]["pass"] and doc["step_check"]["pass"]
    assert set(doc["summary"]) == {"0.00", "0.45"}
    assert set(doc["summary"]["0.45"]["survival_minus_untrained"]) == {str(t) for t in sr.STEPS}
    bad = _run(tmp_path / "b", shift=1e-6)
    assert sr.main(["--states", str(bad / "states"), "--out-dir", str(tmp_path / "g04b")]) == 1
    assert not (tmp_path / "g04b").exists()
    assert sr.main(["--states", str(bad / "states"), "--out-dir", str(tmp_path / "g04b"), "--dry-run"]) == 1
