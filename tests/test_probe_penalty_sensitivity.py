"""Closing item G01 (runbook 1.6, R11): scripts/run_probe_penalty_sensitivity.py.

Smoke on tmp dump sets: C = 1 is the registered probe bit for bit, the reference of each kind
of states dir is resolved and checked within 1e-9 before anything is written, the G57 rows carry
the tolerance label, and a dir that does not reproduce its reference is VOID (nothing written,
exit 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_probe_penalty_sensitivity as ps  # noqa: E402
from itasorl import folds  # noqa: E402
from itasorl.experiment_b import episode_features, probe_auroc  # noqa: E402

ARMS = ("survival", "predictor", "untrained")


def _dumps(states, tags=("0.00", "0.45"), seeds=(0,)) -> dict:
    states.mkdir(parents=True)
    targets = {}
    for tag in tags:
        for s in seeds:
            for k, arm in enumerate(ARMS):
                rng = np.random.default_rng(10 * s + k + int(float(tag) * 100))
                Ha = rng.normal(size=(20, 3, 4)).astype(np.float32)
                Hs = (rng.normal(size=(20, 3, 4)) + 0.5 * (arm == "survival")).astype(np.float32)
                np.savez(states / f"states_d{tag}_s{s}_{arm}.npz", Ha=Ha, Hs=Hs)
                with folds.fold_scheme("explicit"):
                    targets[(tag, s, arm)] = float(probe_auroc(
                        episode_features(np.concatenate([Ha, Hs])), np.r_[np.zeros(20), np.ones(20)].astype(int)))
    return targets


def _cells(run: Path, targets: dict, shift=0.0):
    (run / "cells").mkdir()
    for tag, s in sorted({(k[0], k[1]) for k in targets}):
        agents = {arm: {"pool": {"target": targets[(tag, s, arm)] + (shift if arm == "survival" else 0.0)}}
                  for arm in ARMS}
        (run / "cells" / f"cell_d{tag}_s{s}.json").write_text(json.dumps({"cell": {"agents": agents}}))


@pytest.fixture(autouse=True)
def _small_grid(monkeypatch):
    monkeypatch.setattr(ps, "C_GRID", (0.1, 1.0))


def test_c1_is_the_registered_probe_bit_for_bit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.r_[np.zeros(20), np.ones(20)].astype(int)
    X[y == 1] += 0.4
    with folds.fold_scheme("explicit"):
        assert ps.cv_auroc(X, y, np.arange(40), 1.0)[0] == probe_auroc(X, y)
    nested = ps.nested_auroc(X, y, np.arange(40), (0.01, 1.0))
    assert len(nested["chosen_C"]) == 5 and set(nested["chosen_C"]) <= {0.01, 1.0}
    assert np.isfinite(nested["target"])


def test_cell_target_dir_and_regenerated_dir(tmp_path):
    run = tmp_path / "corrected_l1_heldout"
    t = _dumps(run / "states")
    _cells(run, t)
    g57 = tmp_path / "g57"
    t2 = _dumps(g57 / "states_local")
    report = {"run_dir": str(tmp_path / "corrected_l3_h8_wm"), "mode": "cross_stack_replay", "tolerance": 0.002,
              "cells": [{"drift": float(k[0]), "seed": k[1], "arm": k[2], "target": v,
                         "recorded_target": v + 0.001} for k, v in t2.items()]}
    (g57 / "c1_regen.json").write_text(json.dumps(report))
    out = tmp_path / "g01"
    argv = ["--states", str(run / "states"), str(g57 / "states_local"), "--out-dir", str(out)]
    assert ps.main([*argv, "--dry-run"]) == 0 and not out.exists()
    assert ps.main(argv) == 0
    doc = json.loads((out / "corrected_l1_heldout.json").read_text())
    assert doc["provenance"]["git_commit_at_promotion"] and "git_commit" not in doc["provenance"]
    assert {"tracked_changes_at_generation", "stack", "argv"} <= set(doc["provenance"])
    assert doc["integrity"]["pass"] and doc["integrity"]["n_checked"] == 6
    assert doc["integrity"]["worst_abs_diff"] == 0.0
    c = doc["cells"][0]
    assert set(c["grid"]) == {"0.1", "1"} and len(c["grid"]["1"]["shuffled_draws"]) == 8
    assert c["grid"]["1"]["target"] == t[(c["drift"], c["seed"], c["arm"])]
    assert "label" not in c
    reg = json.loads((out / "corrected_l3_h8_wm_regenerated.json").read_text())
    assert reg["integrity"]["kind"] == "regenerated" and reg["integrity"]["regen_within_tolerance"]
    assert all(c["label"] == "cross-stack regenerated, max abs target delta = 0.001" for c in reg["cells"])
    assert set(reg["summary"]) == {f"d={tag} {arm}" for tag in ("0.00", "0.45") for arm in ARMS}


def test_a_dir_off_its_reference_is_void(tmp_path):
    run = tmp_path / "balanced_confirmation_wm"
    _cells(run, _dumps(run / "states"), shift=1e-6)
    out = tmp_path / "g01"
    assert ps.main(["--states", str(run / "states"), "--out-dir", str(out)]) == 1
    assert not out.exists()


def test_a_regeneration_outside_its_tolerance_is_void(tmp_path):
    g57 = tmp_path / "g57"
    t = _dumps(g57 / "states_local")
    (g57 / "c1_regen.json").write_text(json.dumps({"run_dir": "x/corrected_l3_h8_wm", "tolerance": 0.002, "cells": [
        {"drift": float(k[0]), "seed": k[1], "arm": k[2], "target": v, "recorded_target": v + 0.01}
        for k, v in t.items()]}))
    assert ps.main(["--states", str(g57 / "states_local"), "--out-dir", str(tmp_path / "o")]) == 1


def test_fail_closed_on_an_unplaced_dir_and_a_missing_drift(tmp_path):
    _dumps(tmp_path / "somewhere" / "states")
    with pytest.raises(SystemExit, match="regenerate_state_dumps report"):
        ps.main(["--states", str(tmp_path / "somewhere" / "states"), "--out-dir", str(tmp_path / "o")])
    run = tmp_path / "corrected_l1_heldout"
    _cells(run, _dumps(run / "states", tags=("0.00",)))
    with pytest.raises(SystemExit, match=r"0\.023 .*d0\.0230.*corrected_l1_heldout"):
        ps.main(["--states", str(run / "states"), "--drifts", "0.00", "0.023", "--out-dir", str(tmp_path / "o")])
