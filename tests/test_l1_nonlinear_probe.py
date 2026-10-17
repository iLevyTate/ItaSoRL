"""Closing item H53 (runbook 1.6, R17): scripts/run_l1_nonlinear_probe.py.

Smoke on a tmp dump set: the linear C = 1 gate against the recorded cells, the three probe
classes, the drift tag rule (0.023 -> 0.0230) and the fail-closed paths (a reference off by
1e-6 reports nothing; a drift with no dump is a hard error naming the dir and the drift)."""

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

import run_l1_nonlinear_probe as nl  # noqa: E402
from itasorl import folds  # noqa: E402
from itasorl.experiment_b import episode_features, probe_auroc  # noqa: E402

ARMS = ("survival", "untrained")


def _run_dir(tmp_path, tags=("0.00", "0.0230"), seeds=(0, 1), shift=None) -> Path:
    run = tmp_path / "corrected_l1_heldout"
    (run / "states").mkdir(parents=True)
    (run / "cells").mkdir()
    for tag in tags:
        for s in seeds:
            agents = {}
            for k, arm in enumerate(ARMS):
                rng = np.random.default_rng(100 * s + k + int(float(tag) * 1e4))
                Ha = rng.normal(size=(20, 3, 4)).astype(np.float32)
                Hs = (rng.normal(size=(20, 3, 4)) + (0.6 if arm == "survival" else 0.0)).astype(np.float32)
                np.savez(run / "states" / f"states_d{tag}_s{s}_{arm}.npz", Ha=Ha, Hs=Hs)
                with folds.fold_scheme("explicit"):
                    t = float(probe_auroc(episode_features(np.concatenate([Ha, Hs])),
                                          np.r_[np.zeros(20), np.ones(20)].astype(int)))
                if shift and (tag, s, arm) == shift[0]:
                    t += shift[1]
                agents[arm] = {"pool": {"target": t, "shuffled": 0.5}}
            (run / "cells" / f"cell_d{tag}_s{s}.json").write_text(
                json.dumps({"cell": {"agents": agents}}), encoding="utf-8")
    return run


@pytest.fixture(autouse=True)
def _small_models(monkeypatch):
    from itasorl.post_review import mlp_probe_auroc
    monkeypatch.setattr(nl, "RF_TREES", 5)
    monkeypatch.setitem(nl.SCORERS, "mlp", lambda X, y, groups=None, *, seed=0, **_: mlp_probe_auroc(
        X, y, groups=np.arange(len(y)), seed=seed, max_iter=20))


def test_smoke_scores_every_class_after_the_gate(tmp_path):
    run = _run_dir(tmp_path)
    out = tmp_path / "out.json"
    argv = ["--states-dir", str(run / "states"), "--arms", *ARMS, "--drifts", "0.00", "0.023",
            "--out", str(out)]
    assert nl.main([*argv, "--dry-run"]) == 0 and not out.exists()
    assert nl.main(argv) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    prov = doc["provenance"]
    assert prov["git_commit_at_promotion"] and "git_commit" not in prov
    assert {"tracked_changes_at_generation", "stack", "argv", "states_dir"} <= set(prov)
    assert doc["gate"]["pass"] and doc["gate"]["gate_drift"] == "0.0230"
    assert doc["gate"]["mean_abs_diff"] == 0.0
    assert set(doc["drifts"]) == {"0.00", "0.0230"} and len(doc["cells"]) == 8
    for c in doc["cells"]:
        assert set(c["classes"]) == {"linear", "rf", "mlp"}
        assert c["classes"]["linear"]["target"] == c["recorded_target"]
        for v in c["classes"].values():
            assert np.isfinite(v["target"]) and len(v["shuffled_draws"]) == 8
    s = doc["summary"]["0.0230"]
    assert set(s["arms"]["survival"]) == {"linear", "rf", "mlp"}
    assert set(s["class_minus_linear"]["untrained"]) == {"rf", "mlp"}
    assert set(s["survival_minus"]) == {"untrained"}


def test_a_reference_off_by_1e6_reports_nothing(tmp_path):
    run = _run_dir(tmp_path, shift=(("0.0230", 1, "survival"), 1e-6))
    out = tmp_path / "out.json"
    argv = ["--states-dir", str(run / "states"), "--arms", *ARMS, "--drifts", "0.00", "0.023",
            "--out", str(out)]
    assert nl.main(argv) == 1 and not out.exists()
    assert nl.main([*argv, "--dry-run"]) == 1


def test_a_missing_drift_names_the_dir_and_the_drift(tmp_path):
    run = _run_dir(tmp_path, tags=("0.00",))
    with pytest.raises(SystemExit, match=r"0\.023 .*d0\.0230.*corrected_l1_heldout"):
        nl.main(["--states-dir", str(run / "states"), "--arms", *ARMS, "--drifts", "0.00", "0.023",
                 "--out", str(tmp_path / "o.json")])


def test_the_dump_pattern_is_the_historical_one():
    import rf_probe_historical_h8 as rf
    assert nl.FNAME is rf.FNAME
    m = nl.FNAME.search(nl.dump_name("0.0230", 3, "survival"))
    assert (m["drift"], m["seed"], m["agent"]) == ("0.0230", "3", "survival")
    assert nl.FNAME.search("states_d0.00_s0_predictor_cg.npz") is None
