"""Round-trip tests for scripts/promote_bv3_gates.py on a tiny synthetic B-v3 bundle.

Contract under test: the gate artifact copies the bundle's per-seed lists and per-cell
engagement / matched-pair leakage receipts verbatim, and the four gate verdicts follow
PREREGISTRATION_Bv3 section 7 exactly (engagement all-engaged; L0 TOST at margin 0.05;
speed probe >= 0.75 in every pool; mp leakage clean in every cell, with each exception
named by drift / seed / arm / worst channel)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_bv3_gates as pb  # noqa: E402

ARMS = ("survival", "untrained", "predictor")
SEEDS = (0, 1, 2)


def _mp(target: float, reward_sum: float) -> dict:
    dev = abs(reward_sum - 0.5)
    return {"target": target, "shuffled": 0.5, "speed": 0.95, "n": 120,
            "leakage_clean": dev <= 0.1, "leakage_max_dev": dev,
            "leakage": {"reward_sum": reward_sum, "length": 0.5, "lifetime": 0.5,
                        "max_abs_dev": dev, "margin": 0.1, "clean": dev <= 0.1},
            "n_pairs": 60}


def _bundle(run_dir: Path, leak_seed: int | None = 1, low_speed: bool = False) -> dict:
    (run_dir / "artifacts" / "cells").mkdir(parents=True)
    results = {}
    for dk, base in (("0.0", 0.50), ("0.45", 0.61)):
        results[dk] = {}
        for a_i, arm in enumerate(ARMS):
            tgt = [base - 0.03 * a_i + 0.005 * s for s in SEEDS]
            speed = [0.7 if (low_speed and arm == "untrained" and s == 2) else 0.9 + 0.01 * s
                     for s in SEEDS]
            mp_clean = [not (dk == "0.45" and arm == "survival" and s == leak_seed) for s in SEEDS]
            results[dk][arm] = {
                "pool_target": tgt, "pool_target_lo": [v - 0.07 for v in tgt],
                "pool_target_hi": [v + 0.07 for v in tgt], "pool_target_var": tgt,
                "pool_target_full": tgt, "pool_selectivity": [0.0] * 3,
                "pool_selectivity_var": [0.0] * 3, "pool_selectivity_full": [0.0] * 3,
                "pool_speed": speed, "pool_shuffled": [0.5] * 3,
                "pool_anchor_energy": [0.8] * 3, "pool_anchor_food": [0.8] * 3,
                "pool_ceiling_drag": [float("nan")] * 3 if dk == "0.0" else [0.5] * 3,
                "mp_target": [0.7] * 3, "mp_leak_clean": mp_clean, "xeval_return": []}
    (run_dir / "artifacts" / "expB2_results.json").write_text(json.dumps(results), encoding="utf-8")
    for dk in results:
        for s in SEEDS:
            agents = {}
            for arm in ARMS:
                bad = dk == "0.45" and arm == "survival" and s == leak_seed
                agents[arm] = {"pool": {"target": results[dk][arm]["pool_target"][s],
                                        "speed": results[dk][arm]["pool_speed"][s],
                                        "n": 220, "n_auth": 110, "n_surr": 110,
                                        "too_few_survivors": False},
                               "mp": _mp(0.7, 0.393 if bad else 0.52)}
            cell = {"fingerprint": "356be26d94199878", "git_commit": "820849f",
                    "cell": {"drift": float(dk), "seed": s,
                             "eng": {"trained_return": -0.2, "random_return": -1.0,
                                     "scripted_return": -1.4, "trained_len": 74.0,
                                     "random_len": 69.0, "scripted_len": 63.0,
                                     "better_return": True, "not_worse_life": True, "engaged": True},
                             "xeval": {"0.00": 0.2, dk: 0.1}, "agents": agents}}
            (run_dir / "artifacts" / "cells" / f"cell_d{float(dk):.2f}_s{s}.json").write_text(
                json.dumps(cell), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "20260707_015006", "git_commit": "820849f", "quick": False,
         "environment": {"python": "3.13.2", "cuda_available": True}}), encoding="utf-8")
    (run_dir / "b2_flags.json").write_text(json.dumps(["--seeds", "0", "1", "2", "--drift-mode", "regime"]),
                                           encoding="utf-8")
    return results


def test_round_trip_with_one_leakage_exception(tmp_path: Path):
    run_dir = tmp_path / "fullruns" / "07062026"
    results = _bundle(run_dir, leak_seed=1)
    out = tmp_path / "artifacts" / "expB2" / "bv3_n10_gates.json"
    assert pb.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["source_run"] == "fullruns/07062026"
    assert doc["generated_by"] == "scripts/promote_bv3_gates.py"
    assert doc["fingerprint"] == "356be26d94199878"
    assert doc["git_commit_at_run"] == ["820849f"]
    assert doc["run_id"] == "20260707_015006" and doc["environment"]["python"] == "3.13.2"
    assert doc["flags"][-1] == "regime"
    assert doc["drifts"] == ["0.0", "0.45"]
    for dk in results:
        for arm in ARMS:
            src, row = results[dk][arm], doc["arms"][dk][arm]
            for k in pb.RESULT_KEYS:
                if k == "pool_ceiling_drag" and dk == "0.0":
                    assert row[k] == [None] * 3
                else:
                    assert row[k] == src[k], (dk, arm, k)
            assert "pool_selectivity" not in row and "xeval_return" not in row
            agg = row["aggregate"]["pool_target"]
            assert agg["mean"] == pytest.approx(float(np.mean(src["pool_target"])))
            assert agg["n_seeds"] == 3
    # per-cell receipts
    assert len(doc["engagement"]) == 6 and all(e["engaged"] for e in doc["engagement"])
    assert doc["engagement"][0] == {"drift": "0.0", "seed": 0, "trained_return": -0.2,
                                    "random_return": -1.0, "scripted_return": -1.4,
                                    "trained_len": 74.0, "random_len": 69.0, "scripted_len": 63.0,
                                    "better_return": True, "not_worse_life": True, "engaged": True}
    assert len(doc["mp_leakage"]) == 18 and len(doc["pools"]) == 18
    bad = [r for r in doc["mp_leakage"] if not r["leakage_clean"]]
    assert len(bad) == 1 and (bad[0]["drift"], bad[0]["seed"], bad[0]["agent"]) == ("0.45", 1, "survival")
    assert bad[0]["reward_sum"] == 0.393 and bad[0]["margin"] == 0.1
    # gate verdicts per PREREGISTRATION_Bv3 section 7
    g = doc["gates"]
    assert g["engagement"]["pass"] and g["engagement"]["n_engaged"] == 6 and g["engagement"]["n_cells"] == 6
    assert g["engagement"]["per_drift"]["0.45"]["engaged_per_seed"] == [True, True, True]
    assert g["l0_control"]["drift"] == "0.0"
    assert g["l0_control"]["survival_pool_target_per_seed"] == results["0.0"]["survival"]["pool_target"]
    assert g["l0_control"]["tost"]["margin"] == 0.05
    assert isinstance(g["l0_control"]["tost"]["equivalent"], bool)
    assert g["l0_control"]["pass"] == g["l0_control"]["tost"]["equivalent"]
    assert g["speed_positive_control"]["threshold"] == 0.75
    assert g["speed_positive_control"]["min_speed"]["d=0.45 predictor"] == 0.9
    assert g["speed_positive_control"]["pass"] is True
    assert g["leakage"]["n_clean"] == 17 and g["leakage"]["n_cells"] == 18
    assert g["leakage"]["per_arm"]["d=0.45 survival"] == {"n_clean": 2, "n_cells": 3}
    assert g["leakage"]["failures"] == [{"drift": "0.45", "seed": 1, "agent": "survival",
                                         "channel": "reward_sum", "value": 0.393,
                                         "max_abs_dev": pytest.approx(0.107), "margin": 0.1}]
    assert g["leakage"]["pass"] is False and g["all_pass"] is False


def test_all_gates_pass_when_bundle_is_clean(tmp_path: Path):
    run_dir = tmp_path / "fullruns" / "07062026"
    _bundle(run_dir, leak_seed=None)
    out = tmp_path / "out.json"
    assert pb.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    g = json.loads(out.read_text(encoding="utf-8"))["gates"]
    assert g["leakage"]["pass"] is True and g["leakage"]["failures"] == []
    assert g["speed_positive_control"]["pass"] is True
    # the L0 arm in the synthetic bundle sits at 0.50 +- 0.005: TOST accepts at n=3
    assert g["l0_control"]["tost"]["equivalent"] is True
    assert g["all_pass"] is True


def test_speed_gate_fails_on_a_slow_pool(tmp_path: Path):
    run_dir = tmp_path / "fullruns" / "07062026"
    _bundle(run_dir, leak_seed=None, low_speed=True)
    out = tmp_path / "out.json"
    assert pb.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    g = json.loads(out.read_text(encoding="utf-8"))["gates"]
    assert g["speed_positive_control"]["min_speed"]["d=0.45 untrained"] == 0.7
    assert g["speed_positive_control"]["pass"] is False and g["all_pass"] is False


def test_missing_cells_and_mixed_fingerprints(tmp_path: Path):
    empty = tmp_path / "fullruns" / "empty"
    (empty / "artifacts").mkdir(parents=True)
    assert pb.main(["--run-dir", str(empty), "--out", str(tmp_path / "x.json")]) == 1
    run_dir = tmp_path / "fullruns" / "07062026"
    _bundle(run_dir, leak_seed=None)
    p = run_dir / "artifacts" / "cells" / "cell_d0.45_s2.json"
    d = json.loads(p.read_text())
    d["fingerprint"] = "other"
    p.write_text(json.dumps(d))
    cells = [json.loads(q.read_text()) for q in sorted((run_dir / "artifacts" / "cells").glob("*.json"))]
    with pytest.raises(ValueError, match="mixed fingerprints"):
        pb.build_gates_doc(json.loads((run_dir / "artifacts" / "expB2_results.json").read_text()),
                           cells, None, None)


def test_source_label():
    assert pb.source_label("fullruns/07062026") == "fullruns/07062026"
    assert pb.source_label("C:/x/y/fullruns/07062026") == "fullruns/07062026"
    assert pb.source_label("D:\\bundles\\run7") == "D:/bundles/run7"
