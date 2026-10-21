"""Routes of the prospective balanced promotion (spec
docs/specs/2026-10-20-prospective-balanced-confirmation-design.md), on synthetic run folders."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
_spec = importlib.util.spec_from_file_location(
    "ppb", os.path.join(ROOT, "scripts", "promote_prospective_balanced.py"))
ppb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppb)

SEEDS = list(range(20, 30))
ARMS = ("survival", "predictor", "untrained")


def _gates(speed, leak=0.52):
    return {"population": "matched pairs", "n_pairs": 110, "speed": speed,
            "leakage": {"reward_sum": leak, "pellets_consumed": leak, "length": 0.5, "lifetime": 0.5}}


def make_run(tmp, run_id="PB1", *, speeds=None, cell_speeds=None, untrained_resid=0.58,
             trained_resid=0.76, shuffled=0.5, engaged=True, missing_cell=False, recon_shift=0.0):
    info = ppb.RUNS[run_id]
    rd = tmp / info["dir"]
    (rd / "cells").mkdir(parents=True)
    speeds = speeds or {"survival": 0.9, "predictor": 0.85, "untrained": 0.79}
    cell_speeds = cell_speeds or speeds
    targets = {"survival": 0.80, "predictor": 0.66, "untrained": 0.61}
    for d in (0.0, 0.45):
        for s in SEEDS:
            if missing_cell and d == 0.45 and s == 29:
                continue
            agents = {}
            for a in ARMS:
                agents[a] = {"balanced": {"seed_base": 3_000_000, "n_pairs": 110, "dropped_pairs": 0,
                                          "identical_first_state": True, "shuffled_within_pair": shuffled,
                                          "target": targets[a],
                                          "matched_gates": _gates(cell_speeds[a] if not callable(cell_speeds[a])
                                                                  else cell_speeds[a](s))}}
            cell = {"drift": d, "seed": s, "objective": info["objective"], "mortal": info["mortal"],
                    "eng": {"engaged": engaged}, "agents": agents, "stack": {"device": "cuda"}}
            with open(rd / "cells" / f"cell_d{d:.2f}_s{s}.json", "w") as fh:
                json.dump({"git_commit": "abc", "fingerprint": "fp", "cell": cell}, fh)
    rows = [{"seed": s, "arm": a, "drift": 0.45, "seed_base": 3_000_000, "n_pairs": 110, "dropped_pairs": 0,
             "target": targets[a] + recon_shift,
             "matched_gates": _gates(speeds[a] if not callable(speeds[a]) else speeds[a](s))}
            for s in SEEDS for a in ARMS]
    with open(rd / "matched_gates.json", "w") as fh:
        json.dump({"population": "retained pairs", "seed_base": 3_000_000, "objective": info["objective"],
                   "mortal": info["mortal"], "rows": rows}, fh)
    se = rd / "sensory_echo"
    se.mkdir()
    lvl = {"survival": trained_resid, "predictor": 0.64, "untrained": untrained_resid}
    se_cells = [{"drift": "0.45", "seed": s, "agent": a, "resid_obs": lvl[a] + 0.001 * (s - 24),
                 "identical_first_state": True, "dropped_pairs": 0, "n_pairs": 110}
                for s in SEEDS for a in ARMS]
    with open(se / "cells.json", "w") as fh:
        json.dump(se_cells, fh)

    def block(a, m="resid_obs"):
        vals = [r["resid_obs"] for r in se_cells if r["agent"] == a]
        mean = sum(vals) / len(vals)
        return {"per_seed": vals, "mean": mean, "t90": [mean - 0.02, mean + 0.02]}

    u = block("untrained")["mean"]
    t = block("survival")
    agg = {"integrity": {"all_match": True, "mismatches": [], "adjudication_blockers": []},
           **{f"d=0.45 {a}": {"resid_obs": block(a), **{m: block(a) for m in ppb.S2_METRICS}} for a in ARMS},
           "decision_balanced": {"untrained_resid_obs": u, "untrained_dev": abs(u - 0.5),
                                 "floor_explained": abs(u - 0.5) <= 0.1, "trained_resid_obs": t["mean"],
                                 "trained_t90": t["t90"], "pass_bar": t["mean"] >= 0.65,
                                 "pass_margin": t["mean"] > u + 0.05, "t90_lo_above_untrained": t["t90"][0] > u,
                                 "zone": "x"}}
    with open(se / "aggregate.json", "w") as fh:
        json.dump(agg, fh)
    return rd


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(ppb, "ROOT", str(tmp_path))
    return tmp_path


def test_met_when_every_gate_and_clause_hold(root):
    make_run(root)
    doc = ppb.promote("PB1")
    assert doc["decision"] == "MET", doc["reason"]
    assert doc["S1_strict"]["label"] in ("STRICT MET", "STRICT NOT MET")
    assert doc["S2_controls"]["resid_obs_int_mlp"]["label"] == "SURVIVES"


def test_not_run_when_a_cell_is_missing(root):
    make_run(root, missing_cell=True)
    assert ppb.promote("PB1")["decision"] == "NOT RUN"


def test_one_trained_seed_under_075_fails_gate_2(root):
    make_run(root, speeds={"survival": lambda s: 0.749 if s == 25 else 0.9,
                           "predictor": 0.85, "untrained": 0.79})
    doc = ppb.promote("PB1")
    assert doc["decision"] == "UNINFORMATIVE" and "gate2_speed_ok" in doc["reason"]


def test_one_low_untrained_seed_does_not_fail_gate_2_when_its_mean_holds(root):
    make_run(root, speeds={"survival": 0.9, "predictor": 0.85,
                           "untrained": lambda s: 0.749 if s == 29 else 0.80})
    assert ppb.promote("PB1")["decision"] == "MET"


def test_untrained_mean_under_075_fails_gate_2(root):
    make_run(root, speeds={"survival": 0.9, "predictor": 0.85, "untrained": 0.74})
    assert ppb.promote("PB1")["decision"] == "UNINFORMATIVE"


def test_in_cell_copy_disagreeing_on_a_call_fails_gate_2(root):
    make_run(root, cell_speeds={"survival": lambda s: 0.74 if s == 22 else 0.9,
                                "predictor": 0.85, "untrained": 0.79})
    doc = ppb.promote("PB1")
    assert doc["decision"] == "UNINFORMATIVE"
    assert doc["gates"]["matched"]["in_cell_agrees_speed"] is False


def test_reconstruction_over_tolerance_fails_gates_2_and_3(root):
    make_run(root, recon_shift=0.006)
    doc = ppb.promote("PB1")
    assert doc["decision"] == "UNINFORMATIVE" and not doc["gates"]["matched"]["reconstruction_ok"]


def test_shuffled_past_01_fails_gate_4(root):
    make_run(root, shuffled=0.61)
    assert "gate4_shuffled_ok" in ppb.promote("PB1")["reason"]


def test_engagement_failure_fails_gate_1(root):
    make_run(root, engaged=False)
    assert "gate1_engagement_ok" in ppb.promote("PB1")["reason"]


def test_floor_failure_is_uninformative(root):
    make_run(root, untrained_resid=0.62)
    doc = ppb.promote("PB1")
    assert doc["decision"] == "UNINFORMATIVE" and "floor" in doc["reason"]


def test_trained_clause_failure_is_not_met(root):
    make_run(root, trained_resid=0.63)
    assert ppb.promote("PB1")["decision"] == "NOT MET"


def test_s2_label_gap_closed():
    agg = {"d=0.45 survival": {m: {"mean": 0.66, "t90": [0.6, 0.7]} for m in ppb.S2_METRICS},
           "d=0.45 untrained": {m: {"mean": 0.63} for m in ppb.S2_METRICS}}
    assert all(v["label"] == "ATTENUATED" for v in ppb.s2_labels(agg).values())


def test_pb2_reads_its_own_objective(root):
    make_run(root, "PB2")
    assert ppb.promote("PB2")["decision"] == "MET"
