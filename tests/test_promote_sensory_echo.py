"""Round-trip of the sensory-echo promotion on a tiny synthetic bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_sensory_echo as pse  # noqa: E402


def test_promote_round_trip(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    cells = [{"drift": "0.45", "seed": s, "agent": "survival", "n_auth": 110, "n_surr": 110,
              "integrity_match": True, **{m: 0.7 for m in pse.METRICS}, "extra": "dropped"}
             for s in range(3)]
    agg = {"hidden": 8, "g_seed": 0, "n_eps": 110, "steps": 24, "quick": False, "device": "cuda",
           "bar": 0.65, "margin": 0.05,
           "integrity": {"checked": True, "mismatches": [], "all_match": True},
           "d=0.45 survival": {"resid_obs": {"mean": 0.7, "per_seed": [0.7] * 3}},
           "decision": {"survival_resid_obs": 0.7, "untrained_resid_obs": 0.5, "zone": "SENSORY-INDEPENDENT"}}
    (run / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    (run / "aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    out = pse.promote(str(run), str(tmp_path / "art" / "x.json"), head="abc123")
    back = json.loads((tmp_path / "art" / "x.json").read_text(encoding="utf-8"))
    assert back == out
    assert back["git_commit_at_promotion"] == "abc123"
    assert back["aggregate"]["d=0.45 survival"]["resid_obs"]["mean"] == 0.7
    assert back["decision"]["zone"] == "SENSORY-INDEPENDENT"
    assert len(back["cells"]) == 3 and "extra" not in back["cells"][0]
    assert back["integrity"]["all_match"] is True


def test_balanced_run_promotes_pair_bookkeeping_and_balanced_decision(tmp_path):
    """A balanced aggregate (spec 2026-10-13) carries its own spec, the run's agents
    directory, the stack, the per-cell integrity gap and decision_balanced."""
    run = tmp_path / "fullruns" / "balanced_confirmation_touch" / "sensory_echo"
    run.mkdir(parents=True)
    cells = [{"drift": "0.45", "seed": 10, "agent": "untrained", "n_auth": 110, "n_surr": 110,
              "n_pairs": 110, "dropped_pairs": 0, "identical_first_state": True,
              "recorded_target": 0.5938, "target_abs_diff": 0.0, "integrity_match": True,
              **{m: 0.55 for m in pse.METRICS}, "extra": "dropped"}]
    agg = {"hidden": 8, "g_seed": 0, "n_eps": 110, "steps": 24, "quick": False, "device": "cuda",
           "bar": 0.65, "margin": 0.05, "mode": "balanced", "balanced_seed_base": 910000,
           "objective": "touch", "mortal": False, "stack": {"torch": "2.7.0+cu126"},
           "integrity": {"checked": True, "mismatches": [], "all_match": True},
           "d=0.45 untrained": {"resid_obs": {"mean": 0.55, "per_seed": [0.55]}},
           "decision_balanced": {"floor_explained": True, "zone": "SENSORY-INDEPENDENT under matched worlds"}}
    (run / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    (run / "aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    out = pse.promote(str(run), str(tmp_path / "o.json"), head="abc")
    assert out["spec"].endswith("2026-10-13-balanced-sensory-echo-design.md")
    assert "balanced_confirmation_touch/agents" in out["agents"]
    assert out["mode"] == "balanced" and out["objective"] == "touch" and out["mortal"] is False
    assert out["balanced_seed_base"] == 910000 and out["stack"]["torch"] == "2.7.0+cu126"
    assert out["decision"] is None
    assert out["decision_balanced"]["zone"].startswith("WITHHELD")
    assert out["decision_balanced"]["pass_bar"] is False
    cell = out["cells"][0]
    assert cell["recorded_target"] == 0.5938 and cell["n_pairs"] == 110 and "extra" not in cell


def _balanced_cells(resid_obs=0.76, untrained=0.52):
    cells = []
    for seed in range(10, 20):
        for arm, level in (("survival", resid_obs), ("predictor", 0.60), ("untrained", untrained)):
            cells.append({
                "drift": "0.45", "seed": seed, "agent": arm, "n_auth": 110, "n_surr": 110,
                "n_pairs": 110, "dropped_pairs": 0, "identical_first_state": True,
                "recorded_target": 0.70, "target_abs_diff": 0.0, "integrity_match": True,
                **{metric: level for metric in pse.METRICS},
            })
    return cells


def test_complete_balanced_bundle_keeps_the_recorded_pass(tmp_path):
    run = tmp_path / "fullruns" / "balanced_confirmation_wm" / "sensory_echo"
    run.mkdir(parents=True)
    cells = _balanced_cells()
    agg = {"hidden": 8, "g_seed": 0, "n_eps": 110, "steps": 24, "quick": False, "device": "cuda",
           "bar": 0.65, "margin": 0.05, "mode": "balanced", "balanced_seed_base": 910000,
           "objective": "survival", "mortal": True, "stack": {"torch": "2.7.0+cu126"},
           "integrity": {"checked": True, "mismatches": [], "all_match": True},
           "decision_balanced": {"zone": "SENSORY-INDEPENDENT under matched worlds (rule passes)"}}
    (run / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    (run / "aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    out = pse.promote(str(run), str(tmp_path / "o.json"), head="abc")
    assert out["decision_balanced"]["zone"].startswith("SENSORY-INDEPENDENT")
    assert out["decision_balanced"]["pass_bar"] is True
    assert abs(out["decision_balanced"]["trained_resid_obs"] - 0.76) < 1e-9
    assert out["integrity"]["adjudication_blockers"] == []


def test_balanced_promotion_rejects_a_passing_zone_on_a_nan_cell(tmp_path):
    run = tmp_path / "fullruns" / "balanced_confirmation_touch" / "sensory_echo"
    run.mkdir(parents=True)
    cells = _balanced_cells()
    cells[0]["resid_obs"] = float("nan")
    agg = {"hidden": 8, "g_seed": 0, "n_eps": 110, "steps": 24, "quick": False, "device": "cuda",
           "bar": 0.65, "margin": 0.05, "mode": "balanced", "balanced_seed_base": 910000,
           "objective": "touch", "mortal": False, "stack": {"torch": "2.7.0+cu126"},
           "integrity": {"checked": True, "mismatches": [], "all_match": True},
           "decision_balanced": {"pass_bar": True, "pass_margin": True,
                                 "zone": "SENSORY-INDEPENDENT under matched worlds (rule passes)"}}
    (run / "cells.json").write_text(json.dumps(cells), encoding="utf-8")
    (run / "aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    out = pse.promote(str(run), str(tmp_path / "o.json"), head="abc")
    assert out["decision_balanced"]["zone"].startswith("WITHHELD")
    assert out["decision_balanced"]["pass_bar"] is False


def test_agents_line_follows_the_run_capacity(tmp_path):
    """A hidden 7 run reads the hidden 7 held-out bundles; the artifact must not say hidden 8."""
    run = tmp_path / "run"
    run.mkdir()
    agg = {"hidden": 7, "g_seed": 0, "n_eps": 110, "steps": 24, "quick": False, "device": "cuda",
           "bar": 0.65, "margin": 0.05,
           "integrity": {"checked": True, "mismatches": [], "all_match": True},
           "d=0.45 survival": {"resid_obs": {"mean": 0.684, "per_seed": [0.684] * 3}},
           "decision": {"survival_resid_obs": 0.684, "untrained_resid_obs": 0.549, "zone": "SENSORY-INDEPENDENT"}}
    (run / "cells.json").write_text("[]", encoding="utf-8")
    (run / "aggregate.json").write_text(json.dumps(agg), encoding="utf-8")
    out = pse.promote(str(run), str(tmp_path / "o.json"), head="abc")
    assert "l3_h7_heldout" in out["agents"]
    assert "hidden=7" in out["surrogate"]
