"""The comparator-trained promoter reads the agent trunk width from the saved agents.

The white-jitter run (artifacts/texture/T_gn_l3_h8_wm.json) ran an 8-unit trunk instead of the
registered 96. Its promoter now records that from the agents themselves, so re-running the
documented command reproduces the deviation note instead of dropping it, and a future run at
the wrong width cannot be promoted as the registered protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_texture_trained as ptt  # noqa: E402


def _agent(path, hidden, whh_hidden=None):
    h = whh_hidden if whh_hidden is not None else hidden
    torch.save({"ctor": {"hidden": hidden, "obs_dim": 4, "act_dim": 2, "embed": 3,
                         "world_model": True, "sysid_aux": False},
                "state_dict": {"cell.weight_hh": torch.zeros(3 * h, h),
                               "cell.weight_ih": torch.zeros(3 * h, 3)},
                "norm": {"mean": None, "var": None, "count": 0}}, path)


def test_trunk_check_reads_every_agent(tmp_path):
    for i in range(3):
        _agent(tmp_path / f"agent_d0.45_s{i}_survival.pt", 96)
    r = ptt.trunk_check(str(tmp_path))
    assert r["verified"] and r["n_agents"] == 3 and r["hidden_values"] == [96]


def test_trunk_check_reports_an_8_unit_trunk_and_refuses_inconsistent_weights(tmp_path):
    _agent(tmp_path / "agent_d0.45_s0_survival.pt", 8)
    assert ptt.trunk_check(str(tmp_path))["hidden_values"] == [8]
    _agent(tmp_path / "agent_d0.45_s1_survival.pt", 96, whh_hidden=8)
    with pytest.raises(SystemExit):
        ptt.trunk_check(str(tmp_path))


def test_missing_agents_leave_the_width_unverified(tmp_path):
    r = ptt.trunk_check(str(tmp_path / "absent"))
    assert r["verified"] is False and r["n_agents"] == 0


def test_the_committed_gn_artifact_records_the_8_unit_trunk_from_its_agents():
    d = json.loads((ROOT / "artifacts" / "texture" / "T_gn_l3_h8_wm.json").read_text(encoding="utf-8"))
    assert d["registered_trunk_hidden"] == ptt.REGISTERED_TRUNK == 96
    assert d["trunk_hidden"] == 8
    assert d["trunk_check"]["verified"] and d["trunk_check"]["n_agents"] == 60
    assert d["trunk_check"]["hidden_values"] == [8]
    assert d["configuration_deviation"].startswith("The agent trunk was 8 units")
    assert d["rule_met"] is False and d["routing"].startswith("UNINFORMATIVE")
    assert d["source_run"] == "fullruns/T_gn_l3_h8_wm/expB2_results.json"
