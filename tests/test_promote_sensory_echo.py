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
