"""The sensory-echo runner's aggregation and frozen decision rule are pure
functions of the saved cells, and their output must be JSON-serializable
(the first full run finished all 60 cells and then failed on a NumPy bool in
the summary; this pins the fix)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_sensory_echo as ase  # noqa: E402


def _cells(surv=0.72, untr=0.50, n=10):
    rng = np.random.default_rng(0)
    out = []
    for d in ("0.00", "0.45"):
        for s in range(n):
            for arm, base in (("survival", surv), ("untrained", untr), ("predictor", 0.57)):
                lvl = base if d == "0.45" else 0.5
                cell = {"drift": d, "seed": s, "agent": arm, "n_auth": 110, "n_surr": 110,
                        "integrity_match": np.bool_(True)}
                for met in ase.METRICS:
                    cell[met] = float(np.clip(lvl + 0.02 * rng.normal(), 0, 1))
                out.append(cell)
    return out


def _agg(cells):
    return ase.aggregate(cells, [0.45, 0.0], ["survival", "untrained", "predictor"], hidden=8,
                         g_seed=0, n_eps=110, steps=24, quick=False, device="cpu", mismatches=[])


def test_aggregate_is_json_serializable_and_passes_rule():
    agg = _agg(_cells())
    json.dumps(agg)                                   # must not raise
    assert agg["decision"]["pass_bar"] is True and agg["decision"]["pass_margin"] is True
    assert agg["decision"]["zone"].startswith("SENSORY-INDEPENDENT")
    assert agg["d=0.45 survival"]["resid_obs"]["n_seeds"] == 10


def test_rule_zones():
    assert _agg(_cells(surv=0.62))["decision"]["zone"].startswith("ATTENUATED")
    assert _agg(_cells(surv=0.55))["decision"]["zone"].startswith("LARGELY")
    # above the bar but not above untrained + margin: rule fails, zone is attenuated-or-worse
    agg = _agg(_cells(surv=0.66, untr=0.63))
    assert agg["decision"]["pass_bar"] and not agg["decision"]["pass_margin"]
