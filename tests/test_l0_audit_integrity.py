"""The L0 audit's integrity gate and the surrogate export's stack record (agent-free).

scripts/run_l0_audit.py retrains G and rescores saved agents. Its standard world-sample
pair must reproduce the drift-0 survival targets the run recorded, or the readout is void.
The committed C1 audit and C1 cells let the gate be checked here without any agent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")

import run_l0_audit as la  # noqa: E402

C1_CELLS = ROOT / "artifacts" / "corrected_runs" / "corrected_l3_h8_wm" / "cells"
C1_AUDIT = ROOT / "artifacts" / "l0_audit" / "corrected_l3_h8_wm.json"


def test_committed_c1_audit_reproduces_the_recorded_drift0_targets_bit_for_bit():
    rows = json.loads(C1_AUDIT.read_text(encoding="utf-8"))["l0"]
    recorded = la.recorded_l0_targets(str(C1_CELLS))
    assert sorted(recorded) == list(range(10))
    gate = la.integrity_gate(rows, recorded, 0.01)
    assert gate["pass"] is True and gate["n_checked"] == 10
    assert gate["bit_identical"] is True and gate["worst_abs_dev"] == 0.0


def test_integrity_gate_fails_beyond_tolerance_and_reports_missing_cells():
    rows = [{"seed": 0, "rows": [{"target": 0.60}]}, {"seed": 1, "rows": [{"target": 0.55}]}]
    gate = la.integrity_gate(rows, {0: 0.585, 1: 0.55}, 0.01)
    assert gate["pass"] is False and gate["worst_abs_dev"] == pytest.approx(0.015)
    gate = la.integrity_gate(rows, {0: 0.595, 1: 0.55}, 0.01)
    assert gate["pass"] is True and gate["bit_identical"] is False
    assert la.integrity_gate(rows, {}, 0.01)["pass"] is None
    assert la.integrity_gate(rows, {7: 0.5}, 0.01)["pass"] is False      # nothing comparable


def test_surrogate_export_records_its_stack_and_claims_a_tolerance():
    import export_surrogates as ex
    st = ex.stack_record()
    assert set(st) == {"python", "torch", "numpy", "os", "torch_threads"}
    assert isinstance(st["torch_threads"], int)
    doc = ex.__doc__
    assert "exactly" not in doc and "atol=1e-6" in doc
