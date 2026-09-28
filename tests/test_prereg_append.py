"""scripts/reviewer_gaps/prereg_append.py: default wording unchanged, new flags honored."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reviewer_gaps"))

import prereg_append as pa  # noqa: E402

ROWS = [{"hidden": 8, "oracle_auroc": 0.9278, "in_band": True, "mech_leak_pass": True,
         "floor": 0.4844, "floor_ok": True, "passes_gate0": True}]


def test_default_wording_is_the_second_instance_entry():
    s = pa.entry_text(ROWS, "1", "8", "2026-09-27")
    assert s.startswith("- **2026-09-27 - SECOND FINGERPRINT INSTANCE, GATE 0 (G seed 1; spec "
                        "`docs/specs/2026-09-26-l3-second-fingerprint-instance-design.md`")
    assert "frozen order 8, 7, 9, 10" in s
    assert "hidden=8: oracle 0.928 (in band True), mech leak pass, floor 0.484 (ok) -> PASS" in s


def test_flags_override_title_spec_and_order():
    s = pa.entry_text(ROWS, "2", "8", "2026-09-28", title="HIDDEN-8 NEW-SEED INSTANCE",
                      spec="docs/specs/x.md", order="G seeds 2, 3, 4 at hidden 8")
    assert "HIDDEN-8 NEW-SEED INSTANCE, GATE 0 (G seed 2; spec `docs/specs/x.md`" in s
    assert "frozen order G seeds 2, 3, 4 at hidden 8" in s
    none = pa.entry_text(ROWS, "3", "NONE", "2026-09-28")
    assert "No candidate at G seed 3 passed gate 0" in none
