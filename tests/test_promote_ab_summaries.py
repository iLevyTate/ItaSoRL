"""promote_ab_summaries writes the committed form itself (Q53): repo-relative flags and a
vendor-only GPU name, so a rerun from the bundles reproduces the committed summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_ab_summaries as pab  # noqa: E402


def test_the_gpu_is_named_by_vendor_only():
    env = {"python": "3.13.2", "cuda_available": True, "cuda_device": "NVIDIA GeForce Model X"}
    assert pab.generic_device(env)["cuda_device"] == "NVIDIA GPU"
    assert pab.generic_device({"cuda_available": False}) == {"cuda_available": False}
    assert pab.generic_device(None) is None


def test_committed_summaries_carry_the_form_the_script_writes():
    for rel in ("expB2/bv3_n10_summary.json", "expB2/sysid_ceiling_n10_summary.json"):
        d = json.loads((ROOT / "artifacts" / rel).read_text(encoding="utf-8"))
        assert d["generated_by"] == "scripts/promote_ab_summaries.py"
        assert d["environment"]["cuda_device"] == "NVIDIA GPU"
        assert d["flags"] == pab.rel_flags(d["flags"])
        assert not any(":" in f for f in d["flags"] if isinstance(f, str)), rel
