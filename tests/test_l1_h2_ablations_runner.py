"""Unit tests for L1 H2 ablation runner helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_l1_h2_ablations as rh  # noqa: E402


def test_parse_agent_filename_subcent_delta():
    d, s, arm = rh.parse_agent_filename("agent_d0.0230_s3_survival.pt")
    assert abs(d - 0.023) < 1e-9
    assert s == 3 and arm == "survival"


def test_spearman_rho_perfect_mono():
    assert abs(rh.spearman_rho([0.0, 0.25, 0.5, 1.0], [0.4, 0.5, 0.6, 0.7]) - 1.0) < 1e-9


def test_gn_verdict_supported():
    v = rh.gn_verdict(0.54, 0.52)
    assert v["gn_verdict"] == "H2_SUPPORTED"
    assert v["gn_rule_pass"] is False


def test_gn_verdict_negative():
    v = rh.gn_verdict(0.70, 0.52)
    assert v["gn_verdict"] == "H2_NEGATIVE"
    assert v["gn_rule_pass"] is True


def test_gn_verdict_partial():
    v = rh.gn_verdict(0.60, 0.50)
    assert v["gn_verdict"] == "PARTIAL"
