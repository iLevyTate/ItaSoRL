"""Plumbing of the prospective balanced confirmation (spec
docs/specs/2026-10-20-prospective-balanced-confirmation-design.md)."""
import importlib.util
import os
import sys

import numpy as np

from itasorl import post_review
from itasorl.behavior_audit import sensory_residual_probe_auroc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
_spec = importlib.util.spec_from_file_location("ase_pb", os.path.join(ROOT, "scripts", "audit_sensory_echo.py"))
ase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ase)


def test_default_population_is_the_2026_10_13_one():
    assert ase.balanced_spec_keys(range(10, 20)) == ase.BALANCED_SPEC_KEYS


def test_prospective_population_is_seeds_20_to_29():
    keys = ase.balanced_spec_keys(range(20, 30))
    assert len(keys) == 30
    assert {k[1] for k in keys} == set(range(20, 30))
    assert not keys & ase.BALANCED_SPEC_KEYS


def test_integrated_mlp_metric_is_listed():
    assert "resid_obs_int_mlp" in ase.METRICS
    assert "resid_obs_int_mlp" not in ase.REQUIRED_METRICS


def test_pb_bases_are_registered_and_fresh():
    names = {name: base for name, base, _ in post_review.REGISTERED_LATER}
    assert names["PB-primary"] == 3_000_000 and names["PB-draw2"] == 3_100_000
    pb = tuple(e for e in post_review.REGISTERED_LATER if e[0].startswith("PB-"))
    others = post_review.all_bases_in_use() + tuple(
        e for e in post_review.REGISTERED_LATER if not e[0].startswith("PB-"))
    assert post_review.base_collisions(new=pb, in_use=others) == []


def test_draw2_read_base_passes_the_fresh_read_check():
    assert ase.read_base_collisions(3_100_000, 110) == []


def test_integrated_mlp_control_runs_and_is_bounded():
    rng = np.random.default_rng(0)
    n, T, obs, hid = 40, 6, 5, 4
    Ot = rng.normal(size=(n, T, obs))
    y = np.repeat([0, 1], n // 2)
    H = rng.normal(size=(n, T, hid)) + 0.5 * y[:, None, None]
    auc = sensory_residual_probe_auroc(H, Ot, y, integrated=True, nonlinear=True)
    assert 0.0 <= auc <= 1.0
