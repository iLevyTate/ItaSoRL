"""Fold-local speed thresholds and grouped inner selection of a ridge penalty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from itasorl import folds
from itasorl.optional_readouts import long_history_ridge, speed_threshold_probes


def test_fold_local_median_does_not_use_the_held_out_episode():
    rng = np.random.default_rng(0)
    n = 20
    speeds = np.linspace(0.2, 1.0, n)
    speeds[-1] = 50.0
    features = np.column_stack([speeds, rng.normal(size=n)])
    groups = np.arange(n)
    out = speed_threshold_probes(features, speeds, groups)
    paired = list(zip(out["train_medians"], folds.split(groups), strict=True))
    assert paired
    for median, (train, _test) in paired:
        assert median == float(np.median(speeds[train]))
    held_out = [median for median, (train, test) in paired if n - 1 in test]
    assert held_out and max(held_out) < 50.0


def test_a_flat_speed_channel_is_not_a_score():
    features = np.zeros((12, 2))
    out = speed_threshold_probes(features, np.ones(12), np.arange(12))
    assert not np.isfinite(out["global"]) and not np.isfinite(out["fold_local"])


def test_pair_groups_stay_together_in_the_speed_split():
    n_pairs = 20
    pair = np.arange(n_pairs)
    groups = np.r_[pair, pair]
    speeds = np.r_[np.linspace(0.2, 1.2, n_pairs), np.linspace(0.25, 1.25, n_pairs)]
    features = np.column_stack([speeds, np.zeros(len(speeds))])
    out = speed_threshold_probes(features, speeds, groups)
    assert np.isfinite(out["global"]) and np.isfinite(out["fold_local"])
    for train, test in folds.split(groups):
        assert set(groups[train]).isdisjoint(groups[test])


def test_inner_ridge_selection_stays_inside_the_outer_training_fold():
    rng = np.random.default_rng(1)
    n, steps, hidden, width = 30, 4, 2, 12
    signal = rng.normal(size=(n, steps, 1))
    H = np.repeat(signal, hidden, axis=2) + 0.01 * rng.normal(size=(n, steps, hidden))
    noise = rng.normal(size=(n, steps, width))
    phi = np.concatenate([signal, noise], axis=2).reshape(n * steps, -1)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    out = long_history_ridge(H, phi, y, alphas=(1e-6, 1e-2, 1.0, 100.0))
    assert out["n_folds"] == 5
    assert np.isfinite(out["auroc"]) and np.isfinite(out["fixed_alpha"]["auroc"])
    assert out["nuisance_r2_heldout"] >= out["fixed_alpha"]["nuisance_r2_heldout"] - 1e-6


def test_provenance_block_names_the_script_and_disclaims_adjudication():
    from itasorl.optional_readouts import provenance
    block = provenance("scripts/run_optional_o1.py", "python scripts/run_optional_o1.py --pooled",
                       {"runs": ["fullruns/balanced_confirmation_wm"]}, device="cpu")
    assert block["generated_by"] == "scripts/run_optional_o1.py"
    assert block["command"] == "python scripts/run_optional_o1.py --pooled"
    assert block["inputs"] == {"runs": ["fullruns/balanced_confirmation_wm"]}
    assert "not an adjudication" in block["status"]
    assert block["stack"]["device"] == "cpu"
    assert set(block) >= {"generated_by", "command", "inputs", "status", "git_head", "stack", "written"}


def test_every_optional_readout_file_carries_its_provenance():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "docs" / "optional").glob("*.json"))
    assert paths
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        block = doc.get("provenance") or {}
        assert block.get("generated_by", "").startswith("scripts/run_optional_o"), path.name
        assert "not an adjudication" in block.get("status", ""), path.name
