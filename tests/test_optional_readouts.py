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
    # The anonymized supplement ships docs/optional too (2026-10-16 review): the manuscript cites
    # O2, so this test runs, and must pass, inside an extracted archive as in a checkout.
    assert {"o1_fold_local_speed.json", "o1_fold_local_speed_matched.json",
            "o2_long_history_ridge.json"} <= {p.name for p in paths}
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        block = doc.get("provenance") or {}
        assert block.get("generated_by", "").startswith("scripts/run_optional_o"), path.name
        assert "not an adjudication" in block.get("status", ""), path.name


def test_o1_summary_uses_seed_blocks_as_its_uncertainty_unit():
    """Rows share seeds, arms and baseline readings, so they are not independent
    replications (2026-10-14 review, C05). The interval is taken over per-seed block means."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from run_optional_o1 import summarize
    rows = []
    for seed, shift in ((0, 0.001), (1, 0.003)):
        for k in range(4):
            rows.append({"run": "B1", "population": "pooled", "drift": 0.45, "seed": seed, "arm": "survival",
                         "call_changed": False, "recorded_global": 0.6, "recomputed_global": 0.6,
                         "fold_local": 0.6 + shift + (0.0005 if k % 2 else -0.0005)})
    out = summarize(rows)
    assert out["unit"] == "seed" and out["n_blocks"] == 2 and out["n"] == 8
    assert abs(out["fold_minus_global_mean"] - 0.002) < 1e-12
    assert abs(out["block_means"]["0"] - 0.001) < 1e-12 and abs(out["block_means"]["1"] - 0.003) < 1e-12
    lo, hi = out["fold_minus_global_t90"]
    assert lo < 0.002 < hi and hi - lo > 0.002   # two blocks: a wide t interval, not the row-level one
