"""Closing items X05 and X06 (E07, E14): scripts/run_mdl_probe.py --mode {mdl,sparse}.

A separable synthetic set compresses and its full-n AUROC is grouped_auroc's to the bit;
shuffled labels code near the uniform length; four informative columns of 192 give k65 <= 4;
and no prefix boundary of the coding order splits a twin pair. A tmp states dir with its own
cells runs end to end in both modes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_mdl_probe as mdl  # noqa: E402
from itasorl.experiment_a import grouped_auroc  # noqa: E402
from itasorl.experiment_b import episode_features  # noqa: E402

N = 110


def _pools(seed: int, shift: float, hid: int = 6, steps: int = 4):
    rng = np.random.default_rng(seed)
    Ha = rng.standard_normal((N, steps, hid)).astype(np.float32)
    Hs = rng.standard_normal((N, steps, hid)).astype(np.float32)
    Hs[:, :, 0] += shift
    return Ha, Hs


def _labels():
    return np.r_[np.zeros(N), np.ones(N)].astype(int)


def test_separable_set_compresses_and_full_n_auroc_is_grouped_auroc(tmp_path):
    Ha, Hs = _pools(0, 2.0)
    p = tmp_path / "states_d0.45_s0_survival.npz"
    np.savez(p, Ha=Ha, Hs=Hs)
    X = episode_features(np.concatenate([Ha, Hs]))
    y = _labels()
    ref = grouped_auroc(X, y, np.arange(2 * N))
    row = mdl.run_cell({"path": str(p), "drift": "0.45", "seed": 0, "arm": "survival", "mode": "mdl",
                        "prefixes": list(mdl.PREFIXES), "reference": ref, "tolerance": mdl.TOL_EXACT})
    assert row["gate"] == "PASS"
    assert row["full_n_auroc"] == ref                       # to the bit
    assert row["compression"] > 1.0
    assert row["prefixes_used"] == list(mdl.PREFIXES)
    assert row["blocks"][0]["coder"] == "uniform" and row["blocks"][0]["bits"] == 12.0
    assert row["curve"]["176"] == ref                       # the full training fold is the registered fit


def test_shuffled_labels_code_within_two_bits_of_uniform():
    # Features that carry nothing about the label: the probe reduces to its intercept, which
    # a pair-balanced prefix pins at 0.5. (A 192-feature noise input with the registered C = 1
    # probe codes well above uniform, about +160 to +220 bits at n = 220: overfitting cost.)
    rng = np.random.default_rng(5)
    g = mdl.twin_groups(N, N)
    X = np.ones((2 * N, 3))
    excess = []
    for _ in range(20):
        y = _labels()
        flip = rng.random(N) < 0.5                           # shuffle within each pair
        y[:N][flip], y[N:][flip] = 1, 0
        r = mdl.prequential(X, y, g)
        assert r["uniform_bits"] == 2 * N
        excess.append(r["code_bits"] - r["uniform_bits"])
    assert max(abs(e) for e in excess) <= 2.0


def test_four_of_192_informative_columns_give_k65_at_most_4():
    rng = np.random.default_rng(0)
    y = _labels()
    X = rng.standard_normal((2 * N, 192))
    X[:, :4] += 0.8 * y[:, None]
    path = mdl.sparse_path(X, y, mdl.twin_groups(N, N), mdl.PENALTIES[:5])
    assert all(r["converged"] for r in path)
    k = mdl.k65(path)
    assert k is not None and k <= 4


@pytest.mark.parametrize("n_auth,n_surr", [(110, 110), (108, 110), (103, 97)])
def test_no_prefix_boundary_separates_a_twin_pair(n_auth, n_surr):
    g = mdl.twin_groups(n_auth, n_surr)
    order, bounds = mdl.coding_order(g)
    assert sorted(order.tolist()) == list(range(n_auth + n_surr))
    cuts = mdl.snap(mdl.PREFIXES, bounds)
    if n_auth == n_surr == N:
        assert cuts == list(mdl.PREFIXES)
    for b in cuts:
        inside = set(g[order[:b]].tolist())
        outside = set(g[order[b:]].tolist())
        assert not inside & outside, b                       # both members of every pair in one block


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "tmp_run"
    (run / "states").mkdir(parents=True)
    (run / "cells").mkdir()
    y = _labels()
    for s in (0, 1):
        agents = {}
        for k, arm in enumerate(mdl.ARMS):
            Ha, Hs = _pools(10 * s + k, 1.5 if arm == "survival" else 0.0)
            np.savez(run / "states" / f"states_d0.0230_s{s}_{arm}.npz", Ha=Ha, Hs=Hs)
            X = episode_features(np.concatenate([Ha, Hs]))
            agents[arm] = {"pool": {"target": float(grouped_auroc(X, y, np.arange(2 * N)))}}
        (run / "cells" / f"cell_d0.0230_s{s}.json").write_text(
            json.dumps({"cell": {"drift": 0.023, "seed": s, "agents": agents}}))
    return run


def test_end_to_end_both_modes_and_the_drift_tag_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(mdl, "PENALTIES", mdl.PENALTIES[2:5])
    run = _run_dir(tmp_path)
    out = tmp_path / "out"
    for mode in ("mdl", "sparse"):
        assert mdl.main(["--mode", mode, "--states", str(run / "states"), "--drifts", "0.023",
                         "--out-dir", str(out)]) == 0
        doc = json.loads((out / f"{mode}_tmp_run.json").read_text())
        assert doc["provenance"]["git_commit_at_promotion"]
        assert "git_commit" not in doc["provenance"]
        assert doc["drifts"] == ["0.0230"]
        assert {r["gate"] for r in doc["rows"]} == {"PASS"}
        assert len(doc["rows"]) == 6
    with pytest.raises(SystemExit, match="no dump for any listed drift"):
        mdl.main(["--mode", "mdl", "--states", str(run / "states"), "--drifts", "0.45",
                  "--out-dir", str(out), "--dry-run"])
    (run / "states" / "states_d0.0230_s0_survival.npz").rename(run / "states" / "states_d0.023_s0_survival.npz")
    with pytest.raises(SystemExit, match="spelled"):
        mdl.main(["--mode", "mdl", "--states", str(run / "states"), "--drifts", "0.023",
                  "--out-dir", str(out), "--dry-run"])
