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


def _agg_balanced(cells):
    return ase.aggregate(cells, [0.45], ["survival", "untrained", "predictor"], hidden=8,
                         g_seed=0, n_eps=110, steps=24, quick=False, device="cpu", mismatches=[],
                         balanced_seed_base=910_000, objective="touch", mortal=False)


def test_balanced_aggregate_reads_floor_then_trained_clause():
    """Spec 2026-10-13: the floor clause gates the trained clause; the pooled rule and
    the pooled published-reference check are absent in balanced mode."""
    agg = _agg_balanced(_cells(surv=0.72, untr=0.52))
    json.dumps(agg)
    assert agg["mode"] == "balanced" and agg["objective"] == "touch" and agg["mortal"] is False
    assert "decision" not in agg and "published_target" not in agg["integrity"]
    d = agg["decision_balanced"]
    assert d["floor_explained"] is True and d["zone"].startswith("SENSORY-INDEPENDENT")
    assert _agg_balanced(_cells(surv=0.62, untr=0.52))["decision_balanced"]["zone"].startswith("ATTENUATED")
    assert _agg_balanced(_cells(surv=0.55, untr=0.52))["decision_balanced"]["zone"].startswith("LARGELY")
    # an unexplained floor is reported and the trained clause is not read, whatever the trained arm shows
    d = _agg_balanced(_cells(surv=0.80, untr=0.63))["decision_balanced"]
    assert d["floor_explained"] is False and d["zone"].startswith("FLOOR NOT EXPLAINED")


def test_balanced_integrity_gate(tmp_path):
    """The regenerated balanced target must reproduce the run's recorded value; identical
    first states and zero dropped pairs are required too."""
    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    rec = {"cell": {"agents": {"untrained": {"balanced": {"target": 0.6140, "n_pairs": 110}}}}}
    (cells_dir / "cell_d0.45_s10.json").write_text(json.dumps(rec), encoding="utf-8")
    good = {"target": 0.6142, "identical_first_state": True, "dropped_pairs": 0, "n_pairs": 110}
    ok, why = ase.balanced_integrity(str(cells_dir), 0.45, 10, "untrained", good)
    assert ok and why == "" and abs(good["target_abs_diff"] - 0.0002) < 1e-9
    assert good["recorded_target"] == 0.614
    bad = {"target": 0.60, "identical_first_state": True, "dropped_pairs": 0, "n_pairs": 110}
    ok, why = ase.balanced_integrity(str(cells_dir), 0.45, 10, "untrained", bad)
    assert not ok and "recorded" in why
    bad = {"target": 0.614, "identical_first_state": False, "dropped_pairs": 1, "n_pairs": 109}
    ok, why = ase.balanced_integrity(str(cells_dir), 0.45, 10, "untrained", bad)
    assert not ok and "first states" in why and "dropped" in why and "n_pairs" in why


def test_collect_cell_balanced_pairs_share_a_group(monkeypatch):
    """Balanced mode keeps the surviving intersection and gives each pair one group id,
    in the authentic-then-surrogate row order the probes expect."""
    steps, hid, obs_dim = 4, 3, 2

    def fake_pool(agent, norm, params, drift, n_eps, steps_, device, seed_base, ray, **kw):
        kept = np.array([0, 1, 2, 4]) if drift == 0.0 else np.array([1, 2, 3, 4])
        k = len(kept)
        H = np.full((k, steps_, hid), drift, np.float32) + kept[:, None, None]
        H[:, 0] = kept[:, None]                      # shared first state across worlds
        return (H, np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k),
                np.zeros((k, steps_, 7), np.float32), np.zeros((k, steps_, obs_dim), np.float32), kept)

    monkeypatch.setattr(ase, "collect_pool", fake_pool)
    H, Ot, Bt, y, groups, meta, (Ha, Hs) = ase.collect_cell(
        None, None, 0.45, n_eps=5, steps=steps, device="cpu", balanced_seed_base=910_000)
    assert meta == {"n_auth": 4, "n_surr": 4, "n_pairs": 3, "dropped_pairs": 2,
                    "identical_first_state": True}
    assert groups.tolist() == [1, 2, 4, 1, 2, 4] and y.tolist() == [0, 0, 0, 1, 1, 1]
    assert H.shape == (6, steps, hid) and Ot.shape == (6, steps, obs_dim) and Bt.shape == (6, steps, 7)
    assert np.array_equal(Ha[:, 0], Hs[:, 0]) and not np.array_equal(Ha[:, 1], Hs[:, 1])
    # pooled mode: historical bases, every episode its own group
    seen = []
    monkeypatch.setattr(ase, "collect_pool",
                        lambda *a, **kw: (seen.append(a[7]) or fake_pool(*a, **kw)))
    _, _, _, y, groups, meta, _ = ase.collect_cell(
        None, None, 0.45, n_eps=5, steps=steps, device="cpu", balanced_seed_base=None)
    assert seen == [800_000, 850_000] and groups.tolist() == list(range(8))
    assert meta == {"n_auth": 4, "n_surr": 4}
