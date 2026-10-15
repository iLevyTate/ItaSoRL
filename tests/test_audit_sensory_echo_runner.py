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


def _keys(cells):
    return {(c["drift"], int(c["seed"]), c["agent"]) for c in cells}


def _agg(cells, *, expected_keys=None, mismatches=None, quick=False):
    return ase.aggregate(cells, [0.45, 0.0], ["survival", "untrained", "predictor"], hidden=8,
                         g_seed=0, n_eps=110, steps=24, quick=quick, device="cpu",
                         mismatches=[] if mismatches is None else mismatches,
                         expected_keys=_keys(cells) if expected_keys is None else expected_keys)


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


def _prepare_balanced(cells):
    prepared = []
    for cell in cells:
        if cell["drift"] != "0.45":
            continue
        row = dict(cell)
        row.setdefault("n_pairs", 110)
        row.setdefault("dropped_pairs", 0)
        row.setdefault("identical_first_state", True)
        row.setdefault("recorded_target", row["target"])
        row.setdefault("target_abs_diff", 0.0)
        if row.get("integrity_match") is None:
            row["integrity_match"] = True
        prepared.append(row)
    return prepared


def _agg_balanced(cells, *, expected_keys=None, mismatches=None, quick=False):
    prepared = _prepare_balanced(cells)
    return ase.aggregate(prepared, [0.45], ["survival", "untrained", "predictor"], hidden=8,
                         g_seed=0, n_eps=110, steps=24, quick=quick, device="cpu",
                         mismatches=[] if mismatches is None else mismatches,
                         balanced_seed_base=910_000, objective="touch", mortal=False,
                         expected_keys=_keys(prepared) if expected_keys is None else expected_keys)


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
    nan_cell = {"target": float("nan"), "identical_first_state": True, "dropped_pairs": 0, "n_pairs": 110}
    ok, why = ase.balanced_integrity(str(cells_dir), 0.45, 10, "untrained", nan_cell)
    assert not ok and "finite" in why
    rec["cell"]["agents"]["untrained"]["balanced"]["target"] = float("nan")
    (cells_dir / "cell_d0.45_s10.json").write_text(json.dumps(rec), encoding="utf-8")
    ok, why = ase.balanced_integrity(str(cells_dir), 0.45, 10, "untrained", good)
    assert not ok and "finite" in why


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


def test_incomplete_or_unchecked_cells_cannot_pass():
    """A NaN residual, a missing or duplicate seed, a mismatch list, or a quick run
    keeps its scores and does not receive a passing zone."""
    full = _prepare_balanced(_cells(surv=0.80, untr=0.52))
    expected = _keys(full)

    broken = [dict(cell) for cell in full]
    broken[0]["resid_obs"] = float("nan")
    withheld = _agg_balanced(broken, expected_keys=expected)
    assert withheld["decision_balanced"]["zone"].startswith("WITHHELD")
    assert withheld["decision_balanced"]["pass_bar"] is False
    assert "mean" not in withheld["d=0.45 survival"]["resid_obs"]

    short = [cell for cell in full if not (cell["seed"] == 0 and cell["agent"] == "survival")]
    assert _agg_balanced(short, expected_keys=expected)["decision_balanced"]["zone"].startswith("WITHHELD")

    duplicated = full + [dict(full[0])]
    assert _agg_balanced(duplicated, expected_keys=expected)["decision_balanced"]["zone"].startswith("WITHHELD")

    mismatched = _agg_balanced(full, mismatches=["agent_d0.45_s0_survival.pt"])
    assert mismatched["integrity"]["all_match"] is False
    assert mismatched["decision_balanced"]["zone"].startswith("WITHHELD")

    quick = _agg_balanced(full, quick=True)
    assert quick["decision_balanced"]["zone"].startswith("WITHHELD")

    pooled = _cells(surv=0.80, untr=0.52)
    nan_pooled = [dict(cell) for cell in pooled]
    nan_pooled[0]["resid_obs"] = float("nan")
    assert _agg(nan_pooled, expected_keys=_keys(pooled))["decision"]["zone"].startswith("WITHHELD")


def test_committed_balanced_controls_still_pass():
    """Re-adjudicating the committed B1 and B2 cell lists keeps their recorded passes."""
    root = ROOT / "artifacts" / "balanced_confirmation"
    for name in ("sensory_echo_b1.json", "sensory_echo_b2.json"):
        doc = json.loads((root / name).read_text(encoding="utf-8"))
        agg = ase.aggregate(
            doc["cells"], [0.45], ["survival", "predictor", "untrained"],
            hidden=8, g_seed=0, n_eps=110, steps=24, quick=False, device="cpu",
            mismatches=[], balanced_seed_base=910000,
            objective=doc["objective"], mortal=bool(doc["mortal"]),
            expected_keys=set(ase.BALANCED_SPEC_KEYS))
        assert agg["integrity"]["adjudication_blockers"] == []
        assert agg["decision_balanced"]["zone"] == doc["decision_balanced"]["zone"]
        assert agg["decision_balanced"]["pass_bar"] is True
        assert abs(agg["decision_balanced"]["trained_resid_obs"]
                   - doc["decision_balanced"]["trained_resid_obs"]) < 1e-9
