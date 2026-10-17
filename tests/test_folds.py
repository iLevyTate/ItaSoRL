"""itasorl.folds: explicit, stack-independent GroupKFold (spec 2026-09-28-explicit-cv-folds)."""

from __future__ import annotations

import inspect
import json
import os

import numpy as np
import pytest

from itasorl import folds

sklearn_ms = pytest.importorskip("sklearn.model_selection")


def _sklearn_sorts_stably() -> bool:
    src = inspect.getsource(sklearn_ms.GroupKFold._iter_test_indices)
    return 'kind="stable"' in src


GROUPINGS = {
    "one group per episode (pooled readout, 110 + 110)": np.arange(220),
    "matched pairs (60 pairs)": np.repeat(np.arange(60), 2),
    "matched pairs (25 pairs)": np.repeat(np.arange(25), 2),
    "unequal group sizes": np.random.default_rng(3).integers(0, 17, size=203),
    "non-contiguous labels": np.repeat(np.array([7, 3, 99, 42, 5, 11, 8]), 3),
}


@pytest.mark.parametrize("name", list(GROUPINGS))
def test_explicit_matches_current_sklearn(name):
    """With a scikit-learn that sorts stably, the explicit scheme IS its GroupKFold."""
    if not _sklearn_sorts_stably():
        pytest.skip("installed scikit-learn sorts groups unstably; the explicit scheme differs by design")
    g = GROUPINGS[name]
    ours = [te.tolist() for _, te in folds.split(g, 5, scheme="explicit")]
    ref = [te.tolist() for _, te in sklearn_ms.GroupKFold(n_splits=5).split(np.zeros(len(g)), None, g)]
    assert ours == ref


def test_pooled_readout_folds_are_balanced():
    y = np.r_[np.zeros(110), np.ones(110)].astype(int)
    assert folds.class_counts(y, np.arange(220), scheme="explicit") == [(22, 22)] * 5


def test_train_and_test_partition_every_sample():
    g = GROUPINGS["unequal group sizes"]
    seen = np.zeros(len(g), dtype=int)
    for tr, te in folds.split(g, 5, scheme="explicit"):
        assert len(np.intersect1d(tr, te)) == 0
        assert len(tr) + len(te) == len(g)
        assert set(np.unique(g[te])).isdisjoint(set(np.unique(g[tr])))   # groups never split
        seen[te] += 1
    assert (seen == 1).all()


def test_legacy_routes_through_installed_sklearn():
    g = np.arange(220)
    ours = [te.tolist() for _, te in folds.split(g, 5, scheme="legacy")]
    ref = [te.tolist() for _, te in sklearn_ms.GroupKFold(n_splits=5).split(np.zeros(220), None, g)]
    assert ours == ref


def test_scheme_selection(monkeypatch):
    monkeypatch.delenv("ITASORL_FOLDS", raising=False)
    assert folds.current_scheme() == "explicit"
    monkeypatch.setenv("ITASORL_FOLDS", "legacy")
    assert folds.current_scheme() == "legacy"
    with folds.fold_scheme("explicit"):
        assert folds.current_scheme() == "explicit"
    assert folds.current_scheme() == "legacy"
    monkeypatch.setenv("ITASORL_FOLDS", "bogus")
    with pytest.raises(ValueError):
        folds.current_scheme()
    with pytest.raises(ValueError):
        with folds.fold_scheme("bogus"):
            pass


def test_too_many_splits_raises():
    with pytest.raises(ValueError):
        list(folds.split(np.arange(3), 5, scheme="explicit"))


def test_grouped_auroc_uses_the_active_scheme():
    """grouped_auroc must follow itasorl.folds; the two schemes agree whenever the
    installed scikit-learn sorts stably."""
    from itasorl.experiment_a import grouped_auroc
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(110), np.ones(110)].astype(int)
    X = rng.normal(size=(220, 12)) + 0.35 * y[:, None]
    g = np.arange(220)
    with folds.fold_scheme("explicit"):
        a = grouped_auroc(X, y, g)
    with folds.fold_scheme("legacy"):
        b = grouped_auroc(X, y, g)
    assert 0.5 < a < 1.0
    if _sklearn_sorts_stably():
        assert a == b


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# The re-score artifacts folds.py cites for its references (folds.py, REFERENCE_SURVIVAL_TARGET
# comment); the held-out hidden-8 re-score holds the same drift-0.45 survival cells.
REFERENCE_ARTIFACTS = {8: ("l3_h8_traces.json", "l3_h8_heldout.json"), 7: ("l3_h7_traces.json",)}


def _artifact_survival_mean(name: str, scheme: str) -> float:
    path = os.path.join(ROOT, "artifacts", "fold_rescore", name)
    if not os.path.exists(path):
        pytest.skip(f"artifacts/fold_rescore/{name} is not in this tree")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    vals = [c[scheme]["target"] for c in doc["cells"]
            if c["agent"] == "survival" and float(c["drift"]) == 0.45]
    assert len(vals) == 10, name
    return float(np.mean(vals))


def test_integrity_reference_is_keyed_by_scheme():
    # The references are read from the re-score artifacts, not from test constants, so a
    # drift between folds.py and the artifact it cites fails here (H44).
    for scheme in ("legacy", "explicit"):
        with folds.fold_scheme(scheme):
            for hidden, names in REFERENCE_ARTIFACTS.items():
                for name in names:
                    want = round(_artifact_survival_mean(name, scheme), 3)
                    assert folds.reference_survival_target(hidden) == want, (scheme, hidden, name)
    with folds.fold_scheme("explicit"):
        with pytest.raises(SystemExit, match="ITASORL_FOLDS=legacy"):
            folds.reference_survival_target(99)
        assert folds.reference_survival_target(99, strict=False) is None
