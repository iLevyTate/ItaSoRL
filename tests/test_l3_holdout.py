"""Tests for the held-out fingerprint probe's building blocks: G-seed variation (unseen
fingerprints must actually differ), the install/swap hook, the frozen transfer probe
(no train->test leakage), and the Student-t across-seed CI."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl import experiment_b2 as b2  # noqa: E402
from itasorl.experiment_b import probe_transfer_auroc  # noqa: E402
from itasorl.stats import mean_ci_t  # noqa: E402
from itasorl.surrogate_l3 import GMotion, train_g_motion  # noqa: E402

TINY = dict(hidden=4, n_eps=8, steps=8, epochs=15, ray_steps=4)


def test_g_motion_seed_changes_the_fingerprint():
    """Different training seeds must yield different nets (different data AND init), and
    the same seed must retrain bit-identically - the holdout design depends on both."""
    g1 = train_g_motion(seed=1, **TINY)
    g2 = train_g_motion(seed=2, **TINY)
    g1b = train_g_motion(seed=1, **TINY)
    x = (np.array([0.1, -0.2]), np.array([0.3, 0.0]))
    assert np.allclose(g1(*x), g1b(*x)), "same seed must be deterministic"
    assert not np.allclose(g1(*x), g2(*x)), "different seeds must differ"
    for w1, w1b in zip(g1._W, g1b._W):
        assert np.array_equal(w1, w1b), "same-seed weights must be bit-identical"


def test_setup_l3_surrogate_returns_net_and_install_swaps(monkeypatch):
    monkeypatch.setattr(b2, "_L3_GMOTION", None)
    g = b2.setup_l3_surrogate(seed=0, **TINY)
    assert isinstance(g, GMotion)
    assert b2._L3_GMOTION is g
    g2 = train_g_motion(seed=3, **TINY)
    b2.install_l3_surrogate(g2)
    assert b2._L3_GMOTION is g2
    b2.install_l3_surrogate(None)
    assert b2._L3_GMOTION is None


def _shared_direction_pools(rng, n=80, dim=6, shift=2.5):
    """Two pools whose class signal lives on the same direction (transfer should work)."""
    def pool(seed_shift):
        X0 = rng.normal(size=(n, dim))
        X1 = rng.normal(size=(n, dim))
        X1[:, 0] += shift + seed_shift
        return np.vstack([X0, X1]), np.r_[np.zeros(n), np.ones(n)].astype(int)
    return pool(0.0), pool(0.3)


def test_probe_transfer_auroc_transfers_shared_signal():
    rng = np.random.default_rng(0)
    (Xtr, ytr), (Xte, yte) = _shared_direction_pools(rng)
    assert probe_transfer_auroc(Xtr, ytr, Xte, yte) > 0.9


def test_probe_transfer_auroc_no_leakage_from_train_fit():
    """If the test pool carries NO class signal, a probe fit on a strongly-signalled train
    pool must score ~0.5 - anything else means train->test contamination."""
    rng = np.random.default_rng(1)
    (Xtr, ytr), _ = _shared_direction_pools(rng)
    Xte = rng.normal(size=(160, Xtr.shape[1]))          # pure noise
    yte = np.r_[np.zeros(80), np.ones(80)].astype(int)
    auc = probe_transfer_auroc(Xtr, ytr, Xte, yte)
    assert abs(auc - 0.5) < 0.12


def test_probe_transfer_auroc_single_class_is_nan():
    rng = np.random.default_rng(2)
    (Xtr, ytr), (Xte, _) = _shared_direction_pools(rng)
    assert np.isnan(probe_transfer_auroc(Xtr, ytr, Xte, np.zeros(len(Xte), int)))


def test_mean_ci_t_matches_hand_computation():
    m, lo, hi = mean_ci_t([0.6, 0.7, 0.8], level=0.90)
    assert m == pytest.approx(0.7)
    assert hi - m == pytest.approx(m - lo), "t interval is symmetric"
    scipy = pytest.importorskip("scipy")
    from scipy.stats import t as student_t
    half = float(student_t.ppf(0.95, 2)) * 0.1 / np.sqrt(3)
    assert hi - m == pytest.approx(half, rel=1e-9)
    assert scipy is not None


def test_mean_ci_t_degenerate_inputs():
    assert all(np.isnan(v) for v in mean_ci_t([]))
    m, lo, hi = mean_ci_t([0.7])
    assert m == lo == hi == 0.7
