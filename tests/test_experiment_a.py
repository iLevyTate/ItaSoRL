"""Functional regression tests for the Experiment A oracle and leakage gate.

These guard the core detection methodology end to end: an L1 grid is detectable
while L0 is at chance, the detectability ceiling falls as the grid approaches the
sensor noise, and the leakage audit catches a contaminated (confounded) reward.
Deterministic - run_experiment_a and generate_clean use fixed seeds.
"""

import numpy as np
import pytest

from itasorl.experiment_a import generate_clean, grouped_auroc, run_experiment_a

pytest.importorskip("sklearn")

SIGMA = 0.01


def _clean():
    return generate_clean(n_pairs=20, steps=20, seed0=1000, ray_steps=10)


def test_L1_detectable_while_L0_at_chance():
    clean = _clean()
    l0 = run_experiment_a(clean, "L0", delta=0.06, sigma=SIGMA)
    l1 = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA)
    assert l1["oracle_auroc"] > 0.9   # a clearly-resolved grid is detectable
    assert l0["oracle_auroc"] < 0.7   # an identical world is ~chance
    assert l1["oracle_auroc"] > l0["oracle_auroc"]


def test_detectability_ceiling_falls_as_delta_approaches_sigma():
    clean = _clean()
    coarse = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA)["oracle_auroc"]
    fine = run_experiment_a(clean, "L1", delta=SIGMA, sigma=SIGMA)["oracle_auroc"]
    assert coarse > fine


def test_leakage_gate_catches_reward_contamination():
    clean = _clean()
    clean_run = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA)
    dirty_run = run_experiment_a(clean, "L1", delta=0.06, sigma=SIGMA, contaminate=0.05)
    assert clean_run["leakage_pass"] is True    # no confound
    assert dirty_run["leakage_pass"] is False   # reward offset is caught


# ---------------------------------------------------------------------------
# Scaler fold discipline: the docstring promise "scaler fit on TRAIN folds only"
# is load-bearing (every AUROC in the project flows through grouped_auroc), so
# it gets both a behavioral and a structural pin.
# ---------------------------------------------------------------------------

def _fold_discipline_dataset(seed: int = 0, scale: float = 1000.0):
    """25 matched pairs, 2 features. Feature 0 is informative everywhere; feature 1
    is informative in four folds but carries huge ANTI-informative values in exactly
    one test fold (labels balanced there). Fold membership is looked up from the
    deterministic GroupKFold split itself, so the outliers land in one fold by
    construction, not by luck."""
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(seed)
    n_groups = 25
    y = np.tile([0, 1], n_groups)
    g = np.repeat(np.arange(n_groups), 2)
    X = np.empty((len(y), 2))
    X[:, 0] = (2 * y - 1) + rng.normal(0.0, 0.3, len(y))
    X[:, 1] = (2 * y - 1) + rng.normal(0.0, 0.3, len(y))
    te = list(GroupKFold(n_splits=5).split(X, y, g))[0][1]
    X[te, 1] = -(2 * y[te] - 1) * scale     # huge, anti-informative, one fold only
    return X, y, g


def test_grouped_auroc_scaler_fit_on_train_folds_only_behavioral():
    """Behavioral pin of the no-leakage promise. With train-only scaling, the fold
    holding the huge anti-informative feature-1 values scores ~0 (the probe trusts
    the direction it learned on the clean train folds and feature 1 betrays it),
    while the other four folds score ~1 with feature 1 neutralized by its huge train
    std - mean 0.8. A scaler fit on ALL data before CV would see the outliers,
    crush feature 1 everywhere, ride feature 0 alone, and report ~1.0."""
    X, y, g = _fold_discipline_dataset()
    auc = grouped_auroc(X, y, g)
    assert auc == pytest.approx(0.8, abs=0.05), \
        f"expected ~0.8 under train-only scaling (got {auc}); ~1.0 means the scaler saw test rows"


def test_grouped_auroc_scaler_never_sees_test_rows_structural(monkeypatch):
    """Structural pin: every StandardScaler fit inside grouped_auroc must receive
    exactly the corresponding TRAIN fold - never the full dataset - and its fitted
    mean_ must be that train fold's mean, not the full-data mean (grouped_auroc
    visits folds in gkf.split order and no fold is skipped here, so the recorded
    fits align 1:1 with the split)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X, y, g = _fold_discipline_dataset()
    expected = [(len(tr), X[tr].mean(0)) for tr, _ in GroupKFold(n_splits=5).split(X, y, g)]

    seen = []
    orig_fit = StandardScaler.fit

    def spy_fit(self, Xf, y=None, sample_weight=None):
        out = orig_fit(self, Xf, y=y, sample_weight=sample_weight)
        seen.append((np.asarray(Xf).shape[0], np.array(self.mean_)))
        return out

    monkeypatch.setattr(StandardScaler, "fit", spy_fit)
    grouped_auroc(X, y, g)

    assert len(seen) == len(expected), "one scaler fit per fold"
    for (n_rows, mean), (n_exp, mean_exp) in zip(seen, expected):
        assert n_rows < len(y), "scaler was fit on the FULL dataset - test rows leaked"
        assert n_rows == n_exp, f"scaler fit on {n_rows} rows, expected the {n_exp}-row train fold"
        assert np.allclose(mean, mean_exp), \
            "scaler.mean_ is not the train-fold mean - it saw other rows"
