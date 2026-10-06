"""Revision step 13: intervals aligned with their estimators, paired contrasts, honest labels."""

from __future__ import annotations

import numpy as np
import pytest

from itasorl.stats import auroc, fold_mean_auroc_ci, paired_contrast, rope_test


def test_fold_mean_interval_brackets_the_fold_mean_estimator():
    rng = np.random.default_rng(0)
    fy, fp = [], []
    for _ in range(5):
        y = np.r_[np.zeros(20), np.ones(20)].astype(int)
        fy.append(y)
        fp.append(np.r_[rng.normal(0, 1, 20), rng.normal(0.8, 1, 20)])
    point = float(np.mean([auroc(y, p) for y, p in zip(fy, fp)]))
    lo, hi = fold_mean_auroc_ci(fy, fp, seed=0)
    assert lo < point < hi
    glo, ghi = fold_mean_auroc_ci(fy, fp, seed=0,
                                  fold_groups=[np.r_[np.arange(20), np.arange(20)]] * 5)
    assert glo < point < ghi


def test_paired_contrast_tests_the_margin_on_the_difference():
    a = [0.75, 0.70, 0.72, 0.78, 0.69]
    b = [0.60, 0.62, 0.58, 0.61, 0.63]
    c = paired_contrast(a, b)
    assert c["mean"] == pytest.approx(np.mean(a) - np.mean(b))
    assert c["t90_lower_ge_margin"] and c["mean_ge_margin"]
    thin = paired_contrast([0.66, 0.64, 0.70, 0.62], [0.60, 0.61, 0.60, 0.61])
    assert thin["mean_ge_margin"] and not thin["t90_lower_ge_margin"]   # rule met, evidence thin


def test_rope_result_is_labeled_as_a_bootstrap_not_a_posterior():
    r = rope_test([0.50, 0.49, 0.51, 0.50, 0.52])
    assert r.boot_interval == r.hdi and r.boot_share_in_rope == r.p_in_rope
    assert "HDI" not in str(r) and "P(in ROPE)" not in str(r)
    assert "percentile-bootstrap" in str(r)
