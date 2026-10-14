"""Optional sensitivity readouts. They do not change a frozen verdict.

O1. The speed positive control labels an episode "fast" when its mean speed
exceeds one threshold, then asks whether the state reads that label. The
recorded call uses the median of the whole pool, including episodes the probe
later holds out. The fold-local call computes that median on the outer
training fold only and applies it to the training and held-out episodes.
Outer groups are the probe's groups, so a matched pair stays in one fold.

O2. The long-history controls used one ridge penalty, 1e-3, and their held-out
fit was negative. The penalty here is chosen inside each outer training fold
by an episode-grouped inner split. Rows from the same episode, and both
members of a pair, never cross from a training fold into its validation fold.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone

import numpy as np
import scipy
import sklearn
from scipy.linalg import svd as scipy_svd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from itasorl import folds
from itasorl.control_diagnostics import _r2
from itasorl.experiment_a import grouped_auroc
from itasorl.experiment_b import episode_features
from itasorl.results_io import git_head

SPEED_BAR = 0.75
FIXED_RIDGE_ALPHA = 1e-3
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


def _probe_fold(features: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(features[train], labels[train])
    return float(roc_auc_score(labels[test], clf.predict_proba(features[test])[:, 1]))


def speed_threshold_probes(features: np.ndarray, speeds: np.ndarray, groups: np.ndarray) -> dict:
    """Global-median speed probe beside the fold-local one, on the same episodes.

    `groups` is the outer grouping. Independent episodes use one group each.
    Matched pairs pass the same group id for both members.
    """
    speeds = np.asarray(speeds, float)
    groups = np.asarray(groups)
    features = np.asarray(features, float)
    if len(speeds) != len(features) or len(groups) != len(speeds):
        raise ValueError("features, speeds, and groups must describe the same episodes")
    if not np.isfinite(speeds).all() or float(np.ptp(speeds)) == 0.0:
        empty = float("nan")
        return {"global": empty, "fold_local": empty, "n_folds": 0, "n_folds_unscored": 0,
                "train_medians": [], "global_median": empty}
    global_median = float(np.median(speeds))
    global_labels = (speeds > global_median).astype(int)
    global_auroc = float(grouped_auroc(features, global_labels, groups))
    local_aucs = []
    medians = []
    unscored = 0
    for train, test in folds.split(groups):
        median = float(np.median(speeds[train]))
        medians.append(median)
        score = _probe_fold(features, (speeds > median).astype(int), train, test)
        if np.isfinite(score):
            local_aucs.append(score)
        else:
            unscored += 1
    return {
        "global": global_auroc,
        "fold_local": float(np.mean(local_aucs)) if local_aucs else float("nan"),
        "n_folds": len(local_aucs),
        "n_folds_unscored": unscored,
        "train_medians": medians,
        "global_median": global_median,
    }


def _rows_for(row_episode: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    return np.isin(row_episode, episodes)


def _ridge_factor(x_train: np.ndarray, y_train: np.ndarray):
    """SVD of the standardized training design and the centered target projection."""
    x_mean = x_train.mean(0)
    x_scale = x_train.std(0)
    x_scale[x_scale == 0.0] = 1.0
    train = (x_train - x_mean) / x_scale
    y_mean = y_train.mean(0)
    try:
        u, singular, vt = np.linalg.svd(train, full_matrices=False)
    except np.linalg.LinAlgError:
        u, singular, vt = scipy_svd(train, full_matrices=False, lapack_driver="gesvd")
    return x_mean, x_scale, singular, vt, u.T @ (y_train - y_mean), y_mean


def _predict_from_factor(x: np.ndarray, factor, alpha: float) -> np.ndarray:
    x_mean, x_scale, singular, vt, projected, y_mean = factor
    filt = singular / (singular ** 2 + float(alpha))
    coef = (vt.T * filt) @ projected
    return ((x - x_mean) / x_scale) @ coef + y_mean


def _ridge_r2_grid(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                   y_test: np.ndarray, alphas: tuple[float, ...]) -> list[float]:
    """Held-out R^2 at every penalty from one SVD of the standardized training design."""
    factor = _ridge_factor(x_train, y_train)
    return [_r2(y_test, _predict_from_factor(x_test, factor, alpha)) for alpha in alphas]


def _ridge_predict_grid(x_train: np.ndarray, y_train: np.ndarray, x_all: np.ndarray,
                        alphas: tuple[float, ...]) -> list[np.ndarray]:
    factor = _ridge_factor(x_train, y_train)
    return [_predict_from_factor(x_all, factor, alpha) for alpha in alphas]


def select_ridge_alpha(phi: np.ndarray, target: np.ndarray, row_episode: np.ndarray,
                       train_episodes: np.ndarray, train_groups: np.ndarray,
                       alphas: tuple[float, ...] = RIDGE_ALPHAS, n_inner: int = 3) -> float:
    """Pick a ridge penalty by grouped inner validation on the outer training episodes only."""
    if len(train_episodes) < n_inner:
        n_inner = max(2, len(train_episodes))
    scores = {float(alpha): [] for alpha in alphas}
    for inner_train, inner_test in folds.split(train_groups, n_inner):
        tr_rows = _rows_for(row_episode, train_episodes[inner_train])
        te_rows = _rows_for(row_episode, train_episodes[inner_test])
        if not tr_rows.any() or not te_rows.any():
            continue
        for alpha, score in zip(alphas, _ridge_r2_grid(
                phi[tr_rows], target[tr_rows], phi[te_rows], target[te_rows], alphas)):
            scores[float(alpha)].append(score)
    means = {alpha: float(np.mean(vals)) for alpha, vals in scores.items() if vals}
    if not means:
        return float(FIXED_RIDGE_ALPHA)
    return max(means, key=means.get)


def long_history_ridge(H: np.ndarray, phi: np.ndarray, y: np.ndarray,
                       groups: np.ndarray | None = None,
                       alphas: tuple[float, ...] = RIDGE_ALPHAS) -> dict:
    """Outer-fold residual probe whose ridge penalty is chosen inside the training fold.

    Also reports the same probe at the recorded penalty 1e-3, on these same arrays.
    """
    y = np.asarray(y, int)
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    n, steps, hidden = H.shape
    target = np.asarray(H, float).reshape(n * steps, hidden)
    phi = np.asarray(phi, float)
    row_episode = np.repeat(np.arange(n), steps)
    chosen_aucs, chosen_r2, chosen_alphas = [], [], []
    fixed_aucs, fixed_r2 = [], []
    for train, test in folds.split(groups):
        if len(np.unique(y[test])) < 2:
            continue
        alpha = select_ridge_alpha(phi, target, row_episode, train, groups[train], alphas)
        chosen_alphas.append(alpha)
        tr_rows = _rows_for(row_episode, train)
        te_rows = _rows_for(row_episode, test)
        predictions = _ridge_predict_grid(phi[tr_rows], target[tr_rows], phi, (alpha, FIXED_RIDGE_ALPHA))
        for pred, aucs, r2s in (
            (predictions[0], chosen_aucs, chosen_r2),
            (predictions[1], fixed_aucs, fixed_r2),
        ):
            r2s.append(_r2(target[te_rows], pred[te_rows]))
            residual = (target - pred).reshape(n, steps, hidden)
            aucs.append(_probe_fold(episode_features(residual), y, train, test))
    def _mean(vals: list[float]) -> float:
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")
    return {
        "auroc": _mean(chosen_aucs),
        "nuisance_r2_heldout": _mean(chosen_r2),
        "alphas": chosen_alphas,
        "fixed_alpha": {
            "alpha": FIXED_RIDGE_ALPHA,
            "auroc": _mean(fixed_aucs),
            "nuisance_r2_heldout": _mean(fixed_r2),
        },
        "basis_dim": int(phi.shape[1]),
        "n_folds": len(chosen_alphas),
    }


NOT_ADJUDICATION = ("optional readout outside the 2026-10-13 plan; not an adjudication; "
                    "does not replace a recorded control or verdict")


def provenance(script: str, command: str, inputs: dict, *, device: str) -> dict:
    """The block written beside an optional readout's summary: which script and command
    produced it, what it read, the stack it ran on, and the standing it does not have."""
    import torch
    return {
        "generated_by": script,
        "command": command,
        "inputs": inputs,
        "status": NOT_ADJUDICATION,
        "git_head": git_head(),
        "stack": {"python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                  "torch": torch.__version__, "device": device},
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
