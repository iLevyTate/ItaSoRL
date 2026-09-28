"""Cross-validation folds that do not depend on the installed scikit-learn or numpy.

Every grouped probe in the project (pooled readout, matched-pair oracles, behavior
and sensory controls) splits episodes with a 5-fold GroupKFold. scikit-learn's
GroupKFold orders equal-sized groups with `np.argsort`, which before the stable
sort landed in scikit-learn was the default (unstable) kind. With one group per
episode every group has the same size, so fold membership then depended on the
numpy build and the CPU's sort path. The published GPU numbers were scored on
the split (24,20), (22,22), (21,23), (21,23), (22,22) for a 110 + 110 pool; current
scikit-learn and the 2026-09 cloud runs use five balanced 22/22 folds.

`split` reproduces current scikit-learn's non-shuffled GroupKFold exactly (stable
sort of group sizes, largest first, each group to the lightest fold, ties to the
lowest fold index) in plain numpy, so the partition is the same on every stack.
Setting ITASORL_FOLDS=legacy (or `fold_scheme("legacy")`) routes through the
installed scikit-learn GroupKFold instead, which reproduces whatever split that
stack produces; on the stack that generated the published GPU numbers it
reproduces them. Spec: docs/specs/2026-09-28-explicit-cv-folds.md.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import numpy as np

SCHEMES = ("explicit", "legacy")
_override: str | None = None


def current_scheme() -> str:
    """The active fold scheme: a `fold_scheme` override, else ITASORL_FOLDS, else explicit."""
    scheme = _override or os.environ.get("ITASORL_FOLDS", "explicit").strip().lower()
    if scheme not in SCHEMES:
        raise ValueError(f"unknown fold scheme {scheme!r}; expected one of {SCHEMES}")
    return scheme


@contextlib.contextmanager
def fold_scheme(scheme: str):
    """Temporarily force a fold scheme (used to score one dump under both)."""
    global _override
    if scheme not in SCHEMES:
        raise ValueError(f"unknown fold scheme {scheme!r}; expected one of {SCHEMES}")
    prev, _override = _override, scheme
    try:
        yield
    finally:
        _override = prev


def fold_of_sample(groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Fold index per sample under the explicit scheme."""
    groups = np.asarray(groups)
    unique_groups, group_idx = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    if n_splits > n_groups:
        raise ValueError(f"Cannot have n_splits={n_splits} greater than the number of groups: {n_groups}.")
    counts = np.bincount(group_idx)
    order = np.argsort(counts, kind="stable")[::-1]
    load = np.zeros(n_splits)
    group_to_fold = np.zeros(n_groups, dtype=int)
    for g in order:
        f = int(np.argmin(load))
        load[f] += counts[g]
        group_to_fold[g] = f
    return group_to_fold[group_idx.ravel()]


def split(groups: np.ndarray, n_splits: int = 5, scheme: str | None = None
          ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) per fold, like GroupKFold(n_splits).split(X, y, groups)."""
    scheme = scheme or current_scheme()
    groups = np.asarray(groups)
    if scheme == "legacy":
        from sklearn.model_selection import GroupKFold
        yield from GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), None, groups)
        return
    fos = fold_of_sample(groups, n_splits)
    for f in range(n_splits):
        yield np.where(fos != f)[0], np.where(fos == f)[0]


# Drift-0.45 survival pooled-target means (3 dp) that the regeneration integrity
# gates must reproduce, per fold scheme and L3 capacity. The legacy values are the
# published GPU numbers (FINDINGS 10.2, 10.5), reproducible under
# ITASORL_FOLDS=legacy on the stack that produced them. The explicit values come from
# re-scoring the saved dumps with scripts/rescore_fold_split.py; until one is recorded
# here, an explicit-scheme gate stops and says so rather than skipping the check.
REFERENCE_SURVIVAL_TARGET: dict[str, dict[int, float]] = {
    "legacy": {8: 0.752, 7: 0.737},
    "explicit": {},
}


def reference_survival_target(hidden: int, strict: bool = True) -> float | None:
    """Integrity-gate reference for the active scheme. strict=True raises SystemExit
    with instructions when none is recorded; strict=False returns None instead."""
    scheme = current_scheme()
    ref = REFERENCE_SURVIVAL_TARGET[scheme].get(int(hidden))
    if ref is None and strict:
        raise SystemExit(
            f"No {scheme}-scheme integrity reference for the hidden={hidden} survival target "
            "yet. Either run with ITASORL_FOLDS=legacy on the stack that produced the "
            "published numbers, or re-score the saved dumps first "
            "(scripts/rescore_fold_split.py) and record the explicit reference in "
            "itasorl/folds.py. Spec: docs/specs/2026-09-28-explicit-cv-folds.md.")
    return ref


def class_counts(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                 scheme: str | None = None) -> list[tuple[int, int]]:
    """(label-0, label-1) counts per test fold: a fingerprint of the partition in use."""
    y = np.asarray(y)
    return [(int((y[te] == 0).sum()), int((y[te] == 1).sum()))
            for _, te in split(groups, n_splits, scheme)]
