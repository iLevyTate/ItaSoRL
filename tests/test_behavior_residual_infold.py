"""Structural pin: the per-timestep residual controls fit their nuisance regression in-fold.

trace_residual_probe_auroc, sensory_residual_probe_auroc (default, integrated=True,
nonlinear=True) and control_diagnostics.residual_probe_with_diagnostics fit phi_t -> h_t on
the TRAIN episodes' pooled timesteps only. An in-sample mutant (`reg.fit(Phi, Hflat)`) passed
every behavioral test, because on synthetic data the in-fold and in-sample AUROCs differ by
only a few hundredths. This pin checks the rows instead, modeled on
test_experiment_a.py::test_grouped_auroc_scaler_never_sees_test_rows_structural: the last
input channel carries the episode index, so every per-timestep fit can be traced back to the
episodes it saw.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.pipeline import Pipeline  # noqa: E402

from itasorl import folds  # noqa: E402
from itasorl.behavior_audit import sensory_residual_probe_auroc, trace_residual_probe_auroc  # noqa: E402
from itasorl.control_diagnostics import history_basis, residual_probe_with_diagnostics  # noqa: E402

K, T, C, HID = 25, 6, 4, 6          # C real channels plus one episode-id channel
ID_COL = C                          # column of the id channel in every basis built here


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 2 * K
    y = np.r_[np.zeros(K), np.ones(K)].astype(int)
    ep = np.broadcast_to(np.arange(n, dtype=float)[:, None, None], (n, T, 1))
    X = np.concatenate([rng.normal(size=(n, T, C)), ep], axis=2)    # id constant over t
    H = rng.normal(size=(n, T, HID)) + 0.5 * y[:, None, None]
    return H, X, y


def _expected_train_sets(y: np.ndarray) -> list[set]:
    """Train episodes per fold, in folds.split order, with the functions' skip rule."""
    return [set(tr.tolist()) for tr, te in folds.split(np.arange(len(y)), 5)
            if len(np.unique(y[te])) >= 2]


def _spy(monkeypatch, n: int, width: int) -> list:
    """Record the episode ids of every per-timestep fit. A fit is per-timestep when it gets
    more rows than there are episodes; the episode-level probe (n rows at most) is ignored.
    The ids come from whichever side of the fit holds the basis (`width` columns): X for the
    nuisance regression, the target for control_diagnostics' basis-from-residual check."""
    seen = []
    orig = Pipeline.fit

    def spy(self, Xf, yf=None, **kw):
        Xa = np.asarray(Xf)
        if Xa.shape[0] > n:
            side = Xa if Xa.shape[1] == width else np.asarray(yf)
            assert side.shape[1] == width, "per-timestep fit holds the basis on neither side"
            ids = np.rint(side[:, ID_COL]).astype(int)
            seen.append((Xa.shape[0], set(ids.tolist())))
        return orig(self, Xf, yf, **kw)

    monkeypatch.setattr(Pipeline, "fit", spy)
    return seen


def _assert_infold(seen: list, expected: list, per_fold: int = 1):
    assert len(seen) == per_fold * len(expected), "unexpected number of per-timestep fits"
    for i, (rows, ids) in enumerate(seen):
        tr = expected[i // per_fold]
        assert ids <= tr, f"fit {i} saw held-out episodes {sorted(ids - tr)[:5]}: in-sample residualization"
        assert ids == tr and rows == len(tr) * T, \
            f"fit {i} saw {rows} rows over {len(ids)} episodes, expected the {len(tr)}-episode train fold"


def test_trace_residualizer_never_sees_test_episodes(monkeypatch):
    H, B, y = _data()
    seen = _spy(monkeypatch, len(y), 3 * (C + 1))            # _trace_phi: [b_t, b_{t-1}, cummean]
    trace_residual_probe_auroc(H, B, y)
    _assert_infold(seen, _expected_train_sets(y))


@pytest.mark.parametrize("kw,width", [({}, 2 * (C + 1)),                       # _echo_phi
                                      ({"integrated": True}, 3 * (C + 1)),     # _trace_phi
                                      ({"nonlinear": True}, 2 * (C + 1))])
def test_sensory_residualizer_never_sees_test_episodes(monkeypatch, kw, width):
    H, O, y = _data()
    seen = _spy(monkeypatch, len(y), width)
    sensory_residual_probe_auroc(H, O, y, **kw)
    _assert_infold(seen, _expected_train_sets(y))


def test_sensory_joint_basis_never_sees_test_episodes(monkeypatch):
    """With Bt joined, the observation block comes first, so the id column is unchanged."""
    H, O, y = _data()
    B = np.random.default_rng(1).normal(size=(len(y), T, 3))
    seen = _spy(monkeypatch, len(y), 2 * (C + 1) + 3 * 3)
    sensory_residual_probe_auroc(H, O, y, Bt=B)
    _assert_infold(seen, _expected_train_sets(y))


@pytest.mark.parametrize("model", ["ridge", "mlp"])
def test_diagnostics_residualizer_never_sees_test_episodes(monkeypatch, model):
    """Each fold makes two per-timestep fits: the nuisance regression and the
    basis-from-residual check. Both must use training episodes only."""
    H, O, y = _data()
    Phi = history_basis(O, lags=1)                          # [x_t, x_{t-1}]: id at column C
    seen = _spy(monkeypatch, len(y), Phi.shape[1])
    residual_probe_with_diagnostics(H, Phi, y, model=model, mlp_iter=50)
    _assert_infold(seen, _expected_train_sets(y), per_fold=2)
