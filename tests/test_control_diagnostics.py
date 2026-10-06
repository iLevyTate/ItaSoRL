"""Revision step 8: control-quality diagnostics, wider bases, and sequence readouts."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.control_diagnostics import (  # noqa: E402
    flat_sequence_linear_auroc,
    history_basis,
    residual_probe_with_diagnostics,
    sequence_gru_auroc,
)


def test_history_basis_lags_traces_and_actions():
    O = np.arange(2 * 4 * 1, dtype=float).reshape(2, 4, 1)
    A = np.ones((2, 4, 1))
    Phi = history_basis(O, lags=2, ema_taus=(2,), At=A).reshape(2, 4, -1)
    np.testing.assert_array_equal(Phi[0, :, 0], [0, 1, 2, 3])            # x_t
    np.testing.assert_array_equal(Phi[0, :, 1], [0, 0, 1, 2])            # x_{t-1}, edge padded
    np.testing.assert_array_equal(Phi[0, :, 2], [0, 0, 0, 1])            # x_{t-2}
    np.testing.assert_allclose(Phi[0, :, 3], [0, 0.5, 1.25, 2.125])      # causal EMA, tau 2
    np.testing.assert_array_equal(Phi[0, :, 5], [0, 1, 1, 1])            # a_{t-1}, zero at t = 0


def _synthetic(label_in_state: bool, n=60, T=6, C=3, hid=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    O = rng.normal(size=(2 * n, T, C)) + 0.8 * y[:, None, None]       # the input carries the label
    W = rng.normal(size=(C, hid))
    H = O @ W + 0.05 * rng.normal(size=(2 * n, T, hid))
    if label_in_state:
        H[:, :, 0] += 1.5 * y[:, None]                                  # a component inputs do not explain
    return H, O, y


def test_a_pure_echo_is_removed_and_the_diagnostics_say_so():
    H, O, y = _synthetic(label_in_state=False)
    r = residual_probe_with_diagnostics(H, history_basis(O), y)
    assert r["auroc"] < 0.65
    assert r["nuisance_r2_heldout"] > 0.9
    assert r["nuisance_from_residual_r2_heldout"] < 0.2


def test_a_component_beyond_the_basis_survives():
    H, O, y = _synthetic(label_in_state=True)
    r = residual_probe_with_diagnostics(H, history_basis(O), y)
    assert r["auroc"] > 0.9


def test_mlp_control_reports_its_optimizer():
    H, O, y = _synthetic(label_in_state=True, n=30)
    r = residual_probe_with_diagnostics(H, history_basis(O), y, model="mlp", mlp_iter=20)
    assert r["optimizer"]["folds"] == 5 and len(r["optimizer"]["n_iter"]) == 5
    assert r["optimizer"]["converged_folds"] <= 5


def test_sequence_readouts_decode_a_sequence_signal_and_are_deterministic():
    rng = np.random.default_rng(1)
    n, T, C = 40, 6, 3
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    S = rng.normal(size=(2 * n, T, C))
    S[:, -1, 0] += 3.0 * y                                             # signal at the last step only
    a = sequence_gru_auroc(S, y, epochs=40, hidden=8, embed=8)
    b = sequence_gru_auroc(S, y, epochs=40, hidden=8, embed=8)
    assert a["auroc"] > 0.8 and a["auroc"] == b["auroc"]
    assert flat_sequence_linear_auroc(S, y, n_components=10) > 0.8
