"""Diagnostics and stronger bases for the behavior and sensory controls (revision step 8).

The published controls (`itasorl.behavior_audit`) regress a nuisance basis out of every h_t
in-fold and probe what is left. What they establish is that a signal remains after
controlling for that basis with that model; this module reports how good the control was and
widens the basis, so each control states exactly which alternative it addresses.

  residual_probe_with_diagnostics  the in-fold residual probe of `behavior_audit`, plus per
        fold: held-out R^2 of the nuisance regression (how much of h_t the basis explains on
        unseen episodes), held-out R^2 of predicting the basis back from the residuals (does
        nuisance information survive residualization), and, for the MLP, iterations used,
        final loss, and whether the optimizer converged. Scalers and regressions are fit on
        training folds only.
  history_basis  [x_t, x_{t-1}, ..., x_{t-L}] plus causal exponential traces of x at several
        timescales, optionally with the env action a_t and the previous action a_{t-1} that
        the GRU received.
  sequence_gru_auroc  an actual SEQUENCE readout: a supervised GRU classifier on the
        (observation, previous action) sequence with the agent trunk's capacity (64-unit
        encoder, 96-unit GRU), trained per fold for a fixed number of epochs. A comparator
        built from summary features (means, finals, dispersions) is labeled a summary-feature
        comparator, never "full sequence".
  flat_sequence_linear_auroc  the standard linear probe on the flattened sequence after an
        in-fold PCA to the hidden-state probe's dimensionality.

The recurrent state is built from the observation and action history, so independence from
the whole history is not the question. The question is what the state carries beyond a named
baseline, and how a decoder with comparable capacity does on that history directly.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import folds
from .experiment_b import episode_features


def history_basis(Ot: np.ndarray, lags: int = 1, ema_taus=(), At: np.ndarray | None = None,
                  Bt: np.ndarray | None = None) -> np.ndarray:
    """Per-timestep regressors, rows = (episode, t). Lags are edge padded. EMA traces are
    causal: e_t = e_{t-1} + (x_t - e_{t-1}) / tau with e_0 = x_0. At adds [a_t, a_{t-1}]
    (a_{-1} = 0, as the GRU receives it); Bt adds the behavior trace [b_t, b_{t-1}]."""
    O = np.asarray(Ot, float)
    n, T, C = O.shape
    cols = [O]
    for k in range(1, lags + 1):
        cols.append(np.concatenate([np.repeat(O[:, :1], min(k, T), axis=1), O[:, :max(T - k, 0)]],
                                   axis=1)[:, :T])
    for tau in ema_taus:
        e = np.empty_like(O)
        e[:, 0] = O[:, 0]
        for t in range(1, T):
            e[:, t] = e[:, t - 1] + (O[:, t] - e[:, t - 1]) / float(tau)
        cols.append(e)
    if At is not None:
        A = np.asarray(At, float)
        cols += [A, np.concatenate([np.zeros_like(A[:, :1]), A[:, :-1]], axis=1)]
    if Bt is not None:
        B = np.asarray(Bt, float)
        cols += [B, np.concatenate([B[:, :1], B[:, :-1]], axis=1)]
    return np.concatenate(cols, axis=2).reshape(n * T, -1)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-weighted R^2 across output columns."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def residual_probe_with_diagnostics(H: np.ndarray, Phi: np.ndarray, y: np.ndarray,
                                    groups: np.ndarray | None = None, *, model: str = "ridge",
                                    alpha: float = 1e-3, mlp_iter: int = 300, seed: int = 0,
                                    n_splits: int = 5) -> dict:
    """In-fold per-timestep residual probe with control-quality diagnostics."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if groups is None:
        groups = np.arange(len(y))
    n, T, hid = H.shape
    Hflat = np.asarray(H, float).reshape(n * T, hid)
    row_ep = np.repeat(np.arange(n), T)
    aucs, r2_heldout, r2_back, opt = [], [], [], []
    for tr, te in folds.split(groups, n_splits):
        if len(np.unique(y[te])) < 2:
            continue
        tr_rows, te_rows = np.isin(row_ep, tr), np.isin(row_ep, te)
        if model == "mlp":
            reg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3,
                                                               max_iter=mlp_iter, random_state=seed))
        else:
            reg = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            reg.fit(Phi[tr_rows], Hflat[tr_rows])
        if model == "mlp":
            m = reg[-1]
            opt.append({"n_iter": int(m.n_iter_), "final_loss": float(m.loss_),
                        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught)})
        pred = reg.predict(Phi)
        r2_heldout.append(_r2(Hflat[te_rows], pred[te_rows]))
        R = Hflat - pred
        back = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(R[tr_rows], Phi[tr_rows])
        r2_back.append(_r2(Phi[te_rows], back.predict(R[te_rows])))
        F = episode_features(R.reshape(n, T, hid))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(F[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]))
    out = {"auroc": float(np.mean(aucs)) if aucs else float("nan"),
           "nuisance_r2_heldout": float(np.mean(r2_heldout)) if r2_heldout else float("nan"),
           "nuisance_from_residual_r2_heldout": float(np.mean(r2_back)) if r2_back else float("nan"),
           "basis_dim": int(Phi.shape[1]), "model": model}
    if opt:
        out["optimizer"] = {"n_iter": [o["n_iter"] for o in opt],
                            "final_loss": [o["final_loss"] for o in opt],
                            "converged_folds": int(sum(o["converged"] for o in opt)),
                            "folds": len(opt)}
    return out


def sequence_gru_auroc(Seq: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, *,
                       embed: int = 64, hidden: int = 96, epochs: int = 60, lr: float = 1e-3,
                       batch: int = 32, seed: int = 0, n_splits: int = 5) -> dict:
    """Supervised sequence readout: encoder (2 x Linear+ReLU, `embed`) -> GRU(`hidden`) ->
    linear logit on the final state, trained per fold with Adam for a fixed number of epochs
    on the training episodes only (inputs standardized with training-fold statistics).
    Reports out-of-fold mean AUROC and the per-fold final training loss."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    if groups is None:
        groups = np.arange(len(y))
    S = np.asarray(Seq, np.float32)
    n, T, C = S.shape
    aucs, losses = [], []
    for f, (tr, te) in enumerate(folds.split(groups, n_splits)):
        if len(np.unique(y[te])) < 2:
            continue
        mu = S[tr].reshape(-1, C).mean(0)
        sd = S[tr].reshape(-1, C).std(0) + 1e-6
        Xtr = torch.as_tensor((S[tr] - mu) / sd)
        Xte = torch.as_tensor((S[te] - mu) / sd)
        ytr = torch.as_tensor(y[tr], dtype=torch.float32)
        torch.manual_seed(seed * 100 + f)
        enc = nn.Sequential(nn.Linear(C, embed), nn.ReLU(), nn.Linear(embed, embed), nn.ReLU())
        gru = nn.GRU(embed, hidden, batch_first=True)
        head = nn.Linear(hidden, 1)
        params = list(enc.parameters()) + list(gru.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        g = torch.Generator().manual_seed(seed * 100 + f)
        last = float("nan")
        for _ in range(epochs):
            perm = torch.randperm(len(tr), generator=g)
            tot = 0.0
            for i in range(0, len(tr), batch):
                idx = perm[i:i + batch]
                _, hT = gru(enc(Xtr[idx]))
                loss = nn.functional.binary_cross_entropy_with_logits(head(hT[0]).squeeze(-1), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            last = tot / len(tr)
        with torch.no_grad():
            _, hT = gru(enc(Xte))
            p = torch.sigmoid(head(hT[0]).squeeze(-1)).numpy()
        aucs.append(roc_auc_score(y[te], p))
        losses.append(last)
    return {"auroc": float(np.mean(aucs)) if aucs else float("nan"),
            "final_train_loss": losses, "epochs": epochs, "hidden": hidden, "embed": embed}


def flat_sequence_linear_auroc(Seq: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                               n_components: int = 192, n_splits: int = 5) -> float:
    """Standard linear probe on the flattened (T * C) sequence after an in-fold PCA to
    `n_components` (the [mean h, final h] probe's width for a 96-unit state)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if groups is None:
        groups = np.arange(len(y))
    F = np.asarray(Seq, float).reshape(len(y), -1)
    aucs = []
    for tr, te in folds.split(groups, n_splits):
        if len(np.unique(y[te])) < 2:
            continue
        k = int(min(n_components, len(tr) - 1, F.shape[1]))
        clf = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                            LogisticRegression(max_iter=2000))
        clf.fit(F[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")
