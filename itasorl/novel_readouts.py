"""Pure pieces of the four readout-only specs frozen on 2026-10-15.

- docs/specs/2026-10-15-salience-design.md: label-free prominence of world identity.
- docs/specs/2026-10-15-world-draw-features-design.md: features of the agent's own world
  decoded from its state, with the senses regressed out.
- docs/specs/2026-10-15-online-erasure-design.md: the rank-two erasure subspace.
- docs/specs/2026-10-15-what-develops-design.md: direction cosines across snapshots.

Nothing here adjudicates a registered rule. The scripts that call these functions write
their outputs with a provenance block saying so.
"""

from __future__ import annotations

import numpy as np

from itasorl.experiment_b import episode_features


# ---------------------------------------------------------------------------
# shared statistics
# ---------------------------------------------------------------------------

def paired_t90(a, b=None) -> dict:
    """Seed-paired difference a - b (or a alone) with its t-based 90% interval."""
    from itasorl.stats import t_ci90
    x = np.asarray(a, float) if b is None else np.asarray(a, float) - np.asarray(b, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float("nan"), "t90": [float("nan"), float("nan")], "n": int(len(x))}
    lo, hi = t_ci90(x)
    return {"mean": float(x.mean()), "t90": [float(lo), float(hi)], "n": int(len(x))}


def _folds(n: int):
    from itasorl import folds
    return list(folds.split(np.arange(n), 5))


# ---------------------------------------------------------------------------
# salience
# ---------------------------------------------------------------------------

def _agreement(labels: np.ndarray, y: np.ndarray) -> float:
    acc = float(np.mean(np.asarray(labels) == np.asarray(y)))
    return max(acc, 1.0 - acc)


def salience_measures(Ha: np.ndarray, Hs: np.ndarray, seed: int = 0) -> dict:
    """Label-free and effect-size measures of world identity in the registered pooled
    features [mean h, final h], standardized over the pooled episodes."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    X = episode_features(np.concatenate([Ha, Hs])).astype(np.float64)
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    Z = StandardScaler().fit_transform(X)
    keep = Z.std(0) > 0
    Z = Z[:, keep]
    total = float(Z.var(0).sum())
    w = LogisticRegression(max_iter=2000).fit(Z, y).coef_[0]
    w = w / np.linalg.norm(w)
    pca = PCA().fit(Z)
    align = np.abs(pca.components_ @ w)
    out = {"n": [int(len(Ha)), int(len(Hs))],
           "share_probe_dir": float((Z @ w).var() / total),
           "share_pc1": float(pca.explained_variance_ratio_[0]),
           "pc_most_aligned": int(np.argmax(align)) + 1,
           "pc_most_aligned_cos": float(align.max())}
    # out-of-fold projection and Cohen's d
    oof = np.full(len(y), np.nan)
    for tr, te in _folds(len(y)):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    a, s = oof[y == 0], oof[y == 1]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(s) - 1) * s.var(ddof=1)) / (len(a) + len(s) - 2))
    out["cohen_d_oof"] = float((s.mean() - a.mean()) / sp) if sp > 0 else float("nan")
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Z)
    out["kmeans_agreement"] = _agreement(km.labels_, y)
    out["kmeans_ari"] = float(adjusted_rand_score(y, km.labels_))
    out["pc1_sign_agreement"] = _agreement((pca.transform(Z)[:, 0] > 0).astype(int), y)
    return out


# ---------------------------------------------------------------------------
# world-draw features
# ---------------------------------------------------------------------------

GRID = 128


def arena_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Euclidean distance in the unit arena. The world has walls (patch_of_earth clips a
    position that crosses an edge and stops its normal velocity), so no distance wraps.
    The 2026-10-15 run used a toroidal distance here; see the spec's third amendment."""
    d = np.asarray(p, float) - np.asarray(q, float)
    return np.sqrt((d ** 2).sum(-1))


def exceedance_rate(max_disp: np.ndarray, r: float) -> float:
    """Share of episodes whose largest displacement from the start passes r. r is the 95th
    percentile of the calibration episodes, a percentile and not a bound (review N01)."""
    d = np.asarray(max_disp, float)
    return float(np.mean(d > r)) if d.size else float("nan")


def terrain_grid(world) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, H) on a GRID x GRID lattice of the unit arena, with H the world's own height
    function evaluated in closed form. Consumes no random draw."""
    g = (np.arange(GRID) + 0.5) / GRID
    x, y = np.meshgrid(g, g, indexing="ij")
    arg = (world._tf[:, 0][:, None, None] * x + world._tf[:, 1][:, None, None] * y
           + world._tph[:, None, None])
    H = 0.5 + 0.5 * np.tanh(np.sum(world._ta[:, None, None] * np.sin(arg), axis=0))
    return x, y, H


def water_mask(world, H: np.ndarray) -> np.ndarray:
    wet = 1.0 / (1.0 + np.exp(-(world.params.sea_level - H) / 0.05))
    return wet >= world.water_thresh


def draw_features(world, r: float) -> dict:
    """The six features of the frozen spec, from a world just after reset."""
    x, y, H = terrain_grid(world)
    water = water_mask(world, H)
    start = np.asarray(world.pos, float)
    dg = arena_dist(np.stack([x, y], -1), start)
    near = dg <= r
    far = ~near
    pel = np.asarray(world.pellets, float)
    dp = arena_dist(pel, start)
    pf = pel[dp > r]
    if len(pf) >= 2:
        i, j = np.triu_indices(len(pf), 1)
        disp = float(arena_dist(pf[i], pf[j]).mean())
    else:
        disp = float("nan")
    return {"I1_nearest_pellet": float(dp.min()),
            "I2_pellets_in_reach": float((dp <= r).sum()),
            "I3_water_near": float(water[near].mean()) if near.any() else float("nan"),
            "N1_water_far": float(water[far].mean()),
            "N2_roughness_far": float(H[far].std()),
            "N3_pellet_dispersion_far": disp}


INSTRUMENTAL = ("I1_nearest_pellet", "I2_pellets_in_reach", "I3_water_near")
INCIDENTAL = ("N1_water_far", "N2_roughness_far", "N3_pellet_dispersion_far")


def residualize_incidental(F: dict) -> dict:
    """Each incidental feature minus its linear prediction from the three instrumental
    ones (with intercept), over the episodes."""
    Xi = np.column_stack([np.ones(len(F[INSTRUMENTAL[0]]))] + [F[k] for k in INSTRUMENTAL])
    out = dict(F)
    for k in INCIDENTAL:
        v = np.asarray(F[k], float)
        beta, *_ = np.linalg.lstsq(Xi, v, rcond=None)
        out[k] = v - Xi @ beta
    return out


def residual_targets(F: dict, k: str):
    """A fold-target callable for feature `k`: for a far feature, its linear prediction from
    the three near features (with intercept) is fit on the training rows alone and
    subtracted from both row sets; for a near feature the raw values. Third amendment,
    2026-10-15: the first run fit this regression over every episode before the decoder's
    cross-validation, so held-out episodes shaped the training targets."""
    v = np.asarray(F[k], float)
    if k not in INCIDENTAL:
        return lambda tr, te: (v[tr], v[te])
    Xi = np.column_stack([np.ones(len(v))] + [np.asarray(F[j], float) for j in INSTRUMENTAL])

    def fn(tr, te):
        beta, *_ = np.linalg.lstsq(Xi[tr], v[tr], rcond=None)
        return v[tr] - Xi[tr] @ beta, v[te] - Xi[te] @ beta
    return fn


ALPHAS = np.logspace(-2, 5, 15)


def ridge_r2_cv(X: np.ndarray, t: np.ndarray, target=None) -> float:
    """Held-out R^2 pooled over the five folds: ridge on columns standardized in fold, its
    strength chosen inside each training fold by leave-one-out over ALPHAS (spec amendment
    2026-10-15: a fixed alpha of 1.0 read far below the constant baseline on the permuted
    target). `target(tr, te)`, when given, returns the training and held-out targets of a
    fold, so a target transform can be fit on the training rows alone (residual_targets)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    pred, truth = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for tr, te in _folds(len(t)):
        t_tr, t_te = (t[tr], t[te]) if target is None else target(tr, te)
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), t_tr)
        pred[te] = m.predict(sc.transform(X[te]))
        truth[te] = t_te
    sst = float(((truth - truth.mean()) ** 2).sum())
    return float(1.0 - ((truth - pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def residual_r2_cv(Xh: np.ndarray, Xo: np.ndarray, t: np.ndarray, target=None) -> float:
    """Held-out R^2 of the state features after the observation summary is regressed out
    in fold, then ridge from the residual, in the standardized state's units, to the target.
    Both ridge strengths are chosen inside the training fold by leave-one-out over ALPHAS.
    `target` as in ridge_r2_cv."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    pred, truth = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for tr, te in _folds(len(t)):
        t_tr, t_te = (t[tr], t[te]) if target is None else target(tr, te)
        so = StandardScaler().fit(Xo[tr])
        sh = StandardScaler().fit(Xh[tr])
        Otr, Ote = so.transform(Xo[tr]), so.transform(Xo[te])
        Htr, Hte = sh.transform(Xh[tr]), sh.transform(Xh[te])
        rm = RidgeCV(alphas=ALPHAS).fit(Otr, Htr)
        # The residual stays in the standardized state's units. Re-standardizing it would
        # inflate the small leftover of the sensory regression back to full scale.
        Rtr, Rte = Htr - rm.predict(Otr), Hte - rm.predict(Ote)
        m = RidgeCV(alphas=ALPHAS).fit(Rtr, t_tr)
        pred[te] = m.predict(Rte)
        truth[te] = t_te
    sst = float(((truth - truth.mean()) ** 2).sum())
    return float(1.0 - ((truth - pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def obs_summary(O: np.ndarray) -> np.ndarray:
    """[mean o, final o] per episode, constant columns dropped."""
    X = np.concatenate([O.mean(1), O[:, -1]], axis=-1).astype(np.float64)
    return X[:, X.std(0) > 0]


# ---------------------------------------------------------------------------
# erasure
# ---------------------------------------------------------------------------

def probe_block_directions(Ha: np.ndarray, Hs: np.ndarray) -> np.ndarray:
    """(2, hidden): the registered probe's mean-block and final-block coefficient vectors,
    each divided by its scaler scale (raw state space), fit on every episode."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    hid = Ha.shape[-1]
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    coef = clf[-1].coef_[0] / clf[0].scale_
    return np.stack([coef[:hid], coef[hid:]])


def projector(vectors: np.ndarray, rank: int | None = None, tol: float = 1e-10) -> np.ndarray:
    """Orthogonal projector onto the span of the rows of `vectors`, built from the left
    singular vectors whose singular value exceeds tol times the largest. Reduced QR returned
    extra arbitrary directions for rank-deficient rows (review N10, 2026-10-15). With `rank`,
    a span of any other dimension raises."""
    V = np.asarray(vectors, float).T
    if not np.any(V):
        found, P = 0, np.zeros((V.shape[0], V.shape[0]))
    else:
        U, s, _ = np.linalg.svd(V, full_matrices=False)
        keep = s > tol * s[0]
        found = int(keep.sum())
        P = U[:, keep] @ U[:, keep].T
    if rank is not None and found != rank:
        raise ValueError(f"projector: the rows span rank {found}, rank {rank} required")
    return P


def sham_projector(P: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """A rank-`rank` projector onto a random subspace orthogonal to P's range, drawn from the
    RNG of mechanism_readouts.sham_direction (seed 1000003 plus the agent seed)."""
    rng = np.random.default_rng(1_000_003 + seed)
    V = rng.normal(size=(P.shape[0], rank))
    V = V - P @ V
    return projector(V.T)


def erase(H: np.ndarray, P: np.ndarray, m: np.ndarray) -> np.ndarray:
    """h - P (h - m) on the last axis."""
    return H - (H - m) @ P


# ---------------------------------------------------------------------------
# development
# ---------------------------------------------------------------------------

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")
