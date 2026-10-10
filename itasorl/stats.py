"""
ITASORL - small statistics helpers for methodologically-tight readouts.

The headline tool here is an EQUIVALENCE test (two one-sided tests, TOST). The L0
control must be shown to sit AT chance, and "we failed to reject a difference from
0.5" is not the same claim as "it is equivalent to 0.5". TOST makes the at-chance
claim positively: it rejects the null of a meaningful difference in favour of
equivalence within a pre-registered margin. (ITASORL.md sec.13 item #2.)

Deliberately dependency-light (numpy + scipy.stats.t if available, else a normal
approximation) so it runs anywhere the rest of the pipeline runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import t as _student_t
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy optional
    _HAVE_SCIPY = False


def _t_sf(tstat: float, df: int) -> float:
    """Upper-tail P(T > tstat). Student-t if scipy is present, else normal approx."""
    if df <= 0:
        return float("nan")
    if _HAVE_SCIPY:
        return float(_student_t.sf(tstat, df))
    # normal approximation to the t survival function
    return float(0.5 * math.erfc(tstat / math.sqrt(2.0)))


@dataclass
class EquivalenceResult:
    mean: float
    margin: float
    lower: float          # (h0 - margin)
    upper: float          # (h0 + margin)
    p_lower: float        # H0: mean <= lower   (one-sided)
    p_upper: float        # H0: mean >= upper   (one-sided)
    p_value: float        # max(p_lower, p_upper) - the TOST p
    equivalent: bool      # p_value < alpha AND the mean lies inside the band
    n: int

    def __str__(self) -> str:
        verdict = "EQUIVALENT to chance" if self.equivalent else "NOT shown equivalent"
        return (f"mean={self.mean:.3f}  band=[{self.lower:.3f},{self.upper:.3f}]  "
                f"TOST p={self.p_value:.4f}  -> {verdict} (n={self.n})")


def equivalence_test(values, h0: float = 0.5, margin: float = 0.05,
                     alpha: float = 0.05) -> EquivalenceResult:
    """TOST equivalence test that `values` are within +/- margin of h0.

    values: a sample of summary statistics (e.g. one AUROC per seed). We test the
    composite null "the true mean is OUTSIDE [h0-margin, h0+margin]" with two
    one-sided t-tests; rejecting both (p<alpha) concludes practical equivalence.

    Returns an EquivalenceResult; `.equivalent` is the gate the L0 control uses.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    lower, upper = h0 - margin, h0 + margin
    mean = float(x.mean())
    if n < 2:
        # With no variance there is no test. A point inside the band is not equivalence, so
        # this fails closed: the gate stays unmet and the caller sees NaN p-values.
        return EquivalenceResult(mean, margin, lower, upper, float("nan"),
                                 float("nan"), float("nan"), False, n)
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 1e-12
    df = n - 1
    # H0a: mean <= lower  -> reject if mean is significantly ABOVE lower
    t_lower = (mean - lower) / se
    p_lower = _t_sf(t_lower, df)
    # H0b: mean >= upper  -> reject if mean is significantly BELOW upper
    t_upper = (upper - mean) / se
    p_upper = _t_sf(t_upper, df)
    p = max(p_lower, p_upper)
    equivalent = bool(p < alpha and lower <= mean <= upper)
    return EquivalenceResult(mean, margin, lower, upper, p_lower, p_upper, p, equivalent, n)


# ---------------------------------------------------------------------------
# AUROC uncertainty. A point AUROC says nothing about precision; reviewers of a
# null result will (rightly) ask for an interval. These are dependency-light
# (numpy only) so they run wherever the pipeline runs.
# ---------------------------------------------------------------------------
def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks (1-based), matching scipy.stats.rankdata(method='average')."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(a.size, dtype=np.intp)
    inv[sorter] = np.arange(a.size)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], a.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auroc(y_true, y_score) -> float:
    """AUROC via the Mann-Whitney U / rank statistic (handles ties). NaN if one class."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(y_score)
    sum_pos = float(ranks[y_true == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_ci(y_true, y_score, level: float = 0.95, n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float]:
    """Stratified bootstrap CI for a single AUROC. Resamples positives and negatives
    with replacement (keeps both classes present) and recomputes the rank-AUROC; no
    model refit, so it is cheap enough to attach to every reported number."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if pos.size == 0 or neg.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([
            pos[rng.integers(0, pos.size, pos.size)],
            neg[rng.integers(0, neg.size, neg.size)],
        ])
        aucs[b] = auroc(y_true[idx], y_score[idx])
    a = (1.0 - level) / 2.0
    return (float(np.nanpercentile(aucs, 100 * a)), float(np.nanpercentile(aucs, 100 * (1 - a))))


def cluster_auroc_ci(y_true, y_score, clusters, level: float = 0.95, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for one AUROC that resamples CLUSTERS (e.g. the two members
    of a matched pair, which share a world seed) with replacement and keeps every member of
    a drawn cluster, so dependence inside a pair is preserved. auroc_ci treats every sample
    as independent and is only right when they are. Conditional on the fitted probe scores
    (no refit)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    if len(np.unique(y_true)) < 2 or len(uniq) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        aucs[b] = auroc(y_true[idx], y_score[idx])
    a = (1.0 - level) / 2.0
    return (float(np.nanpercentile(aucs, 100 * a)), float(np.nanpercentile(aucs, 100 * (1 - a))))


def fold_mean_auroc_ci(fold_y: list, fold_p: list, level: float = 0.95, n_boot: int = 2000,
                       seed: int = 0, fold_groups: list | None = None) -> tuple[float, float]:
    """Percentile bootstrap interval ALIGNED with the mean-of-fold-AUROCs point estimator
    (revision step 13): each draw resamples within every test fold (whole groups when
    `fold_groups` is given, else stratified by class) and averages the per-fold AUROCs.
    `auroc_ci` on the pooled out-of-fold scores is an interval for a different quantity, the
    pooled out-of-fold AUROC. Both are conditional on the fitted probes (no refit)."""
    rng = np.random.default_rng(seed)
    folds_ = [(np.asarray(y).astype(int), np.asarray(p, float)) for y, p in zip(fold_y, fold_p)]
    if not folds_:
        return (float("nan"), float("nan"))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for k, (y, p) in enumerate(folds_):
            if fold_groups is not None:
                g = np.asarray(fold_groups[k])
                ug = np.unique(g)
                pick = ug[rng.integers(0, len(ug), len(ug))]
                idx = np.concatenate([np.flatnonzero(g == u) for u in pick])
            else:
                pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
                idx = np.concatenate([pos[rng.integers(0, pos.size, pos.size)],
                                      neg[rng.integers(0, neg.size, neg.size)]])
            vals.append(auroc(y[idx], p[idx]))
        draws[b] = np.nanmean(vals)
    a = (1.0 - level) / 2.0
    return (float(np.nanpercentile(draws, 100 * a)), float(np.nanpercentile(draws, 100 * (1 - a))))


def paired_contrast(a, b, margin: float = 0.05) -> dict:
    """Seed-paired contrast a - b with its t-based 90% CI and whether the CI clears `margin`
    (revision step 13: the registered 0.05 margins are claims about a difference, so they get
    an interval of the difference, not two separate intervals). Conditional on the evaluation
    worlds and surrogate the seeds share."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    d = a - b
    lo, hi = t_ci90(d) if d.size > 1 else (float("nan"), float("nan"))
    return {"diff_per_seed": d.tolist(), "mean": float(d.mean()), "t90": [float(lo), float(hi)],
            "mean_ge_margin": bool(d.mean() >= margin), "t90_lower_ge_margin": bool(lo >= margin),
            "margin": float(margin), "n": int(d.size)}


def mean_ci(values, level: float = 0.90, n_boot: int = 10000,
            seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI of the across-seed mean. Seeds are the replication unit for a null
    claim (cf. Colas et al., 'How many random seeds?'), so this is the decision-relevant
    interval. Returns (mean, lo, hi)."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    a = (1.0 - level) / 2.0
    return (float(x.mean()), float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a))))


def t_ci90(values) -> tuple[float, float]:
    """Student-t 90% CI of the across-seed mean - the repo's decision interval.

    Unlike mean_ci (a bootstrap percentile band), this is the parametric t interval the
    FINDINGS decision layer quotes for per-seed AUROCs. Kept here as the single audited
    path so experiments and the site generator agree. NaNs when n<2 (variance undefined).
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(x.mean())
    se = float(x.std(ddof=1)) / np.sqrt(n)
    if not _HAVE_SCIPY:
        # The normal quantile (1.645) would narrow this by about a quarter at n = 10 while
        # every artifact and the manuscript label the result t-based. Refuse rather than
        # relabel: scipy is a declared dependency of this project.
        raise RuntimeError(
            "t_ci90 needs scipy for the Student-t quantile; the normal approximation would "
            "silently report a narrower interval under a t label. Install scipy.")
    crit = float(_student_t.ppf(0.95, n - 1))
    return (mean - crit * se, mean + crit * se)


@dataclass
class RopeResult:
    """Bootstrap ROPE check. NOT a Bayesian analysis (revision step 13): there is no prior and
    no posterior. `hdi` and `p_in_rope` keep their historical field names because committed
    artifacts store them under those keys; read them as `boot_interval` (a percentile
    bootstrap interval of the across-seed mean, not a highest-density interval) and
    `boot_share_in_rope` (the share of bootstrap means inside the ROPE, not a posterior
    probability). TOST (`equivalence_test`) is the formal equivalence test."""
    mean: float
    rope: tuple[float, float]
    hdi: tuple[float, float]      # percentile bootstrap interval of the mean (historical key)
    p_in_rope: float             # share of bootstrap means inside the ROPE (historical key)
    accept: bool                 # the bootstrap interval lies entirely inside the ROPE
    n: int

    @property
    def boot_interval(self) -> tuple[float, float]:
        return self.hdi

    @property
    def boot_share_in_rope(self) -> float:
        return self.p_in_rope

    def __str__(self) -> str:
        verdict = "bootstrap interval inside ROPE" if self.accept else "not inside ROPE"
        return (f"mean={self.mean:.3f}  ROPE=[{self.rope[0]:.3f},{self.rope[1]:.3f}]  "
                f"95% percentile-bootstrap interval=[{self.hdi[0]:.3f},{self.hdi[1]:.3f}]  "
                f"share of bootstrap means in ROPE={self.p_in_rope:.3f}  -> {verdict} (n={self.n})")


def rope_test(values, rope: tuple[float, float] = (0.45, 0.55), level: float = 0.95,
              n_boot: int = 20000, seed: int = 0) -> RopeResult:
    """Descriptive equivalence leg beside TOST: resample the per-seed values, take the
    percentile interval of the bootstrap means, and check whether it lies inside the ROPE.
    Seeds are resampled as independent; when they share evaluation worlds or a surrogate the
    interval is conditional on those (revision step 13). Not a posterior; TOST is the test."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    lo_r, hi_r = rope
    if n < 2:
        # Fails closed, as equivalence_test does: a single point cannot bound a mean, so the
        # interval is undefined and acceptance is refused rather than inferred from the point.
        m = float(x.mean()) if n else float("nan")
        return RopeResult(m, (lo_r, hi_r), (float("nan"), float("nan")), float("nan"), False, n)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    a = (1.0 - level) / 2.0
    hdi = (float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a))))
    p_in = float(np.mean((means >= lo_r) & (means <= hi_r)))
    accept = bool(hdi[0] >= lo_r and hdi[1] <= hi_r)
    return RopeResult(float(x.mean()), (lo_r, hi_r), hdi, p_in, accept, n)
