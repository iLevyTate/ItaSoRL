"""L0 and evaluation-world audit (revision step 5, docs/REVISION_2026-10.md).

The pooled readout scores every agent seed on the same two world samples: an authentic
pool from seed base 800000 and a surrogate pool from seed base 850000. Three consequences
are measured here.

1. Pre-intervention decodability. If the two world samples differ in a way the probe can
   read before any surrogate dynamics act, part of a "world identity" signal is a sampling
   difference. `pre_intervention_probe` decodes pool membership from the observation at
   reset, which no dynamics have touched, using the standard probe and partition.
2. World-sample dependence of L0. At drift 0 both pools are authentic, so the L0 target
   measures how separable two finite samples of worlds look to the agent's state. Because
   the samples are fixed, the ten agent seeds are not independent draws of that quantity.
   `l0_world_samples` rescores a drift-0 agent on independent pairs of world samples.
3. Balanced sampling. `paired_pooled_readout` draws both pools from the SAME seeds, so each
   authentic episode has a surrogate twin with an identical initial state, and both twins
   share a CV group. At drift 0 the twins are identical and the target is exactly 0.5 by
   construction; at drift > 0 any separation comes from the dynamics alone.

All readouts use the standard probe family (`grouped_auroc`) and the active fold scheme,
and record the partition they used (`itasorl.folds.partition_record`).
"""

from __future__ import annotations

import numpy as np

from . import folds
from .experiment_a import grouped_auroc
from .experiment_b import episode_features
from .experiment_b2 import _seeds, collect_pool, make_world, pooled_readout
from .stats import auroc_ci, cluster_auroc_ci, t_ci90

STANDARD_BASES = (800_000, 850_000)
# Independent world-sample pairs for the L0 audit, disjoint from every seed base the
# pipeline uses (training <= 285_800; evaluation 555_000 to 970_109).
AUDIT_BASES = tuple((1_000_000 + 100_000 * k, 1_050_000 + 100_000 * k) for k in range(8))


def initial_observations(seed_base: int, n: int, params, ray_steps: int = 5,
                         drift_sigma: float = 0.0) -> np.ndarray:
    """Observation at reset for worlds seed_base + i, i < n: before any transition."""
    out = []
    for i in range(n):
        w = make_world(params, drift_sigma, ray_steps)
        out.append(np.asarray(w.reset(_seeds(seed_base + i)).obs, np.float64))
    return np.stack(out)


def pre_intervention_probe(params, bases=STANDARD_BASES, n: int = 110, ray_steps: int = 5,
                           seed: int = 0) -> dict:
    """Decode pool membership (base a vs base b) from the reset observation alone."""
    Xa = initial_observations(bases[0], n, params, ray_steps)
    Xb = initial_observations(bases[1], n, params, ray_steps)
    X = np.concatenate([Xa, Xb])
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    g = np.arange(2 * n)
    auc, yv, pv = grouped_auroc(X, y, g, return_oof=True)
    lo, hi = auroc_ci(yv, pv, seed=seed) if yv.size else (float("nan"), float("nan"))
    return {"bases": list(bases), "n_per_pool": n, "target": auc, "oof_ci95": [lo, hi],
            "partition": folds.partition_record(g, y)}


def world_sample_scan(agent, norm, params, drift_sigma: float, *, bases=AUDIT_BASES,
                      n_eps: int = 110, steps: int = 24, ray_steps: int = 5,
                      device: str = "cpu", seed: int = 0,
                      first_state: bool = False) -> list[dict]:
    """Pooled target of one agent on each pair of independent world samples, at any drift.

    At drift 0 both pools are authentic and this measures how separable two finite samples
    of worlds look (the L0 use, via `l0_world_samples`). At drift > 0 it measures how much
    the headline reading itself moves with the draw of evaluation worlds, which is the
    question frozen in docs/specs/2026-10-08-drift-045-world-sample-sensitivity-design.md.
    """
    rows = []
    for a, b in bases:
        r = pooled_readout(agent, norm, params, drift_sigma=drift_sigma, n_eps=n_eps,
                           steps=steps, ray_steps=ray_steps, device=device, seed=seed,
                           seed_base_auth=a, seed_base_surr=b)
        row = {"bases": [a, b], "target": r["target"], "n": r["n"]}
        if first_state:
            row["first_state_target"] = _first_state_target(agent, norm, params, a, b,
                                                            n_eps, ray_steps, device)
        rows.append(row)
    return rows


def l0_world_samples(agent, norm, params, *, bases=AUDIT_BASES, n_eps: int = 110,
                     steps: int = 24, ray_steps: int = 5, device: str = "cpu",
                     seed: int = 0) -> list[dict]:
    """L0 target of one drift-0 agent on each pair of independent world samples."""
    return world_sample_scan(agent, norm, params, 0.0, bases=bases, n_eps=n_eps,
                             steps=steps, ray_steps=ray_steps, device=device, seed=seed,
                             first_state=True)


def world_sample_summary(independent_means, registered_mean: float, *,
                         bar: float = 0.65) -> dict:
    """Adjudicate the drift-0.45 world-sample scan under the rule frozen on 2026-10-08.

    The means are SIGNED and are never folded about chance. Folding is the conservative
    choice at drift 0, where the null is chance and a reading of 0.406 is as much
    separation as 0.594; where a real signed effect exists it is anti-conservative,
    because it converts a weak draw into a strong number and would manufacture a pass.

    Three outcomes, mutually exclusive and jointly exhaustive. Let m be the independent
    draw means and r the registered draw's rank among all nine, 1 being the largest:
      DRAW-DEPENDENT  min(m) < bar
      SECURE          min(m) >= bar and r >= 5   (registered at or below the median)
      INDETERMINATE   min(m) >= bar and r <= 4   (registered above the median)
    """
    m = np.asarray(list(independent_means), dtype=float)
    if m.size == 0:
        raise ValueError("world_sample_summary needs at least one independent draw")
    allm = np.append(m, float(registered_mean))
    rank = int((allm > float(registered_mean)).sum()) + 1
    below = bool(m.min() < bar)
    verdict = "DRAW-DEPENDENT" if below else ("SECURE" if rank >= 5 else "INDETERMINATE")
    lo, hi = t_ci90(m) if m.size > 1 else (float("nan"), float("nan"))
    return {
        "n_draws": int(m.size),
        "per_draw": [float(x) for x in m],
        "registered": float(registered_mean),
        "registered_rank": rank,
        "min": float(m.min()),
        "max": float(m.max()),
        "mean": float(m.mean()),
        "between_draw_sd": float(m.std(ddof=1)) if m.size > 1 else float("nan"),
        "t90_over_draws": [float(lo), float(hi)],
        "bar": float(bar),
        "n_at_or_above_bar": int((m >= bar).sum()),
        "verdict": verdict,
    }


def _first_state_target(agent, norm, params, base_a, base_b, n, ray_steps, device) -> float:
    """Decode pool membership from h_1, the state after the reset observation only."""
    Ha, _ = collect_pool(agent, norm, params, 0.0, n, 1, device, base_a, ray_steps)
    Hb, _ = collect_pool(agent, norm, params, 0.0, n, 1, device, base_b, ray_steps)
    X = np.concatenate([Ha[:, 0], Hb[:, 0]])
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hb))].astype(int)
    return grouped_auroc(X, y, np.arange(len(y)))


def paired_pooled_readout(agent, norm, params, drift_sigma: float, *, n_eps: int = 110,
                          steps: int = 24, ray_steps: int = 5, device: str = "cpu",
                          seed_base: int = 800_000, seed: int = 0) -> dict:
    """Pooled readout with balanced initial states: authentic and surrogate episodes come
    from the same world seeds, both members of a pair share a CV group, and a pair is kept
    only if both members survive (symmetric, so survivorship cannot separate the pools).
    The CI resamples pairs (cluster bootstrap) and is conditional on the fitted probes."""
    Ha, _, ia = collect_pool(agent, norm, params, 0.0, n_eps, steps, device, seed_base,
                             ray_steps, return_index=True)
    Hs, _, is_ = collect_pool(agent, norm, params, drift_sigma, n_eps, steps, device,
                              seed_base, ray_steps, return_index=True)
    common = np.intersect1d(ia, is_)
    sel_a = np.searchsorted(ia, common)
    sel_s = np.searchsorted(is_, common)
    out = {"drift": float(drift_sigma), "seed_base": int(seed_base), "n_pairs": int(len(common)),
           "dropped_pairs": int(n_eps - len(common)), "target": float("nan"),
           "cluster_ci95": [float("nan"), float("nan")]}
    if len(common) < 5:
        return out
    X = episode_features(np.concatenate([Ha[sel_a], Hs[sel_s]]))
    y = np.r_[np.zeros(len(common)), np.ones(len(common))].astype(int)
    g = np.r_[common, common]
    auc, yv, pv = grouped_auroc(X, y, g, return_oof=True)
    # out-of-fold rows come back fold by fold; recover their groups for the cluster bootstrap
    gv = np.concatenate([g[te] for _, te in folds.split(g) if len(np.unique(y[te])) > 1])
    out.update(target=auc, cluster_ci95=list(cluster_auroc_ci(yv, pv, gv, seed=seed)),
               partition=folds.partition_record(g, y),
               identical_first_state=bool(np.array_equal(Ha[sel_a, 0], Hs[sel_s, 0])))
    return out
