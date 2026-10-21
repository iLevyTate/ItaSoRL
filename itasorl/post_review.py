"""Post-review readouts on the saved agents of a corrected-trainer run
(docs/specs/2026-10-11-post-review-readouts-design.md, frozen before any run).

Five readout-only analyses, each named by the review finding it answers:

  r210  probe classes: a random forest and a small MLP beside the registered linear probe,
        on the registered pools regenerated from the saved agents (bit-checked), same
        features, same in-fold partition, same eight label permutations
  r35   normalizer-matched replay: the replay protocol of itasorl/eval_protocols.py with
        each trunk's own normalizer, the survival agent's, and one refit on the replayed
        streams (pooled and in-fold)
  r206  law deviation along visited states: |G(v, a) - f(v, a)| at every visited step of
        surrogate-world walks, G the run's learned velocity law, f the authentic law
  r33   draw-free balanced readout (itasorl.l0_audit.paired_pooled_readout) on every arm at
        both agent drifts, plus the C1 minus C2 survival contrast across two outputs
  r31   stream-decoder capacity curve: the 17.8 supervised GRU at 1x, 2x, 4x and 8x the
        trunk's recurrent width, and one 1D-conv reader, on the survival agent's own
        drift-0.45 streams and on the scripted walk
  r205  which half of the registered feature carries the signal: the linear probe on mean h
        alone, on h_T alone and on the registered concatenation, on the same regenerated and
        bit-checked pools as r210 (amendment of 2026-10-12)

None of these changes a registered verdict. This module holds the pure pieces (scorers,
normalizers, the deviation integral, summaries, the partial-file store, the seed bases);
scripts/run_post_review_readouts.py drives the rollouts. Torch, the worlds and the agents
are imported lazily, inside the functions that need them.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from . import folds

SPEC = "docs/specs/2026-10-11-post-review-readouts-design.md"
READOUTS = ("r210", "r205", "r35", "r206", "r33", "r31")
ARMS = ("survival", "predictor", "untrained")

# ---------------------------------------------------------------------------
# Seed bases
# ---------------------------------------------------------------------------
# Fresh evaluation worlds, shared by r35, r206 and r31 (one authentic and one surrogate
# sample, so the three readouts see the same episodes), and one base for the balanced pairs
# of r33. Every base registered before these three sits below 2_000_000; the bases
# registered after them (the 2026-10-15 readouts at the tail of BASES_IN_USE, and
# REGISTERED_LATER) were placed to miss them, which base_collisions checks.
FRESH_AUTH_BASE = 2_100_000
FRESH_SURR_BASE = 2_150_000
BALANCED_FRESH_BASE = 2_200_000
# The registered pair. r210 re-scores it by design (the bit check needs it); r33 reads the
# balanced readout at 800000 as well, as artifacts/l0_audit/corrected_l3_h8_wm.json did, as a
# continuity entry beside the fresh base.
REGISTERED_AUTH_BASE = 800_000
REGISTERED_SURR_BASE = 850_000
# Action RNG of the scripted walk (eval_protocols.POLICY_SEED_BASE): action stream i is
# default_rng(600000 + i). It seeds actions, never a world.
SCRIPTED_POLICY_BASE = 600_000
POOL_N = 110

# experiment_b2._seeds(b) is SeedBundle(world=b, weather=b+7000, ecology=b+13000), and
# SeedBundle.streams() adds 10001 to the weather seed and 20002 to the ecology seed. Episode
# i of a pool at base b therefore draws from default_rng(b+i), default_rng(b+17001+i) and
# default_rng(b+33002+i).
STREAM_OFFSETS = (0, 7_000 + 10_001, 13_000 + 20_002)

#: every world-seed base the code and the record use, as (name, base, episodes). Spans are
#: generous where a runner adds a per-seed offset. Overlaps among these are documented in
#: itasorl/l0_audit.py and do not concern this module; the new bases must miss all of them.
BASES_IN_USE: tuple = (
    ("survival and predictor training (100000 + 10000 s + 16 u, 200000 + 9000 s + 16 u)", 100_000, 185_800),
    # The entry above spans the survival schedule of seeds 0 to 19 and the predictor schedule
    # of seeds 0 to 9 only. The predictor schedule of seeds 14 to 19 (B1 and B2, 2026-10-12)
    # runs from 200000 + 9000 x 14 = 326000 to 200000 + 9000 x 19 + 16 x 300 = 375800, past
    # that span and inside the Experiment C panel entry below. The overlap is a historical
    # fact of two training schedules, not a new draw: Experiment C trains its own populations
    # and no B1 or B2 readout draws from those panels (H25, 2026-10-17).
    ("predictor training, seeds 14 to 19 (200000 + 9000 s + 16 u)", 326_000, 49_800),
    ("experiment C panels (320000 + 10000 s + p)", 320_000, 100_000),
    ("untrained normalizer warmup", 555_000, 8),
    ("matched pairs", 700_000, 60),
    ("registered authentic pool", REGISTERED_AUTH_BASE, POOL_N),
    ("registered surrogate pool", REGISTERED_SURR_BASE, POOL_N),
    ("transfer readout authentic", 860_000, POOL_N),
    ("transfer readout surrogate", 870_000, POOL_N),
    ("cross-recipe rff authentic", 880_000, POOL_N),
    ("cross-recipe rff surrogate", 890_000, POOL_N),
    ("engagement metric", 900_000, 64),
    ("2026-10-12 balanced confirmation pairs", 910_000, 110),
    ("survival return", 920_000, 64),
    ("common garden", 930_000, POOL_N),
    ("cross-recipe cd authentic", 940_000, POOL_N),
    ("cross-recipe cd surrogate", 950_000, POOL_N),
    ("gn comparator and L1 H2 authentic (960000 + seed)", 960_000, POOL_N + 10),
    ("gn comparator and L1 H2 surrogate (970000 + seed)", 970_000, POOL_N + 10),
    ("persistence and H2 h16 authentic", 980_000, POOL_N),
    ("H2 h16 surrogate", 990_000, POOL_N),
    ("H2 h32 authentic", 1_000_000, POOL_N),
    ("H2 h32 surrogate", 1_010_000, POOL_N),
    ("H2 h64 authentic", 1_020_000, POOL_N),
    ("H2 h64 surrogate", 1_030_000, POOL_N),
    *((f"L0 audit draw {k + 1} authentic", 1_000_000 + 100_000 * k, POOL_N) for k in range(8)),
    *((f"L0 audit draw {k + 1} surrogate", 1_050_000 + 100_000 * k, POOL_N) for k in range(8)),
    ("scarcity calibration", 1_300_000, 200),
    ("experiment C survival rollouts (panel base + 1000000)", 1_320_000, 100_000),
    ("mechanism readouts (authentic and surrogate share it)", 1_400_000, 64),
    ("mechanism surrogate base named in the results manifest", 1_450_000, 64),
    ("qd comparator authentic", 1_900_000, POOL_N),
    ("qd comparator surrogate", 1_950_000, POOL_N),
    # correlated-jitter sweep, spec 2026-10-15 (registered after the post-review bases, above them)
    ("ar comparator authentic", 2_400_000, POOL_N),
    ("ar comparator surrogate", 2_450_000, POOL_N),
    ("2026-10-15 world-feature readout (both worlds share it)", 2_600_000, 1_100),
    ("2026-10-15 world-feature reach calibration", 2_650_000, POOL_N),
    ("sweep C authentic (companion spec 2026-10-15)", 2_500_000, POOL_N),
    ("sweep C G_ar, drawn afresh per episode (companion spec 2026-10-15)", 2_550_000, POOL_N),
)

NEW_BASES: tuple = (
    ("post-review fresh authentic", FRESH_AUTH_BASE, POOL_N),
    ("post-review fresh surrogate", FRESH_SURR_BASE, POOL_N),
    ("post-review balanced pairs", BALANCED_FRESH_BASE, POOL_N),
)

#: Bases registered after the post-review readouts, for the closing readouts of 2026-10
#: (runbook stage 1.5): the B2 walk-speed pools, the B6 development transfer pools and the
#: fresh read base of the G03 sensory-echo repeat. Each of their runners refuses to start
#: when base_collisions(new=own, in_use=all_bases_in_use()) is non-empty. They are not
#: copied into BASES_IN_USE or NEW_BASES: those two tuples are the record of the bases
#: the registered readouts drew, and tests/test_world_seed_bases.py checks these against
#: them and against each other.
REGISTERED_LATER: tuple = (
    ("B2-auth", 2_800_000, POOL_N),
    ("B2-surr", 2_850_000, POOL_N),
    ("B6-auth", 2_700_000, POOL_N),
    ("B6-surr", 2_750_000, POOL_N),
    ("G03-fresh", 2_250_000, POOL_N),
    # Prospective balanced confirmation, spec 2026-10-20: PB1 and PB2 matched pairs, and the
    # descriptive second draw of their sensory-echo readout.
    ("PB-primary", 3_000_000, POOL_N),
    ("PB-draw2", 3_100_000, POOL_N),
)


def all_bases_in_use() -> tuple:
    """Every base a registered readout has drawn: BASES_IN_USE plus the post-review
    NEW_BASES. The `in_use` argument for a base registered later."""
    return BASES_IN_USE + NEW_BASES


def registered_later(name: str) -> tuple[str, int, int]:
    """The REGISTERED_LATER entry called `name` (KeyError when there is none)."""
    for entry in REGISTERED_LATER:
        if entry[0] == name:
            return entry
    raise KeyError(name)


def seed_footprint(base: int, n: int) -> list[tuple[int, int]]:
    """Half-open ranges of the integer RNG seeds a pool of n episodes at `base` draws from."""
    return [(base + o, base + o + n) for o in STREAM_OFFSETS]


def footprints_overlap(base_a: int, n_a: int, base_b: int, n_b: int) -> bool:
    return any(lo1 < hi2 and lo2 < hi1 for lo1, hi1 in seed_footprint(base_a, n_a)
               for lo2, hi2 in seed_footprint(base_b, n_b))


def base_collisions(new=NEW_BASES, in_use=BASES_IN_USE) -> list[tuple[str, str]]:
    """Every (new, existing) pair whose seed footprints intersect, and every pair of new
    bases that intersect each other. Empty when the new bases are fresh."""
    hits = [(a, b) for a, ba, na in new for b, bb, nb in in_use if footprints_overlap(ba, na, bb, nb)]
    for i, (a, ba, na) in enumerate(new):
        for b, bb, nb in new[i + 1:]:
            if footprints_overlap(ba, na, bb, nb):
                hits.append((a, b))
    return hits


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def seed_summary(values) -> dict:
    """Across-seed mean with the t-based 90% interval over the finite values."""
    from .stats import t_ci90
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    return {"n": int(v.size), "mean": float(v.mean()) if v.size else float("nan"),
            "t90": [float(x) for x in t_ci90(v)] if v.size > 1 else [float("nan")] * 2}


def paired_summary(a: dict, b: dict) -> dict:
    """Seed-paired a minus b over the seeds where both are finite; keys are seed labels."""
    seeds = sorted(set(a) & set(b), key=lambda s: int(s))
    d = {s: float(a[s]) - float(b[s]) for s in seeds
         if a[s] is not None and b[s] is not None and np.isfinite(a[s]) and np.isfinite(b[s])}
    return {**seed_summary(list(d.values())), "seeds": sorted(d, key=int),
            "per_seed": [d[s] for s in sorted(d, key=int)]}


def interval_label(summary: dict, *, above: float = 0.0) -> str:
    """ABOVE when the t90 interval lies above `above`, BELOW when it lies below, else
    INCLUDES; NA without an interval."""
    lo, hi = summary.get("t90", [float("nan")] * 2)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    if lo > above:
        return "ABOVE"
    if hi < above:
        return "BELOW"
    return "INCLUDES"


# ---------------------------------------------------------------------------
# r210: probe classes
# ---------------------------------------------------------------------------
N_PERM = 8                 # pooled_readout's label permutations
RF_TREES = 200
MLP_HIDDEN = (64,)
MLP_ALPHA = 1e-3
MLP_LR = 1e-3
MLP_MAX_ITER = 500


def registered_permutations(y: np.ndarray, seed: int, n_perm: int = N_PERM) -> list[np.ndarray]:
    """The label permutations pooled_readout draws for its shuffled control: default_rng(seed),
    then n_perm successive permutations of y. Nothing else draws from that generator first."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(y) for _ in range(n_perm)]


def linear_probe_auroc(X, y, groups=None, *, seed: int = 0, **_) -> dict:
    """The registered probe (scaler + logistic regression, in-fold), as grouped_auroc runs it."""
    from .experiment_a import grouped_auroc
    g = np.arange(len(y)) if groups is None else groups
    return {"auroc": float(grouped_auroc(X, y, g))}


def rf_probe_auroc(X, y, groups=None, *, seed: int = 0, n_estimators: int = RF_TREES, **_) -> dict:
    """Random forest probe. With the default 200 trees this is behavior_audit's RF path
    (behavior_only_auroc, nonlinear=True) applied to the state features."""
    from threadpoolctl import threadpool_limits

    from .behavior_audit import behavior_only_auroc
    with threadpool_limits(1):
        if n_estimators == RF_TREES:
            return {"auroc": float(behavior_only_auroc(X, y, groups, nonlinear=True, seed=seed))}
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        g = np.arange(len(y)) if groups is None else groups
        aucs = []
        for tr, te in folds.split(g):
            if len(np.unique(y[te])) < 2:
                continue
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        return {"auroc": float(np.mean(aucs)) if aucs else float("nan")}


def mlp_probe_auroc(X, y, groups=None, *, seed: int = 0, max_iter: int = MLP_MAX_ITER, **_) -> dict:
    """Small MLP probe, fixed architecture: scaler, one hidden layer of 64 ReLU units, L2 1e-3,
    Adam at 1e-3, at most 500 iterations, random_state = seed; scaler and net fit in-fold.
    Reports per-fold iterations and how many folds stopped at the iteration cap."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    g = np.arange(len(y)) if groups is None else groups
    aucs, iters = [], []
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for tr, te in folds.split(g):
            if len(np.unique(y[te])) < 2:
                continue
            clf = make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=MLP_ALPHA,
                                              learning_rate_init=MLP_LR, max_iter=max_iter,
                                              random_state=seed))
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
            iters.append(int(clf[-1].n_iter_))
    return {"auroc": float(np.mean(aucs)) if aucs else float("nan"), "n_iter": iters,
            "hit_max_iter": int(sum(i >= max_iter for i in iters))}


PROBE_CLASSES = {"linear": linear_probe_auroc, "rf": rf_probe_auroc, "mlp": mlp_probe_auroc}


def score_probe_classes(X, y, *, seed: int, perms, classes=("rf", "mlp"), scorers=None,
                        **kw) -> dict:
    """Each class's in-fold AUROC on (X, y), its shuffled control (mean over the given label
    permutations, every one scored with the same class and fold partition), and the
    selectivity target minus shuffled. Extra keywords reach the scorers (rf n_estimators)."""
    scorers = scorers or PROBE_CLASSES
    out = {}
    for c in classes:
        f = scorers[c]
        res = dict(f(X, y, seed=seed, **kw))
        target = float(res.pop("auroc"))
        draws = [float(f(X, yp, seed=seed, **kw)["auroc"]) for yp in perms]
        sh = float(np.mean(draws)) if draws else float("nan")
        out[c] = {"target": target, "shuffled": sh, "selectivity": target - sh,
                  "shuffled_draws": draws, **res}
    return out


def r210_cell(Ha: np.ndarray, Hs: np.ndarray, *, pool: dict, recorded: dict, seed: int,
              n_perm: int = N_PERM, rf_trees: int = RF_TREES, scorers=None) -> dict:
    """One (drift, seed, arm) cell of r210 from the regenerated pools and pooled_readout's
    output. The cell is reproduced only when the regenerated target equals the recorded one
    to the bit; an unreproduced cell is not scored with the other classes. The permutation
    check recomputes the linear shuffled control from registered_permutations and must equal
    pooled_readout's own value, so the nonlinear classes see the same eight label draws."""
    from .experiment_b import episode_features
    out = {"recorded_target": recorded.get("target"), "target": float(pool["target"]),
           "recorded_shuffled": recorded.get("shuffled"), "shuffled": float(pool["shuffled"]),
           "n_auth": int(len(Ha)), "n_surr": int(len(Hs))}
    out["reproduced"] = bool(recorded.get("target") is not None
                             and float(pool["target"]) == float(recorded["target"]))
    out["shuffled_reproduced"] = bool(recorded.get("shuffled") is not None
                                      and float(pool["shuffled"]) == float(recorded["shuffled"]))
    if not out["reproduced"] or len(Ha) < 5 or len(Hs) < 5:
        return out
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    perms = registered_permutations(y, seed, n_perm)
    scorers = scorers or PROBE_CLASSES
    lin_draws = [scorers["linear"](X, yp, seed=seed)["auroc"] for yp in perms]
    out["perm_check"] = bool(n_perm == N_PERM and float(np.mean(lin_draws)) == float(pool["shuffled"]))
    out["classes"] = {"linear": {"target": float(pool["target"]), "shuffled": float(pool["shuffled"]),
                                 "selectivity": float(pool["target"]) - float(pool["shuffled"])}}
    out["classes"].update(score_probe_classes(X, y, seed=seed, perms=perms, scorers=scorers,
                                              n_estimators=rf_trees))
    return out


def _perm_ok(per_seed: dict, d: str, s) -> bool:
    """A seed's perm_check holds when every recorded arm agrees with pooled_readout's shuffled
    control. Seeds recorded before the check existed have no flag and pass."""
    arms = per_seed[s]["cells"][d]
    flags = [a["perm_check"] for a in arms.values() if isinstance(a, dict) and "perm_check" in a]
    return all(flags)


def summarize_r210(per_seed: dict) -> dict:
    """Per drift: the seeds whose three arms all reproduced, and on those seeds every class's
    target, shuffled control and selectivity per arm, each class minus linear per arm, and the
    survival margins under each class. Seeds with any unreproduced arm are listed, excluded."""
    drifts = sorted({d for r in per_seed.values() for d in r.get("cells", {})}, key=float)
    out = {}
    for d in drifts:
        ok = [s for s, r in per_seed.items()
              if r.get("reproduced", {}).get(d) and _perm_ok(per_seed, d, s)]
        bad = [s for s, r in per_seed.items()
               if d in r.get("cells", {}) and not (r["reproduced"].get(d) and _perm_ok(per_seed, d, s))]
        block = {"seeds_used": sorted(ok, key=int), "seeds_excluded": sorted(bad, key=int), "arms": {},
                 "class_minus_linear": {}, "margins": {}}

        def val(s, arm, c, key="target"):
            return per_seed[s]["cells"][d][arm].get("classes", {}).get(c, {}).get(key, float("nan"))

        classes = sorted({c for s in ok for arm in ARMS
                          for c in per_seed[s]["cells"][d][arm].get("classes", {})})
        for arm in ARMS:
            block["arms"][arm] = {c: {k: seed_summary([val(s, arm, c, k) for s in ok])
                                      for k in ("target", "shuffled", "selectivity")} for c in classes}
            block["class_minus_linear"][arm] = {
                c: paired_summary({s: val(s, arm, c) for s in ok}, {s: val(s, arm, "linear") for s in ok})
                for c in classes if c != "linear"}
            for c, v in block["class_minus_linear"][arm].items():
                v["reading"] = interval_label(v)
        for other in ("predictor", "untrained"):
            block["margins"][f"survival_minus_{other}"] = {
                c: paired_summary({s: val(s, "survival", c) for s in ok}, {s: val(s, other, c) for s in ok})
                for c in classes}
        out[d] = block
    return out


# ---------------------------------------------------------------------------
# r205: which half of [mean h, h_T] carries the signal
# ---------------------------------------------------------------------------
R205_SETS = ("mean", "final", "concat")
R205_BAR = 0.65          # the registered bar, for the count of seeds at or above it
R205_CLOSE = 0.02        # "within 0.02 of the concatenation"
R205_GAP = 0.05          # "at least 0.05 below the concatenation"


def feature_halves(H: np.ndarray) -> dict:
    """experiment_b.episode_features split into its two halves: the mean-h block, the h_T block,
    and the registered concatenation itself, so the halves are exactly what the probe reads."""
    from .experiment_b import episode_features
    X = episode_features(H)
    hid = np.asarray(H).shape[-1]
    return {"mean": X[:, :hid], "final": X[:, hid:], "concat": X}


def r205_cell(Ha: np.ndarray, Hs: np.ndarray, *, pool: dict, recorded: dict, seed: int,
              n_perm: int = N_PERM, scorer=None) -> dict:
    """One (drift, seed, arm) cell of r205 from the regenerated pools and pooled_readout's
    output. The bit check is r210's: the concatenated target must equal the recorded one, and
    an unreproduced cell is not scored. The registered linear probe (scaler and logistic
    regression in-fold, explicit-v1 folds, one group per episode) is scored on the mean-h half
    and on the h_T half, each with its shuffled control over pooled_readout's eight label
    permutations; the concatenation's target and shuffled control are pooled_readout's own."""
    out = {"recorded_target": recorded.get("target"), "target": float(pool["target"]),
           "recorded_shuffled": recorded.get("shuffled"), "shuffled": float(pool["shuffled"]),
           "n_auth": int(len(Ha)), "n_surr": int(len(Hs))}
    out["reproduced"] = bool(recorded.get("target") is not None
                             and float(pool["target"]) == float(recorded["target"]))
    out["shuffled_reproduced"] = bool(recorded.get("shuffled") is not None
                                      and float(pool["shuffled"]) == float(recorded["shuffled"]))
    if not out["reproduced"] or len(Ha) < 5 or len(Hs) < 5:
        return out
    scorer = scorer or linear_probe_auroc
    sets = feature_halves(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    perms = registered_permutations(y, seed, n_perm)
    out["sets"] = {}
    for name in ("mean", "final"):
        t = float(scorer(sets[name], y, seed=seed)["auroc"])
        draws = [float(scorer(sets[name], yp, seed=seed)["auroc"]) for yp in perms]
        sh = float(np.mean(draws)) if draws else float("nan")
        out["sets"][name] = {"target": t, "shuffled": sh, "selectivity": t - sh, "shuffled_draws": draws}
    lin = [float(scorer(sets["concat"], yp, seed=seed)["auroc"]) for yp in perms]
    out["perm_check"] = bool(n_perm == N_PERM and float(np.mean(lin)) == float(pool["shuffled"]))
    out["sets"]["concat"] = {"target": float(pool["target"]), "shuffled": float(pool["shuffled"]),
                             "selectivity": float(pool["target"]) - float(pool["shuffled"])}
    return out


def r205_reading(diffs: dict) -> str:
    """MEAN CARRIES: mean h alone reads within 0.02 of the concatenation or above it, and h_T
    alone reads at least 0.05 below it (seed-paired mean differences). FINAL CARRIES: the
    mirror case. BOTH: anything else. NA without finite differences. The two named cases
    cannot hold together."""
    cm, cf = diffs["concat_minus_mean"]["mean"], diffs["concat_minus_final"]["mean"]
    if not (np.isfinite(cm) and np.isfinite(cf)):
        return "NA"
    if cm <= R205_CLOSE and cf >= R205_GAP:
        return "MEAN CARRIES"
    if cf <= R205_CLOSE and cm >= R205_GAP:
        return "FINAL CARRIES"
    return "BOTH"


def summarize_r205(per_seed: dict) -> dict:
    """Per drift, on the seeds whose three arms all reproduced (the others listed, excluded):
    per arm and feature set the target with its t90 and the count of seeds at or above 0.65,
    the shuffled control and the selectivity; per arm the seed-paired mean minus final,
    concatenation minus mean and concatenation minus final; and the pre-stated reading."""
    drifts = sorted({d for r in per_seed.values() for d in r.get("cells", {})}, key=float)
    out = {}
    for d in drifts:
        ok = [s for s, r in per_seed.items()
              if r.get("reproduced", {}).get(d) and _perm_ok(per_seed, d, s)]
        bad = [s for s, r in per_seed.items()
               if d in r.get("cells", {}) and not (r["reproduced"].get(d) and _perm_ok(per_seed, d, s))]
        block = {"seeds_used": sorted(ok, key=int), "seeds_excluded": sorted(bad, key=int),
                 "arms": {}, "differences": {}, "reading": {}}

        def val(s, arm, name, key="target"):
            return per_seed[s]["cells"][d][arm].get("sets", {}).get(name, {}).get(key, float("nan"))

        for arm in ARMS:
            block["arms"][arm] = {}
            for name in R205_SETS:
                t = [val(s, arm, name) for s in ok]
                block["arms"][arm][name] = {
                    "target": {**seed_summary(t), "n_at_or_above_bar": int(sum(
                        x is not None and np.isfinite(x) and x >= R205_BAR for x in t))},
                    "shuffled": seed_summary([val(s, arm, name, "shuffled") for s in ok]),
                    "selectivity": seed_summary([val(s, arm, name, "selectivity") for s in ok])}
            per = {name: {s: val(s, arm, name) for s in ok} for name in R205_SETS}
            diffs = {"mean_minus_final": paired_summary(per["mean"], per["final"]),
                     "concat_minus_mean": paired_summary(per["concat"], per["mean"]),
                     "concat_minus_final": paired_summary(per["concat"], per["final"])}
            block["differences"][arm] = diffs
            block["reading"][arm] = r205_reading(diffs)
        out[d] = block
    return out


# ---------------------------------------------------------------------------
# r35: normalizer-matched replay
# ---------------------------------------------------------------------------
R35_VARIANTS = ("own", "survival_norm", "refit", "refit_infold")


def refit_normalizer(obs_raw_list):
    """A frozen RunningNorm whose mean and variance are the empirical per-channel moments of
    every (episode, step) row of the given raw observation arrays, pooled. Applied with
    RunningNorm's own formula, (x - mean) / (sqrt(var) + 1e-6)."""
    from .experiment_b2 import RunningNorm
    rows = np.concatenate([np.asarray(o, np.float64).reshape(-1, np.asarray(o).shape[-1])
                           for o in obs_raw_list])
    norm = RunningNorm(rows.shape[1])
    norm.mean, norm.var, norm.count = rows.mean(0), rows.var(0), float(len(rows))
    return norm.freeze()


def infold_normalizer_auroc(obs_raw: np.ndarray, y: np.ndarray, states_fn, groups=None,
                            n_splits: int = 5) -> float:
    """The registered probe with a normalizer refit inside each fold: the normalizer is fit
    on the training episodes' raw observations only (both classes), every episode's states
    are recomputed through `states_fn(norm)` (all episodes, (n, T, hidden)), and the
    scaler and logistic regression are fit on the training episodes. Mean fold AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .experiment_b import episode_features
    g = np.arange(len(y)) if groups is None else groups
    aucs = []
    for tr, te in folds.split(g, n_splits):
        if len(np.unique(y[te])) < 2:
            continue
        F = episode_features(np.asarray(states_fn(refit_normalizer([obs_raw[tr]]))))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def summarize_r35(per_seed: dict) -> dict:
    """Per arm and variant, the replay target; the survival-minus-arm margin under each
    variant (own: the survival reference against the arm's own normalizer; survival_norm:
    every trunk on the survival normalizer; refit and refit_infold: every trunk, survival
    included, on the refit normalizer); and the pre-stated reading per arm."""
    seeds = sorted(per_seed, key=int)

    def t(s, arm, v):
        return per_seed[s]["arms"][arm].get(v, float("nan"))

    out = {"arms": {arm: {v: seed_summary([t(s, arm, v) for s in seeds]) for v in R35_VARIANTS}
                    for arm in ARMS}, "margins": {}, "reading": {}}
    for other in ("predictor", "untrained"):
        m = {}
        for v in R35_VARIANTS:
            ref = "own" if v == "survival_norm" else v
            m[v] = paired_summary({s: t(s, "survival", ref) for s in seeds}, {s: t(s, other, v) for s in seeds})
        own = m["own"]["mean"]
        for v in ("survival_norm", "refit", "refit_infold"):
            m[v]["share_of_own_margin"] = (m[v]["mean"] / own) if np.isfinite(own) and own != 0 else float("nan")
        out["margins"][f"survival_minus_{other}"] = m
        out["reading"][other] = r35_reading(m)
    return out


def r35_reading(m: dict) -> str:
    """TRUNK: under both matched variants (the survival normalizer and the refit) the
    survival-minus-arm interval lies above zero and keeps at least half of the own-normalizer
    margin. NORMALIZER: under both it includes zero or lies below it. SPLIT: anything else.
    NA when an interval is missing. The refit entry is the pooled refit unless its label
    differs from the in-fold refit's, in which case the in-fold refit governs."""
    refit = "refit" if interval_label(m["refit"]) == interval_label(m["refit_infold"]) else "refit_infold"
    keys = ("survival_norm", refit)
    labels = [interval_label(m[v]) for v in keys]
    if "NA" in labels:
        return "NA"
    if all(lab == "ABOVE" for lab in labels) and all(
            np.isfinite(m[v].get("share_of_own_margin", np.nan)) and m[v]["share_of_own_margin"] >= 0.5
            for v in keys):
        return "TRUNK"
    if all(lab != "ABOVE" for lab in labels):
        return "NORMALIZER"
    return "SPLIT"


# ---------------------------------------------------------------------------
# r206: law deviation along visited states
# ---------------------------------------------------------------------------
R206_WALKS = ("survival_own", "predictor_own", "untrained_own", "scripted")
#: which walk each (protocol, arm) is read on. Replay feeds the survival agent's recorded
#: walk to every trunk, so its visited states are the survival walk by construction.
R206_PROTOCOL_WALK = {
    "own": {"survival": "survival_own", "predictor": "predictor_own", "untrained": "untrained_own"},
    "scripted": {arm: "scripted" for arm in ARMS},
    "replay": {arm: "survival_own" for arm in ARMS},
}


def law_deviation(motion_log, dt: float) -> np.ndarray:
    """Per-step |G(v, a) - f(v, a)| from a PatchOfEarthV0 motion log. Each entry is
    (vel, a, drag, vel_next) as _integrate_motion records it: in an l3 surrogate world
    vel_next is G's output, and f is the authentic law (1 - drag dt) vel + a dt at the same
    (vel, a, drag). Euclidean norm of the 2-vector difference."""
    out = np.empty(len(motion_log))
    for t, (vel, a, drag, vnext) in enumerate(motion_log):
        f = (1.0 - float(drag) * dt) * np.asarray(vel, float) + np.asarray(a, float) * dt
        out[t] = float(np.linalg.norm(np.asarray(vnext, float) - f))
    return out


def walk_deviation(policy_factory, params, drift_sigma: float, *, n_eps: int, steps: int,
                   seed_base: int, ray_steps: int, world_factory=None, seeds_fn=None) -> dict:
    """Walk `n_eps` worlds (seed_base + i) for exactly `steps` steps under the policy
    `policy_factory(i)` (a callable obs -> env action, fresh per episode), as collect_pool
    steps them, and return the per-step law deviation of every full-length episode
    (D (k, steps)), mean |v| and |a| per episode, and the kept indices. Episodes that end
    early are dropped, as in the pools. The motion log is a passive record and changes
    nothing in the world."""
    if world_factory is None or seeds_fn is None:
        from .experiment_b2 import _seeds, make_world
        world_factory = world_factory or make_world
        seeds_fn = seeds_fn or _seeds
    D, V, A, kept = [], [], [], []
    for i in range(n_eps):
        w = world_factory(params, drift_sigma, ray_steps)
        w._log_motion = []
        w.reset(seeds_fn(seed_base + i))
        obs = w.observe().astype(np.float64)
        act = policy_factory(i)
        died = False
        for _ in range(steps):
            r = w.step(act(obs))
            obs = r.obs.astype(np.float64)
            if r.terminated:
                died = True
                break
        if died or len(w._log_motion) != steps:
            continue
        D.append(law_deviation(w._log_motion, float(w.params.dt)))
        V.append(float(np.mean([np.linalg.norm(m[0]) for m in w._log_motion])))
        A.append(float(np.mean([np.linalg.norm(m[1]) for m in w._log_motion])))
        kept.append(i)
    return {"D": np.stack(D) if D else np.zeros((0, steps)), "speed": np.asarray(V),
            "acc": np.asarray(A), "kept": np.asarray(kept, int)}


def deviation_summary(walk: dict) -> dict:
    """Per walk: episodes kept, the mean and quantiles of the per-episode deviation summed
    over the pooled steps, per-step quantiles, and mean |v| and |a|."""
    D = np.asarray(walk["D"], float)
    nan = float("nan")
    if D.shape[0] == 0:
        return {"n_eps": 0, "sum_mean": nan, "sum_q10": nan, "sum_q50": nan, "sum_q90": nan,
                "step_mean": nan, "step_q50": nan, "step_q90": nan, "step_q99": nan,
                "speed_mean": nan, "acc_mean": nan}
    s = D.sum(1)
    q = np.quantile(s, [0.1, 0.5, 0.9])
    qs = np.quantile(D.ravel(), [0.5, 0.9, 0.99])
    return {"n_eps": int(D.shape[0]), "sum_mean": float(s.mean()), "sum_q10": float(q[0]),
            "sum_q50": float(q[1]), "sum_q90": float(q[2]), "step_mean": float(D.mean()),
            "step_q50": float(qs[0]), "step_q90": float(qs[1]), "step_q99": float(qs[2]),
            "speed_mean": float(np.mean(walk["speed"])), "acc_mean": float(np.mean(walk["acc"]))}


def summarize_r206(per_seed: dict, recorded_survival_targets: dict | None = None) -> dict:
    """Per walk, the across-seed summary of the mean summed deviation; the survival walk minus
    each other walk, seed-paired; the pre-stated reading; and, descriptively, the Pearson r
    across seeds between the survival walk's mean summed deviation and the run's recorded
    survival pooled target at the strongest drift."""
    seeds = sorted(per_seed, key=int)

    def v(s, walk, key="sum_mean"):
        return per_seed[s]["walks"][walk][key]

    out = {"walks": {w: {k: seed_summary([v(s, w, k) for s in seeds])
                         for k in ("sum_mean", "step_mean", "step_q90", "speed_mean", "acc_mean")}
                     for w in R206_WALKS},
           "protocol_walk": R206_PROTOCOL_WALK, "survival_minus": {}}
    for w in R206_WALKS[1:]:
        out["survival_minus"][w] = paired_summary({s: v(s, "survival_own") for s in seeds},
                                                  {s: v(s, w) for s in seeds})
    labels = [interval_label(out["survival_minus"][w]) for w in R206_WALKS[1:]]
    out["reading"] = ("NA" if "NA" in labels else "HIGHER" if all(x == "ABOVE" for x in labels)
                      else "NOT HIGHER" if all(x != "ABOVE" for x in labels) else "MIXED")
    if recorded_survival_targets:
        xs = [(v(s, "survival_own"), recorded_survival_targets[s]) for s in seeds
              if s in recorded_survival_targets and np.isfinite(v(s, "survival_own"))]
        if len(xs) >= 3:
            a, b = np.asarray(xs, float).T
            out["corr_with_recorded_target"] = {"n": len(xs), "pearson_r": float(np.corrcoef(a, b)[0, 1])}
    return out


# ---------------------------------------------------------------------------
# r33: draw-free balanced readout
# ---------------------------------------------------------------------------
R33_BASES = {"fresh": BALANCED_FRESH_BASE, "registered": REGISTERED_AUTH_BASE}


def r33_key(agent_drift: float, base: str) -> str:
    return f"d{float(agent_drift):.2f}/{base}"


def r33_values(per_seed: dict, arm: str) -> dict:
    """{key: {seed: target}} for one arm across every (agent drift, base) key."""
    out: dict = {}
    for s, r in per_seed.items():
        for c in r["cells"]:
            if c["arm"] == arm:
                out.setdefault(r33_key(c["agent_drift"], c["base"]), {})[s] = c["target"]
    return out


def summarize_r33(per_seed: dict) -> dict:
    vals = {arm: r33_values(per_seed, arm) for arm in ARMS}
    keys = sorted(vals["survival"])
    out = {}
    for k in keys:
        out[k] = {"arms": {arm: seed_summary(list(vals[arm].get(k, {}).values())) for arm in ARMS}}
        for other in ("predictor", "untrained"):
            out[k][f"survival_minus_{other}"] = paired_summary(vals["survival"][k], vals[other].get(k, {}))
    return out


def balanced_reference_check(per_seed: dict, ref_doc: dict, *, drift: float) -> dict:
    """Registered-base balanced targets of the agents at `drift` against a run_l0_audit.py
    output's `paired` entries (the same agents, if the run is the one audited)."""
    ref = {(str(int(p["seed"])), p["arm"]): float(p["target"]) for p in ref_doc.get("paired", [])
           if abs(float(p["drift"]) - drift) < 1e-9 and int(p.get("seed_base", REGISTERED_AUTH_BASE)) == REGISTERED_AUTH_BASE}
    rows = []
    for s, r in per_seed.items():
        for c in r["cells"]:
            key = (s, c["arm"])
            if c["base"] == "registered" and abs(float(c["agent_drift"]) - drift) < 1e-9 and key in ref:
                rows.append({"seed": s, "arm": c["arm"], "reference": ref[key], "rescored": c["target"],
                             "abs_dev": abs(c["target"] - ref[key]), "bit_equal": c["target"] == ref[key]})
    rows.sort(key=lambda x: (x["arm"], int(x["seed"])))
    return {"n_checked": len(rows), "n_bit_equal": int(sum(x["bit_equal"] for x in rows)),
            "max_abs_dev": max((x["abs_dev"] for x in rows), default=float("nan")), "per_cell": rows}


def balanced_contrast(doc_a: dict, doc_b: dict, *, arm: str = "survival") -> dict:
    """Seed-paired doc_a minus doc_b of the r33 balanced target for one arm, for every
    (agent drift, base) key present in both. Refuses outputs that used different bases or
    settings. The primary key is the strongest agent drift on the fresh base."""
    for field in ("seed_bases", "settings"):
        if doc_a.get(field) != doc_b.get(field):
            raise ValueError(f"the two outputs differ in {field}; the contrast needs one estimator")
    va = r33_values(doc_a["per_seed"].get("r33", {}), arm)
    vb = r33_values(doc_b["per_seed"].get("r33", {}), arm)
    keys = sorted(set(va) & set(vb))
    out = {"arm": arm, "a": doc_a.get("run_dir"), "b": doc_b.get("run_dir"), "contrasts": {}}
    for k in keys:
        c = paired_summary(va[k], vb[k])
        c["a_mean"], c["b_mean"] = seed_summary(list(va[k].values()))["mean"], seed_summary(list(vb[k].values()))["mean"]
        c["reading"] = contrast_reading(c)
        out["contrasts"][k] = c
    dmax = max((float(k.split("/")[0][1:]) for k in keys), default=float("nan"))
    out["primary"] = r33_key(dmax, "fresh") if keys else None
    return out


def contrast_reading(c: dict, margin: float = 0.05) -> str:
    """HOLDS: the interval lies above `margin` (the auxiliary rule's 0.05). DIRECTION ONLY: it
    lies above zero but not above the margin. REVERSED: it lies below zero. NOT RESOLVED:
    it includes zero."""
    lo, hi = c.get("t90", [float("nan")] * 2)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    if lo > margin:
        return "HOLDS"
    if lo > 0:
        return "DIRECTION ONLY"
    if hi < 0:
        return "REVERSED"
    return "NOT RESOLVED"


# ---------------------------------------------------------------------------
# r31: stream-decoder capacity curve
# ---------------------------------------------------------------------------
TRUNK_EMBED, TRUNK_HIDDEN = 64, 96
WIDTH_MULTS = (1, 2, 4, 8)
GRU_EPOCHS = 60
CONV_CHANNELS = (64, 96)
CONV_KERNEL = 5
R31_STREAMS = ("survival_own", "scripted")


def capacity_table(mults=WIDTH_MULTS, embed: int = TRUNK_EMBED, hidden: int = TRUNK_HIDDEN) -> list[dict]:
    """The GRU decoders of the curve: the recurrent width scales with the multiplier, the
    per-step encoder keeps the trunk's 64 units (the trunk's recurrent width is what the
    multiplier names). 1x is FINDINGS 17.8's decoder exactly."""
    return [{"label": f"{int(m)}x", "mult": int(m), "embed": int(embed), "hidden": int(hidden * m)}
            for m in mults]


def _conv_reader(n_in: int, channels=CONV_CHANNELS, kernel: int = CONV_KERNEL):
    """Two temporal convolutions (same padding, ReLU), a mean over time, a linear logit.
    Input (batch, T, C); output (batch,) logits."""
    import torch.nn as nn

    class ConvReader(nn.Module):
        def __init__(self):
            super().__init__()
            pad = kernel // 2
            self.net = nn.Sequential(nn.Conv1d(n_in, channels[0], kernel, padding=pad), nn.ReLU(),
                                     nn.Conv1d(channels[0], channels[1], kernel, padding=pad), nn.ReLU())
            self.head = nn.Linear(channels[1], 1)

        def forward(self, x):
            return self.head(self.net(x.transpose(1, 2)).mean(-1)).squeeze(-1)

    return ConvReader()


def conv_sequence_auroc(Seq: np.ndarray, y: np.ndarray, groups=None, *, epochs: int = GRU_EPOCHS,
                        lr: float = 1e-3, batch: int = 32, seed: int = 0, n_splits: int = 5) -> dict:
    """The 1D-conv reader trained exactly as control_diagnostics.sequence_gru_auroc trains
    the GRU: per fold, inputs standardized with training-fold statistics, torch seed
    seed * 100 + fold, Adam, fixed epochs, batch 32, no validation split. Out-of-fold mean
    AUROC and the per-fold final training loss."""
    import torch
    from sklearn.metrics import roc_auc_score

    g = np.arange(len(y)) if groups is None else groups
    S = np.asarray(Seq, np.float32)
    n, T, C = S.shape
    aucs, losses = [], []
    for f, (tr, te) in enumerate(folds.split(g, n_splits)):
        if len(np.unique(y[te])) < 2:
            continue
        mu = S[tr].reshape(-1, C).mean(0)
        sd = S[tr].reshape(-1, C).std(0) + 1e-6
        Xtr = torch.as_tensor((S[tr] - mu) / sd)
        Xte = torch.as_tensor((S[te] - mu) / sd)
        ytr = torch.as_tensor(y[tr], dtype=torch.float32)
        torch.manual_seed(seed * 100 + f)
        model = _conv_reader(C)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(seed * 100 + f)
        last = float("nan")
        for _ in range(epochs):
            perm = torch.randperm(len(tr), generator=gen)
            tot = 0.0
            for i in range(0, len(tr), batch):
                idx = perm[i:i + batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xtr[idx]), ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            last = tot / len(tr)
        with torch.no_grad():
            p = torch.sigmoid(model(Xte)).numpy()
        aucs.append(roc_auc_score(y[te], p))
        losses.append(last)
    return {"auroc": float(np.mean(aucs)) if aucs else float("nan"), "final_train_loss": losses,
            "epochs": epochs, "channels": list(CONV_CHANNELS), "kernel": CONV_KERNEL}


def stream_decoder_curve(seq: np.ndarray, y: np.ndarray, *, seed: int, epochs: int = GRU_EPOCHS,
                         mults=WIDTH_MULTS, gru_fn=None, conv_fn=None) -> dict:
    """Every decoder of the capacity table plus the conv reader on one stream, same folds."""
    if gru_fn is None:
        from .control_diagnostics import sequence_gru_auroc as gru_fn
    conv_fn = conv_fn or conv_sequence_auroc
    out = {"gru": {}}
    for row in capacity_table(mults):
        r = gru_fn(seq, y, embed=row["embed"], hidden=row["hidden"], epochs=epochs, seed=seed)
        out["gru"][row["label"]] = {**r, "mult": row["mult"]}
    out["conv"] = conv_fn(seq, y, epochs=epochs, seed=seed)
    return out


def summarize_r31(per_seed: dict) -> dict:
    """Per stream: the state reference and every decoder across seeds; each wider decoder
    minus 1x, seed-paired; the state minus the best decoder (highest mean, chosen after the
    fact, so the comparison leans toward the decoder); and the two pre-stated readings."""
    seeds = sorted(per_seed, key=int)
    out = {}
    for st in R31_STREAMS:
        rows = {s: per_seed[s][st] for s in seeds if "decoders" in per_seed[s].get(st, {})}
        if not rows:
            continue
        labels = list(next(iter(rows.values()))["decoders"]["gru"])
        dec = {lab: {s: r["decoders"]["gru"][lab]["auroc"] for s, r in rows.items()} for lab in labels}
        dec["conv"] = {s: r["decoders"]["conv"]["auroc"] for s, r in rows.items()}
        state = {s: r["state"] for s, r in rows.items()}
        block = {"state": seed_summary(list(state.values())),
                 "decoders": {k: seed_summary(list(v.values())) for k, v in dec.items()},
                 "minus_1x": {k: paired_summary(v, dec["1x"]) for k, v in dec.items() if k != "1x"}}
        best = max(dec, key=lambda k: block["decoders"][k]["mean"]
                   if np.isfinite(block["decoders"][k]["mean"]) else -np.inf)
        block["best_decoder"] = best
        block["state_minus_best"] = paired_summary(state, dec[best])
        wider = [block["minus_1x"][k] for k in labels if k != "1x"]
        lab_w = [interval_label(w) for w in wider]
        block["capacity_reading"] = ("NA" if not wider or "NA" in lab_w else
                                     "CAPACITY-LIMITED" if any(x == "ABOVE" and w["mean"] >= 0.05
                                                               for x, w in zip(lab_w, wider))
                                     else "NOT CAPACITY-LIMITED")
        lo, hi = block["state_minus_best"]["t90"]
        block["state_reading"] = ("NA" if not (np.isfinite(lo) and np.isfinite(hi)) else
                                  "EQUIVALENT" if lo > -0.05 and hi < 0.05 else
                                  "DECODER ABOVE" if hi < 0 else
                                  "STATE ABOVE" if lo > 0 else "UNRESOLVED")
        out[st] = block
    return out


# ---------------------------------------------------------------------------
# Partial file: one JSON line per finished (readout, seed)
# ---------------------------------------------------------------------------
def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _canon(d) -> str:
    return json.dumps(d, sort_keys=True, default=_jsonable)


class PartialStore:
    """Append-only JSON-lines record of finished tasks, so an interrupted session resumes.

    Line 1 is {"header": ...}: the run's cell fingerprints, the settings and the seed bases.
    A resume with a different header is refused, so numbers computed under one setting can
    never be mixed with another's. Each later line is {"readout", "seed", "result"} or
    {"readout", "seed", "error"}; an error line does not count as done, so the task is retried.
    A line cut off by a kill is dropped (the file is truncated back to its last newline)."""

    def __init__(self, path: str, header: dict):
        self.path, self.header = path, header

    def load(self) -> dict:
        """{(readout, seed): result} of finished tasks (latest line wins)."""
        if not os.path.exists(self.path):
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"header": self.header}, sort_keys=True, default=_jsonable) + "\n")
            return {}
        with open(self.path, "rb") as fh:
            raw = fh.read()
        if raw and not raw.endswith(b"\n"):
            raw = raw[: raw.rfind(b"\n") + 1]
            with open(self.path, "wb") as fh:
                fh.write(raw)
        lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"{self.path} has no header line; delete it to start over")
        head = json.loads(lines[0]).get("header")
        if _canon(head) != _canon(self.header):
            raise ValueError(f"{self.path} was written for a different run or setting; "
                             "pass another --partial or delete it")
        done: dict = {}
        for ln in lines[1:]:
            rec = json.loads(ln)
            key = (rec["readout"], str(rec["seed"]))
            if "result" in rec:
                done[key] = rec["result"]
            else:
                done.pop(key, None)
        return done

    def append(self, readout: str, seed, *, result: dict | None = None, error: str | None = None) -> None:
        rec = {"readout": readout, "seed": str(seed)}
        if error is not None:
            rec["error"] = error
        else:
            rec["result"] = result
        line = json.dumps(rec, default=_jsonable) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
