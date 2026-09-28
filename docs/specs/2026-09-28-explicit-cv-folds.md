# Explicit cross-validation folds and the re-score of published numbers

Date: 2026-09-28
Status: draft for the owner to freeze before any re-score is run

## Problem

Every grouped probe splits episodes with a 5-fold GroupKFold: the pooled
readout, the behavior and sensory controls, the matched-pair oracles and
leakage channels, and the common-garden and matched-pair readouts. scikit-learn
orders equal-sized groups with `np.argsort`. Before the stable sort landed in
scikit-learn that call used numpy's default unstable kind, so when every group
has the same size (one group per episode, or two per matched pair) the fold each
episode lands in depended on the numpy build and the CPU's sort path.

The committed numbers show which split produced them. For a 110 + 110 pool:

| Source | Partition (authentic, surrogate) per fold | Evidence |
|---|---|---|
| Published GPU runs (L3 hidden 8 and 7, B-v3, owner's stack: scikit-learn 1.5.2) | (24,20), (22,22), (21,23), (21,23), (22,22) | all 60 per-seed values in `heldout_l3_h8_summary.json` and all 30 in `bv3_n10_summary.json` are exact fold means under this split |
| 2026-09 cloud runs (10.8, 10.9, device control; scikit-learn 1.9.1) | (22,22) x 5 | all 180 per-seed values sit on the 1/2420 grid only the balanced split produces |

scikit-learn 1.5.2 with numpy 1.26.4 reproduces the first split; scikit-learn
1.6.1 with numpy 2.2.6, and 1.9.1 (which sorts stably), give the second. This is
a noise problem, not a bias: both are valid 5-fold partitions. A simulation at
AUROC near 0.73 (300 synthetic datasets, low-rank 192-dimensional features, not
project data) puts the shift at about 0.011 per seed and 0.004 on a 10-seed mean,
one standard deviation, with no consistent direction. It does mean the published
pooled numbers reproduce bit-exactly only on a stack that happens to sort the way
the owner's did, and `scripts/audit_stats_recheck.py` cannot see this because it
compares docs with committed artifacts.

## Change (implemented on `claude/recent-research-status-2ki8yg`)

- `itasorl/folds.py` builds the partition in plain numpy with the algorithm
  current scikit-learn uses (stable sort of group sizes, largest first, each group
  to the lightest fold, ties to the lowest index). It is the default scheme,
  `explicit`. On a stack whose scikit-learn sorts stably it is identical to
  GroupKFold; `tests/test_folds.py` checks that on five group structures.
- `grouped_auroc` and every control in `itasorl/behavior_audit.py` take their
  folds from it, so every probe in the project follows one switch.
- `ITASORL_FOLDS=legacy` (or `folds.fold_scheme("legacy")`) routes through the
  installed scikit-learn GroupKFold instead. On the owner's stack that reproduces
  the published numbers.
- The regeneration integrity gates (`run_expH2_ablation.py`,
  `run_l3_h2_ablations.py`, `run_l3_crossrecipe.py`, `audit_sensory_echo.py`) read
  their reference from `folds.REFERENCE_SURVIVAL_TARGET`, keyed by scheme. The
  legacy entries are the published 0.752 (hidden 8) and 0.737 (hidden 7). The
  explicit entries are empty until the re-score below fills them; until then an
  explicit-scheme gate stops with instructions rather than skipping the check.
  `audit_sensory_echo.py` records `target_reproduced: null` instead.

Pinning scikit-learn and numpy is not needed for this issue: the partition no
longer depends on either. The re-score output records the stack anyway.

What does not change: the three 2026-09 cloud runs were already scored on the
explicit partition, so their committed numbers are what the new code produces.
`transfer_probe` (held-out and cross-recipe transfer) fits once and scores a
disjoint test set with no cross-validation, so 0.773, 0.638, and 0.684 are
untouched.

## Re-score protocol (frozen with this spec; run on the owner's machine)

Readout-only, no training, no GPU needed. On the owner's stack, for every
published run with saved pooled dumps, at minimum the L3 hidden 8 and hidden 7
trace runs:

    python scripts/rescore_fold_split.py fullruns/l3_h8_traces/states \
        --json fullruns/fold_rescore/l3_h8_traces.json --label "L3 hidden 8 headline"
    python scripts/rescore_fold_split.py fullruns/l3_h7_traces/states \
        --json fullruns/fold_rescore/l3_h7_traces.json --label "L3 hidden 7"

and, where the dumps exist, the B-v3 n = 10, capacity-ceiling, and L1 organism
runs. Each output carries both partitions, per-cell values under both schemes,
and per (drift, agent) the two means, t-based 90% CIs, seed counts at the bar,
and the mean shift.

Readouts that are not pooled dumps are re-scored by running their existing
reanalysis script twice, once with `ITASORL_FOLDS=legacy` and once with
`ITASORL_FOLDS=explicit`: the common-garden tails (`reanalyze_cg_states.py`), the
matched-pair readouts (`reanalyze_mp_readout.py`), and gate 0 for hidden 8 and 7
(`run_expA_l3.py --hiddens 8 7`, agent-free, minutes).

**Integrity check first.** The legacy column must reproduce the published values
exactly (0.752 and 0.726 at hidden 8; 0.737 and 0.722 at hidden 7). If it does
not, the stack is not the one that produced them, and the re-score stops there.

## Reporting rule (frozen with this spec)

1. The published numbers stay in FINDINGS as the record. The explicit values are
   reported beside them in one dated methods note in FINDINGS section 11, with the
   per-run shift, and promoted to a committed artifact.
2. The explicit hidden 8 and hidden 7 survival means become the explicit entries
   of `REFERENCE_SURVIVAL_TARGET`, so the integrity gates work on any stack.
3. If any pre-registered verdict changes under the explicit split (a mean crosses
   0.65, a t-CI bound crosses it, or a margin clause flips), that is reported as a
   finding in its own dated section, not folded into the methods note. The numbers
   nearest a boundary, to read first: the common-garden forward tail 0.666 (bar
   0.65), the joint sensory-plus-behavior control's lower bound 0.638 (bar 0.65),
   the hidden 7 survival-minus-predictor lead +0.023 (margin 0.05), and the B-v3
   upper bound 0.634 (bar 0.65).
4. Future runs use the explicit scheme. Nothing is re-run to chase a number.

## Out of scope

Other numerical dependence on the stack (BLAS, the logistic solver) is at the
1e-6 level, well below the partition effect, and is not addressed.
