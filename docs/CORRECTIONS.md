# Corrections and provenance record

Dated entries, newest last. Each entry says what was wrong, which results it touches, what
was changed, and what the change does not establish. The plan these belong to is
`docs/REVISION_2026-10.md`; which implementation produced each committed number is
`docs/RESULTS_MANIFEST.md`.

Earlier corrections are recorded where they were made and are not repeated here: the GAE
padding fix (`679fee6`, `docs/PREREGISTRATION.md` 2026-06-28), the L3 world-params bug
(`docs/PREREGISTRATION_L3.md` 2026-07-10), the Experiment C estimator invalidation
(`docs/FINDINGS.md` 13.C), the matched-pair fold-safety fix (`docs/PREREGISTRATION_L3.md`
2026-07-22), and the explicit fold partition (`docs/FINDINGS.md` methods note 8 and section 16).

## 2026-10-06: results manifest

**What.** No single record said which code, configuration, seeds, fold partition, and device
produced each committed number, so a corrected result could not be told apart from a historical
one by reading the artifacts.

**Change.** `scripts/build_results_manifest.py` assigns every file under `artifacts/` to exactly
one run and records the run's commit, configuration, agent seeds, surrogate seed and capacity,
training budget, evaluation seed bases, fold partition, device, and dependency versions where
those were recorded. `--check` fails when a committed artifact belongs to no run or the
generated pages are stale. Every existing row is marked `historical`.

**Found while building it.** All survival-trained runs from `679fee6` (2026-06-28) to `4b6e1f3`
used the same `compute_gae` bootstrap, the one corrected in the next entry. The predictor arms,
the untrained arms, Experiment A, Experiment B, and Experiment C train no actor-critic and are
unaffected by it. Every readout-only analysis (H2, cross-recipe, sensory echo, observation
localization) reads saved survival agents and inherits the issue. Evaluation worlds are fixed
seed bases shared by every agent seed, and the pooled readout draws its authentic pool
(seed base 800000) and its surrogate pool (850000) from different world samples; both facts
bear on step 5 and step 13.

## 2026-10-06: GAE truncation bootstrap

**What was wrong.** `compute_gae` in `itasorl/experiment_b2.py` bootstrapped every episode
still alive at the 80-step rollout cutoff from `value[:, last]`, the critic value at the
episode's last stored step. That value belongs to the state before the final transition, so
the final residual was r_T + gamma V(h_T) - V(h_T), which is (gamma - 1) V(h_T) + r_T. Neither
treatment of the cutoff justifies it: a continuing task needs V of the successor state, and a
true terminal horizon needs 0. The test `test_compute_gae_truncation_bootstraps_last_value`
asserted the wrong value, so the suite protected the bug.

**Which cutoff this is.** A sampling truncation of a continuing task. `PatchOfEarthV0` has no
horizon (`age_max` is 1e9 and `StepResult.truncated` is always false), death is its only
terminal event, and nothing in the observation counts rollout steps. The corrected bootstrap
is therefore the successor value; deaths keep a zero bootstrap.

**Change.** The collector keeps, for each episode alive at the cutoff, the successor
observation and the final environment action. After `score_actions`, the trainer takes the
recurrent state after the final stored observation, runs one GRU step on the successor
observation with the final action as the previous action, and passes the critic's value of
that state to `compute_gae` through a now-required `bootstrap` argument. No gradient flows
through it and no environment step is taken. Normalization convention: the successor
observation is normalized with the normalizer as it stood for the episode's final stored
observation and is not added to the running statistics. The next-step mask that keeps the
GAE carry from crossing padded steps (`679fee6`) is unchanged.

**Results touched.** Every survival-trained run in `docs/RESULTS_MANIFEST.md` (trainer
`pre_transition_value`) and every readout-only analysis of those agents. The predictor and
untrained arms, Experiments A, B and C, and the oracle gates train no actor-critic and are
unaffected.

**Reproduction of the record.** `train_actor_critic(..., gae_bootstrap="pre_transition")` and
`run_expB2.py --gae-bootstrap pre_transition` keep the historical trainer. On a 12-update
check it produces agent weights and normalizer statistics bit-identical to the original
`4b6e1f3` code, and a reproduction run hashes to the same config fingerprint as the
published runs. The corrected default hashes apart, so corrected cells can never resume into
or mix with a historical run. Each new cell records `gae_bootstrap`.

**What this does not establish.** Whether the published effect sizes or verdicts survive.
That is measured in revision step 4; until then every historical number stays as published,
labeled historical.

## 2026-10-06: the GAE tests

**What was wrong.** `test_compute_gae_truncation_bootstraps_last_value` expected the
pre-transition bootstrap, so it would have failed on a correct implementation and passed on
the bug.

**Change.** `tests/test_gae_bootstrap.py` replaces it. It checks a one-step truncated episode
(reward 1, current value 0.3, successor value 2, gamma 0.99: residual **2.68**, where the old
rule gives 0.997); the same step ending in death, which must ignore any successor value; a
batch of mixed lengths and endings against an independent reference written from the
definition of GAE; padded slots holding 7, 1e30, plus or minus infinity, and NaN; a 12-step
truncated episode against the reference; and, on a real recurrent agent and collector, that
the stored successor observation and final action replay exactly, that the bootstrap equals
one GRU step plus the critic, and that changing only the successor observation changes the
bootstrap and the final advantage. Mutation checks: reverting to the current value, to a
padded slot, to multiplicative masking, or dropping the terminal zero each make the file fail.

**Hardening found by the tests.** Padded slots were masked by multiplying by zero, so an
infinite or NaN padded value would have turned a valid advantage into NaN. `compute_gae` now
masks by selection. On finite inputs the arithmetic is unchanged: the historical trainer is
still bit-identical to `4b6e1f3`, re-checked on a run with deaths and padding.

## 2026-10-06: fold partition, gate table, and what L0 conditions on

**What was wrong.** The partition was frozen as a rule (`itasorl.folds`) but not versioned or
serialized, so a later change to the rule would have moved every number silently. No single
table showed the gates of every run under both partitions, and the manifest marked the
hidden 10 GPU re-measure as scored on the legacy partition; it was scored on explicit, the
`itasorl.folds` default at its run commit `283f3ab` (the gate table exposed the mismatch).
Every L0 statement treated the ten agent seeds as independent replications, although all ten
are scored on one fixed pair of world samples.

**Change.** `itasorl.folds` gains a version string (`explicit-v1`), `fold_index`, and
`partition_record` (scheme, version, class counts, whether groups stay intact, a digest);
`artifacts/folds/explicit_v1.json` serializes the standard designs and a test regenerates it.
`scripts/build_gate_table.py` writes `docs/GATE_TABLE.md`: L0 TOST and ROPE and the untrained
floor recomputed from committed per-seed values for every run and partition, other gates
carried from the summaries. Ten of its twenty rows do not show L0 equivalence; they read
"inconclusive". `itasorl/l0_audit.py` adds the pre-intervention probe, the world-sample
rescoring, and a balanced paired-seed readout with a pair-level bootstrap
(`itasorl.stats.cluster_auroc_ci`). The primary analysis is frozen in
`docs/specs/2026-10-06-primary-analysis-and-l0.md`.

**What the audit found so far.** From the reset observation alone the standard world-sample
pair reads 0.453 (chance). Across eight independent pairs the same probe reads 0.419 to 0.612,
sd 0.066, larger than the 0.05 equivalence margin. A seed-level L0 interval is therefore
conditional on the fixed world pair.

## 2026-10-06: statistical labels and estimators

**What was wrong.** The ROPE leg was described as Bayesian: its percentile bootstrap interval
was called a 95% HDI and the share of bootstrap means inside the ROPE was called P(in ROPE),
a posterior probability. FINDINGS 14 called the alpha 0 interval a "90% bootstrap HDI"; it is
a 95% percentile bootstrap interval. Per-cell AUROC intervals bootstrapped the pooled
out-of-fold AUROC while the point estimate was the mean of fold AUROCs. The paired readouts'
per-cell intervals resampled rows, treating the two members of a pair as independent. The
0.05 margins were judged on separate intervals rather than on an interval of the difference,
and across-seed intervals did not say they condition on one surrogate and one set of
evaluation worlds.

**Change.** `RopeResult` keeps its stored field names (artifacts use them) but is documented,
printed, and aliased as a bootstrap check (`boot_interval`, `boot_share_in_rope`). FINDINGS
labels are corrected and methods note 9 states the conventions, the conditioning, the
difference between a rule met and evidence shown, and the analysis tiers.
`itasorl.stats.fold_mean_auroc_ci` gives an interval aligned with the fold-mean estimator;
`_auroc_with_ci` resamples pairs when groups are paired (`cluster_auroc_ci`);
`paired_contrast` and `docs/CONTRAST_INTERVALS.md` give seed-paired margin intervals for
every run. TOST stays the formal equivalence test. No decision rule used a per-cell
interval, so no verdict moves.

## 2026-10-06: the surrogate and the detector, described as implemented

**What was wrong.** PREREGISTRATION_L3 section 9 describes the surrogate as a recurrent
predictor trained to early stopping; the code is a feed-forward MLP trained for a fixed 300
epochs with no held-out set. The surrogate's observations were said to stay "on the authentic
manifold", which was never established. The gate-0 oracle's score was called a detectability
ceiling, and section 4 said the agent sees raycasts, not velocity; interoception carries the
velocity and the applied acceleration exactly.

**Change.** A dated PREREGISTRATION_L3 amendment, FINDINGS 10.1.1, and corrected docstrings.
The new measurements (`artifacts/surrogate_diagnostics.json`, audited): in world P the
authentic law is linear with constant drag and a linear fit recovers it to rounding, so the
fingerprint is the approximation error of a finite-trained network; the headline G errs by
0.0349 RMS per step on held-out authentic transitions, with an open-loop rollout gap that
compounds and then saturates; and a detector that reads only the agent's own observations,
with no detector-side noise, scores 0.991 against every learned fingerprint and the Gaussian
comparator. The privileged gate-0 score (0.928) is a score at a detector-side handicap of
sigma 0.02, not a universal ceiling.

**What this does not change.** No organism number. Introducing early stopping now would make
a different fingerprint, with its own gate 0 and its own provenance.

## 2026-10-06: what the bootstrap correction did to the primary comparisons

**Runs.** `C1` and `C2` (`docs/specs/2026-10-06-corrected-trainer-confirmation-design.md`)
retrained the CPU device control and the CPU no-auxiliary run with the successor bootstrap and
nothing else changed. The original code reproduced a committed historical cell bit for bit on
the same container first, and in both corrected runs the predictor and untrained arms, which
train no actor-critic, came out bit-identical to the historical cells. FINDINGS 17.

**Effect, paired by seed against the historical runs.**

| run | quantity | corrected | historical | difference |
|---|---|---|---|---|
| `C1` (auxiliary on) | survival target, drift 0.45 | 0.733 | 0.730 | +0.003 [-0.017, +0.024] |
| `C1` | survival target, drift 0 (L0) | 0.559 | 0.529 | +0.031 [+0.011, +0.050] |
| `C2` (auxiliary off) | survival target, drift 0.45 | 0.613 | 0.601 | +0.012 [-0.019, +0.044] |
| `C2` | survival target, drift 0 (L0) | 0.515 | 0.514 | +0.001 [-0.014, +0.016] |

**Verdict changes.** One. The historical device control met its rule with every gate; the
corrected `C1` meets the decodability clauses and leaves the L0 gate open on the registered
world-sample pair (TOST p = 0.939), so its verdict is now conditional on L0. Across eight
independent world-sample pairs the same agents average 0.493 at drift 0 (FINDINGS 17.5). The
no-auxiliary run is NOT MET before and after. The auxiliary contrast is +0.120 [+0.065,
+0.174] on the corrected runs against +0.128 historically.

**What this does not establish.** The other survival-trained results (L2, B-v3, L1, the
second capacity, the transfer probes, the second and third instances, the skill-matched
baseline, the H2 batteries) were not rerun and keep their historical label. That the
correction barely moved the two runs that were rerun does not show it would barely move
those.

**Found alongside, not caused by the correction.** The readouts on the corrected agents
narrowed four claims independently of the trainer: the survival-over-predictor margin is not a
difference of objective at matched input (17.6), the strongest well-fit control leaves a
signal under the bar (17.8), the common garden reads the physical footprint (17.7), and a
hand-authored coherent perturbation is read like the learned one (17.9). These are recorded as
claim decisions in `docs/REVISION_2026-10.md`.

## 2026-10-06: branch history rewritten to keep the manuscript tables out of main

**What.** Three commits on the revision branch had committed the LaTeX tables generated for the
manuscript (`docs/paper_tables/*.tex`). The manuscript and everything generated for it stay
local (`AGENTS.md`), and a regular merge would have carried those commits into main's history.

**Change.** The branch was rewritten from the first commit that added the tables onward, with
the tables removed from every commit and each rewritten commit signed again. The final tree is
byte-identical to the one before the rewrite. The 13 earlier commits keep their hashes,
including `f676b95`, the code every corrected-run cell records. Two later commits that the
provenance records name changed hash; their code is identical, only the generated tables
dropped out:

| before | after | what records it |
|---|---|---|
| `39c1e5d` | `34e2c0d` | the C1 readouts (`commit_at_run` in the manifest) and the C1 promotion (`git_commit_at_promotion`) |
| `0894d6c` | `7870bba` | the exploratory cross-run replay and the C2 promotion |

The references were updated to the new hashes. No number changed.

