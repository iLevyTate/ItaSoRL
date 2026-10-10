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
carried from the summaries. Eleven of its twenty-two rows do not show L0 equivalence; they
read "inconclusive" (the count the generated page itself reports; ten of the twenty historical
rows, plus the corrected `C1` row). `itasorl/l0_audit.py` adds the pre-intervention probe, the world-sample
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

*(2026-10-10: the held-out sample in this entry overlapped G's training episodes; see the
2026-10-10 entry on the surrogate diagnostics below.)*

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
including `f676b95`, the code every corrected-run cell records. Two later commits named in
the provenance records changed hash with identical code: the C1 readouts and the C1
promotion now name `34e2c0d`, and the exploratory cross-run replay and the C2 promotion now
name `7870bba`. No number changed. The pre-rewrite hashes are deliberately not listed here,
because they still resolve on GitHub to commits that carry the tables.

## 2026-10-10: the surrogate's held-out diagnostics were in-sample

**What was wrong.** `surrogate_diagnostics` scored `G`'s held-out error on 60 episodes that
start at seed 77. Episode i of a sample is seeded by the sample's base seed plus i (world,
weather, ecology and the scripted action stream), and every `G` was trained on 250 episodes
that start at its own seed (0, 1 or 2). The 60 episodes were therefore episodes 77 to 136 of
each `G`'s training data, bit for bit. The train column also used seed 0 for every `G`, which
is the training data of the two seed-0 instances only. FINDINGS 10.1.1, the 2026-10-06 entry
above, the docstrings and the texture comparator spec called the sample held out at disjoint
seeds.

**Change.** The held-out sample now starts at seed 100000, `surrogate_diagnostics` refuses a
held-out range that overlaps the training range, and `scripts/run_surrogate_diagnostics.py`
scores each `G`'s train column on its own training seed. The committed artifact
`artifacts/surrogate_diagnostics.json` keeps every stored value and gains a `disjoint_heldout`
block. The runner writes that block only after the rerun reproduces the stored
agent-accessible detector scores exactly and the stored deviation profiles to within 1e-6;
the largest profile difference was 6.5e-8, from retraining `G` on the owner's workstation
stack (Python 3.13.2, numpy 1.26.4, torch 2.7.0). FINDINGS 10.1.1 now leads with the
held-out table and keeps the seed-77 table, relabeled as the calibration sample: the
quadratic drag's eps 6.0 was matched on that sample (FINDINGS 17.9,
`artifacts/texture/qd_eps1_calibration.json`), so it stays as recorded.

**Effect.** The headline `G` errs by 0.0354 RMS per step held out, against 0.0349 on the
calibration sample. Its open-loop rollout gap at 24 steps is 0.076 against 0.072, and the
lag-1 autocorrelation of its deviation is 0.83 against 0.84. The other three instances and
the two comparators move by similar amounts (FINDINGS 10.1.1). The agent-accessible
detector (0.991, and 0.967 for the quadratic drag) reads neither sample and is unchanged.

**What this does not change.** No organism number, no gate and no verdict. The surrogates
are the same frozen nets, and the gate-0 oracle never read this sample.

## 2026-10-10: both comparator-trained runs had an 8-unit agent trunk

**What was wrong.** Two comparator-trained runs were executed: `T-qd` (quadratic drag,
2026-10-06, set aside in FINDINGS 17.9) and `T-gn` (white jitter, 2026-10-08 and 09, FINDINGS
14.5.1). The record said that only `T-gn` had an 8-unit agent trunk, that its launch command
caused it, and that every other run in the project had the registered 96-unit trunk. All 60
saved agents of `T-qd` also have `cell.weight_hh` `(24, 8)`, its recorded config fingerprint
reproduces only with `--hidden 8`, and its speed positive control is short in 8 of 60 pools
(worst 0.7018), which no document reported. The error was written into the specs before either
run: the 2026-10-06 texture comparator spec calls the protocol a "hidden 8 trunk", and the
frozen command block of the 2026-10-09 `gn` spec reads `--l3-hidden 8 --hidden 8`.

**Change.** Dated corrections under FINDINGS 14.5.1 and the 17.9 addendum, a dated entry in
PREREGISTRATION_L3 section 12, and dated notes at the end of both specs; no frozen text is
rewritten. The 17.9 addendum's seed-bootstrap interval for `T-qd` now has the t-based interval
beside it, [0.537, 0.624], computed from the local run bundle.

**What this does not change.** No verdict. `T-gn` was already uninformative and `T-qd` already
excluded, and no claim rests on either. Runs whose agents were not kept cannot be checked from
their weights; the other seven saved agent sets checked on 2026-10-10 all have the registered
trunk.

## 2026-10-10: the held-out hidden-8 run was not the only in-band hidden-8 instance with saved agents

**What was wrong.** FINDINGS 17.5.1 and the frozen specs of the drift-0.45 world-sample
sensitivity run (2026-10-08) and the reverse-direction readout (2026-10-09) say the historical
held-out run is "the only in-band hidden-8 instance with saved weights". The seed-2 hidden-8
instance (`L3-H8-GS2-GPU`, oracle 0.939, in band, FINDINGS 10.9) also kept its 60 agents in
`fullruns/l3_h8_gseed2_gpu/agents/` (10 seeds at drifts 0 and 0.45, three arms), as
`fullruns/l3_h8_heldout/agents/` does.

**Change.** FINDINGS 17.5.1 now says the held-out run is the only retained instance of the
headline fingerprint (hidden 8, G seed 0), the fingerprint the corrected `C1` agents were raised
under, and names the seed-2 instance; both specs carry a dated note at their end and are
otherwise left as frozen.

**What this does not change.** Both readouts scored the agents their specs named, and neither
result moves. The seed-2 agents have not been scored by either design. Like every saved agent
set, both directories are gitignored local files and are not in the repository.

## 2026-10-10: three Experiment B cells in FINDINGS 3.1 were off by 0.001

**What was wrong.** The FINDINGS 3.1 table gave the drift-0 target as 0.460 and the shuffled
control as 0.522 at drift 0.20 and 0.548 plus or minus 0.059 at drift 0.45. The promoted
artifact `artifacts/expB/summary.json` and the run log it was promoted from
(`fullruns/06302026/steps/expB_full.log`) read 0.459, 0.523 and 0.547 plus or minus 0.058.
The run recorded no per-seed values, so the artifact's three-decimal means are the source.
The audit checked three of the twelve table values the artifact stores.

**Change.** The three cells now match the artifact and the log, with a dated note under the
table giving the old values, and the audit checks all twelve stored values (target mean and
standard deviation, shuffled mean, speed mean, at each drift).

**What this does not change.** Every reading stays at chance, and no verdict moves.

## 2026-10-10: the random-forest control value in FINDINGS 3.4

**What was wrong.** FINDINGS 3.4 gave the random-forest probe's drift-0 target as 0.459 plus or
minus 0.009. That mean is the linear probe's (3.1). The random-forest value is 0.460 plus or
minus 0.009 in `artifacts/expB/summary.json` (`expB_nonlinear`) and in the run log.

**Change.** FINDINGS 3.4 quotes 0.460 with a dated note, which also says that the speed
standard deviations of the 3.1 table, like its shuffled ones, exist only in the gitignored run
log. The audit compares the 3.4 values with the artifact.

**What this does not change.** The reading is at chance either way; no verdict moves.

## 2026-10-10: per-individual means of the Experiment C rerun quoted off by 0.001

**What was wrong.** The 2026-10-07 amendment in `docs/PREREGISTRATION_C.md` gives the final
treatment per-individual means as 0.535, 0.539 and 0.537 and the generation-0 means as 0.527,
0.521 and 0.527. The promoted summary (`artifacts/expC/emergence_pilot_per_individual_summary.json`)
and the run file give 0.534, 0.539 and 0.536 and 0.526, 0.521 and 0.527. FINDINGS 17.11 gave
the final treatment range as 0.535 to 0.539, and FINDINGS 13.F says the individuals "average
0.535", which no grouping of the stored means reproduces; the final treatment populations
average 0.537.

**Change.** FINDINGS 17.11 quotes 0.534 to 0.539 with a dated note. The 13.F figure and the
amendment's values are corrected by the second 2026-10-10 note under 13.F and a dated entry in
`docs/PREREGISTRATION_C.md`; neither earlier text is edited. The audit recomputes the 17.11
range.

**What this does not change.** No individual reaches 0.65 in any population, and the run's
routing does not move.

## 2026-10-10: which gate makes Experiment C uninformative, and what the run record claimed

**What was wrong.** The summary surfaces of FINDINGS, `docs/PAPER_OUTLINE.md` and the 13.F note
said the verdict is uninformative because "gates 1 and 5 fail" and that the preregistration
"routes a gate-1 failure to uninformative". Section 7 of `docs/PREREGISTRATION_C.md`, the frozen
text, gives gate 1 no numeric margin. The 0.005 margin is in section 14, which was never
ratified, and against the frozen wording the treatment gap (0.00228, 90% CI [0.00109, 0.00377])
is a weak positive. The routing rests on gate 5, the speed positive control (0.527 to 0.631
against 0.75), because section 7 interprets no contrast unless every gate passes. A review of
the run file and the runner found more. Gate 3 is satisfied by construction (identical
branches, every panel 0.500). The pooled-AUROC shift from 13.D comes from the fold partition
alone. The rerun's summary was promoted although 13.F and the amendment say it was not.
"Bit-reproducible" means reproducible to 8 decimals. The reproduction threshold saturates
within three to seven generations. Every lifetime is scored on the same two layouts, against
the per-episode keyed draw of section 4, a fourth deviation from the preregistration. The
separate survival series uses two fixed layouts for 44 steps and records no death while the
panel drops about 6% of its pairs.

**Change.** The FINDINGS summary surfaces (TL;DR, section 1, section 6, 17.12) and
`docs/PAPER_OUTLINE.md` (section 4.7, row 27) name gate 5 and the draft margin. A second dated
note under FINDINGS 13.F records each point with its numbers and attributes the estimand and
the gates to their two runs (`a0cb850` and `06ecc10`), and adds the descriptive per-individual
contrast, +0.006 [-0.012, +0.024]. A dated section-12 amendment in `docs/PREREGISTRATION_C.md`
records the same points, the fourth deviation and the in-place edits to that file.

**What this does not change.** The verdict stays uninformative, now on gate 5 under section 7.
No number of the estimand moves.

## 2026-10-10: H2 readings narrowed (graded seam at hidden 7, observation masks, L1 calibration)

**What was wrong.** FINDINGS 14 argued specificity from a flat untrained floor and said the
graded-seam result holds at both capacities. At hidden 7 the untrained floor rises with the
seam, from 0.474 to 0.586 (local run bundle), and the predictor from 0.500 to 0.714. The
alpha-0 equivalence rested on the bootstrap ROPE check; the TOST accepts at hidden 8 (p = 0.005)
and not at hidden 7 (p = 0.086). In FINDINGS 14.6 and the two 2026-07-24 entries of
`docs/PREREGISTRATION_L3.md`, the all-mask collapse to 0.500 was read as a sanity check,
although a constant input makes every probe read 0.500, the positive controls included. A
masked channel enters the trunk as the constant -mean/sd (as far as 31.8 sd out for the
interoceptive channels), so the single-channel masks are constant-input lesions. At hidden 7
they lift the untrained arm to 0.724 (vision) and 0.646 (interoception). FINDINGS 14.7's "leakage
clean" for the L1 calibrations cannot fail, because both members of each pair come from one
fixed-policy trajectory.

**Change.** Dated notes under FINDINGS 14, 14.6 and 14.7, a dated entry in PREREGISTRATION_L3,
and section 6 of FINDINGS adds the TOST. The audit recomputes the hidden-8 TOST, the hidden-8
margins and the hidden-7 mask readings from committed artifacts; the hidden-7 graded-seam
numbers live only in the gitignored run bundle.

**What this does not change.** The survival-minus-untrained margin falls with the seam at both
capacities, and the hidden-8 collapse and its TOST stand. No run is repeated.

## 2026-10-10: gate statements narrowed (hidden 7, the hidden-8 regression check, B-v2, the supervised control)

**What was wrong.** The hidden-7 run's reward-leak (0.567, clean 10/10) and speed (0.959)
figures in FINDINGS 10.5 and the 2026-07-14 entry are the survival arm's; counted over every
arm, one predictor pool (drift 0.45, seed 8) reads 0.616, outside the 0.1 tolerance. The hidden-8
gate-0 regression is called "exact", while it is a tolerance check whose floor, 0.482, is a
three-seed mean against a frozen seed-0 value of 0.483. FINDINGS 9 gave the B-v2 speed control
as 0.84 to 0.96 (the pools run from 0.790 to 0.968) and did not say its plus-or-minus values are
population standard deviations. FINDINGS 7.1 quoted the supervised control's matched-pair
channel at about 0.70 without saying it was scored by the pre-fix matched-pair estimator, whose
drift-zero floors read 0.137 in that run; the B-v3 matched-pair leakage audit used the same
grouping.

**Change.** Dated notes under FINDINGS 7.1, 9 and 10.5, and dated entries in
`docs/PREREGISTRATION_L3.md` and `docs/PREREGISTRATION_Bv3.md`. The audit recomputes the
hidden-7 leakage count, the B-v2 speed range and the population standard deviations.

**What this does not change.** Every gate named still passes on the survival arm, the B-v2
speed control passes in every pool, and the B-v3 pooled verdict does not use the matched-pair
channel. No verdict moves.

## 2026-10-10: the speed positive control was read on arm means

**What was wrong.** `scripts/promote_reviewer_gaps_runs.py` recorded `gates.speed_min` as the
lowest arm mean, while the registered gate 3 asks for at least 0.75 in every pool. Eight promoted
artifacts carried the higher figure, the gate table copied it, and FINDINGS 10.8, 10.8.1 and
10.9 and six entries of `docs/PREREGISTRATION_L3.md` quoted it (0.814 to 0.837). The lowest
pools read 0.748 to 0.790. On the hidden-10 GPU re-measure one pool of 60 reads 0.748, under
the bar, and no document reported it.

**Change.** The promoter computes the per-pool minimum and records where it sits, and
`--refresh-gates` rewrote the gates block of each promoted artifact, keeping the earlier figure
beside it and leaving the decision block as promoted. The audit pin for the skill-matched run
moved from 0.837 to 0.790 and the gate table is rebuilt. FINDINGS 10.9 has a dated note with
every run's two figures, and `docs/PREREGISTRATION_L3.md` a dated entry.

**What this does not change.** Every other run passes in every pool. The GPU re-measure's rule
was not met before; with its positive control short in one pool, its readings are descriptive.
The 0.75 threshold is not relaxed, and the corrected verdicts already used per-pool values.

## 2026-10-10: the 156-value reproduction check cannot be backed

**What was wrong.** FINDINGS 17, `docs/REPRODUCE.md` and the corrected-trainer spec say that
the original code at `4b6e1f3`, rerun in the revision container on one committed cell of the
historical CPU device control, reproduced all 156 of its recorded values. The rerun cell was
not saved, and no local bundle or log holds it. The committed cell
(`artifacts/reviewer_gaps_runs/l3_h8_wm_cpu/cells/cell_d0.45_s0.json`) has 157 leaves under
`cell`, 139 numbers (3 of them NaN) and 18 booleans, so the count cannot be rebuilt from it.

**Change.** FINDINGS 17 and `docs/REPRODUCE.md` say the rerun reproduced the committed cell's
recorded values, including survival target 0.66570 and engagement return -0.65194, and that
the rerun cell was not saved. The frozen spec has a dated note.

**What this does not change.** The two quoted values match the committed cell. Whether every
other value matched cannot be shown without a new rerun at `4b6e1f3`.

## 2026-10-10: what the Experiment C payoff comparison measures, and what the record left unlabeled

**What was wrong.** FINDINGS 13.E and 17.11 called the matched assignment minus the better
single policy (-0.043 on the historical CPU cells, -0.056 on `C1`) a value of world
information, and the docstring of `itasorl.experiment_c.value_of_world_information` called it
a lower bound on what an ideal world-conditional policy could gain. The quantity is negative
whenever one policy returns more in both worlds, so it bounds nothing. The best-of-two value,
which cannot be negative, is +0.013 on the historical cells and +0.003 on `C1` (zero in 8 and
in 9 of 10 seeds). Both comparisons, and the 17.11 validation behind the estimate of 0.61 under
the pooled probe, use 96-unit gradient-trained agents; the evolved Experiment C policies have an
8-unit trunk. `docs/PREREGISTRATION_C.md` does not label that trunk width, the 2 + 2 episode
lifetime, the missing energy cost of reproduction, the panel cadence, or the missing
generational curve and untrained floor. The results manifest row for the per-individual rerun
recorded a commit, `3c6f0aa`, that is not a revision of this repository (the artifact and the
checkpoints record `06ecc10`), named claims row 28 (the sensory-echo row) and gave its fold
scheme as n/a. The site gave the individuals' average as 0.535, the figure the second 13.F note
of 2026-10-10 corrects to 0.537.

**Change.** Dated notes under FINDINGS 13.E, 13.F (third note) and 17.11, and a second
2026-10-10 entry in `docs/PREREGISTRATION_C.md`. The docstring is corrected and the function's
output is unchanged. `docs/PAPER_OUTLINE.md` row 49 names the quantity. The manifest row records
`06ecc10`, claims row 27 and the explicit partition, and the site says 0.537. The audit
recomputes both best-of-two values from the committed cells and checks both runs' wall times.

**What this does not change.** No verdict moves. Experiment C stays uninformative on gate 5, and
-0.043 and -0.056 stand as measured.

## 2026-10-10: the results manifest's note on the legacy fold split

**What was wrong.** The legacy entry of `FOLDS_NOTE` in `scripts/build_results_manifest.py`
gave the published GPU numbers' split as (24,20), (22,22), (21,23), (21,23), (22,22), the split
first inferred from fold means. The 2026-09-29 re-score measured (22,22), (22,22), (22,22),
(21,23), (23,21) (`partition_110_110.legacy` in `artifacts/fold_rescore/`), and the explicit
CV-folds spec and the `itasorl/folds.py` docstring already say the measurement supersedes the
inference.

**Change.** The note gives the measured split and calls the inferred one superseded;
`docs/RESULTS_MANIFEST.md` is regenerated.

**What this does not change.** No number. The note describes a partition the re-score records.

## 2026-10-10: the hidden-7 graded seam has a committed source

**What was wrong.** The H2 entry above says the hidden-7 graded-seam numbers live only in the
gitignored run bundle. That bundle has since been promoted as
`artifacts/expH2/summary_h7.json`.

**Change.** FINDINGS 14 notes the promotion, and the audit recomputes the hidden-7 numbers of
the 14 note from the artifact.

**What this does not change.** Every number the note quotes matches the artifact.
