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
read "inconclusive" (the count the generated page reported when this entry was written; ten of the twenty historical
rows, plus the corrected `C1` row). As of 2026-10-12 the generated page reports 14 of 26 rows.
`itasorl/l0_audit.py` adds the pre-intervention probe, the world-sample
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

*(Follow-up 2026-10-19: the MLP `train_g_motion` was first committed in `489a772` (2026-07-10,
14:00 UTC), after the L3 registration `0217263` (10:23 UTC the same day) and before the first L3
organism run, recorded in `ced27f4` (21:23 UTC). See the 2026-10-19 entry on registration timing below.)*

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
those. *(Updated 2026-10-16: the second capacity and the two further instances were rerun on
the corrected trainer (FINDINGS 20), and so was the L1 organism run, `L1-CORRECTED` (FINDINGS
22.1): 0.551 [0.531, 0.570], 0 of 10 seeds at 0.65, against 0.552 for the historical run on the
same partition. The L1 negative reproduces. The other rows of this list keep their label.)*

**Found alongside, not caused by the correction.** The readouts on the corrected agents
narrowed four claims independently of the trainer: the survival-over-predictor margin is not a
difference of objective at matched input (17.6), the strongest well-fit control leaves a
signal under the bar (17.8), the common garden reads the physical footprint (17.7), and a
hand-authored coherent perturbation is read like the learned one (17.9). These are recorded as
claim decisions in `docs/REVISION_2026-10.md`.

*(Follow-up 2026-10-19: the 0.493 is the average of the eight signed pair means. The
two-one-sided test over them, a diagnostic that the L0 audit artifact carries and the frozen
spec does not name, accepts equivalence to 0.5 at p = 0.022 on those signed means. Folded about
0.5, the same eight draws average 0.537 and give p = 0.129, not equivalent. The registered L0
gate is the TOST on the registered pair, so neither form changes a gate or verdict.)*

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

**Change.** FINDINGS 17.5.1 now says the held-out run is the only retained agent set of the
headline fingerprint (hidden 8, G seed 0, the one the corrected `C1` agents were raised under)
trained under `C1`'s protocol, with the next-observation auxiliary on at 300 updates. It names
the seed-2 instance and the two no-auxiliary runs of the headline fingerprint that also kept
agents: `l3_h8_nowm` (six drift-0 agents) and `l3_h8_nowm_skill_u450` (60 agents, the survival
arm at 450 updates). Both specs carry a dated note at their end and are otherwise left as
frozen.

**What this does not change.** Both readouts scored the agents their specs named, and neither
result moves. None of the other agent sets has been scored by either design. Like every saved
agent set, these directories are gitignored local files and are not in the repository.

*(Follow-up 2026-10-19: both designs have since scored the corrected `C1` agents, on 2026-10-18,
as post hoc readouts (`docs/specs/2026-10-17-closing-readouts-design.md`, sections G02 and B3).
The world-sample readout reads C1's eight independent drift-0.45 draws at 0.685 to 0.829, label
R1 SECURE, with C1's own between-draw SD 0.052
(`artifacts/l0_audit/d045_world_samples_corrected_l3_h8_wm.json`); the record's 0.042
propagation stays. The reverse-direction readout reads the drift-0 agents at 0.754 [0.721,
0.787], label R1 CARRIED, descriptive on C1 with no rule-ledger row
(`artifacts/l0_audit/reverse_direction_corrected_l3_h8_wm.json`). No stored value or verdict
changes.)*

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

## 2026-10-10: Experiment C's registered verdict is uninformative, not a validated null

**What was wrong.** Before this pass the summary surfaces read the fixed-code re-run of 13.D as
a null. The README said "Negative under the pooled population readout" and "validated null";
`docs/PAPER_OUTLINE.md` row 27 said "the fixed-code re-run is a validated null under the pooled
population readout"; FINDINGS said "validated null" in section 1 and in the H3 bullet of section
6, "the evolutionary null is restricted to the pooled population readout" in the revision
banner, and "The evolutionary null stays restricted" in 17.12. The per-individual rerun of
2026-10-07 ran the full gate battery, and its records (the 13.F gate paragraph, the 2026-10-07
amendment in `docs/PREREGISTRATION_C.md` and step 12 of `docs/REVISION_2026-10.md`) said that
gates 1 and 5 fail. Section 7 of `docs/PREREGISTRATION_C.md`, the frozen text, interprets no
contrast unless every gate passes, so the registered verdict is uninformative. It gives gate 1
no numeric margin. The 0.005 margin is in section 14, which was never ratified, and against the
frozen wording the treatment gap (0.00228, 90% CI [0.00109, 0.00377]) is a weak positive. The
routing therefore rests on gate 5, the speed positive control (0.527 to 0.631 against 0.75). A
review of the run file and the runner found more. Gate 3 is satisfied by construction
(identical branches, every panel 0.500). The pooled-AUROC shift from 13.D comes from the fold
partition alone. The rerun's summary was promoted although 13.F and the amendment say it was
not. "Bit-reproducible" means reproducible to 8 decimals. The reproduction threshold saturates
within three to seven generations. Every lifetime is scored on the same two layouts, against
the per-episode keyed draw of section 4. The separate survival series uses two fixed layouts
for 44 steps and records no death while the panel drops about 6% of its pairs.

**Change.** The FINDINGS summary surfaces (revision banner, TL;DR, sections 1 and 6, the 13
banner, the scope paragraph of 14, 17.11 and 17.12), `docs/PAPER_OUTLINE.md` (section 4.7, row
27) and `docs/REVISION_2026-10.md` (step 12) state one verdict: the contrast is flat under both
readouts, and the registered verdict is uninformative, decided by gate 5, with gate 1 a weak
positive under the frozen wording that falls short only of the draft margin. The README, the
site and `docs/LEARNING.md` say the same. The audit holds the FINDINGS banner, TL;DR, section
1, section 6 and 17.12, `docs/PAPER_OUTLINE.md`, the README, the site and `docs/LEARNING.md` to
it. The site's Experiment C card footer ("uninformative (gates 1, 5)") and item 6 of the
cover page of both July 2026 walkthrough PDFs ("because registered gates 1 and 5 fail") had
kept the old routing. The footer now names gate 5 and the covers name it as the gate that
decides, and the audit guard matches any listing of gates 1 and 5 together on the surfaces
it reads. Two dated notes under FINDINGS 13.F record each point with its numbers, attribute the
estimand and the gates to their two runs (`a0cb850` and `06ecc10`), and add the descriptive
per-individual contrast, +0.006 [-0.012, +0.024]. Two dated section-12 entries in
`docs/PREREGISTRATION_C.md` record the same points and the in-place edits to that file, and
count seven deviations from its frozen text: the four of the first entry, the fixed lifetime
layouts among them, and in the second the missing energy cost of reproduction, the panel
cadence and the unreported section-10 analysis.

**What this does not change.** No number of the estimand moves. 13.D keeps "validated null" as
a description of the pooled estimand, and the verdict bears on H3 in neither direction.

## 2026-10-10: H2 readings narrowed (graded seam at hidden 7, observation masks, L1 calibration)

**What was wrong.** FINDINGS 14 argued specificity from a flat untrained floor and said the
graded-seam result holds at both capacities. At hidden 7 the untrained floor rises with the
seam, from 0.474 to 0.586 (`artifacts/expH2/summary_h7.json`), and the predictor from 0.500
to 0.714. The
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
and section 6 of FINDINGS adds the TOST. The hidden-7 graded-seam run is promoted from its
local bundle as `artifacts/expH2/summary_h7.json`. The audit recomputes the hidden-8 TOST, the
hidden-8 margins, the hidden-7 mask readings and the hidden-7 graded-seam numbers of the note
from committed artifacts.

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

## 2026-10-10: the C1 verdict label led with MET

**What was wrong.** The corrected `C1` run was labeled "MET on the decodability clauses,
conditional on the open L0 gate" in `artifacts/corrected_verdicts.json`, FINDINGS (TL;DR, 17.2,
17.12), `docs/PAPER_OUTLINE.md` (section 4.3, row 41), `docs/REVISION_2026-10.md` and the
2026-10-06 results entry of `docs/PREREGISTRATION_L3.md`. FINDINGS 17.4 said the
auxiliary-conditional reading "holds on the decodability clauses, conditional on `C1`'s open
gate". The frozen rule counts every gate, so a run whose L0 gate is open does not meet it, and
the auxiliary rule asks for `C1` to meet its primary rule. Leading with MET stated both
verdicts more favorably than the rules do.

**Change.** The artifact's label reads "NOT MET as registered (open gate(s): l0); decodability
clauses MET", and the generated verdict table reports the auxiliary rule as not met as
registered before the decodability lead. FINDINGS (TL;DR, 17.2, 17.3, 17.4, 17.12),
`docs/PAPER_OUTLINE.md` and `docs/REVISION_2026-10.md` lead with the registered outcome, with a
dated marker where a label is replaced, and a dated entry in `docs/PREREGISTRATION_L3.md`
restates the 2026-10-06 wording. The audit's 17.4 check reads the new wording.

**What this does not change.** No reading, interval or margin. The decodability clauses are met
and the L0 gate is open, as before.

## 2026-10-10: the L1 organism result was worded as no encoding

**What was wrong.** FINDINGS 14.7 said the oracle-detectable L1 artifact "leaves no decodable
world-identity trace in the survival state", section 6 called H2 at L1 "a strengthened
negative", and the 2026-07-25 entry of `docs/PREREGISTRATION_L3.md` says the artifact "leaves
no incidental world-identity encoding". The survival mean, 0.533, sits above chance: its seed
bootstrap interval is [0.509, 0.556], its t-based 90% interval [0.506, 0.561], and the TOST
against 0.5 at the 0.05 margin does not accept equivalence (p = 0.147). The strengthened
negative of the L3 decision table needs survival at chance.

**Change.** FINDINGS 14.7 says the mean sits above chance and below the bar, and section 6 calls
the L1 result negative at the registered bar, both with dated markers. A dated entry in
`docs/PREREGISTRATION_L3.md` restates the 2026-07-25 wording. The audit checks the t interval
and the TOST p that 14.7 quotes.

**What this does not change.** Primary H_B2 is not met, as recorded, and 0 of 10 seeds reach
0.65.

## 2026-10-10: what the matched-input readouts of 17.6 show

**What was wrong.** FINDINGS 17.6 said that on replayed survival streams a prediction-trained
trunk reads the world "as well as" the survival trunk, credited the roughly 0.1 by which the
trained agents read above the untrained one to training of the trunk, and said the survival
policy "visits states where they differ" more than the scripted policy does. Section 6 said the
flipping condition is a dynamics difference "reached by the foraging trajectories". The replay
readouts put the predictor within about 0.01 of the survival trunk with intervals that include
zero, which does not show equality. In `replay` each trunk reads the stream through its own
frozen normalizer, so the 0.1 belongs to trunk and normalizer together. The size of the law
deviation along each policy's trajectories was never recorded.

**Change.** 17.6 says "within about 0.01" with the paired contrasts, adds the
prediction-trained arms against the untrained arm, assigns the 0.1 to the trained agent as a
whole, and says that why trunks keep more from the survival streams is not measured, with dated
markers where a sentence is replaced. The TL;DR, the ladder table, section 6 and 17.12 use the
same wording, as do `docs/PAPER_OUTLINE.md` section 4.5 and row 44. The audit recomputes the
contrasts from `artifacts/policy_controls/corrected_l3_h8_wm.json`.

**What this does not change.** No number. The primary verdict of 17.2 is defined on the `own`
protocol, and "survival-specific" stays withdrawn.

## 2026-10-10: which historical runs the explicit partition moves off L0 equivalence

**What was wrong.** FINDINGS methods note 8 said the L0 equivalence test flips on "every
GPU-scored L3 and L1 organism run". The L1 organism run was trained on CPU. The runs whose L0
test does not accept under the explicit partition are the ones scored on the owner's
workstation stack, the GPU-trained L3 runs and the CPU-trained L1 organism run among them; the
runs scored on the cloud CPU stack pass.

**Change.** Note 8 names the runs by the stack they were scored on, says the stack is named as
provenance and not as the cause (the explicit partition is the same on every stack), and dates
the wording change.

**What this does not change.** No gate value and no verdict.

## 2026-10-10: specs said to be frozen before runs that git cannot place before them

**What was wrong.** FINDINGS 10.4.2 said the sensory-echo spec was "frozen before the run", and
the spec says the same of itself. Its spec, runner, preregistration entry and artifact first
appear in one commit, about an hour after the run ended. Three more runs show the same gap
between the record and git: the hidden = 7 selection (already recorded on 2026-07-16), the
nonlinear joint control, whose spec was committed about seven minutes after the run started,
and the A1 graded-seam ablation, whose spec was committed 87 seconds after the run wrote its
aggregate.

**Change.** FINDINGS 10.4.2 says the spec was first committed together with its result, and a
dated entry in `docs/PREREGISTRATION_L3.md` gives the commit and run times of all four.

**What this does not change.** No number, rule or verdict. The written-before-launch statements
may be true; git cannot show it. The review checked these four cases and did not audit the
timing of every other spec.

## 2026-10-10: the figures of sections 2 and 3 plotted a different run

**What was wrong.** The committed `expA_ceiling.png`, `expA_L2_ceiling.png`,
`expB_incidental.png` and `expB_channels.png` were images committed on 2026-06-28 from the
owner's local stack, which the local quick-mode run `fullruns/07042026` reproduces byte for
byte. The FINDINGS text and tables quote the promoted run `fullruns/06302026` (Tesla T4).
Plotted and quoted values differed by up to 0.055 (L1 at Δ = 0.015: 0.457 plotted, 0.402
quoted). The images also carried the project name in a raster title, and `expB_incidental.png`
left out the speed positive control.

**Change.** `scripts/plot_record_figures.py` redraws the five figures of sections 2 and 3 and
the two B-v2 figures in `artifacts/expB2/` from the committed artifacts.
`docs/figures/README.md` and a dated note at the head of FINDINGS section 2 record the
provenance.

**What this does not change.** No number in the text or tables, and the qualitative reading of
every figure.

## 2026-10-10: the status page put the hidden 8 new-seed instance under the bar

**What was wrong.** `docs/STATUS_2026-09-27.md` says the hidden 8 instance at G seed 2 (0.676)
lands "under the 0.65 bar with the CI excluded". Its mean is above 0.65; the lower bound of its
t-based 90% interval, 0.636, is below (FINDINGS 10.9).

**Change.** A dated correction at the head of the status page. The recorded sentence is kept.

**What this does not change.** The frozen rule's reading: replication at hidden 8 is not
claimed, because the interval's lower bound sits under 0.65 and the margin over the predictor
misses 0.05.

## 2026-10-10: Experiment B's bar was called registered

**What was wrong.** `docs/PAPER_OUTLINE.md` row 4 said the prediction-only L2 result "did not
meet the registered encoding criterion". Experiment B ran before the first preregistration
(`docs/PREREGISTRATION.md`, 2026-06-28, which cites its result), so no criterion was registered
for it.

**Change.** Row 4 and section 4.2 of `docs/PAPER_OUTLINE.md` say the 0.65 bar is applied to
Experiment B after the fact.

**What this does not change.** The reading: 0.510 at drift 0.45, at chance.

## 2026-10-10: promoted artifacts named the registered verdict with a gate open

**What was wrong.** `scripts/promote_reviewer_gaps_runs.py` wrote the zone "ENCODING INDUCED
(all clauses)" whenever the survival mean reached 0.65 and both margins held. It read neither
the registered gates, which the decision matrix (`docs/PREREGISTRATION_L3.md` section 8)
requires to pass, nor the t-based 90% interval that section 10 adjudicates the bar on. Two
promoted artifacts carried that name with the L0 gate open: `C1`'s
(`artifacts/expB2/corrected_l3_h8_wm.json`), whose verdict in
`artifacts/corrected_verdicts.json` reads NOT MET as registered, and the skill-matched run's
(`artifacts/expB2/skill_matched_l3_h8_nowm_u450.json`).

**Change.** The zone reads "ENCODING INDUCED" only when the mean is at the bar, the t-based 90%
interval excludes it, both margins hold, and every gate passes. Clauses met with a gate open
read "DECODABILITY CLAUSES MET, GATE OPEN: <gates>", and a mean at the bar whose interval
straddles it reads "MEAN AT BAR, T90 STRADDLES 0.65". A speed gate with a pool that could not
be scored, or a leakage gate with an arm that carries no flag, reads as not recorded instead of
passed or open. `--relabel` rewrote the zone of both artifacts, which now read "DECODABILITY
CLAUSES MET, GATE OPEN: l0", and each keeps its earlier zone in a `zone_relabelled` note.
Before writing, it checked every number against a re-promotion from the committed cells.

**What this does not change.** No number moves, and no other committed artifact's zone
changes. The corrected verdicts, the gate table and FINDINGS already read `C1`'s registered
rule as not met.

## 2026-10-11: the random-forest floor of FINDINGS 10.3 has no committed source

**What was wrong.** FINDINGS 10.3 and the `docs/PREREGISTRATION_L3.md` entry of 2026-07-12
report a random-forest probe on the untrained hidden-8 agent at 0.517 (shuffled baseline 0.506,
L0 untrained 0.483) as a control that passes. No committed code produced these numbers. The
committed random forest (`itasorl/behavior_audit.py`) fits behavior features, not recurrent
states, and the check on the states was run post hoc and never committed. Re-scored on the same
dumps (`fullruns/l3_n10/states`) with the Experiment B forest of `scripts/run_expB_nonlinear.py`
on the legacy partition, the three values come close and none reproduces exactly: 0.514, 0.508
and 0.485 with random_state 0, and 0.518, 0.502 and 0.483 with random_state set to the seed. On
the explicit partition the untrained forest at drift 0.45 reads 0.557 [0.545, 0.569], an
interval that excludes 0.5, against a shuffled control of 0.516.

**Change.** `artifacts/agent_free/rf_probe_l3_n10_historical.json`
(`scripts/rf_probe_historical_h8.py`) records the forest and the registered linear probe, with
their shuffled controls, per arm and drift under both partitions; its linear values reproduce
the run's record exactly. A dated note at FINDINGS 10.3 and FINDINGS 19.3 report it. The
preregistration entry is left as written.

**What this does not change.** The historical headline (0.752 on the legacy partition) and the
survival-versus-untrained gap, which the forest puts at 0.23 to 0.26 under both partitions. No
verdict. The near-chance reading of the untrained state under a nonlinear probe holds on the
published partition only.

## 2026-10-11: the 10.5 scope note credited the shared training set

**What was wrong.** The FINDINGS 10.5 scope note (2026-07-18) quotes a residual-field
correlation of about +0.60 between the hidden 8 and hidden 7 maps and about +0.20 between
hidden 8 and `G_rff`, attributes the first to the shared training set (both maps were fit on
the bit-identical authentic transitions), and committed no artifact. Measured on 250 fresh
episodes from seed 100000, both values reproduce: +0.607 [+0.573, +0.639] and +0.174
[+0.133, +0.217]. The attribution does not. `G_rff` is fit on exactly the hidden 8 seed-0
episodes and shares 3% of variance with that map, while the MLP maps at G seeds 1 and 2, which
share 249 and 248 of their 250 training episodes with it and differ mainly in their weight
initialisation, correlate +0.43 and +0.52 with it. The recipe the MLP maps share explains their
overlap better than the shared training set does; the matrix cannot separate recipe from data
further.

**Change.** `artifacts/agent_free/residual_field_correlation.json`
(`scripts/residual_field_correlation.py`) holds the full correlation matrix of every exported
fingerprint and `G_rff` with episode-bootstrap intervals. A dated note at the scope note and
FINDINGS 19.2 report it. The scope note is left as written.

**What this does not change.** The scope note's conclusion: hidden 7 is a capacity variant of
one recipe, and the recipe-level generalization claim rests on the `G_rff` transfer of
FINDINGS 10.7. No number or rule outcome.

## 2026-10-11: the per-value reproduction check, rerun and saved

**What was wrong.** The 2026-10-10 entry above found that the step-4 machine check (the
original code at `4b6e1f3` rerunning one committed cell of the historical CPU device control)
could not be backed value by value, because the rerun cell was not saved.

**Change.** The check was run again on 2026-10-11 in a cloud session on the same stack (torch
2.14.1 CPU, numpy 2.4.6, scikit-learn 1.9.1), and this time saved. The code at `4b6e1f3` reran
both seed-0 cells of that run, drift 0 and drift 0.45. Every one of the 157 recorded values of
each cell (139 numbers, 3 of them NaN, and 18 booleans) equals the committed cell, and the
config fingerprints match (`artifacts/repro_check/per_value_4b6e1f3_l3_h8_wm_cpu_s0.json`, with
the rerun cells beside it). FINDINGS 17.1, `docs/REPRODUCE.md` and the frozen spec carry a dated
note.

**What this does not change.** No number moves. The like-for-like comparison of step 4 stood on
the two quoted values before; it now stands on every recorded value of two cells.

## 2026-10-12: equivalence_test passed on zero variance

**What was wrong.** `itasorl/stats.py` `equivalence_test` computed `se = sd / sqrt(n) if sd > 0
else 1e-12`. A constant sample therefore got a standard error of 1e-12 and passed the TOST
whenever the constant sat inside the band. Experiment C gate 3 is exactly this case: the L0
sub-panel reads 0.5 by construction, so the gate passed with a p-value of zero and certified
nothing. No committed artifact stores that p-value, so no recorded number changes.

**Change.** Zero variance now fails closed, like n < 2: NaN p-values and `equivalent=False`.
`scripts/run_expC_milestone3.py` records the gate as `identity_check_untestable` when every
panel value is exactly 0.5, and the gate no longer counts as passed.

**What this does not change.** Every equivalence result in the record has nonzero spread, so
every committed TOST p-value and every `equivalent` flag stands. The Experiment C routing
already ends UNINFORMATIVE on other gates; this removes the one gate that could not fail.

## 2026-10-12: the L2 AR(1) rung is a level shift, not volatility at a shared level

**What was wrong.** `PatchOfEarthV0` clips the AR(1) drag walk to [-0.8, 8.0]. The clip is
asymmetric, so the walk's mean is not zero. At drift_sigma 0.45, 20000 steps give mean drag
factor 1.30 (median 1.165, sd 0.885) with 11.2% of steps pinned at the lower clip. The
docstrings in `patch_of_earth.py` and `experiment_b.py` and the paper described the rung as
drag volatility at a shared central level.
(2026-10-16 addendum: the 20000 steps are 24-step episodes from reset, pooled. The walk
restarts at zero at every reset and its level rises with episode time, so 1.30 is the mean over
the readout's 24-step window and not a stationary level; the docstrings and the paper now say
so.)

**Change.** Both docstrings now state the level shift and the measured 1.30. No code in the
walk changed, so every recorded L2 number stands.

**What this does not change.** The L2 negative. A level shift is a stronger, more persistent
tell than pure volatility, so a failure to encode it is a stronger negative, not a weaker one.

## 2026-10-12: advantage normalization divides by an unmasked standard deviation

**What.** `train_actor_critic` standardizes advantages by the standard deviation of the whole
B x T tensor, including the zero-padded slots, and centres on the all-slot mean. The masked
standard deviation over valid slots only is larger by about sqrt(valid fraction), roughly 7
to 10 percent at the usual episode lengths and up to 50 percent early in training. The
policy-gradient term is therefore scaled relative to the value, entropy and world-model
terms by a batch-dependent factor. It is not a sign error.

**Change.** `adv_norm` selects the formula: "historical" (the default, every recorded run)
or "masked" (valid slots only). `--adv-norm` on `scripts/run_expB2.py` passes it through, and
"masked" changes the config fingerprint so such a run cannot resume into a recorded one.

**What this does not change.** Any recorded number. The historical path is the formula those
runs trained under, and the 2026-10-12 confirmation runs use it so they stay a replication of
the corrected trainer.

## 2026-10-12: the L1 tell is a static offset the registered oracle does not measure

**What.** `Hyd/Hydmax` sits near 0.800 and moves about 0.0005 per step, so quantization at
delta 0.023 snaps it to one grid cell for a whole episode and every surrogate observation
carries a constant offset on that channel. The registered grid-residual oracle
(`oracle_features_L1`) is blind to a constant. Measured on the registered scripted pools
(110 x 24, seed bases 800000 and 850000, delta 0.023, sensor sigma 0.01): the oracle reads
0.782, a linear probe on the per-episode mean of the continuous channels reads 0.977, and
the same probe with the hydration channel removed reads 0.627. The L0 control reads 0.478
and same-seed twins read 0.995 (`artifacts/expL1/observation_ceiling.json`).

**Change.** The ceiling readout is committed and checked by `audit_stats_recheck.py`. The
observation-noise stream can now be seeded before the drift stream (`_obs_before_drift`),
so same-seed twins share their noise; the historical order stays the default, and every
recorded L1 number stands.

**What this does not change.** The L1 negative (state 0.533, stream 0.590). The ceiling the
paper quotes (0.873) is the grid-residual oracle, which understates what the agent's own
stream carries. The negative is read against the higher ceiling from here on.

*(Follow-up 2026-10-19: the 0.873 ceiling is the oracle on the legacy fold partition. Rerun
agent-free on 2026-10-17 under both partitions
(`docs/specs/2026-10-17-closing-readouts-design.md`, section G47;
`artifacts/fold_rescore/gate0_l1_{legacy,explicit}.json`), it reproduces 0.872875 at grid 0.023
on the legacy partition, bit-equal to `artifacts/expA/l1_calib.json`, and reads 0.871125 on the
explicit one: REPRODUCED, 0.873 legacy and 0.871 explicit. No stored value or verdict changes.)*

*(Follow-up 2026-10-19: the results manifest filed `artifacts/expL1/observation_ceiling.json`
under the July L1 organism run, `L1-ORGANISM` (commit `8a71593`), although
`scripts/run_l1_observation_ceiling.py` wrote it for this entry. It now has its own manifest
entry, `L1-OBSERVATION-CEILING`, which names that runner and `2c170f0`, the commit that added
the file. The file is unchanged.)*

## 2026-10-12: the pellet run's leakage gate summed the homeostatic reward

**What was wrong.** `collect_pool` and the matched-pair branch added `r.reward` regardless of
the training objective. Under the pellet objective the trained reward is 1.0 per pellet
consumed. `T-touch`, reported at 0.796 with every gate passing, never had that reward gated.
The stored leakage block on those cells has `reward_sum` and no `pellets_consumed` key.

**Change.** Both accumulators now add `task_reward(r)`, and the leakage audit gates
`pellets_consumed` when the pool records it. The 0.796 figure stays as recorded. The
matched-world rerun is FINDINGS 21.

**What this does not change.** No committed `T-touch` number. The historical leakage readings
on survival and predictor runs used the reward those runs trained on, so they stand.

*(Follow-up 2026-10-19: commit `2c170f0` of 2026-10-12, one day after `T-touch` was promoted,
also changed the touch engagement gate: it gained an absolute floor of 0.25 pellets per episode
beside the 1.5x ratio (`TOUCH_ENGAGE_MIN_PELLETS` in `itasorl/experiment_b2.py`). No frozen
text, section-12 entry or entry here recorded it at the time. The goal-and-stakes spec now
carries a dated post-run note on it (`docs/specs/2026-10-07-goal-and-stakes-design.md`,
2026-10-17). `T-touch` was adjudicated under the ratio rule, and every one of its cells also
clears the later floor; its verdict is not re-adjudicated and no recorded value changes.)*

## 2026-10-14: the workstation NumPy pin is not an official Python 3.13 wheel

**What was wrong.** `constraints-workstation.txt` and the balanced-confirmation stack blocks
name Python 3.13.2 with NumPy 1.26.4. NumPy 1.26.4 supports Python 3.9 through 3.12.
PyPI has wheels for cp39 through cp312 and a source tarball, and no cp313 wheel.

**What ran.** `C:\Python313\python.exe`, Python 3.13.2, on Windows 11. NumPy 1.26.4
is a local meson build installed by pip, wheel tag `cp313-cp313-win_amd64`, compiled
with MSVC 19.29.30159 and Cython 3.0.12, at
`%APPDATA%\Python\Python313\site-packages\numpy`. The installed tree, 1209 files,
hashes to `675bd7f1f96e6fe6fba53bf7db13ddd50aba480cd34d44dce2ada19af9472c2a`.
A binary-only install of the official 1.26.4 package on this interpreter finds no wheel.

**What this does not change.** No recorded cell, target, or verdict. The original stack
lines in the artifacts stay as written. A newer NumPy is not a substitute for this build
if the claim is bit-exact.

## 2026-10-14: balanced speed and leakage were scored on the pooled worlds

**What was wrong.** The 2026-10-12 specification asks for the speed probe and the
leakage channels on the retained pairs at base 910000. `promote_balanced_confirmation.py`
read `pool.speed` and `pool.pool_leakage`, which are the independent pools at 800000
and 850000. Engagement stays on its own protocol (the cell's `eng` block, 64 episodes).

**Change.** The promoter keeps those pooled checks and labels their population. A MET
decision now requires finite matched-pair speed and leakage gates. Missing or non-finite
channels cannot pass. `paired_pooled_readout` records those channels on the retained
pairs. The readout script is `scripts/audit_balanced_matched_gates.py`. It does not
retrain and does not rewrite the historical cells.

**What this does not change.** B1 and B2 stay UNINFORMATIVE. The untrained floor already
fails, at 0.114 from 0.5. The recorded targets stay.

A later readout on the saved agents, `scripts/audit_balanced_matched_gates.py`,
scored the retained pairs. Leakage stays within 0.1 of 0.5 in every pool. The
speed probe is under 0.75 on the untrained arm at seed 19 (0.749) in both runs.
That readout is not folded back into the historical cell files.

*(Follow-up 2026-10-19: the promoter's `reason` field named only the untrained floor, though the
matched speed gate fails as well. It now lists every failed gate: "gates not held: matched speed
(min 0.749 under 0.75); untrained floor (0.114 from 0.5, outside the 0.1 floor)". `b1.json` and
`b2.json` were regenerated in place on 2026-10-17 (`4a4bb13`), adding a `fingerprint` key read
from the cells and changing no other field. The decision, the gates and every number are
unchanged.)*

## 2026-10-15: world-feature distances were toroidal in a walled world

**What was wrong.** The world-feature readout (FINDINGS 23.2) measured every distance with
a wrapping torus metric, while the simulator clips the agent at the walls and never wraps
(`patch_of_earth.py`). The far targets were also residualized on the near set over all
episodes before cross-validation, so held-out episodes shaped the training targets. The
manifest note said every arm's residual sat within 0.005 of zero; the `C2` predictor arm's
far water and roughness read +0.0094 and +0.0088 on the same run. The deep revision review
found all three on 2026-10-15.

**Change.** The third amendment to the spec, written before the rerun, uses the arena's
Euclidean distance, residualizes the far targets inside each training fold, and reports the
reach radius as a calibration percentile with the readout episodes' exceedance rate. The
readout was rerun on the same saved `C1` and `C2` agents, with no retraining. The first
run's artifacts are kept under `artifacts/world_features/superseded_toroidal_2026-10-15/`
with a superseded manifest row.

**What this does not change.** The label is the same on both runs: keeps nothing beyond the
senses. The correction does not establish that the state keeps nothing of its world. The
label says only that no near feature met both clauses of reading 1; the residual R squared
values run from -0.006 to +0.009 across arms, with the largest upper bound +0.013.

## 2026-10-15: online erasure was scored on common survivors only

**What was wrong.** The online erasure readout (FINDINGS 23.3) scored behavior, return and
probes on episodes alive to the horizon in all six conditions, and recorded neither how
many episodes that excluded nor any outcome over the episodes removal ended early. Its
sentence that the decodable direction can be changed without changing what the agent does
outran that subset.

**Change.** `rollout_behavior` keeps every attempted episode's return through termination,
its termination flag and its steps alive. The runner reports attrition per condition and
the population return and termination contrasts beside the common-survivor readings. The
readout was rerun on the same saved `C1` agents; the first artifact is kept under
`artifacts/mechanism/erasure/superseded_common_survivor_2026-10-15/`.

**What this does not change.** Every common-survivor value reproduces the first run to the
float, apart from the projection residual's rounding, and every reading is unchanged.
Removal excluded 26.6 percent of attempted episodes. On every attempted episode the return
and termination contrasts against the sham include zero in both worlds.


## 2026-10-16: reproduction off the writing machine is to a tolerance, not to the bit

**What was wrong.** The record said the corrected runs reproduce on the stack that produced
them, and the reproducibility statement read as if any copy of that stack would give the same
bits. The workstation GPU results also depend on a local NumPy build (the 2026-10-14 item above)
that no one else can install from PyPI.

**What ran.** A clean Python 3.11 environment on `constraints-cpu-corrected.txt` (torch
2.14.1+cpu, NumPy 2.4.6, scikit-learn 1.9.1, SciPy 1.17.1) on the owner's Windows CPU replayed
the C1 L0 audit, the C1 control diagnostics and the B1 sensory-echo control. Per-seed AUROCs
moved by up to 0.0058 on the linear readouts and up to 0.028 on the trained MLP control;
reported means moved by at most 0.0041, and the deciding B1 mean is 0.7608 on both. Eight of
the 30 B1 cells miss the 0.0005 integrity gate on CPU, so the replay withholds that decision.
`docs/REPRODUCE.md` gives the commands and the full table under "Supported-stack replay route".

**What this does not change.** No recorded cell, target, or verdict. Bit-exact claims are
bounded to the stack and machine that wrote each result, and no replayed value crosses a
threshold a committed reading relies on.

## 2026-10-16: the L2 prediction-only world model predicted each step without its action

**What was wrong.** `experiment_b.collect_episodes` discards the reset observation and records
each observation after the step that produced it, so `O[t]` is the observation that action
`A[t]` produced. `RecurrentWorldModel.forward` (`itasorl/agent.py`) feeds the pair `(O[t], A[t])`
and `prediction_loss` scores the prediction against `O[t+1]`, which `A[t+1]` produces. The model
therefore predicts every next observation without the action that causes it, and the scripted
policy draws each action independently, so it cannot infer that action. `forward_rollout` uses
the same pairing in open loop. Measured with the repo code on 60 scripted episodes of 30 steps
(`seed0` 10000), the heading change from `O[t]` to `O[t+1]` correlates 1.000 with the turn in
`A[t+1]` and 0.015 with the turn in `A[t]`. The B-v2 and L3 predictor arm is aligned correctly:
`agent_ac.world_model_loss` feeds the previous action into the trunk and decodes `O[t+1]` from
the state and the current action.

**Which results it touches.** The L2 prediction-only numbers of `scripts/run_expB_full.py`,
`run_expB_kstep.py`, `run_expB_surprise.py`, `run_expB_nonlinear.py` and `run_expB_gap.py`
(`artifacts/expB/summary.json`; FINDINGS 3 and 4). The prediction objective, the prediction-error
channel (0.596 at drift 0.45) and the open-loop horizon check are the most exposed. The state
probe still saw the full observation and action history, so the recurrent-state target (0.51 at
drift 0.45) may move little under a corrected alignment. That has not been measured.

**Change.** None to code or to stored numbers; this entry is the disclosure. The manuscript's L2
appendix now says that the L2 world model receives each action with the observation it already
produced, a design difference that cuts against decodability, unlike the two it lists that
favor it. A corrected run would be a new, separately labeled run under an opt-in alignment that
keeps the reset observation and pairs each observation with the action taken from it. The
default path stays as it is so the stored numbers reproduce. FINDINGS carries the caveat in the
TL;DR's L2 sentence, in a dated note at the head of section 3 (covering 3.1 to 3.4), and at
section 4 item 2, whose "the objective is local" explanation the same timing confounds.

**What this does not change.** No recorded number or verdict. The L2 negative stays as recorded,
read with the caveat that its prediction-only arm was a weaker predictor than designed. The L3
results, the B-v2 predictor arm and the survival-trained L2 run (B-v3, FINDINGS 9) do not use
this path.

*(Follow-up 2026-10-19: the corrected run this entry anticipates ran on 2026-10-18 on this
machine under the opt-in alignment and the legacy one, three seeds each, with no saved agent
(`docs/specs/2026-10-17-closing-readouts-design.md`, section C1;
`artifacts/expB_legacy_local_2026-10-18/` and `artifacts/expB_aligned_2026-10-18/`, JSON only).
At drift 0.45 the prediction-only state reads 0.544 ± 0.037 with correct timing and
0.517 ± 0.056 under the legacy pairing, both inside the recorded row's interval [0.429, 0.591] around
0.510 ± 0.039, so the label is RESTORE and the L2 negative stands with correct timing. The
prediction-error channel reads 0.599 ± 0.007 aligned and 0.596 ± 0.004 legacy. The effect on
decodability is therefore measured, and the weaker-predictor caveat above is read with it. The
frozen spec printed the recorded interval as [0.430, 0.591], a rounding slip: its formula gives
[0.429, 0.591], and no label moves, since 0.517 and 0.544 lie inside either way. No stored
number or verdict changes.)*

*(Follow-up 2026-10-19: the survival-trained L2 run named above, B-v3, is reported in
FINDINGS 7.1, not section 9, which holds the B-v2 arc.)*

## 2026-10-16: the development readout's transfer column is in-sample by world seed

**What was wrong.** The development readout (FINDINGS 23.4) reports that the frozen 300-update
probe reads the 450-update states at 0.785 on `C1` and 0.689 on `C2`.
`scripts/run_development_readout.py` reads every snapshot on the same registered pools (world
bases 800000 and 850000) and calls `transfer_probe(*feats[300], *feats[b])`, which fits on all
of the 300-update episodes and scores the other snapshot's states of the same world seeds, which each snapshot walks with
its own policy. The
docstring of `transfer_probe` says train and test worlds are disjoint by construction, and this
call breaks that. At 300 updates the episodes are identical, so the column is the probe scored
on its own training rows, 0.971
on `C1` and 0.949 on `C2`, against cross-validated targets of 0.733 and 0.613.

**Change.** FINDINGS 23.4 and the manuscript's development paragraph say that the 0.785 and
0.689 readings score the 450-update states of the same world seeds the 300-update probe was fit
on, so they are upper bounds and not held-out transfer, and the self-transfer at 300 updates
shows the size of the in-sample advantage. The runner carries a comment at the call site. The
spec froze this estimator, so the stored values and labels stay as recorded. A held-out transfer
reading, if one is run, is a separate post hoc readout beside this one.

**What this does not change.** The "reorganizes" label, which the frozen spec decides by the
cosine between the 300 and 450-update directions alone (0.40 on `C1`, 0.42 on `C2`). The transfer
reading enters only as an extra condition of the "one axis sharpens" label, which the cosines
already rule out. The residual growth readings and the auxiliary reading do not use the column.

*(Follow-up 2026-10-19: the held-out reading ran on 2026-10-18 as a separate post hoc readout
(`docs/specs/2026-10-17-closing-readouts-design.md`, section B6;
`artifacts/development/heldout_transfer.json`), on fresh pools at bases 2700000 and 2750000,
after reproducing the in-sample column within 1e-9. The frozen 300-update probe reads the
450-update states at 0.736 [0.680, 0.791] on `C1` and 0.632 [0.594, 0.670] on `C2`, against the
in-sample 0.785 and 0.689 (label REPORTED). No threshold reads the column; the stored values and
the "reorganizes" label stay as recorded.)*

## 2026-10-16: the control diagnostics called a_t an action the trunk received

**What was wrong.** FINDINGS 17.8 and the manuscript said that adding "the actions the GRU
received" (the manuscript: "the actions the trunk received") lowers the behavior control to
0.681. `collect_pool` records `a_t` after `agent.act` has updated the state, so at readout, where
the policy is deterministic, `a_t` is a function of `h_t`. The trunk receives only `a_{t-1}`. The
behavior trace at t is recorded after `w.step(a_t)`. `scripts/run_control_diagnostics.py` builds
the 31-column control from the trace, `a_t` and `a_{t-1}`, and `history_basis` appends the same
two actions to the basis of the 608-column MLP control. The table row already named `a_t,
a_{t-1}`; the prose did not match it.

**Change.** FINDINGS 17.8, the manuscript and the `itasorl/control_diagnostics.py` docstrings
now name `a_t` as the action the policy emitted at t and `a_{t-1}` as the one the trunk received.
The text says these controls are conservative: a basis that holds `a_t` or the post-step trace
also removes part of what the state itself drives, so their drop, and part of their held-out R²,
is not input dependence alone.

**What this does not change.** The recorded 0.681 and 0.620 and every verdict. The reading that
the state is not independent of the agent's inputs rests on the stream decoder of FINDINGS 17.8
(0.724), which uses only the observations and the previous action.

## 2026-10-16: the world section said every comparison branches from byte-identical state

**What was wrong.** The manuscript's world section said that dissipation does not by itself rule
out sensitive dependence on initial conditions, "which is why every comparison branches two
worlds from byte-identical state". Experiment A's matched pairs
(`experiment_a_l3.generate_l3_pairs`, a snapshotted shared prefix) and the matched-pair readout
branch that way. The primary pooled readout (`experiment_b2.pooled_readout`) does not. It
collects the authentic and surrogate pools from fresh resets at different world-seed bases,
800000 and 850000. The methods section and the Figure 1 caption already said so.

**Change.** The manuscript sentence names Experiment A and the matched-pair readout as the
channels that branch, and says that the primary pooled readout compares independent world
samples and that the L0 gate measures what that draw contributes.

**What this does not change.** No number or verdict. The L0 gate and the world-sample audit of
FINDINGS 17.5 were already the record's account of the world draw.

## 2026-10-16: scripted action streams share a PCG64 stream with the terrain

**What was wrong.** `SeedBundle.streams()` (`itasorl/world.py`) seeds the world stream with the
raw world integer and offsets only the weather, ecology and policy streams. Its comment says the
offsets keep the streams independent. Several collectors seed the scripted action generator
with that same integer, so the actions and the terrain come from identical streams:
`experiment_b2._collect_scripted` (the predictor arm's training episodes),
`surrogate_l3.collect_authentic_transitions` (the transitions `G` was trained on),
`experiment_b.collect_episodes` (the Experiment B pools of the L2 arc), and the shared prefixes
of `experiment_a_l3` (Experiment A's L3 pairs). `reset` first draws 16 uniforms from the world stream
for the terrain fields, starting with the terrain frequencies (`_init_static_fields` in
`itasorl/patch_of_earth.py`), and then the start heading (`_init_creatures`, two more draws
before it for each retry when the default start is wet). The scripted policy draws three per
step, so the first five scripted actions of each such episode are exact functions of that
episode's terrain parameters and the sixth draws partly on the heading (for the two-draw policy of
the Experiment A prefixes, the first eight, with the ninth drawing partly on the heading). Checked with the repo code over 55 world seeds: the first thrust divided by 0.6
equals the uniform behind the first terrain frequency to within 4.1e-8 (float32 rounding). The
engagement baseline (`_baseline_return`) uses one generator, seeded with the base world integer,
for all of its episodes, so only its first episode is coupled this way.

Separately, the ecology stream adds 13000 and then 20002 to the episode integer, so in the
survival training schedule (`100000 + 10000*seed + 16*update + i`, 16 episodes per update) the
pellet stream of seed s at update u is the terrain stream of seed s+3 about 187 updates later.
This happens only at budgets of 188 updates or more (the registered 300 and the 450-update
runs). The same random numbers then become pellet positions in one episode and terrain
frequencies in another, two different transforms. The training schedules meet in other places of
the same kind (for example the survival schedule's pellet stream and the predictor schedule's
terrain stream); this entry does not list them all.

**Which results it touches, and which it does not.** Within each collector the coupling is the
same rule for the authentic and the surrogate class, and the label never enters a seed, so it
adds no label signal by itself. It does mean that the early scripted actions are not independent
of the world they run in, for the predictor arm's training data, `G`'s training transitions, the
L2 pools and the Experiment A L3 prefixes. The pooled readouts of trained arms take their actions
from the network and are not coupled. The scripted walk of the matched-input readouts (FINDINGS
17.6) uses a disjoint policy base, 600000, and is not coupled either. The size of any effect has
not been measured.

**Change.** None to seeding. Changing it would change the stored bits of every run listed above.
Any future opt-in flag that decouples the streams must default off, so stored fingerprints and
bit-exact reproduction claims hold. The 2026-10-10 entry on the surrogate's held-out diagnostics
already names the shared base integer for world, weather, ecology and the scripted action
stream; this entry states what that sharing does.

**What this does not change.** No recorded number or verdict.

## 2026-10-16: the erasure refit probe split same-world twins across folds

**What was wrong.** The online erasure readout (FINDINGS 23.3) refits a probe on the erased
states and attaches the spec's "partial erasure" label when it reads 0.65 or above. Row i of the
erased authentic and surrogate rollouts runs on the same world seed from the same first state,
but `scripts/run_erasure_readout.py` gave every episode its own cross-validation group. Whenever
the number of common survivors was not a multiple of five, twins landed in different folds and
the probe read each test twin's label off its training twin, which pulls the AUROC toward zero.
This breaks the pairing rule stated in `itasorl/experiment_b2.py`; the surprise readout of FINDINGS
18.3 had the same defect (next entry). The recorded refit was 0.646
[0.562, 0.729], under the label by 0.004, and the untrained arm read 0.258, far below chance.

**Change.** The refit passes one group per twin pair (`1dc44e4`), and
`tests/test_readout_twin_groups.py` shows the leak and its fix on synthetic twins. The readout
was rerun on the same saved `C1` agents. Every other recorded reading of every row reproduces
the superseded artifact bit for bit; only the refit and the wall-clock `seconds` field differ.
The refit reads 0.747 [0.702, 0.791] (seeds 1
and 5, whose twins already shared a fold, unchanged at 0.810 and 0.644; the other eight rise),
and the untrained arm reads 0.576 [0.530, 0.623] over its 6 informative seeds. The label
"partial erasure" is now attached to every reading, as the spec requires. The superseded
artifact is kept under `artifacts/mechanism/erasure/superseded_split_twins_2026-10-16/`.

**What this does not change.** The frozen-probe readings, which fit on the registered pools and
only score the rollouts; erasure held in 10 of 10 seeds; and readings 1 to 3 (not used under
removal MET, used and costly NOT MET). The conclusion drawn from the refit, that removal bounds
what the decoded direction carries and not what the state carries, is stronger with the
corrected value.

## 2026-10-16: the surprise readout's AUROC split same-world twins

**What was wrong.** Readout 8(a) of the mechanism readouts (FINDINGS 18.3) scores the agent's own
next-observation error with a cross-validated probe. `scripts/run_mechanism_readouts.py` rolls the
authentic and surrogate episodes from one seed base, 1400000, so row j of each runs on the same
world seed from the same first state, and `common_rows` keeps the rows aligned. `surprise_auroc`
in `itasorl/mechanism_readouts.py` gave every episode its own group, so twins could land in
different folds, the defect of the previous entry. A probe that reads a test twin's label off its
training twin pulls the AUROC down, so the stored surprise AUROCs are biased low: 0.846 for `C1`,
0.848 for `S-immortal` and 0.912 for `T-touch` (`artifacts/goal_stakes/*/mechanism.json`), and the
untrained floors recorded beside them.

**Change.** `surprise_auroc` takes a `groups` argument and drops it with the non-finite rows, and
the runner passes one group per twin pair. `tests/test_mechanism_surprise_groups.py` shows the
leak and its fix on synthetic twins and checks that the runner passes the pair groups. The stored
values are not rescored and are read as lower bounds. The `T-touch` and `S-immortal` agents were
trained in a cloud container and are not on this machine, and `C1` is kept as recorded with them
so the three columns stay comparable.

**What this does not change.** The surprise reading needs a mean AUROC of at least 0.65
(`scripts/decide_goal_stakes.py`), and every stored mean is above it, so removing a downward bias
changes no 18.3 surprise wording. The correlation with the direction, the nudge and the
adaptation readouts fit no cross-validated probe, and the direction itself is fit on the
registered pools, whose authentic and surrogate episodes come from separate seed bases. The
replay readout of FINDINGS 22.2 draws its two pools from separate seed bases (800000 and 850000),
so its episodes have no twins and it is not affected.

*(Erratum 2026-10-19: two statements of the Change paragraph no longer hold, that the stored
values are not rescored and that the `T-touch` and `S-immortal` agents are not on this machine.
The agents are on this machine (from the run branches `run/T-touch` and `run/S-immortal`, copied
under `fullruns/branch_preserve/`), and the three runs were rescored with twin groups on
2026-10-18, readout only, into new files (`docs/specs/2026-10-17-closing-readouts-design.md`,
section B5; `artifacts/goal_stakes_grouped_surprise/{C1,T-touch,S-immortal}.json`). Grouped, the
surprise AUROC reads 0.856 [0.791, 0.922] for `C1`, 0.917 [0.897, 0.937] for `T-touch` and 0.859
[0.814, 0.904] for `S-immortal`: GROUPED ABOVE 0.65 in all three, so each stored lower bound and
its verdict agree with the grouped value. The stored 0.846, 0.912 and 0.848 are not rewritten
and stay the recorded values. `T-touch` and `S-immortal` reproduce their stored ungrouped values
exactly (worst 0.0000); `C1`'s stored file came from another stack, and its ungrouped surprise misses 0.01
on five survival seeds (worst 0.051), a cross-stack note that does not void the comparison.)*

## 2026-10-16: two seeds of the seeds 10 to 19 runs trained on the same worlds

**What.** Survival training worlds start at `100000 + 10000 * seed` and predictor training
worlds at `200000 + 9000 * seed` (`itasorl/experiment_b2.py`), each advancing by the batch size
per update. In the seeds 10 to 19 runs (B1 and B2), survival seed 19 and predictor seed 10
therefore train on the same world integers. Seeds 0 to 9 have no such overlap.

**What this does not change.** Every readout is computed within one seed, and the seed-paired
contrasts pair arms of the same seed index, so no probe is fit and scored across the shared
worlds. The two agents share training worlds, not weights or episodes of the other arm's policy.

## 2026-10-16: the historical L1 organism run had no rule registered before it ran

**What was wrong.** The L1 organism run (FINDINGS 14.7) is read under H_B2, and the record calls
its outcome "the registered encoding criterion" not met. No L1 rule was registered before the
run. `docs/PREREGISTRATION.md`, where H_B2 was frozen on 2026-06-28, covers the L2 run only (its
section 11: "this world family, and L2 only"). The 2026-07-21 H2 spec names the run only as
"A3 - Alternate-substrate-primitive positive control (DEFERRED, needs a train)", with no
estimand, gate-0 rule or decision rule. The rule ledger row `L1-PRIMARY` already says so in its
freeze note. The L1 code, the gate-0 selection of Δ = 0.023 by a fine sweep, the σ_o noise
comparator and the result were first committed together in `01e88e4` (2026-07-25 00:14), and
the historical cells record `8a71593`, a commit with no L1 drift mode (corrected-L1 spec,
`docs/specs/2026-10-15-corrected-l1-organism-rerun-design.md`). The noise-knockout rule
`NOISE-KNOCKOUT-L1` is likewise borrowed from the 2026-07-22 spec, whose only mention of L1 is
that it does not collide with it.

**Which results it touches.** The historical L1 primary reading, its A1 and A2 ablations, and
the corrected L1 rerun of FINDINGS 22.1, whose spec names the rule the historical run "was
already read under" and so inherits a rule that was applied after the fact.

**Change.** A dated entry in the deviation log of `docs/PREREGISTRATION.md` records that
`L1-PRIMARY`, `L1-CORRECTED` and `NOISE-KNOCKOUT-L1` apply rules frozen for other runs. FINDINGS
carries a dated note in the TL;DR and in 14.7 saying that H_B2 was registered for L2 and
applied unchanged to L1 after the run. The manuscript's list of runs without a pre-run freeze
names the historical L1 organism run. Later the same day, after the third review pass, the
README status table and key result, the `CITATION.cff` abstract and `docs/PAPER_OUTLINE.md`
were given the same wording; they had still said the L1 run did not meet the registered
criterion.

**What this does not change.** No number or verdict. H_B2 was applied with its thresholds
unchanged, and the 2026-07-21 spec had framed the run as a positive control expected to show
encoding. The not-met reading stands, at 0.533 and, on the corrected trainer, 0.551.

*(Follow-up 2026-10-19: two post hoc readouts on the corrected L1 agents ran on 2026-10-18
(`docs/specs/2026-10-17-closing-readouts-design.md`, sections B1 and H53). At the supra-band
grid 0.06 the state readout registers this class of seam: the untrained arm reads 0.672 [0.640,
0.704], label (a) SENSITIVE (`artifacts/expL1/l1_coarse_grid_{corrected,historical}.json`), so
the not-met reading is a negative at grid 0.023, not an insensitivity of the readout to the
class; the survival arm reads below the untrained arm there (-0.029 [-0.062, +0.005]). A random
forest and an MLP read the survival state at 0.529 and 0.534 at grid 0.023, both under 0.65,
label (b) (`artifacts/expL1/l1_nonlinear_corrected.json`). No rule, number or verdict above
changes.)*

## 2026-10-16: the development readout left out its walk-held-fixed column

**What was wrong.** The frozen spec of the development readout
(`docs/specs/2026-10-15-what-develops-design.md`) puts a matched-input column beside the
own-episode target: the 300-update agent's recorded episodes replayed through every snapshot of
the same seed, so the development is also read with the walk held fixed.
`scripts/run_development_readout.py` computes it (`replay_target`) and both artifacts store it,
but FINDINGS 23.4 and the manuscript reported only the own-episode target.

**What the column shows.** From `summary.aggregate['<budget> replay_target']` of
`artifacts/development/corrected_l3_h8_wm.json` and `corrected_l3_h8_nowm.json`, at 0, 100,
200, 300 and 450 updates: `C1` 0.636, 0.659, 0.719, 0.733, 0.742; `C2` 0.610, 0.624, 0.608,
0.613, 0.642. Seed-paired from 100 to 450 updates (t-based 90% interval over the `per_seed`
arrays, n = 10), the target rises +0.083 [+0.059, +0.107] on `C1` with the walk fixed against
+0.273 [+0.222, +0.325] on own episodes, and +0.018 [-0.008, +0.044] on `C2` against +0.157
[+0.100, +0.214]. On identical input most of the `C1` rise and nearly all of the `C2` rise is
gone.

**Change.** FINDINGS 23.4 reports the column with these contrasts, labeled descriptive as the
spec frames it, and says that the fixed walk is the 300-update agent's and that `resid_obs` and
`resid_trace` have no walk-fixed counterpart, so readings 1 and 2 are read on changing walks.

**What this does not change.** No reading or label. Readings 1 to 4 are defined on own episodes
and stay as recorded.

## 2026-10-16: the erasure population contrasts cite an amendment the erasure spec lacks

**What was wrong.** The online erasure readout's every-attempted-episode return and termination
contrasts (FINDINGS 23.3) are labeled "descriptive, no pre-stated reading (third amendment)" in
`scripts/run_erasure_readout.py` (lines 59 and 151), in the artifact the runner writes and in
FINDINGS 23.3. `docs/specs/2026-10-15-online-erasure-design.md` has two amendments, one before
any code or number and one after a two-seed setup check. It has no third. The only 2026-10-15
third amendment is the world-features spec's. The superseded first full-run artifact
(`artifacts/mechanism/erasure/superseded_common_survivor_2026-10-15/`) has no `population` or
`attrition` keys, so these contrasts were defined after the first full run had been read, as
the 2026-10-15 entry above records.

**Change.** FINDINGS 23.3 calls the population lines post hoc and descriptive, says they were
defined after the first full run, and drops the citation of a third amendment, with a dated
marker. The runner comment and the artifact label are left for the code owner to correct, or
for a dated amendment to the erasure spec that says it was written after the first full run.

**What this does not change.** No number. The contrasts were never read against a threshold,
and the common-survivor readings, which the spec froze, are unchanged.

*(Follow-up 2026-10-19: the runner's source now labels the contrasts "post hoc, defined after
the first full run", and the erasure spec carries a dated post-run note (2026-10-19) that says
the same; the committed artifact keeps the label it was written with.)*

*(Follow-up 2026-10-19: the full-rank extension of this readout, run on 2026-10-18
(`docs/specs/2026-10-17-extension-readouts-design.md`, section X10), is VOID for survival under
its gate E. Gate (I) passed: the rank-two values reproduced the stored artifact within 0.01.
Gate E failed, so no use reading was made: the in-fold refit reached the 0.55 stop by rank 16 in
1 of 10 seeds (seed 5, at rank 2; the rule needs 8); in the other 9 it stayed above 0.55 up to
the rank cap of 16. Nothing was promoted; the per-seed values live in the run output
(`fullruns/closing_2026-10/x10/corrected_l3_h8_wm.json`), not in a committed artifact. The
rank-two reading stands as recorded. The prediction (gate E passes with k between 4 and 10)
was not borne out.)*

## 2026-10-16: a better-fitting long-history control is on record

**What was wrong.** FINDINGS 17.8 and the manuscript call the 64-unit MLP joint control "the
control that fits best" (held-out R² 0.887, residual 0.620), and the manuscript's control-ladder
caption counts the two long-history ridge bases among the bases that put the residual below
the bar. Those two bases fit at a fixed penalty of 0.001 with negative held-out R², which the
same text calls failed fits. The optional sensitivity readout O2
(`docs/optional/o2_long_history_ridge.json`, `itasorl/optional_readouts.py` `long_history_ridge`)
chose the penalty inside each outer training fold by an episode-grouped inner split. The two
bases then fit held-out states at R² 0.906 and 0.909, better than the MLP, and leave 0.737
[0.690, 0.785] and 0.691 [0.650, 0.732] on the `C1` survival arm. Its fixed-penalty rows
reproduce the recorded battery (0.640 and 0.606).

**Change.** FINDINGS 17.8 carries a dated note that scopes "fits best" to the battery at its
recorded penalty and gives the O2 values. The manuscript cites O2 as a sensitivity readout,
scopes its "fits best" wording the same way, and no longer counts the failed fits as readings
under the bar. `docs/optional/README.md` says O2 is now cited.

**What this does not change.** No recorded row, threshold or verdict. O2 ran on CUDA on the
survival arm only, with no untrained or predictor floor, so it is reported beside the battery
and does not enter its table. The better-fitting linear controls remove less of the signal, not
more.

## 2026-10-16: the L1 noise knockout gave every surrogate episode of a seed one noise draw

**What was wrong.** The A2 noise knockout at L1 (FINDINGS 14.7) transfers the frozen L1 probe
onto worlds with iid observation noise at σ_o = 0.01. In `scripts/run_l1_h2_ablations.py`,
`make_noise_world` builds a new `LObsNoise` wrapper for each surrogate world and reseeds it to
Philox(970000 + agent seed). `collect_pool` creates one world per episode, and
`SurrogateWrapper.reset` never reseeds the wrapper (`itasorl/surrogate_l1_families.py`,
`itasorl/world.py`), so every surrogate episode of a seed starts the same stream and draws the
same noise at each step: iid in time within an episode, identical across episodes. The gate-0
calibration that matched σ_o to the band (`scripts/run_expA_l1_noise.py`, oracle 0.873) drew
fresh noise for each pair from one continuing generator, so the tested comparator is not the
calibrated one. The 2026-07-22 spec states its reseed rule for the L3 `GNoise` family, where the
runner resets the stream once per pool, which gives one continuing stream across a pool's
episodes; it names no rule for `LObsNoise`. The L1 code resets the stream at every surrogate
world, which is every episode, and that reset is the defect.

**Which results it touches.** The `a2_noise_knockout` block of `artifacts/expL1/h2_ablations.json`
(survival 0.525, untrained 0.509, predictor 0.516) and the `NOISE-KNOCKOUT-L1` reading. A
repeated draw lowers the variety of noise the probe meets, and its effect on the transfer reading
has not been measured.

**Change.** FINDINGS 14.7 carries a dated note. Any rerun would be a separately labeled post hoc
readout behind an opt-in flag that keys each wrapper on its episode's world seed, defaulting
off so the stored bits reproduce.

**What this does not change.** No recorded number or verdict. The knockout was already read as
vacuous, because the L1 primary probe does not meet the bar and has no structured direction to
transfer.

*(Follow-up 2026-10-19: the opt-in rerun ran on 2026-10-18 with fresh noise per episode on the
same 30 agents (`docs/specs/2026-10-17-closing-readouts-design.md`, section B7;
`artifacts/expL1/l1_h2_noise_per_episode.json`). Survival reads 0.508 [0.485, 0.531] and
untrained 0.513 [0.488, 0.538], and the call is H2_SUPPORTED, as stored (label SAME CALL): the
shared draw does not change the call. The stored 0.525, 0.509 and 0.516 stay as recorded.)*

## 2026-10-16: sweep C draws one jitter realization for every agent seed and arm

**What was wrong.** Sweep C of FINDINGS 22.3 draws the correlated jitter afresh for every
surrogate episode. `per_episode_draw` in `scripts/run_texture_fresh_probe.py` reseeds the
`G_ar` family from Philox(2550000 + 1000 + k) for the k-th world created with positive drift,
with no agent-seed term, and `collect_pool` creates every world before it drops deaths. Each
task (agent seed, arm and ρ) therefore meets the same Philox seeds, 2551000 onward, in the same
order, and the jitter's innovations depend on no action, so every agent seed and arm sees the
same per-episode jitter paths. Across ρ the standard normals behind them are also shared. The
earlier per-pool calls `fam.reseed(bs + s)` are overwritten at the next world creation. Sweeps A
and B (family `ar`) reseed once per pool at a base plus the agent seed, so their jitter differs
across seeds. The frozen spec says the draw is seeded by the pool's seed base plus the episode
index, so the code follows the spec; neither said what that implies for the interval.

**Which results it touches.** Sweep C's seed-level intervals are conditional on that one jitter
realization, as well as on the shared evaluation-world draw. Seed-to-seed variation in them is
variation over agents, not over jitter draws.

**Change.** Disclosure only. Sweep C's intervals, and those of the readout-only follow-up at the
matched gap, which reuses the same seeding, are read with this condition wherever FINDINGS 22.3
or the manuscript quotes them.

**What this does not change.** No recorded number or reading of sweep C.

*(Follow-up 2026-10-19: a second jitter realization, keyed by agent seed, was drawn on
2026-10-17 to 2026-10-18 at the two points the stored contrast reads, rho 0 and 0.99, on the same `C1` agents
(`docs/specs/2026-10-17-closing-readouts-design.md`, section B8;
`artifacts/texture/ar_sweep_c_realization2/C_rho{0.0,0.99}.json`). Its contrast M(0.99) - M(0)
reads -0.0002 [-0.027, +0.027], reading (2), within the margin unit, and CONSISTENT with the
stored matched-gap follow-up -0.017 [-0.051, +0.016]. The stored intervals stay conditional on
their one realization each.)*

## 2026-10-16: the `T-qd` artifact names the wrong reference for its sizing

**What was wrong.** The gate-0 note of `artifacts/texture/T_qd_l3_h8_wm.json`
(`gates.gate0_oracle_band.note`), written by `scripts/promote_off_design_runs.py`, says that
eps 6.0 "was matched to the learned map's one-step magnitude". It was matched to the white
jitter `G_gn`. In `artifacts/surrogate_diagnostics.json` the one-step RMS deviation of `qd` at
eps 6.0 is 0.0143 (train) and 0.0141 (held out), that of `gn` at σ 0.01 is 0.0141 and 0.0143,
and that of the learned map `gmotion_h8_s0` is 0.0361 and 0.0349, so `qd` sits at about 0.4 of
the learned map's size. The texture-comparator spec, FINDINGS 17.9 and the manuscript all name
`G_gn` as the match.

**Change.** The promoter's note names `G_gn`'s one-step RMS magnitude as the match, and the
artifact is re-promoted so that only this note changes.

**What this does not change.** No number or reading. The run is off-design, excluded, and bears
on no rule.

*(Follow-up 2026-10-19: the gate-0 oracle at the eps 6.0 setting was measured agent-free on
2026-10-17 (`docs/specs/2026-10-17-closing-readouts-design.md`, section H31;
`artifacts/agent_free/qd_eps6_oracle.json`): 0.671, outside the [0.85, 0.95] band, label (b)
OUTSIDE; the stored eps 4 and 8 readings, 0.656 and 0.697, reproduce bit for bit. The run stays
off-design, and the drag's gate 0 stays failed.)*

## 2026-10-16: the pooled population probe groups folds by pooled row

**What was wrong.** `cg_probe` (`itasorl/experiment_b2.py`) groups its cross-validation folds by
pooled row. The pooled population panels pool every individual's common-garden pairs, which
run on the same world seeds for each individual: Experiment C's panels and the C1 population
readout of FINDINGS 17.11 (`scripts/validate_population_readout.py`, base 930000). One world
seed's pairs from two individuals can then land in different folds. FINDINGS 13.F discloses
this for Experiment C, where related individuals can lift the pooled AUROC, and
`itasorl/experiment_c.py` carries it as a comment. The C1 population readout did not say so, and
the manuscript's list of places where the twin grouping rule was broken did not include it.

**Change.** FINDINGS 17.11 carries a dated note, and the manuscript's list names the pooled
population probe, which was not changed in code. A rerun should group pooled rows by world-seed
index.

**What this does not change.** No number. The 0.609 and 0.536 of 17.11 are not regrouped. Those
agents were trained independently, so they cannot test the relatedness leak, and the Experiment
C tails were not saved, so its panels cannot be rescored.

*(Follow-up 2026-10-19: an opt-in world-seed grouping was added with the default unchanged, and
the C1 population readout was rerun on 2026-10-19 with both groupings on the same tails
(`docs/specs/2026-10-17-closing-readouts-design.md`, section B4;
`artifacts/population_readout/corrected_l3_h8_wm_world_grouped.json`). Regrouped by world seed,
the C1 readings are 0.617 (survival) and 0.537 (untrained), against 0.609 and 0.536 grouped by
row: label DIFFERS, where the spec expected EQUAL. Claim row 49 holds on the world-grouped
values (0.617 above 0.537 and below 0.649). The label asks for the number of dead pairs, which
the output did not record; it records only the world-group counts (110 survival, 108 untrained).
The stored 0.609 and 0.536 stay as recorded, and the Experiment C panels still cannot be
rescored.)*

## 2026-10-16: sweep C's matched gap was measured with the per-episode draw off

**What was wrong.** Sweep C (FINDINGS 22.3) sets sigma at each rho so the 24-step open-loop
velocity gap matches the shared point, and its runner measures each setting's gap before the
readouts. `profile` in `scripts/run_ar_sweep_c.py` wraps that measurement in `per_episode_draw`,
which patches only the world constructor the readouts use; the measurement builds its worlds
and calls the family directly, so the AR process ran on across the calibration episodes. The
measured gap was sweep A's recorded gap scaled by sigma, and the record's "all six settings
within 0.0001" held by construction.

**Which results it touches.** For `G_ar` the gap from a stationary start is exact in closed
form, sigma * sqrt(2 w'Cw) with w_t = 0.925^(23 - t) and C_ij = rho^|i - j|. The shared point
(rho 0.9 at sweep A's sigma) reads 0.1175; the six recorded settings read 0.1162 to 0.1267, and
rho 0.95 and 0.99 sit 1.6 and 7.8 percent above the shared point, the second outside the spec's
3 percent band. The contrast M(0.99) - M(0) therefore compared a rho 0.99 point with more 24-step
deviation than the rho 0 point.

**Change.** `scripts/run_ar_sweep_c_followup.py` (`6846435`) reruns rho 0.95 and 0.99, readout
only on the same saved C1 agents, at the sigma that matches 0.1175 exactly (0.00869 and 0.00763).
Only those two points were rerun; the stored rho 0 point, the other end of the contrast, sits at
0.1162, 1.1 percent below 0.1175 and inside the 3 percent band.
Margins +0.060 [+0.015, +0.105] and +0.049 [+0.016, +0.081]; with the stored rho 0 point the
contrast is -0.017 [-0.051, +0.016], reading 3 (neither), its lower bound 0.001 outside the
margin unit. The follow-up is in `artifacts/texture/ar_sweep_c_followup/` and the manifest as
AR-SWEEP-C-FOLLOWUP, beside the frozen result. The runner's `profile` carries a comment; its
behavior is unchanged so the recorded run stays reproducible. FINDINGS 22.3 and the manuscript
withdraw "coherence moved the margin by less than 0.05" and keep "coherence did not raise the
margin", which both readings support.

**What this does not change.** The frozen sweep C contrast and its reading 2, which stay as
recorded, and the finding that no jitter setting reaches the bar.

## 2026-10-16: the review archive carried branch names, and its anonymity was overstated

**What was wrong.** The review archive rebuilt after the second review pass replaced commit IDs
and recorded digests with keyed pseudonyms, but its text members still carried four public
branch names with random suffixes, and a pull-request search on any one of them returned the
public repository alone. The builder's leak scan had no rule for branch names, so it passed.
Separately, `docs/REPRODUCE.md` and the submission check described the pseudonyms as if they
left the archive with no public search key. Three remain by design. `SHA256SUMS` lists the
real digest of every shipped file in plain text, and 53 of those digests are recorded in
tracked files of the public repository, so reading it is enough and no hashing is needed. The
16-character configuration fingerprints stay. Apart from the package name, the code, its
class and function names and its docstrings match the public repository verbatim, and the
documents do too apart from the replaced identifiers, so one code search on a class name finds
it.

**Which results it touches.** None. This is submission packaging.

**Change.** The builder replaces branch names with neutral labels, and the build fails if any
listed branch or pull-request head name survives (`docs/REPRODUCE.md`). `docs/REPRODUCE.md` ("Anonymized supplement") and
`docs/SUBMISSION_CHECK_2026-10-16.md` (sections 2a and 2b) now say plainly that the
replacements remove commit IDs, digests and branch names as search keys and that the archive is
not anonymous to a search while the repository and its site are public. The archive built at
the second review pass is not to be submitted.

**What this does not change.** No number, artifact or verdict. Whether the repository and the
site stay public during review, or the submission relies on the venue's policy on prior public
posting, is the owner's decision before upload, and this entry does not make it.

## 2026-10-16: the selection appendix read no specialist advantage from a hindsight benchmark

**What was wrong.** The manuscript's selection appendix said that rearing in a world gives the
C1 survival policies no specialist advantage. Its only basis was the own-world assignment minus
the better single policy of each seed (-0.056), a benchmark chosen with hindsight that is
negative whenever one policy returns more in both worlds, as on 9 of 10 seeds.

**Change.** Recomputed from the C1 cell records' cross-evaluation returns: on the means each
policy earns 0.042 more in its own world, and paired by seed the own-world assignment beats the
average of the two fixed policies by +0.021 [-0.0004, +0.043]. The manuscript now says that a
specialist advantage is neither shown nor ruled out.

**What this does not change.** No stored number.

## 2026-10-16: the matched-world runs' drift-zero pooled reading was not reported

**What was wrong.** The frozen specification of the matched-world confirmation reports the
pooled L0 reading on the registered pair beside its result, without deciding on it. The record
reported only the drift-0.45 pooled values (0.758 and 0.835).

**Change.** Recomputed from the B1 and B2 cell records (`fullruns/balanced_confirmation_wm` and
`_touch`, not committed artifacts): at drift zero on the registered pair the survival agents of
seeds 10 to 19 read 0.561 [0.527, 0.595] (TOST p 0.72) and the pellet-objective agents 0.536
[0.514, 0.558], against 0.498 for the predictor and 0.503 for the untrained arm. The separation of
that pair by the survival arm repeats on a second seed set. FINDINGS 21 and the manuscript report
it as descriptive.

**What this does not change.** The reading decides neither run, and no verdict changes.

## 2026-10-17: the steepness-sweep paragraph of FINDINGS 13 said no cell was steeper than the pilot cell

**What was wrong.** The follow-on de-risk paragraph of FINDINGS 13 says that once the gate-1 gap
is normalized for reach distance "none was steeper than the pilot cell", and it points at the
gitignored `fullruns/expC_gate1_sweep/steepness.json`. The committed grid,
`artifacts/expC/gate1_steepness_sweep.json` (the bias-guarded rerun of 2026-07-20, seeds
7100-7105, disjoint from the certification seeds 7000-7009; `4909f1b`), ranks the cells two
ways. By raw gap the best cell is reach 0.50 / horizon 60 at 0.00445, 1.9 times the 0.0023 pilot
baseline. By reach-normalized gap the best cell is reach 0.15 / horizon 80 at 0.0241, against
0.0132 for the pilot cell (reach 0.25 / horizon 30), so one cell is steeper than the pilot cell
by that measure. The 2026-07-17 sweep on seeds 7000-7005 that the paragraph was written from
read the same way, and `docs/PREREGISTRATION_C.md` section 14 item 4 singles out the same reach
0.15 / horizon 80 cell. That sweep was not promoted, and its output file was overwritten by the
rerun at the same path. The sentence misread the sweep; it was not superseded.

**Change.** FINDINGS 13 keeps the paragraph as written and gains a dated note directly after it
(2026-10-17) that gives these values from the committed artifact, the only grid on record.

**What this does not change.** No cell reaches the 0.005 gate-1 margin, which is the reading the
go/no-go in `docs/PREREGISTRATION_C.md` section 14 item 4 rests on, and the routing to
controller expressiveness stands on that margin, not on the normalized ranking. The manuscript
sentence (at most 0.00445, 1.9 times the baseline, no cell reached 0.005) and the no-launch
disposition do not move. No verdict changes.

## 2026-10-19: follow-up lines for closing-pass readouts that extend no earlier entry

**What.** The closing pass ran post hoc readouts frozen in
`docs/specs/2026-10-17-closing-readouts-design.md`,
`docs/specs/2026-10-17-walk-speed-vs-goal-readout-design.md` and
`docs/specs/2026-10-17-extension-readouts-design.md` and announced in
`docs/PREREGISTRATION_L3.md` section 12 before they ran, all at `75ce098`. A readout that
extends an entry above carries its follow-up line there. Each line below extends no entry; it
names the artifact and the label, and section 12 carries the result entry.

- B2, walk speed against goal direction (run 2026-10-18;
  `artifacts/walk_speed/corrected_l3_h8_wm.json`): NOT SEPARATED. Reading 1 is UNRESOLVED (1 of
  10 seeds speed-matched; the rule needs 7), and reading 2 is CLOSED LOOP ADDS (own walk minus
  replay +0.073 [+0.051, +0.095]).
- G03, the balanced sensory-echo control on a second draw (run 2026-10-19;
  `artifacts/balanced_confirmation/sensory_echo_b{1,2}_fresh.json`): the floor is explained on
  the second draw (untrained `resid_obs` 0.577, 0.077 from 0.5 and inside the 0.1 floor by
  0.023; first draw 0.596, 0.096 and 0.004; one floor read twice, since B1 and B2 share their
  untrained and predictor agents), and the reading is sensory-independent under matched worlds
  for B1 (0.724) and B2 (0.761). The 910000 result is unchanged.
- X03, head overlap (run 2026-10-19; `artifacts/head_overlap/corrected_l3_h8_wm.json`):
  undetermined. The prediction, policy-blind, was not borne out.
- X07, pellet-density dose (run 2026-10-19; `artifacts/pellet_dose/corrected_l3_h8_wm.json`):
  locomotion-carried (survival 0.726 [0.655, 0.796] at 2 pellets). The predicted fall at low
  density was not borne out.
- X08, budget-curve dynamics (run 2026-10-17;
  `artifacts/budget_curve_dynamics/corrected_runs.json`): C1 tracks budget and C2 both; the C2
  100-to-200 segment label is not met, as predicted.
- X09, Bayes factors with an interval null (run 2026-10-17;
  `artifacts/bayes_factors/interval_null.json`): an annotation that relabels nothing, with its
  labels only in the manuscript's appendix table; the C1 L0 registered pair reads BF01 = 0.133
  (annotation, 0.707 scale), and one of the five predictions was borne out.

- H34, budget comparison at 450 updates (FINDINGS 17.10, the manuscript's budget paragraph):
  "forage about equally well" is withdrawn as an equivalence reading. The paired return
  difference -0.054 [-0.198, +0.090] is unchanged; its interval contains the 0.15 engagement
  margin and does not meet the ±0.05 equivalence band, so the arms are not shown equal. Wording
  only.

**What this does not change.** No stored value, label or verdict. X01 stopped at its integrity
gate and reported nothing, so it has no line here.

## 2026-10-19: the manuscript said the rule was registered before any survival-trained run

**What was wrong.** The manuscript's abstract, its section 4.2 and its bar-and-margin
sensitivity paragraph said the 0.65 bar and the 0.05 margins were fixed before any
survival-trained run. A drift-0.45 survival de-risk run of B-v2 and an earlier pilot, whose
outputs were never committed, preceded the first registration, `51e086e` (2026-06-28, 18:48
UTC); the trainer settings were tuned on the de-risk, the code of that commit sets the harsh
metabolism and food density and cites the pilot's outcome, and the bar and margins first appear
in that commit (`git grep 0.65 51e086e^` finds nothing).
The engagement margin (`ENGAGE_MARGIN = 0.15`, `LIFE_TOL = 2.0`) was set at `5c3aca1` (22:13
UTC) after the first confirmatory B-v2 result, `07ef113` (21:18 UTC), had been read and the GAE
fix `679fee6` had landed, and before the rerun `83e9297` (2026-06-29, 00:44 UTC); the
manuscript's appendix said it was fixed in the pilot and never swept. The L3 registration,
`0217263` (2026-07-10, 10:23 UTC), describes a recurrent surrogate trained with held-out early
stopping, and the MLP `train_g_motion` that every L3 result used was first committed in
`489a772` (14:00 UTC the same day), before the first L3 organism run. Gate 0 was frozen at a
detector handicap of sigma_d = 0.05 in `9e64ffa` (16:32 UTC); the first L3 organism run was
recorded in `ced27f4` (21:23 UTC), traced to a world-parameter defect and retracted, and sigma_d
was re-frozen at 0.02 in `6150a15` (22:42 UTC). The manuscript said sigma_d was frozen before the
next organism run and did not say that an organism result had been read first. It also called
the historical sensory-echo control a pre-registered rule in two places, although its rule was
first committed with its result (2026-10-10 entry above).

**Change.** The manuscript revision of 2026-10-19 carries a dated registration timeline in its
section 4.3, built from these commits, and says the bar and margins were registered after a
survival tuning run and before the first confirmatory run. It says the engagement margin was
set after the first confirmatory run had been read, that the executed MLP was first committed
after the L3 registration, and that sigma_d was re-frozen after the retracted organism run. The
historical sensory control's rule is called a rule committed with its result. FINDINGS 11 note
2, `docs/PREREGISTRATION.md` section 12 and `docs/PREREGISTRATION_L3.md` section 12 carry dated
notes of 2026-10-19.

**What this does not change.** No number, rule, gate or verdict. The registered texts stay as
written.

## 2026-10-19: in-sample and biased estimates were called bounds

**What was wrong.** The development readout's frozen-300 transfer at 450 updates (0.785 on
`C1`, 0.689 on `C2`) was called an upper bound, and the ungrouped surprise AUROCs (0.846 for
`C1`, 0.912 for `T-touch`, 0.848 for `S-immortal`) lower bounds. An in-sample estimate is
optimistic, and an estimator that splits same-world twins across folds is biased low in
expectation; neither makes a stored value a numerical bound. On `C1` the grouped minus
ungrouped surprise contrast, both computed on the local stack (the stored 0.846 came from another
stack), is -0.0002 [-0.008, +0.007], slightly negative with an interval
that includes zero. The erasure readout was said to bound what the decoded direction carries,
which a removal and a refit do not establish.

**Change.** The manuscript revision of 2026-10-19 calls the transfer values in-sample,
optimistic estimates and the surprise values values from a biased estimator, and lets the
held-out transfer (FINDINGS 23.4, 2026-10-19 note) and the twin-grouped rescores (FINDINGS
18.3, 2026-10-19 note) carry the inference. The erasure sentence says the removal speaks to
what the direction carries, not to what the state carries. FINDINGS 18.3, 23.3 and 23.4 gain
dated notes, `docs/PREREGISTRATION_L3.md` section 12 (B5 entries) gains one, and the results
manifest's surprise note drops the word. The 2026-10-16 entries
above keep their wording.

**What this does not change.** No stored value or label. The partial-erasure label stands.

## 2026-10-19: record wording stronger than the manuscript after the full-review pass

**What was wrong.** After the manuscript revision of 2026-10-19 (full-review pass), several
record passages stated a result more strongly than the manuscript does. FINDINGS 21 called the
`B1` and `B2` rows "sensory-independent under matched worlds" on both evaluation draws without
saying that the sensory-echo procedure was frozen after the raw outcomes and the untrained
diagnostics, and that the second draw was read under that already fixed procedure. FINDINGS
18.7 said survival is not required and that the walk carries the signal, with replay giving it
to untrained trunks, and 18.4 said an untrained trunk reads most of the signal. FINDINGS 3.1, 3.3
and 3.4 said the L2 prediction-only target sits at chance and that the agent did not encode
world identity, with no equivalence test at three seeds. FINDINGS 23.3 recorded **not used under
removal** without stating the limits of the use test in the same place. The README read the
hand-authored drag with no word on its untrained floor in the hypotheses and the overview, and
gave 0.733 and the auxiliary contrast without the evaluation-draw and tested-regime qualifiers.
`docs/PREREGISTRATION_L3.md` section 12 headed the 2026-10-06 surrogate entry as a description
correction, although the registered configuration is not the one executed. The 2026-10-19
follow-up-lines entry above quotes the sensory-control label without the conditional reading.

**Change.** FINDINGS 3.1, 3.3, 3.4, 18.4, 18.7 (items 1 and 4), 21 (twice) and 23.3 gain dated
notes of 2026-10-19: the L2 target reads near 0.5, below the criterion; the walk and the trained
trunk both carry the L3 reading; `B2`'s 0.780 is a conditional result under the later procedure,
neither a confirmation under the original complete rule nor independence from nonlinear or
longer-history sensory information, and the second draw is not a prospective test of it; the
erasure and nudge use tests are limited by partial erasure, by scoring the behavior gap on common
survivors only (470 of 640 attempted episodes kept in the erasure readout, whose
every-attempted-episode return contrast was added after its first run) and by a fixed
five-measure behavior endpoint, so absence of use is not shown. `docs/PREREGISTRATION_L3.md` section 12 gains two
entries: one reclassifies the 2026-10-06 surrogate entry as a deviation, and one reads the
sensory-control label and the frozen stakes wording as conditional results. The README states
the drag as a margin over an untrained floor (0.612) outside tolerance, holds the jitter readings
to the tested agents, worlds and jitter draws, places 0.733 on the one evaluation-world draw all
seeds share, and reads the auxiliary contrast as a comparison of the tested regimes with the
auxiliary rule unmet and general necessity not established. `docs/LEARNING.md` gives the 0.780
reading with the same conditional status, and the site template qualifies its jitter sentence
as the README does. A second pass of the same date (full-review audit) carries the remaining
qualifiers to the other live copies. The README opening places 73% and 61% on the one pair of
evaluation worlds all seeds share, and its results table adds the draw and the tested-regime
reading of the auxiliary lead. The site template adds the shared draw to the hero caption, the
Experiment B card and the film caveat, reads the auxiliary lead on that card as a comparison of
the tested regimes that does not establish general necessity, and retitles its boundary-check
entry "the predictor leads at 300 updates". `docs/LEARNING.md` places the Act 10 readings on the
shared pair, reads the 300-update auxiliary result as a comparison of the setups tested with its
registered rule unmet, holds the jitter sentence to the tested worlds and jitter draws, scopes
the erasure sentence to the 470 of 640 episodes that survive every rollout, and narrows the
survival caveat to the controlled reading of the pellet creatures with death off.
`docs/PAPER_OUTLINE.md` gives the shared draw in contribution (c) and section 4.3 (with the
0.618 lower bound under the between-draw spread), the auxiliary qualifiers in section 4.4 and a
Known-gaps entry. FINDINGS gains notes of the same date on the TL;DR headline (shared draw),
on two L2 "at chance" copies (sections 5 and 9) and on 17.12 item 6 (the drag's floor and the
jitter scope), and 18.7 item 2 is restored as its own list item.

**What this does not change.** No number, label, rule or verdict. The frozen labels
("sensory-independent under matched worlds", **not used under removal**, the frozen stakes
wording) and the registered texts stay as written.
