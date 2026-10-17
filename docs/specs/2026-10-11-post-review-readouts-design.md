# Post-review readouts on saved agents: frozen design (2026-10-11)

Frozen on 2026-10-11, before any of these readouts was run on any agent. This is the
separate spec that `docs/specs/2026-10-11-corrected-generality-reruns-design.md` ("What this
does not cover") promised. It names five readout-only analyses, each answering one finding of
the post-submission review (R210, R35, R206, R33, R31; the runner calls them `r210`, `r35`,
`r206`, `r33`, `r31`), and fixes for each its inputs, procedure, evaluation worlds, statistics
and the reading that will be written.

**None of these readouts changes a registered verdict.** No bar, margin, gate or tolerance
moves. Every registered rule is evaluated where its preregistration says, on the registered
estimand, and a readout below is reported beside it as a diagnostic. Where a reading below
would contradict a sentence of FINDINGS, that sentence is revised to the measured reading in a
dated edit; the verdict line it supports does not change.

Code: `itasorl/post_review.py` (pure pieces), `scripts/run_post_review_readouts.py` (runner),
`tests/test_post_review_readouts.py`.

## Targets

The first target is the C1 rerun: branch `run/C1-rerun`, `artifacts/goal_stakes/C1-rerun`,
20 cells (drifts 0 and 0.45, seeds 0 to 9), 60 saved agents, registered recipe, corrected
trainer. Its recorded drift-0.45 survival targets equal C1's per-seed values (0.632, 0.805,
0.691, 0.744, 0.662, 0.522, 0.885, 0.783, 0.868, 0.738). The same readouts then run, with no
change, on each later run with the same layout: `C2-rerun` (no next-observation decoder, budget
snapshots at 100, 200 and 450 updates), `H7-corrected` (`--l3-hidden 7`), `H10GS1-corrected`
(`--l3-hidden 10 --l3-seed 1`) and `H8GS2-corrected` (`--l3-seed 2`).

What is read from a run:

- the 60 agents `agents/agent_d{0.00,0.45}_s{0..9}_{survival,predictor,untrained}.pt`. The
  budget snapshots (`*_survival_u100.pt` and the like) are not read;
- the run's knobs from its cells (`knobs`: L3 capacity, L3 seed, scarcity overrides), its
  objective and mortality, as `scripts/regenerate_state_dumps.py` and
  `scripts/run_mechanism_readouts.py` read them. The runner refuses a run whose cells disagree
  on their knobs, or a drift mode other than `l3`, or a surrogate family other than G_motion;
- the recorded pooled target and shuffled control of every (drift, seed, arm) cell, for the
  r210 bit check.

Each worker process refits the run's learned velocity law G_motion from those knobs on one
torch thread (`torch.set_num_threads(1)`), exactly as the run workers fit it, and nothing else
the run trained is retrained. All probes use the explicit-v1 fold partition; the runner
refuses to start unless `ITASORL_FOLDS=explicit`.

## Evaluation worlds and seed bases

A pool of n episodes at base b uses world seeds b + i, weather seeds b + 7000 + i and ecology
seeds b + 13000 + i for i < n (`experiment_b2._seeds`), and `SeedBundle.streams` offsets the
last two by 10001 and 20002. Two bases are called disjoint here when none of the three integer
seed ranges of one meets any of the other's, which is stricter than comparing world seeds
alone.

| base | used by | what it is |
|---|---|---|
| 800000 / 850000 | r210; r33 continuity entry (800000, both members of each pair) | the registered pair, by design |
| 2100000 | r35, r31 | fresh authentic pool, drift 0 |
| 2150000 | r35, r206, r31 | fresh surrogate pool, drift 0.45 |
| 2200000 | r33 primary | fresh balanced pairs, both members |
| 600000 + i | r206, r31 (scripted walk) | the scripted action stream of `itasorl.eval_protocols` (an action RNG, never a world seed) |

Every world-seed base already in use sits below 2000000. The list, with its sources, is
`itasorl.post_review.BASES_IN_USE`: training (100000 to 285800), the matched pairs 700000, the
warmup 555000, the registered pair, the transfer, cross-recipe, engagement, survival-return,
common-garden, gn, persistence and H2 ladder bases 860000 to 1030000, the eight L0 audit pairs
1000000 to 1750000, the scarcity calibration 1300000, the Experiment C panels, the mechanism
readouts' `WORLD_BASE` 1400000 (and 1450000, named in the results manifest) and the qd
comparator 1900000 / 1950000. The three new bases are disjoint from all of these and from
each other with the generous spans listed there; `tests/test_post_review_readouts.py` pins it
and checks that the list carries every base constant the runners define. The runner also
checks it at start and refuses to run on a collision.

r35, r206 and r31 share the fresh pair on purpose: the survival agent's walk that r35 replays,
the survival walk whose law deviation r206 integrates and the survival stream that r31 decodes
are then the same episodes.

## Integrity

The agents' identity with the recorded run rests on the r210 bit check: every regenerated
pooled target must equal the recorded one as a float. That needs the stack the run used (for
these runs the torch 2.14.1 CPU wheel, numpy 2.4.6 and scikit-learn 1.9.1); another stack moves
targets in the last digits, as `scripts/README.md` warns for `regenerate_state_dumps.py`, and
the check then fails. A (drift, seed) whose three arms do not
all reproduce is excluded from the r210 summaries at that drift and listed. The other readouts
do not use any recorded number, so they are not filtered by it, but the output's `integrity`
block lists every unreproduced seed beside them, and an output in which r210 was not run says
the identity is unchecked.

For the C1 rerun, a second check is available. If its agents are C1's, the r33 balanced targets
of the drift-0.45 agents on the registered base equal the `paired` entries of
`artifacts/l0_audit/corrected_l3_h8_wm.json` to the bit (`--balanced-reference`). The result is
reported, with the count of bit-equal cells and the largest deviation.

## Common statistics

The unit of replication is the agent seed (n up to 10). Every summary is the mean over seeds
with the Student-t 90% interval (`itasorl.stats.t_ci90`); every contrast is seed-paired, the
difference taken per seed and then summarized the same way. Intervals are conditional on the
one fingerprint and the evaluation worlds named above. No multiplicity correction is applied:
nothing below is a test of a registered hypothesis. A label such as ABOVE means the t-based
90% interval of the quantity lies entirely above the named value; INCLUDES means it contains
it; BELOW means it lies entirely below.

## R210: probe classes (readout `r210`)

**Finding.** The registered probe is linear (standard scaler and logistic regression on
[mean h, final h]). A nonlinear probe could read more from the same states, which would make the
linear reading a lower bound, or it could read the same, which would say the linear probe is not
what limits the reading.

**Inputs.** Every agent of the run, both drifts, three arms.

**Procedure.**

1. Regenerate the registered pools exactly as the run built them:
   `experiment_b2.pooled_readout(agent, norm, P, drift, n_eps=110, steps=24, ray_steps=5,
   device="cpu", seed=seed)`, authentic pool from 800000 + i at drift 0, surrogate pool from
   850000 + i at the cell's drift, early deaths dropped.
2. Bit check as under Integrity, for the target and, reported only, for the shuffled control.
   An unreproduced cell is not scored with the other classes.
3. Features `experiment_b.episode_features` (mean h and final h, 192 columns), label 0 authentic
   and 1 surrogate, one CV group per episode, the explicit-v1 partition the registered target
   used.
4. Probe classes, each scored as the mean out-of-fold AUROC over the five folds, every model fit
   on the training folds only:
   - linear: the registered value, read from the regenerated pooled readout;
   - random forest: `RandomForestClassifier(n_estimators=200, random_state=agent seed)` through
     `behavior_audit.behavior_only_auroc(nonlinear=True)`, the forest path the behavior audit
     already uses;
   - MLP: a standard scaler and `MLPClassifier` with one hidden layer of 64 ReLU units, L2
     penalty 1e-3, Adam at learning rate 1e-3, at most 500 iterations with scikit-learn's default
     stopping tolerance and no validation split, `random_state` = agent seed. Iterations per fold
     and the number of folds that stopped at the cap are recorded.
5. Shuffled-label control for every class: the same eight label permutations pooled_readout
   draws for its own shuffled control (`numpy.random.default_rng(seed)`, eight successive
   permutations, nothing drawn from that generator before them), each scored with the class
   and the partition above, averaged. A permutation check recomputes the linear shuffled
   control from these permutations; it must equal pooled_readout's value to the bit, and a
   cell where it does not is reported with the flag set. Selectivity is the class target minus
   its shuffled control.

**Seed base.** The registered pair, by design. No fresh worlds.

**Statistics.** Per drift, arm and class: target, shuffled control and selectivity. Per arm:
each nonlinear class minus linear, seed-paired. Per class: survival minus predictor and survival
minus untrained, seed-paired.

**Reading, pre-stated.**

- Drift 0.45, per arm and class. Class minus linear ABOVE: the linear probe understates what
  that arm's state carries, by the reported amount, and the registered number is reported as a
  lower bound in the text that discusses probe choice. INCLUDES: the nonlinear class reads no
  more than the linear probe. BELOW: the linear probe reads more, which is what a nonlinear model
  fit on 176 episodes of 192 features would be expected to do when the signal is linear.
- Margins. Whether survival minus predictor and survival minus untrained lie above 0.05 under each
  nonlinear class, as they do under the linear probe for C1 (+0.144 [+0.072, +0.217] and +0.210
  [+0.135, +0.286]). A margin that holds under the linear probe and not under a nonlinear class
  is reported as specific to the probe class.
- Drift 0. Both pools are authentic and differ only in the world sample. A class that reads the
  drift-0 pools higher than the linear probe (C1: 0.559) reads more of the world-sample difference
  that leaves the L0 gate open (FINDINGS 17.5). Each class's drift-0.45 reading is then read
  against that class's own drift-0 reading, never against the linear one.

**Expected compute.** About 4 CPU minutes per seed (six pooled readouts at about 15 s each, then
nine forest and nine MLP cross-validations per cell); about 40 CPU minutes for ten seeds.

## R35: normalizer-matched replay (readout `r35`)

**Finding.** In the replay protocol (`itasorl.eval_protocols`, FINDINGS 17.6) every trunk reads
the survival agent's recorded streams, normalized with that trunk's own frozen normalizer. The
untrained trunk's normalizer was fitted on 8 episodes of 40 steps of its own random policy, the
trained arms' during training. The replay margin of the trained arms over the untrained one
(survival +0.097 [+0.066, +0.128] for C1) therefore belongs to trunk and normalizer together, and
17.6 records that the split is not measured.

**Inputs.** The three agents of the drift-0.45 cell of every seed.

**Procedure.**

1. Record the survival agent's own evaluation episodes with
   `eval_protocols.record_trajectories` (deterministic policy, 110 episodes of 24 steps, early
   deaths dropped): the authentic pool at drift 0 from 2100000 + i and the surrogate pool at
   drift 0.45 from 2150000 + i. These are raw observations and the env actions taken.
2. Replay them open loop through each trunk with `eval_protocols.replay_states` (the previous
   action at step t is the recorded action of step t - 1, zeros at t = 0), each pool in its own
   batch as `scripts/run_policy_controlled_readouts.py` does, under four normalizers:
   - (a) `own`: the trunk's own frozen normalizer, the recorded convention;
   - (b) `survival_norm`: the survival agent's normalizer;
   - (c) `refit`: one normalizer refit on the replayed streams, the per-channel mean and variance
     over every (episode, step) row of both pools' raw observations, applied with RunningNorm's
     own formula (x - mean) / (sqrt(var) + 1e-6);
   - (d) `refit_infold`: the same refit computed inside each fold from the training episodes'
     observations only, the states of every episode recomputed under it, and the probe fit on the
     training episodes.
3. Probe: the standard pooled probe on the replayed states
   (`eval_protocols.readout_from_states`, explicit-v1); for (d) the same estimator inside the
   fold loop.

**Why the pooled refit (c) cannot leak, and why (d) is computed anyway.** The refit reads no
label. It is one affine map, the same for every episode in both classes, and it is invariant to
any permutation of the labels, so under exchangeable labels the cross-validated AUROC has the
same null distribution as with any fixed normalizer, and the map cannot put a class difference
into the states that the streams do not carry. It does use the test episodes' observations to
set the scale, a label-free transductive step that the registered readouts do not take. (d)
removes that step at the cost of five trunk passes per arm, which is cheap, so both are
reported. The pre-stated expectation is that (c) and (d) agree within fold noise; if their
labels differ, (d) governs the reading.

**Reference.** The survival trunk on its own normalizer. Under (a) and (b) it is the same number.

**Seed bases.** 2100000 (authentic) and 2150000 (surrogate).

**Statistics.** Per arm and normalizer: the replay target. The survival-minus-arm margin under
each normalizer, seed-paired: (a) survival on its own normalizer minus the arm on its own; (b)
the survival reference minus the arm on the survival normalizer; (c) survival on the refit minus
the arm on the refit; (d) the same in-fold. Under (b), (c) and (d) every trunk reads identical
normalized input. For (b) to (d), the share of the (a) margin that remains (ratio of the seed
means).

**Reading, pre-stated, for the untrained arm.**

The refit entry below is (c), unless (c) and (d) carry different labels, in which case it
is (d).

- TRUNK: under both (b) and the refit the margin is ABOVE zero and keeps at least half of the (a)
  margin. On identical normalized input the trained trunk reads more than the untrained one;
  the replay margin comes from training of the trunk.
- NORMALIZER: under both (b) and the refit the margin INCLUDES zero or is BELOW it. Once its input is
  scaled the same way, the untrained trunk reads the replayed streams as well as the trained one;
  the replay margin sits in the normalizer.
- SPLIT: anything else, reported with both margins and shares.

The 17.6 sentence that leaves the split unmeasured is replaced by the reading obtained, with the
numbers. The predictor arm is reported with the same columns and labeled by the same rule, but
descriptively only: its replay margin under (a) is near zero for C1 (+0.012 [-0.002, +0.026]),
so a share of it carries no information.

**Expected compute.** About 0.5 CPU minutes per seed (two recorded pools, about 50 batched
replays, nine pooled probes and three in-fold refit probes); about 5 CPU minutes for ten seeds.

## R206: policy deviation along visited states (readout `r206`)

**Finding.** 17.6 says the survival policy's streams carry the world difference to the trunks
better than the scripted walk's do, and a sentence attributing that to the survival policy
visiting states where the two velocity laws differ more was withdrawn on 2026-10-10 because the
deviation along each policy's trajectories had not been recorded. This readout records it.

**Inputs.** The three agents of the drift-0.45 cell of every seed, and the run's G (refit from
the knobs as above).

**Procedure.** Four walks in surrogate worlds (drift 0.45) from 2150000 + i, 110 episodes of
exactly 24 steps (the pooled length), stepped as `collect_pool` steps them, early deaths dropped:

| walk | policy | protocols it stands for |
|---|---|---|
| `survival_own` | the survival agent's deterministic policy | survival under `own`; every arm under `replay`, which feeds this walk to every trunk |
| `predictor_own` | the predictor's actor head | predictor under `own` |
| `untrained_own` | the untrained actor head | untrained under `own` |
| `scripted` | `experiment_b.scripted_policy`, action stream `default_rng(600000 + i)` | every arm under `scripted` |

At every step the world's motion log (`PatchOfEarthV0._log_motion`, a passive record that changes
nothing in the world) holds the state before the transition and the law's output: velocity v,
acceleration a (thrust and slope gravity), drag, and v' = G(v, a), the learned law the surrogate
world integrates. The authentic law at the same visited state is f(v, a) = (1 - drag dt) v + a dt
(`PatchOfEarthV0._integrate_motion`, the branch taken without a learned law). The deviation at
the step is the Euclidean norm of G(v, a) - f(v, a), and an episode's deviation is its sum over
the 24 steps.

**Seed base.** 2150000, the same surrogate worlds as r35's surrogate pool and r31's.

**Statistics.** Per walk and seed: episodes kept; the mean summed deviation and its 10th, 50th
and 90th percentiles over episodes; the per-step mean and its 50th, 90th and 99th percentiles;
mean |v| and mean |a|. Across seeds: each walk's mean summed deviation; the survival walk minus
each other walk, seed-paired. Descriptive only, with no test at n = 10: the Pearson correlation
across seeds between the survival walk's mean summed deviation and the run's recorded survival
pooled target at drift 0.45.

**Reading, pre-stated.**

- HIGHER: survival walk minus scripted, minus predictor walk and minus untrained walk are all
  ABOVE zero. The survival policy visits states where the learned law departs further from the
  authentic one than the other policies' states do, and the withdrawn 17.6 sentence is restored
  with these numbers.
- NOT HIGHER: none of the three is ABOVE zero. The survival walk does not visit states where the
  laws differ more, so its advantage for the trunks is not explained by the size of the law
  deviation along the visited states, and 17.6 says so.
- MIXED: anything else, reported per comparison.

The deviation grows with speed and acceleration, so mean |v| and |a| are reported beside it; a
HIGHER reading that comes with a faster walk is described as such.

**Expected compute.** About 0.4 CPU minutes per seed (four walks of 110 surrogate episodes);
about 4 CPU minutes for ten seeds.

## R33: draw-free balanced readout, and the C1 minus C2 contrast (readout `r33`)

**Finding.** The registered pooled readout scores an authentic pool and a surrogate pool drawn
from two different world samples, so it confounds the dynamics with the evaluation-world draw
(FINDINGS 17.5). C1's balanced reading exists (survival 0.767 [0.718, 0.816]); C2's does not,
because its agents were not kept, so the auxiliary contrast C1 minus C2 (+0.120 [+0.065,
+0.174], FINDINGS 17.4) has only ever been read on the registered draw.

**Inputs.** Every agent of the run, both drifts, three arms.

**Procedure.** `itasorl.l0_audit.paired_pooled_readout` at readout drift 0.45 (the run's strongest
drift), 110 pairs of 24 steps: both members of a pair start from the same world seed, a pair is
kept only if both members survive, both share a CV group under explicit-v1, and the interval is
the cluster bootstrap over pairs with the agent seed as its seed. The drift-0.45 agents give the
forward direction, as `scripts/run_l0_audit.py` applied it to C1
(`artifacts/l0_audit/corrected_l3_h8_wm.json`); the drift-0 agents read at 0.45 give the reverse
direction, as `scripts/reverse_direction.py` applied it. A balanced readout at readout drift 0 is
not computed: the twins are identical and the target is 0.5 by construction.

**Seed bases.** Primary: 2200000 + i for both members. Continuity entry: 800000 + i for both
members, the base the l0 audit and the reverse-direction readout used, so the C1 rerun can be
checked against the recorded C1 values (Integrity).

**Statistics.** Per agent drift, base and arm: the balanced target. Per agent drift and base:
survival minus predictor and survival minus untrained, seed-paired.

**The contrast.** `scripts/run_post_review_readouts.py --contrast <C1-rerun output> <C2-rerun
output> --out <file>` writes, for every (agent drift, base), the seed-paired C1 minus C2
difference of the survival arm's balanced target with its t-based 90% interval. It refuses two
outputs whose seed bases or settings differ. The primary entry is the drift-0.45 agents on the
fresh base.

**Reading of the contrast, pre-stated.** The auxiliary rule's 0.05 is the reference.

- HOLDS: the interval lies above 0.05. The decoder's lead holds on the draw-free estimator.
- DIRECTION ONLY: the interval lies above zero and not above 0.05.
- NOT RESOLVED: the interval includes zero. The registered-draw lead is not reproduced on the
  draw-free estimator.
- REVERSED: the interval lies below zero.

Whatever the label, the registered auxiliary verdict stays NOT MET as recorded, because C1 does
not meet its primary rule; the contrast is reported beside 17.4's numbers.

**Expected compute.** About 2.2 CPU minutes per seed (12 balanced readouts of about 11 s); about
22 CPU minutes for ten seeds.

## R31: stream-decoder capacity curve, L3 part (readout `r31`)

**Finding.** FINDINGS 17.8 compares the survival state (0.733) with a supervised GRU reading the
survival agent's own observation and action stream at the trunk's capacity (0.724; state minus
decoder +0.009 [-0.011, +0.029]). That decoder's training loss is below 0.001 in every fold, so
17.8 reads its held-out score as limited by the number of training episodes. Whether a wider
decoder reads the stream better, which would make the equivalence a property of the decoder's
size, is not measured. This readout covers the L3 rung only; the L1 stream readout (14.7.1) is
not touched.

**Inputs.** The survival agent of the drift-0.45 cell of every seed.

**Procedure.**

1. Streams on the fresh pair (authentic 2100000 + i at drift 0, surrogate 2150000 + i at drift
   0.45; 110 episodes of 24 steps; early deaths dropped):
   - `survival_own`: the survival agent's own episodes (`collect_pool`); the sequence is the
     normalized observation x_t (survival normalizer) and the previous env action a_{t-1}, zeros
     at t = 0, 151 channels, built exactly as `scripts/run_control_diagnostics.py` built it for
     17.8;
   - `scripted`: the scripted walk on the same worlds
     (`mechanism_readouts.scripted_observation_streams`, action streams 600000 + i), the same
     channels.
2. State reference: the standard linear probe on the survival trunk's states over the same
   episodes (its own walk for `survival_own`, `eval_protocols.collect_pool_scripted` for
   `scripted`).
3. Decoders, every one trained as `control_diagnostics.sequence_gru_auroc` trains the 17.8
   decoder: per fold, inputs standardized with training-fold statistics, Adam at 1e-3, batch 32,
   60 epochs, torch seed agent seed x 100 + fold, no validation split, no weight decay, the
   explicit-v1 partition with one group per episode; mean out-of-fold AUROC and the per-fold
   final training loss.
   - GRU at 1x, 2x, 4x and 8x the trunk's recurrent width: GRU hidden 96, 192, 384 and 768. The
     per-step encoder keeps the trunk's two 64-unit ReLU layers. The 1x decoder is 17.8's
     decoder exactly.
   - One 1D-conv reader: two temporal convolutions (kernel 5, same padding, 64 then 96 channels,
     ReLU), a mean over time, a linear logit.

**Seed bases.** 2100000 and 2150000; scripted actions 600000 + i.

**Statistics.** Per stream and decoder: the decoder's AUROC; the state reference. Seed-paired:
each decoder minus 1x; the state minus the best decoder, where the best decoder is the one with
the highest mean, chosen after the fact (the choice leans the comparison toward the decoder).

**Reading, pre-stated, per stream.**

- Capacity. CAPACITY-LIMITED: some wider GRU minus 1x is ABOVE zero with a mean of at least
  0.05. NOT CAPACITY-LIMITED otherwise.
- State against the best decoder, labels checked in this order. EQUIVALENT: the interval of
  state minus best lies inside (-0.05, +0.05), the smallest effect of interest 17.8 borrowed.
  DECODER ABOVE: it lies below zero. STATE ABOVE: it lies above zero. UNRESOLVED otherwise.

On `survival_own`, NOT CAPACITY-LIMITED with EQUIVALENT keeps 17.8's reading (the state carries
about what the stream carries) at eight times the decoder's recurrent width. DECODER ABOVE says
the stream carries more than the state reads and that 17.8's equivalence belonged to the
trunk-sized decoder; the 17.8 paragraph is revised accordingly. On `scripted`, the law-fitting
detector of 10.1.1 reads 0.991 from scripted streams while the trunks read 0.55 to 0.58.
CAPACITY-LIMITED there says a wider supervised sequence learner trained on about 176 episodes
recovers part of what the hand-built detector reads. NOT CAPACITY-LIMITED says the scripted
streams' signal stays out of reach of these decoders at this sample size.

**Expected compute.** The largest. Per seed: about 0.5 CPU minutes of rollouts, then two streams
times (1x about 0.4, 2x about 0.8, 4x about 2, 8x about 6, conv about 0.2 CPU minutes), so about
19 CPU minutes per seed and about 190 CPU minutes for ten seeds; the 8x GRU is about two thirds
of it. These widths' costs are scaled from the 1x decoder's by multiply-adds, which makes the 8x
figure the least certain number in this section.

## Compute, resume and output

Estimates for a 4-vCPU container of the kind the revision runs used (one torch thread per
worker, four workers), from the recorded cost of a pooled readout (about 15 s:
`artifacts/l0_audit/corrected_l3_h8_wm.json`, 1934 s for about 120 pooled-readout equivalents in
one process) and of the 17.8 cell (about 55 s including the 1x decoder). Nothing was timed
locally for this spec.

| readout | CPU minutes, ten seeds |
|---|---|
| r210 | about 40 |
| r35 | about 5 |
| r206 | about 4 |
| r33 | about 22 |
| r31 | about 190 |
| all five | about 260, so about 65 to 70 minutes of wall clock on four workers |

Tasks are scheduled longest first (r31, r210, r33, r35, r206). If the 8x decoder runs at half the
assumed speed, the full readout takes about 95 minutes; `--readouts r210,r35,r206,r33` (about 20
minutes of wall clock) and `--readouts r31` can then run as two sessions into the same output.

One task is one (readout, seed). Each finished task is appended to a JSON-lines partial file
(default `<out>.partial.jsonl`) whose first line records the run's cell fingerprints, the
settings and the seed bases; a resume with a different record is refused, a line cut off by a
kill is dropped, and a task that raised is retried. The output JSON is rewritten at the end of
every session from everything in the partial file and holds: every per-seed number, the
summaries with their intervals and labels, the seed bases, the settings, the git commit, the
stack (python, torch, numpy, scikit-learn, scipy), the run's cell fingerprints and trainer
bootstrap values, the fold scheme and version, the integrity block, and whether every requested
task finished.

The output lands under the run's own directory on its run branch
(`artifacts/goal_stakes/<run>/post_review.json`, with its partial file), as the generality-reruns
spec fixes for every output of those runs, and is promoted with a manifest entry when it is
recorded.

## What is written in each case

Each readout's reading is entered in FINDINGS 17 beside the section it bears on (r210 and r31
beside 17.8, r35 and r206 beside 17.6, r33 beside 17.4 and 17.5) with its numbers and the label
fixed above, dated, and marked as a post-review diagnostic. No verdict line, abstract claim or
registered number changes on the strength of any of them.

## Amendment 2026-10-11, before any L1 readout ran

Added on 2026-10-11 after the five readouts above were frozen (commit `28ec282`) and handed to
the C1-rerun cloud session, which was set before this amendment was written to start from that
commit at 16:12 UTC the same day. Nothing above changes: no
procedure, base, statistic or reading of r210, r35, r206, r33 or r31 is altered, and the
sentence of R31 that leaves L1 out stays as frozen. This section adds the L1 part of R31 as a
separate runner, `scripts/run_l1_capacity_curve.py`, before any L1 number from it exists. It
changes no registered verdict; the L1 run's own adjudication (registered encoding criterion not
met) stands as recorded.

**Finding.** FINDINGS 14.7.1 scored the L1 stream on the thirty drift-0.023 cells of the
historical L1 held-out run. The supervised GRU at the trunk's capacity read the survival
agents' own streams at 0.590 [0.547, 0.633] against a state of 0.552 [0.530, 0.574], while the
summary-feature reader read 0.795. 14.7.1 concludes that the trunk holds about what a reader of
its capacity extracts, and leaves open whether a recurrent reader with more capacity extracts
more. This readout measures that.

**Inputs.** The sixty saved agents of the historical L1 held-out run (pre-correction trainer;
`drift_mode = l1`, grid spacing 0.023, sensor noise 0.01, ten seeds, three arms), copied with
their cells to a run directory with `agents/` and `cells/` (`states/` optional). Only the
thirty agents at the run's strongest drift, 0.023, are read, the slice 14.7.1 scored. Results
are labelled historical wherever they are reported.

**Procedure.**

1. Regenerate each (arm, seed) pool exactly as `scripts/run_control_diagnostics.py` did for
   14.7.1: one torch thread, its `setup_rung` at the L1 rung, its `collect_pool` call, the
   registered seed bases 800000 (authentic, drift 0) and 850000 (surrogate, drift 0.023), 110
   episodes of 24 steps, early deaths dropped. The sequence is the arm's normalized observation
   (its own normalizer) and the previous env action, zeros at t = 0, 151 channels. Every arm is
   read on its own streams.
2. Decoders, the same as R31's L3 part and trained the same way (explicit-v1 folds, one group
   per episode, training-fold standardization, Adam at 1e-3, batch 32, 60 epochs, torch seed
   agent seed x 100 + fold): the GRU at 1x, 2x, 4x and 8x the trunk's recurrent width (hidden 96,
   192, 384, 768, the per-step encoder at 64; 1x is 14.7.1's decoder exactly) and the 1D-conv
   reader.
3. One task is one (arm, seed, width), appended to a partial file when it finishes, so a session
   that is cut off resumes. Each task regenerates its pools, so every task also reports the state
   target of its cell; tasks of one cell that disagree are reported.

**Seed bases.** The registered pair 800000 / 850000, by design: the integrity gate and the record
check below need the recorded pools. No fresh worlds.

**Integrity gate, checked before the curve is read.** The frozen L1 spec's gate
(`docs/specs/2026-10-09-l1-stream-readout-design.md`): every regenerated state target must
reproduce the recorded explicit-partition value (`artifacts/fold_rescore/l1_heldout.json`, the
values 14.7.1 reproduced exactly) within 0.01, on every cell read. Where the run directory carries
the run's `states/`, the regenerated pools must also bit-match the saved dumps; without them the
output says the dump half was not checked. A failed gate voids the readout and no number from it
is reported.

**Record check.** The 1x decoder is 14.7.1's decoder, so it is compared with the recorded
per-cell values (`artifacts/control_diagnostics/l1_stream_readout.json`), and the output states
whether the survival arm's 1x mean reproduces the recorded 0.590 within 0.01. The record was made
on the owner's laptop CPU (results manifest, `L1-STREAM-READOUT`), and the artifact records no
package versions. A miss on another stack is reported as a stack difference and does not void the
curve, which is read as differences from its own 1x decoder on its own stack. A miss on the
record's stack is a reproduction failure and voids the curve.

**Statistics.** Per arm and width, the decoder's AUROC across the ten seeds with the t-based 90%
interval; seed-paired, each width minus 1x, and the state minus each width.

**Reading, pre-stated, for the survival arm.** The predictor and untrained arms carry the same
labels, descriptively.

- SEAM AVAILABLE AT WIDTH: some wider GRU (2x, 4x or 8x) reads the survival agents' own stream at
  a mean at least 0.05 above the 1x decoder, with the seed-paired interval above zero. The seam is
  available in the stream to a recurrent reader with more capacity than the trunk, and a trunk of
  the agents' capacity does not keep it. 14.7.1 says so beside the 0.590 row.
- READER-RELATIVE: every wider GRU has been scored and none does. A recurrent reader with up to
  eight times the trunk's width extracts no more than one of the trunk's width, so the distance
  between the state (0.552) and the summary-feature reader (0.795) belongs to the reader's shape.
  14.7.1's statement that the trunk holds about what a reader of its capacity extracts is then
  stated as relative to the reader.
- The conv reader is reported beside the GRU widths and does not enter the label. Whether any
  reader reaches 0.65, the L1 spec's bar, is reported descriptively.

**Expected compute.** On a 4-vCPU container with four workers, from the same costs as R31's L3
part: per cell about 6 CPU minutes for 8x, 2 for 4x, 0.8 for 2x, 0.4 for 1x, 0.2 for the conv
reader, and about 10 s of regeneration per task; about 300 CPU minutes for the thirty cells, so
about 75 to 80 minutes of wall clock. The 8x tasks are about 60 percent of it and run first. If
the 8x decoder runs at half the assumed speed, `--widths 8x` and `--widths 4x,2x,1x,conv` can run
as two sessions into the same output.

## Amendment 2026-10-12, before any R205 readout ran

Added on 2026-10-12, after the readouts above ran on the five corrected runs and before any
number from this one exists. It adds one readout, `r205`, to the same runner. No procedure,
base, statistic or reading above changes. Like every readout here it is descriptive, and no
registered verdict changes on its strength; the registered estimand stays the concatenated
feature.

**Finding R205.** The registered probe reads [mean h, h_T], and nothing in the record says which
half carries the signal. The controlled persistence test (FINDINGS 17.7) suggests the time average:
under identical input the per-step reading starts at 0.677 on the first tail step and falls below
0.55 by the fifth. The finding asks for three probes per cell on the corrected agents and a
sentence in the Discussion saying which component carries the signal.

**Inputs.** Every saved agent of the run, both drifts, three arms, as for r210.

**Procedure.**

1. Regenerate each cell's registered 24-step pools exactly as r210 does
   (`experiment_b2.pooled_readout`, authentic 800000 + i at drift 0, surrogate 850000 + i at the
   cell's drift, 110 episodes, early deaths dropped), with r210's bit check: the concatenated
   linear target must equal the cell's recorded pooled target as a float. A (drift, seed) whose
   three arms do not all reproduce is excluded at that drift and listed, and an unreproduced cell
   is not scored. In an output without r210, the integrity block lists r205's unreproduced seeds.
2. Split `experiment_b.episode_features` into its two halves and score three feature sets:
   (a) mean h only, the first 96 columns; (b) h_T only, the last 96; (c) the concatenation, the
   registered feature, whose target is the bit-checked value itself.
3. Each set is read by the registered linear probe with its preprocessing unchanged (standard
   scaler and logistic regression fit in-fold through `experiment_a.grouped_auroc`, explicit-v1
   folds, one group per episode), as the mean of the five fold AUROCs.
4. Shuffled control per set: the eight label permutations pooled_readout draws, each scored with
   the same probe and partition, averaged. For (c) it is pooled_readout's own value, and r210's
   permutation check (the linear control recomputed from those permutations must equal it to the
   bit) is repeated and reported.

**Seed base.** The registered pair, by design. No fresh worlds.

**Statistics.** Per drift, arm and feature set: the mean target with its t-based 90% interval and
the number of seeds at or above 0.65, the shuffled control and the selectivity. Per drift and arm,
seed-paired: (a) minus (b), (c) minus (a), (c) minus (b).

**Reading, pre-stated, per arm at drift 0.45.** On the seed-paired mean differences, with their
intervals reported beside:

- MEAN CARRIES: (c) minus (a) is at most 0.02, so mean h alone reads within 0.02 of the
  concatenation or above it, and (c) minus (b) is at least 0.05.
- FINAL CARRIES: the mirror case, (c) minus (b) at most 0.02 and (c) minus (a) at least 0.05.
- BOTH: anything else.

The two named cases cannot hold together. The drift-0 cells (authentic against authentic on the
registered pair) carry the same labels, descriptively; they say which half reads the world-sample
difference behind the open L0 gate (FINDINGS 17.5).

**What is written.** The label and its numbers go into FINDINGS 20 beside the R210 subsection, and
the Discussion states, for the survival arm at drift 0.45, which component carries the signal:
the time-averaged state (MEAN CARRIES), the final state (FINAL CARRIES), or both halves, with
neither alone accounting for the registered reading (BOTH). The predictor and untrained arms are
reported in the same table.

**Expected compute.** Six pooled-readout regenerations per seed, as in r210, and about 26 further
linear cross-validations per cell. From the costs measured in the C1-rerun session (the balanced
readouts of r33 ran at about 35 to 50 s each on four workers), about 7 to 8 CPU minutes per seed,
about 75 for ten seeds, so about 20 to 25 minutes of wall clock on four workers. It runs alone
with `--readouts r205` into its own output and partial file.

## Note 2026-10-17, before the within-episode readout (not an amendment)

The closing readouts spec (`docs/specs/2026-10-17-closing-readouts-design.md`, section G04) adds a within-episode decoding curve: the
registered linear probe on h_t alone at steps 1, 2, 4, 8, 12, 16 and 24 of the registered pools,
read from stored dumps, with the step-24 column checked against this spec's R205 "final"
feature set where it exists. It adds no readout id to this runner, and no procedure, base,
statistic or reading above changes. A within-episode curve on the historical hidden-8 dumps was
looked at first, in a scratch read before the closing spec was written. That read is not part of
any result, and it makes the G04 readout post hoc and descriptive.
