# Does the agent use the decoded direction? The removal test

Frozen 2026-10-15, before any code for it was written or any number below was computed.
Readout-only on saved agents. It changes no registered rule or verdict.

## Why

The intervention readout of the goal-and-stakes spec (item 6 of
`docs/specs/2026-10-07-goal-and-stakes-design.md`, FINDINGS 18.3) adds a constant shift of
one standard deviation along the decoded direction after every GRU step and asks whether
real-world behavior moves toward fake-world behavior. Real minus sham closed +0.017
[-0.019, +0.054] of the gap on `C1`, -0.001 on the pellet run and -0.029 on the death-off
survival run. The dated note of 2026-10-14 reads this as the registered nudge not showing an
above-sham effect, not as proof that the direction is unused.

A nudge adds the same constant to every state in both worlds. A reader that used the
direction would use its variable part, the difference between the two worlds' projections.
The converse test removes that part while the agent acts and asks whether behavior changes.

## Design

Agents: the `C1` rerun's saved survival agents at drift 0.45
(`fullruns/corrected_l3_h8_wm/agents`, corrected trainer, seeds 0 to 9), the headline arm.
The drift-0.45 untrained arm goes through the same procedure as the floor of the
manipulation: what erasing a probe-fitted direction does to a trunk that keeps little.

Direction: `u` and `s` from `mechanism_readouts.probe_direction` on the registered 24-step
pools (110 episodes, seed bases 800000 authentic and 850000 surrogate, ray steps 5), as the
nudge readout fits them. The erasure target `c` is the midpoint between the two pools' mean
per-step projections onto `u`, computed on the same pools.

Erasure: after every GRU update and before the policy head, so that the action, the recorded
state, the decoder input and the next GRU step all read the erased state (the nudge's
placement),

    h <- h - ((h . u) - c) u

Every state lands at the midpoint along `u`, so the probe's score along `u` is the same in
both worlds by construction. Nothing else in the state is touched: a rank-one removal.

Sham: the same rule along `v = mechanism_readouts.sham_direction(u, seed)`, the unit
direction orthogonal to `u` that the nudge readout used, with its own midpoint `c_v`.

Rollouts: 64 fresh episodes of 80 steps under the agent's own deterministic policy at world
seed base 1400000, the nudge readout's base, so the unerased authentic and surrogate
rollouts are the 18.3 ones. Six variants: authentic and surrogate, each unerased, erased and
sham. Survivors are intersected across all six variants before any mean, as the nudge
readout intersects them. Behavior is the seven-column summary of
`mechanism_readouts.rollout_behavior`, standardized by the pooled across-world scale of the
unerased rollouts.

Per seed:

- Gap closed in the surrogate world: `gap_closed(b_surr, b_auth, b_erased_surr, scale)`, the
  fraction of the surrogate-to-authentic behavior gap that erasing closes while the agent
  acts in the surrogate world; the sham likewise. The authentic world the same way, with
  `gap_closed(b_auth, b_surr, b_erased_auth, scale)`. The primary score is the mean over the
  two worlds of real minus sham; each world is reported beside it.
- Return and pellet intake per world under each variant, erased minus sham.
- Manipulation check: the registered probe fit on the unerased pools, frozen, scored on the
  erased rollouts' pooled features; and a probe refit on the erased states, which measures
  how much world identity a rank-one removal leaves in the rest of the state.
- Informativeness as the nudge readout defines it: a seed whose standardized behavior gap is
  under 0.25 is uninformative, and readings need at least 7 informative seeds.

## Readings, stated before any number

Seed-paired t-based 90% intervals over informative seeds.

1. Used under removal: the primary real-minus-sham score is at least +0.10 with its interval
   excluding zero, the nudge rule's threshold.
2. Costly: erased minus sham return in the surrogate world has an upper bound below 0. The
   authentic world is reported beside it.
3. Not used under removal: both worlds' real-minus-sham scores have intervals inside
   [-0.10, +0.10].
4. Erasure held: the frozen probe reads the erased rollouts within 0.1 of 0.5 in at least 8
   of 10 seeds; otherwise the readout is uninformative. If the refit probe reads the erased
   states at 0.65 or above, the label "partial erasure" is attached to every reading, because
   a rank-one removal left world identity elsewhere in the state and the test then bounds
   only what the decoded direction itself carries.

Each reading is reported as met or not met. None is a registered verdict. A result that
meets neither 1 nor 3 is reported as such. The untrained arm's scores are the floor and
decide nothing.

## Output

`artifacts/mechanism/erasure/corrected_l3_h8_wm.json` with provenance, per-seed values, the
frozen and refit probe readings on erased states, and the four readings.

## Amendment 2026-10-15, before any code or number

Found while writing the code. The registered probe scores the mean-state block and the
final-state block with two coefficient vectors, and `u` is their sum. Removing `u` alone
leaves each block's own projection free to differ between worlds, so the frozen probe could
still separate the erased rollouts and reading 4 would fail for a reason that has nothing to
do with use. The erasure is therefore rank two: `a` and `b`, the mean-block and final-block
coefficient vectors divided by the scaler's scale (the raw-space directions
`probe_direction` sums), span the subspace S, P is the orthogonal projector onto S, and m is
the midpoint of the two pools' per-step mean states. After every GRU update,

    h <- h - P (h - m)

so every linear score the frozen probe can form lies at its midpoint in both worlds. The sham
is a rank-two subspace orthogonal to S, drawn per seed from the RNG of
`mechanism_readouts.sham_direction` (seed 1000003 plus the agent seed), with its own
projector and the same m. The manipulation check scores the first 24 steps of the erased
rollouts, the window the registered features read. Nothing else changes.

## Second amendment 2026-10-15, after a two-seed setup check and before the full run

After erasure every frozen-probe score is equal up to float32 noise, and an AUROC computed on
those scores ranks the noise (0.39 to 0.61 on the check). The frozen probe's scores are
quantized to one thousandth of the unerased rollouts' score spread before the AUROC is taken,
so a pinned score reads 0.5, and the ratio of the erased to the unerased score spread is
recorded beside it. Reading 4 is otherwise unchanged. The two-seed check is not part of the
result.

## Amendment 2026-10-17 for the full-rank extension, before its run

The extension readouts spec (`docs/specs/2026-10-17-extension-readouts-design.md`, section X10) extends the rank-two erasure above to full rank
on the same agents. Its readings are this spec's frozen readings 1 to 3 ("Readings, stated before
any number"), unchanged: used under removal (real minus sham at least +0.10, interval excluding
zero), costly (erased minus sham surrogate-world return with an upper bound below 0), not used
under removal (both worlds' scores inside [-0.10, +0.10]). The rank-two default path, its stored
artifact and its readings do not change. The following are new thresholds and rules, not in the
frozen text, stated here before the run:

1. Refit stop and rank cap. The erased rank grows until the refit probe on the erased pools
   reads at or under 0.55, or the rank reaches 16 of the 96 units. Pooled out-of-fold refit
   scores become unreliable as the erased rank approaches the state width (the in-fold scaler
   blows up on near-null residual dimensions), which is why the cap is 16. A fold that runs out
   of new directions keeps its last subspace at higher ranks.
2. Orthogonalization. Each refit direction is projected onto the complement of the already-erased
   subspace before it is stacked, because a refit direction on erased states lies largely inside
   the erased subspace.
3. In-fold stop statistic. For each outer fold of the explicit twin-grouped partition, the
   subspace is grown from the rank-two projector refit on that fold's training rows only, and the
   refit probe trained on erased training rows is scored on erased test rows; the out-of-fold
   scores are pooled per rank. k is the smallest rank whose pooled in-fold refit is at or under
   0.55, else 16. An erasure fit on every episode, test folds included, pushes the refit AUROC
   toward and below chance once the real signal is erased, which would stop early and understate
   k; that all-episode curve is reported, labeled biased low, and never read. The rollout
   projector is grown on all episodes to that k, and the sham has rank k, orthogonal to it.
4. Capacity-confounded rule. If the sham's surrogate-world return drops by more than 10 percent of
   the unerased return (the upper bound of the paired difference below -0.10 times the unerased
   mean), no use reading is made.
5. Gate E. The in-fold refit must reach 0.55 or less in at least 8 of 10 seeds at rank at most
   16, else the full-rank readout is void and only k is reported.

The refit curve by rank is reported so a reader can apply another stop. This is the only
pre-run entry for the extension; a post-run note on the rank-two run's labels is a separate dated
entry in this file.
