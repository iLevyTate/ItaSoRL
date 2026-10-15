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
