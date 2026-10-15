# Which features of its world does the state keep? Instrumental against incidental in one world

Frozen 2026-10-15, before any code for it was written or any number below was computed.
Readout-only on saved agents. It changes no registered rule or verdict.

## Why

The record asks whether the state keeps a between-world distinction the agent was never
rewarded for. Every episode also draws a world of its own: a terrain basis (frequencies,
phases and amplitudes, `patch_of_earth._init_static_fields`) that fixes the water map, a
start position and heading, and 24 pellet positions on land. Some of those features bear on
foraging inside the 24-step readout window and some cannot. That gives, within one world and
one agent, a ladder from instrumental to incidental features of the same kind, and the
question the paper asks of the fake world can be asked of each rung: kept beyond the senses,
or not.

The open L0 gate is not evidence here. Across eight independent authentic pairs the pair
means average 0.493 with sd 0.049 (FINDINGS 17.5), so the separability of two authentic
pools is a draw fluctuation. This readout decodes named features directly.

## Design

Agents: the `C1` rerun (`fullruns/corrected_l3_h8_wm/agents`, seeds 0 to 9): survival at
drift 0.45 (the headline arm), predictor and untrained at drift 0.45, and survival at drift
0. Second block: the `C2` rerun's survival arm (`fullruns/corrected_l3_h8_nowm/agents`).

Episodes: the headline survival agent's own authentic-world episodes (drift 0), 220
full-length episodes of 24 steps per seed at seed base 2600000 (disjoint from every base in
use), ray steps 5, recorded with `eval_protocols.record_trajectories` and replayed through
every arm with `replay_states`, so every arm reads identical input and the untrained trunk
on that input is the binding floor. Secondary: each arm's own episodes on the same seeds.

Reach: the radius `r` is the 95th percentile of the headline agent's 24-step displacement
over a calibration sample of 110 episodes per seed at seed base 2650000, pooled over seeds,
fixed before any feature is computed and recorded in the output.

Features are computed from the world object at reset and before any step, from the stored
terrain basis, the start position, the pellet array and the deterministic wetness function,
sampled on a 128 by 128 grid with toroidal distances. No random draw is consumed, so the
recordings are unchanged.

Instrumental, inside `r` of the start:

- I1: distance from the start to the nearest pellet.
- I2: number of pellets within `r`.
- I3: water fraction of the grid points within `r`.

Incidental, outside `r`, where nothing inside the window can reach:

- N1: water fraction of the grid points outside `r`.
- N2: terrain roughness outside `r`, the standard deviation of the heightfield there.
- N3: dispersion of the pellets outside `r`, their mean pairwise toroidal distance.

Each incidental feature is decoded as its residual after a linear regression on I1 to I3
over the episodes, so what is decoded is the part the near field does not predict. The
heading is an observation channel and the start is the arena center except when that is
water, so neither is a feature.

Decoding, per arm, seed and feature: ridge regression (alpha 1.0) from the registered pooled
features [mean h, final h], standardized in fold, five-fold cross-validation over episodes,
held-out R squared pooled over folds. Beside it, for the paper's scale, the held-out AUROC of
the registered logistic probe on a median split of the feature.

Echo and residual: the same ridge from the observation summary [mean o, final o], what the
senses carry; and from the state features after the observation summary is regressed out in
fold, what the state keeps beyond the senses.

Sanity: the same decoding with the feature permuted across episodes, whose held-out R
squared must not exceed 0.02 in the mean.

## Readings, stated before any number

Seed-paired t-based 90% intervals, n = 10, on the headline arm.

1. Instrumental kept: for each of I1 to I3, survival R squared minus untrained R squared has
   a lower bound above 0, and the survival residual R squared (beyond the senses) has a lower
   bound above 0.05.
2. Incidental not kept: for each of N1 to N3, the survival residual R squared has an upper
   bound below 0.05.
3. Ladder: the mean residual R squared over I1 to I3 minus the mean over N1 to N3 has a lower
   bound above 0.
4. Echo: a feature whose echo R squared exceeds its state R squared is labeled "carried by
   the senses" and decides nothing.

Labels: "follows usefulness" when 1 and 2 are met; "keeps everything" when 1 is met and 2
fails on at least two of N1 to N3; "keeps nothing beyond the senses" when 1 fails on every
instrumental feature; "mixed" otherwise. Each reading is reported as met or not met. None is
a registered verdict. The predictor, untrained and drift-0 arms and the `C2` block are
reported beside the headline arm.

## Output

`artifacts/world_features/corrected_l3_h8_wm.json` and
`artifacts/world_features/corrected_l3_h8_nowm.json`, each with provenance, the reach
radius, per-episode features, per-seed values for every arm and feature, and the readings.

## Amendment 2026-10-15, before any code or number

Found while writing the code. The observation summary has 292 columns and the residual step
regresses 192 state columns on it in fold. At 220 episodes a training fold has 176 rows, so
the regression interpolates and the residual means nothing. The readout uses 1100 episodes
per seed at seed base 2600000 (880 training rows per fold). Constant observation columns are
dropped before the regression, and the state-on-observation regression is ridge with alpha
1e-3 on columns standardized in fold, the control battery's setting.

The reach radius is the 95th percentile of the largest toroidal distance from the start
reached at any step of the 24-step window, not of the distance at step 24. A feature beyond
that radius was out of reach at every step, which is what "incidental" requires. Both
percentiles are recorded. Distances are toroidal on the unit arena. Nothing else changes.
