# Coherence at a matched finite-horizon gap: the companion sweep (sweep C)

Frozen 2026-10-15, after the deep revision review of the same date and before any code or
number of this sweep. Readout-only, on the saved `C1` agents. Companion to
`docs/specs/2026-10-15-coherence-integration-sweep-design.md`, whose sweeps A and B stand as
their own diagnostic record.

## Why

Sweep B matched the stationary standard deviation of the accumulated velocity offset
(0.0870 at every rho), not the quantity the readout window sees. Its measured 24-step
open-loop gaps run from 0.1205 at rho = 0 to 0.0970 at rho = 0.99, because a 24-step episode
that starts without an accumulated offset does not reach the stationary value. Sweep A
and sweep B also vary one-step size and stationary accumulation separately, so neither
holds the finite-horizon deviation fixed while coherence moves. And in both sweeps the
AR(1) process ran on across episode boundaries after one draw per pool, so at rho = 0.99
adjacent episodes share about 0.79 of its dependence after 24 steps, and the episode-grouped
folds do not remove it.

Sweep C holds the 24-step open-loop velocity gap fixed at the level of the shared point
(sweep A at rho = 0.9) while rho moves, and starts every episode from its own independent
draw of the stationary distribution, so the episodes are independent and the measured
deviation over the readout window is matched.

## The family and its calibration

`G_ar` as defined in the companion spec, with one change in how it is driven: at the start
of every episode the process state is drawn afresh from its stationary distribution, from a
Philox stream seeded by the pool's seed base plus the episode index. Nothing runs on from one
episode to the next. The family's law is unchanged.

rho in {0, 0.5, 0.8, 0.9, 0.95, 0.99}, the six values of sweeps A and B.

sigma(rho) is chosen so that the family's 24-step open-loop RMS velocity gap, measured as the
companion spec measures it (`surrogate_diagnostics` on the seed-77 calibration sample, with
the per-episode draw applied there too), equals the shared point's gap, 0.1150 (sweep A at
rho = 0.9, sigma = 0.01). The perturbation is additive and linear in sigma, so the closed
form sigma(rho) = 0.01 x 0.1150 / gap_A(rho) is used, with gap_A(rho) the recorded 24-step
gap of sweep A at that rho (0.0364, 0.0598, 0.0929, 0.1150, 0.1302, 0.1398). Each point's
measured gap is recorded and must lie within 3 percent of 0.1150; a point outside that band
is recalibrated by one proportional step and the correction is recorded with the point. The
rho = 0.9 point is sweep A's shared setting (sigma = 0.01), run again under the per-episode
draw on the new bases.

## Readout

`scripts/run_texture_fresh_probe.py` with family `arc` (the `ar` family driven as above) at
each of the six settings, on the drift-0.45 survival, predictor and untrained `C1` agents
(`fullruns/corrected_l3_h8_wm/agents`, seeds 0 to 9), 110 episodes of 24 steps, the same
three readings as the companion spec (transfer of the learned-law direction, the fresh
grouped-CV probe on authentic against G_ar pools, the paired-pool fresh readout). Seed bases
2500000 (authentic) and 2550000 (G_ar), disjoint from every base in use and from the
companion sweep's 2400000 and 2450000. The family's measured profile (one-step RMS, lag-1
autocorrelation, 24-step gap) is recorded beside each point. `scripts/run_ar_sweep_c.py`
drives the six points and writes the summary; a point file that exists is reused only if its
recorded family parameters, seed bases and commit equal the invocation's.

## Readings, stated before any number

Primary quantity at each point: the seed-paired survival-minus-untrained margin of the fresh
probe (M), with the survival fresh mean beside it. Contrast: M at rho = 0.99 minus M at
rho = 0, seed-paired, t-based 90 percent interval, n = 10.

1. Coherence raises the reading at a matched finite-horizon gap: the contrast has a lower
   bound above 0.
2. Coherence does not move the reading by more than the margin unit: the contrast's interval
   lies inside [-0.05, +0.05]. This is an equivalence reading, and it is the only reading
   here that can support a statement that coherence has no effect of that size.
3. Neither: the interval reaches outside [-0.05, +0.05] without a lower bound above 0. The
   sweep is then inconclusive on coherence at this gap.

Descriptive, no threshold: the rho = 0.9 point's M and survival fresh mean beside the
companion sweep's shared point (+0.010 and 0.570), as a check that the per-episode draw
changes nothing material. A point where the survival fresh mean reaches 0.65 with M at
least 0.05 is called kept, as before. Nothing here is a registered verdict, and no reading
of sweeps A or B changes.

## Output

`artifacts/texture/ar_sweep_c/C_rho{0.00,0.50,0.80,0.90,0.95,0.99}.json` and
`artifacts/texture/ar_sweep_c/summary.json` with provenance, the calibration (recorded gaps,
sigma per rho, any proportional correction), the per-point profile and margins, the
contrast and the three readings.
