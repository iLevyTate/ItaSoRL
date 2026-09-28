# Local strengthening probes: in-configuration L2 oracle and a nonlinear joint control

Date: 2026-09-27
Status: frozen before any run (both probes are oracle-only or readout-only; no
training anywhere)

## Purpose

Two reviewer-facing gaps that cost minutes of compute, both named in the
manuscript's limitations:

1. **The L2 oracle-versus-agent contrast is not a same-world comparison.** The
   published L2 ceiling (0.993) comes from Experiment A's tamed diagnostic
   configuration, whose hard-coded constants do not run on the foraging world
   `P`; every L2 organism number is measured in `P`. L3 has an in-configuration
   gate (0.928 on `P`); L2 does not. This probe measures the L2 oracle ceiling
   on `P` itself, at the organism's own drift strength, with the same
   detector-side handicap the L3 gate uses.
2. **The behavior and sensory controls are linear.** FINDINGS 10.4 and 10.4.2
   say a full-history or nonlinear residualization could in principle remove
   more. This probe runs the joint sensory-plus-behavior control with a
   nonlinear (MLP) regressor, in-fold, on the same regenerated pools.

## Probe A: in-configuration L2 oracle

**Design.** Matched pairs exactly as the L3 gate builds them
(`generate_l3_pairs`: shared authentic prefix in `P`, snapshot, two branches
under the identical action stream), except the surrogate branch is the L2 world
itself: `PatchOfEarthV0(P, drift_sigma=0.45, drift_mode="ar1")` (the B-v2
artifact) and `drift_mode="regime"` (the B-v3 artifact). The residual oracle
re-applies the AUTHENTIC velocity law with the authentic constant drag of `P`
(`k_land = k_water = 1.5`, so the authentic drag is exactly 1.5 everywhere) to
the logged transitions, under the frozen detector-side noise `sigma_meas =
0.02` of the L3 gate, and scores grouped AUROC with the pair as the unit. The
mechanical leakage channels (length, metadata) must sit at chance; reward is
reported, not gated (dynamics rung).

**Cells (frozen).** 60 pairs, prefix 10, branch 30, seed base 3000, matching
the L3 gate: drift 0.00 (L0 anchor, must read chance), ar1 at 0.45, regime at
0.45. Regime mode keeps the surrogate's reset-drawn offset across the snapshot
restore (the B-v3 apparatus fix of 2026-07-02).

**Decision.** None; this is a measurement. It replaces the tamed-configuration
0.993 in the L2 contrast sentence of FINDINGS section 4 and the paper, and the
scope note is retired if the L0 anchor is at chance and the mechanical channels
are clean. If the in-configuration oracle reads below the L3 band, that is
reported as such: the L2 contrast then compares a weaker oracle against a
chance-level agent, and the "detectability >> encoding" wording is re-checked
against the actual number.

**Extension, 2026-09-27, logged after the first three cells were measured and
before any write-up.** The first pass (60 pairs, sigma 0.02) read the L0
anchor at 0.581, ar1 at 0.646 and regime at 0.618, and at the tamed
configuration's own detector noise (sigma 0.0002) both L2 cells read 1.000.
The published L2 ceiling (0.993) and the L3 gate (0.928) were therefore
measured at detector handicaps a hundredfold apart, so the cross-rung
comparison the manuscript draws ("L2 more detectable than L3, yet only L3
encoded") is not a matched comparison. The probe is extended, still with no
decision rule, to a sigma sweep {0.0002, 0.001, 0.005, 0.01, 0.02, 0.05}
scored identically for the L2 cells and for the L3 fingerprints at hidden 8
and 7 (G seed 0), so the paper can state detectability per rung at a common
handicap. The manuscript's "artifact character" reading is then re-checked
against the matched curve.

## Probe B: nonlinear joint control

**Design.** Same regenerated pools and integrity gate as the sensory-echo
control (spec 2026-09-26; runner `scripts/audit_sensory_echo.py`), now also
saving the observation traces so no further regeneration is needed. The
regressor in the in-fold residualization becomes a one-hidden-layer MLP
(64 units, ReLU, L2 penalty 1e-3, Adam, 300 iterations, fixed seed) fit on the
train folds' pooled timesteps, on two bases: the instantaneous observation
basis alone (`resid_obs_mlp`) and the observation plus full behavior trace
basis (`resid_obs_beh_mlp`). Drift 0.45 only (the decision cell); the
untrained and predictor arms pass through the identical control as the honesty
bound.

**Decision rule (frozen).** On the joint nonlinear control at drift 0.45:

- survival `resid_obs_beh_mlp` >= 0.65 AND > untrained + 0.05: the
  world-signal survives a nonlinear input-plus-behavior control; the "linear
  only" scope limit in methods note 7 and the paper is retired.
- in [0.60, 0.65): attenuated; reported as the new most-conservative number.
- < 0.60: largely explained by nonlinear mirroring of inputs and behavior;
  the paper's "not a linear echo" wording stands and gains the qualifier
  "a nonlinear joint control removes it", which is a real narrowing.

The linear joint control already reads 0.670 (10.4.2); an MLP can only remove
more, so the honest expectation is a lower number, and the rule above is
written for that.

## Code touch-points

- `itasorl/experiment_a_l3.py`: `generate_l2_pairs`, `oracle_features_L2`,
  `run_experiment_a_l2` (reusing the L3 leakage battery).
- `scripts/run_expA_l2_inconfig.py`: the probe A runner; writes a JSON.
- `itasorl/behavior_audit.py`: `sensory_residual_probe_auroc(...,
  nonlinear=True)`.
- `scripts/audit_sensory_echo.py`: `--nonlinear` and `--save-traces`.

## Testing (before launch)

- Probe A: at drift 0 the two branches are bit-identical, so the oracle reads
  exactly chance on the pair structure; at a large drift the oracle reads
  above 0.9 on a tiny pair set. Regime mode preserves the reset-drawn offset
  across `set_state`.
- Probe B: on synthetic data with a nonlinear echo `h = tanh(W x)`, the
  linear control leaves signal the MLP control removes; an orthogonal tag with
  uninformative inputs survives both.

## Run commands

    python scripts/run_expA_l2_inconfig.py --json fullruns/l2_inconfig_oracle.json
    python scripts/audit_sensory_echo.py --agents-dir fullruns/l3_h8_heldout/agents \
        --states-dir fullruns/l3_h8_heldout/states --out-dir fullruns/l3_h8_sensory_echo_mlp \
        --drifts 0.45 --nonlinear --save-traces --device cuda
