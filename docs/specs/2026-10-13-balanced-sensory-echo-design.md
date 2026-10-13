# Balanced sensory-echo control design

Date: 2026-10-13
Status: frozen before any trained arm is examined (readout-only against the
saved `fullruns/balanced_confirmation_{wm,touch}` agents; no training anywhere)

## Why this readout exists

The 2026-10-12 balanced confirmation came back UNINFORMATIVE on both runs. The
trained arms were clear (B1 survival 0.800 [0.772, 0.829], B2 pellets 0.843
[0.825, 0.860], seed-paired margins over the untrained arm with t-90% lower
bounds above 0.05), but the untrained arm read 0.614 [0.599, 0.628] on the
same matched pairs, and the frozen rule required it within 0.1 of 0.5. That
rule is not revisited here. The question this readout answers is narrower:
what does an untrained GRU read off matched worlds, and does the trained
state carry anything the same control does not remove?

## What was looked at before freezing (disclosure)

A read-only diagnostic ran on the ten saved B1 untrained agents only
(`agent_d0.45_s{10..19}_untrained.pt`, drift 0.45, base 910000, CPU). It
reproduced the balanced target at 0.614 and measured: observation trace alone
0.562; action trace alone 0.546; state features with the per-episode
observation features regressed out 0.454; per-step AUROC of the observation
0.500 at t = 0 and 0.639 at t = 1; per-step AUROC of the state 0.500 at t = 0
rising to 0.609 at t = 23; a normalizer refit on both worlds moves the target
to 0.588. The predictor and trained arms were not run under any control. The
pooled sensory-echo result on the 2026-09-26 agents (survival `resid_obs`
0.731, untrained 0.542, hidden 8) was already in the record. The rule below
was fixed with that and nothing else in hand.

## The control, unchanged

`itasorl.behavior_audit.sensory_residual_probe_auroc` as frozen on
2026-09-26 and amended the same day: per fold, ridge-regress the
instantaneous input basis `[x_t, x_{t-1}]` (292 columns) out of every `h_t`
on the training episodes' pooled timesteps, rebuild the level features from
the residual states, probe with the standard logistic pipeline. The
integrated basis `resid_obs_int` and the joint basis `resid_obs_beh` are
reported, not adjudicated. The only changes from the pooled runner are the
pools and the folds:

- Pools: `collect_pool` at drift 0.00 and 0.45 on the SAME seed base 910000,
  n = 110, steps = 24, ray steps 5, `return_obs=True`,
  `return_anchors=True`, `return_index=True`; keep the surviving
  intersection, exactly as `l0_audit.paired_pooled_readout` does.
- Folds: both members of a pair share one group id, so the explicit
  partition never splits a pair (`ITASORL_FOLDS=explicit`).
- Integrity gate: the regenerated balanced `target` of every cell must
  reproduce the value recorded in that run's `cells/cell_d0.45_s{seed}.json`
  under `agents[arm].balanced.target` to within 5e-4, `identical_first_state`
  must be true and `dropped_pairs` zero. Any failure aborts before a control
  number is written.

Agents: `fullruns/balanced_confirmation_wm/agents` (B1, `--objective survival
--mortal on`) and `fullruns/balanced_confirmation_touch/agents` (B2,
`--objective touch --mortal off`), arms `survival` (the trained arm in both
runs; the touch run keeps the key name), `predictor`, `untrained`, seeds 10 to
19. Surrogate `setup_l3_surrogate(hidden=8, seed=0, params=P)`. Device CUDA,
the device the agents were trained and read on. Stack recorded in the output.

## Decision rules (frozen in advance)

Bar 0.65 and margin 0.05 as everywhere else. n = 10 seeds; the t-based 90%
interval adjudicates; the seed bootstrap is reported.

Floor clause (one untrained arm, shared by both runs):

- **Floor explained:** untrained `resid_obs` mean within 0.1 of 0.5. The
  untrained balanced score is a sensory echo that the matched design exposes.
- **Floor not explained:** otherwise. The record says so and nothing further
  is run to rescue it.

Trained clause, per run (B1 trained arm, B2 trained arm), read only if the
floor clause is met:

- **Sensory-independent under matched worlds:** trained `resid_obs` mean at
  or above 0.65 AND above untrained `resid_obs` + 0.05, with the trained
  t-90% lower bound above the untrained mean.
- **Attenuated:** trained `resid_obs` mean in [0.60, 0.65).
- **Largely sensory-mediated:** trained `resid_obs` mean below 0.60.

Honesty checks (reported, no rule reads them): `obs_trace_only` per arm; the
predictor arm under the same control; `resid_obs_int` and `resid_obs_beh`.

## What each outcome changes in the text

The 2026-10-12 routes stay UNINFORMATIVE as recorded; this readout adds one
sentence after them.

- Floor explained and both trained arms sensory-independent: the matched
  worlds section reports that the untrained floor is a passive echo of the
  physics difference and that the trained state carries a world signal the
  same control leaves, with the numbers. The registered headline stays NOT
  MET.
- Floor explained and a trained arm attenuated or sensory-mediated: that
  run's balanced signal is reported as mostly echo, and the pooled-channel
  wording for that objective is tightened to match.
- Floor not explained: the untrained balanced score is reported as
  unexplained, with the diagnostic numbers above, and the matched-worlds
  section ends there.

## Code touch-points

- `scripts/audit_sensory_echo.py`: `--balanced-seed-base`, `--objective`,
  `--mortal`, `--cells-dir` (integrity source in balanced mode). In balanced
  mode the per-cell record adds `n_pairs`, `dropped_pairs`,
  `identical_first_state`, `recorded_target`, `target_abs_diff`. The pooled
  path is unchanged and still bit-compares against the state dumps.
- Tests: balanced mode keeps pairs in one fold group; the integrity gate
  aborts on a mismatched recorded target; the pooled path's outputs are
  unchanged.

## Run commands

    $env:ITASORL_FOLDS = "explicit"
    python scripts/audit_sensory_echo.py --agents-dir fullruns/balanced_confirmation_wm/agents --cells-dir fullruns/balanced_confirmation_wm/cells --out-dir fullruns/balanced_confirmation_wm/sensory_echo --balanced-seed-base 910000 --objective survival --mortal on --drifts 0.45 --device cuda
    python scripts/audit_sensory_echo.py --agents-dir fullruns/balanced_confirmation_touch/agents --cells-dir fullruns/balanced_confirmation_touch/cells --out-dir fullruns/balanced_confirmation_touch/sensory_echo --balanced-seed-base 910000 --objective touch --mortal off --drifts 0.45 --device cuda

## Out of scope

Nonlinear residualization, a new untrained control, any change to the
2026-10-12 rule or to the recorded B1/B2 numbers, and retraining.

## Result (2026-10-13)

Both runs finished on CUDA. Integrity held on all 60 cells: every regenerated balanced
target matched the recorded value, no pair was dropped, and the first states matched.
The floor clause holds. Untrained `resid_obs` is 0.596 [0.577, 0.615], 0.096 from 0.5,
the same ten scores in both runs. The trained clause holds on both runs. `B1` survival
`resid_obs` is 0.761 [0.727, 0.795], 10 of 10 seeds at or above 0.65. `B2` is 0.780
[0.757, 0.804], 10 of 10. Both are sensory-independent under matched worlds. The
2026-10-12 routes stay UNINFORMATIVE as recorded. FINDINGS 21.
