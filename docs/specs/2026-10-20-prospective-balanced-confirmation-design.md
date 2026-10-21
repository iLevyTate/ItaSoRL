# Prospective balanced confirmation

Date: 2026-10-20
Status: frozen before any cell of `PB1` or `PB2` exists. This file was committed and pushed
(7d08686) before the first training command started; the readout plumbing named under "Code" is
committed before any readout below runs. Amended once, 2026-10-20 about 20:00Z, while no cell
file of either run existed, after an independent audit of the first text
(`fullruns/closing_2026-10/handoff/reviews/audit_pb_spec.md`): gate scope and sources, gate 2 on
the trained arm with an untrained mean check, per-run floor, NOT RUN and restart, S2 wording,
the full descriptive commands, and the disclosure list. The first text is 7d08686.
Code: the commit that adds this file, plus the readout commit named in the result section.
`ITASORL_FOLDS=explicit`.

## Why this run exists

The matched-world confirmation of 2026-10-12 (`B1` survival, `B2` pellets with death off) is
UNINFORMATIVE: its untrained arm reads 0.614 on matched pairs, outside the 0.1 floor, and one
untrained matched speed pool reads 0.749 against 0.75. The sensory-echo rule of 2026-10-13
then reads `B1` at 0.761 and `B2` at 0.780 with the untrained floor met by 0.004 at the mean,
its interval crossing 0.60. That rule was frozen after the raw `B1` and `B2` outcomes and an
untrained-only diagnostic were known (`docs/specs/2026-10-13-balanced-sensory-echo-design.md`,
disclosure section). Two referee reviews of the final manuscript (review 5, 2026-10-20) named
this as the result's main limitation and asked for a prospective test with the rule fixed in
advance, and for a control that is nonlinear and reaches further back than the previous
observation.

This run applies the 2026-10-13 rule, unchanged, to new agents on new seeds and a new
evaluation draw, with every gate and reading fixed here before any of those agents exist. It
does not edit `B1`, `B2`, `C1` or any recorded number.

## What was known when this was written

Everything in the record up to paper commit 6ad9577 and ITASORL `main` 0163168 plus the open
records PR #175: the `B1` and `B2` raw and controlled readings at base 910000, the descriptive
second-draw readings at base 2250000 (G03: untrained `resid_obs` 0.577, `B1` survival 0.724,
`B2` 0.761), the predictor `resid_obs` (0.645 at 910000, 0.600 at 2250000), the integrated and
joint linear controls `resid_obs_int` and `resid_obs_beh` on `B1` and `B2` at both bases (at
910000 `resid_obs_int` survival 0.718, pellets 0.748, untrained 0.606), the matched speed and
leakage sidecars of `B1` and `B2` at 910000 (untrained matched speed 0.749 to 0.845, the minimum
on seed 19; trained arms at least 0.888; largest leakage deviation 0.069), the O1 fold-local
speed readout, and the nonlinear sensory-echo readings of the
2026-09-26 historical agents (`fullruns/l3_h8_sensory_echo_mlp`). No agent with seed 20 to 29
has been trained under any configuration, and no pool at base 3000000 or 3100000 has been
drawn.

## What is held fixed

`PB1` repeats the `B1` command and `PB2` the `B2` command
(`docs/specs/2026-10-12-balanced-confirmation-design.md`, "What is held fixed" and
"Commands"), with three changes: seeds 20 to 29, balanced base 3000000, and the output
folders. Drift mode `l3`, surrogate hidden 8, G seed 0, drifts 0 and 0.45, 300 updates,
16 episodes per update, 80 steps, trunk 96, ray steps 5, shaping 1.0, pooled episodes 110 by
24 steps, matched pairs 60 with prefix 20 and branch 24, `--gae-bootstrap successor`,
`--adv-norm historical`, one worker on CUDA, `--save-agents --dump-states`. `PB2` adds
`--objective touch --mortal off`, with the touch engagement rule of `B2`.

Seeds 20 to 29 train survival agents on world seeds 100000 + 10000 s + 16 u, which is 300000
to about 394800, and predictors on 200000 + 9000 s + 16 u, which is 380000 to about 465800.
Both spans overlap training worlds of earlier runs (the `B1` and `B2` predictor schedule and
the Experiment C panels at 320000 and above). That is a fact of the training schedules, as
the 2026-10-17 note in the 2026-10-12 spec records for `B1` and `B2`; the `PB` survival and
predictor spans also overlap each other at 380000 to 394799 (survival seeds 28 and 29, predictor
seeds 20 and 21), the same structure `B1` had at 290000 to 294799; no readout below draws from a
training span.

## Commands

```text
$env:ITASORL_FOLDS = "explicit"
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 20 21 22 23 24 25 26 27 28 29 --gae-bootstrap successor --adv-norm historical --balanced-seed-base 3000000 --save-agents --dump-states fullruns/prospective_balanced_wm/states --out-dir fullruns/prospective_balanced_wm --device cuda --workers 1
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 20 21 22 23 24 25 26 27 28 29 --objective touch --mortal off --gae-bootstrap successor --adv-norm historical --balanced-seed-base 3000000 --save-agents --dump-states fullruns/prospective_balanced_touch/states --out-dir fullruns/prospective_balanced_touch --device cuda --workers 1
python scripts/audit_balanced_matched_gates.py --run fullruns/prospective_balanced_wm --objective survival --mortal on --device cuda --seed-base 3000000 --seeds 20 21 22 23 24 25 26 27 28 29
python scripts/audit_balanced_matched_gates.py --run fullruns/prospective_balanced_touch --objective touch --mortal off --device cuda --seed-base 3000000 --seeds 20 21 22 23 24 25 26 27 28 29
python scripts/audit_sensory_echo.py --agents-dir fullruns/prospective_balanced_wm/agents --cells-dir fullruns/prospective_balanced_wm/cells --out-dir fullruns/prospective_balanced_wm/sensory_echo --balanced-seed-base 3000000 --objective survival --mortal on --drifts 0.45 --device cuda --seeds 20 21 22 23 24 25 26 27 28 29 --nonlinear
python scripts/audit_sensory_echo.py --agents-dir fullruns/prospective_balanced_touch/agents --cells-dir fullruns/prospective_balanced_touch/cells --out-dir fullruns/prospective_balanced_touch/sensory_echo --balanced-seed-base 3000000 --objective touch --mortal off --drifts 0.45 --device cuda --seeds 20 21 22 23 24 25 26 27 28 29 --nonlinear
```

Then, descriptive only: the same two sensory-echo commands at `--balanced-seed-base 3100000
--integrity-seed-base 3000000`, out dirs `.../sensory_echo_draw2`; and the nonlinear controls
on the saved `B1` and `B2` agents at their own base:

```text
python scripts/audit_sensory_echo.py --agents-dir fullruns/balanced_confirmation_wm/agents --cells-dir fullruns/balanced_confirmation_wm/cells --out-dir fullruns/balanced_confirmation_wm/sensory_echo_nonlinear --balanced-seed-base 910000 --objective survival --mortal on --drifts 0.45 --device cuda --nonlinear
python scripts/audit_sensory_echo.py --agents-dir fullruns/balanced_confirmation_touch/agents --cells-dir fullruns/balanced_confirmation_touch/cells --out-dir fullruns/balanced_confirmation_touch/sensory_echo_nonlinear --balanced-seed-base 910000 --objective touch --mortal off --drifts 0.45 --device cuda --nonlinear
```

A restart after an interruption reruns the same command with `--resume` added; the config
fingerprint check in `run_expB2.py` refuses cells written under any other configuration.

Order: `PB1` training, `PB2` training, the two matched-gate audits, the two primary
sensory-echo readouts, then the descriptive readouts. Outputs stay under `fullruns/`
(gitignored) until promotion. If `PB1` training does not finish, `PB2` still runs; each run is
decided on its own.

## Seed bases

The primary matched pairs are drawn at base 3000000 (authentic and surrogate share each world
seed), 110 per cell. The descriptive second draw is base 3100000. Both footprints (base,
base + 17001 and base + 33002, 110 episodes each) miss every base in
`itasorl.post_review.all_bases_in_use()` and `REGISTERED_LATER`, which this spec's code commit
extends with `PB-primary` (3000000) and `PB-draw2` (3100000). The runner's fixed bases
(engagement 900000, matched branch pairs 700000, pooled 800000 and 850000, survival return
920000, the untrained normalizer warmup 555000) are the ones every run uses, as in `B1`.

## Primary decision (per run, `PB1` and `PB2` separately)

The 2026-10-13 rule, with the gates of 2026-10-12 that the matched design can pass. Gate 1 is
read on every cell of the run, both drifts (20 cells), from each cell's `eng.engaged`, as
`scripts/promote_balanced_confirmation.py` counted it for `B1` (20 of 20). Gates 2 to 7 are read
on the drift-0.45 cells, all three arms unless a gate says otherwise, on the retained pairs at
base 3000000.

Gates. Any failure makes the run UNINFORMATIVE:

1. Engagement in every cell, under the rule for the run's objective (survival: as `B1`;
   touch: trained return at least 1.5 times the better baseline and at least 0.25 pellets per
   episode).
2. Matched speed probe at least 0.75 on every drift-0.45 seed of the trained arm (10 values),
   one value per seed and arm over both members of the retained pairs with the median of those
   episodes, as `matched_population_gates` in `itasorl/l0_audit.py` computes it; and the
   untrained arm's mean matched speed over its ten seeds at least 0.75, so that a floor reading
   cannot come from a state the probe cannot read. The positive control asks whether the probe
   reads a known variable from the state the decision reads; in `B1` and `B2` the untrained
   values sat at 0.749 to 0.845 (mean 0.793). Every untrained and predictor value is reported.
   The gate is read from `matched_gates.json` written by `scripts/audit_balanced_matched_gates.py`,
   every row with `reconstruction.abs_diff` at most 0.005; the same block recorded in each
   training cell (`balanced.matched_gates`) is reported beside it, and if the two disagree on
   any pass or fail call the gate fails.
3. Matched leakage within 0.1 of 0.5 on the trained reward, on pellets consumed, and on length
   and lifetime, on the retained pairs, from the same rows and with the same rule for
   disagreement as gate 2. On retained pairs length and lifetime are constant by construction
   (0.5), and under `--objective touch` the trained reward is the pellet count, so in `PB2` the
   first two channels are one.
4. Within-pair label-swap shuffled AUROC (`balanced.shuffled_within_pair` in each training
   cell) less than 0.1 from 0.5 on every drift-0.45 seed and arm.
5. `identical_first_state` true and `dropped_pairs` 0 on every drift-0.45 seed and arm, in the
   training cell and in the sensory-echo regeneration.
6. Every balanced target, every `resid_obs` and every gate value above is finite. A `WITHHELD`
   zone in the sensory-echo `aggregate.json` is a failure of gate 5, 6 or 7.
7. The sensory-echo integrity gate: every regenerated balanced target reproduces the one the
   training cell recorded within 5e-4.

"Within 0.1 of 0.5" in gates 3 and 4 means strictly less than 0.1 from 0.5, as
`scripts/promote_balanced_confirmation.py` computes it for `B1`; in the floor clause it means at
most 0.1, as `balanced_decision()` in `scripts/audit_sensory_echo.py` computes it.

The raw untrained matched floor (gate 4 of 2026-10-12) is not a gate here. It is reported, and
the 2026-10-12 route for the raw readout is reported beside the decision as a description.

Floor clause: untrained `resid_obs` mean within 0.1 of 0.5, read in each run's own
sensory-echo aggregate. The untrained arm does not depend on the objective, and in `B1` and
`B2` it was identical across the two runs because no warmup or readout episode died under death
on; the two `PB` untrained arms are expected to be identical for the same reason, and whether
they are is reported. If they are identical and the floor clause fails, both runs are
UNINFORMATIVE. If they differ, each run's floor clause is read on its own untrained arm. The
same holds for the untrained mean in gate 2.

Trained clause: trained `resid_obs` mean at least 0.65, above the untrained `resid_obs` mean
plus 0.05, and the trained t-based 90% lower bound above the untrained mean.

| Route | Condition |
|---|---|
| MET | Every gate holds, the floor clause holds, and the trained clause holds |
| NOT MET | Every gate holds, the floor clause holds, and the trained clause fails |
| UNINFORMATIVE | Any gate fails or the floor clause fails |
| NOT RUN | The run's 20 cells do not all exist, so no route is read |

NOT MET is reported with the 2026-10-13 zone name that `balanced_decision()` prints
(ATTENUATED or LARGELY SENSORY-MEDIATED), so the two vocabularies stay traceable.

`PB1` does not rescue `PB2` and `PB2` does not rescue `PB1`. Neither changes `B1`, `B2` or
`C1`.

## Pre-stated secondary readings (reported with their own labels, not the decision)

S1, strict interval reading: the same clauses read on intervals. Trained t90 lower bound at
least 0.65; seed-paired (by seed, from `cells.json`) trained minus untrained `resid_obs` t90
lower bound at least 0.05; the
untrained `resid_obs` t90 interval inside [0.40, 0.60]. Labels: STRICT MET, STRICT NOT MET.

S2, controls further from linear and from the last two inputs. For each of `resid_obs_mlp`
(nonlinear, current and previous observation), `resid_obs_int` (linear, current, previous and
the running mean of the observations up to and including the current one), `resid_obs_int_mlp` (the same integrated basis
through the in-fold 64-unit MLP regressor) and `resid_obs_beh_mlp` (nonlinear, instantaneous
observation plus the behavior trace with its running mean): trained mean at least 0.65 and
above the untrained mean under the same control plus 0.05 reads SURVIVES; otherwise a trained
mean at least 0.60 reads ATTENUATED; below 0.60 reads EXPLAINED. In balanced mode these labels
are computed by the promotion below from each metric's `mean` in `aggregate.json`. The same labels are computed on
`B1` and `B2` from their saved agents at base 910000, as descriptions.

S3, raw matched readout: survival or pellets balanced target, predictor and untrained, with
seed-paired margins, under the 2026-10-12 table (which will read UNINFORMATIVE if the raw
untrained floor is outside 0.1).

S4, second draw: the balanced target, `resid_obs` per arm, the floor and trained clauses and
the S2 quantities at base 3100000 (the matched gates are not rerun there), as a description, and the
per-seed mean over the two draws with its t-based 90% interval over seeds, as a description of
the reading with two evaluation draws per seed. It decides nothing.

S5, predictor: the predictor arm under the primary control, as in 2026-10-13.

## What each outcome changes in the text

- `PB2` MET: the manuscript says the controlled pellet reading without survival was confirmed
  on fresh seeds and a fresh draw under the later rule, frozen before those agents existed,
  and keeps every other qualifier (linear control as the primary, one surrogate, one trunk).
  The same for `PB1` and survival.
- NOT MET: the manuscript says the prospective test of the later rule did not confirm the
  controlled reading for that objective, and the `B1` or `B2` controlled reading is described
  as not replicated.
- UNINFORMATIVE: the manuscript says the prospective test could not adjudicate, names the
  failed gate or clause, and keeps the current wording otherwise.
- S1 and S2 are reported in the matched-world section whatever the primary route. A SURVIVES
  under `resid_obs_int_mlp` is described as a reading surviving that control, not as
  independence from all sensory history; an EXPLAINED is described as the signal being
  accounted for by that control.

No prediction about the outcome is stated.

## Code

- `itasorl/post_review.py`: `PB-primary` and `PB-draw2` added to `REGISTERED_LATER`.
- `scripts/audit_sensory_echo.py`: `--seeds` sets the expected balanced population (default
  10 to 19, the 2026-10-13 population), and `resid_obs_int_mlp` is computed under
  `--nonlinear`. The fresh-base read check, which already accepts any base outside
  `all_bases_in_use()`, is tested to accept `PB-draw2`.
- A promotion for `PB1` and `PB2` (`scripts/promote_prospective_balanced.py`: run folders,
  seeds 20 to 29, base 3000000) that computes gates 1 to 6 from the cells and
  `matched_gates.json` exactly as the gate list above states, reads gate 7, the floor clause and
  the trained clause from the sensory-echo `aggregate.json`, computes S1 to S5, and writes the
  route. Committed with tests before the first drift-0.45 cell exists.
- Tests for each change. Committed before any readout runs.

## Out of scope

Retraining `B1`, `B2` or `C1`; a different surrogate, trunk or algorithm; any change to the
2026-10-12 or 2026-10-13 rules or to their recorded results.
