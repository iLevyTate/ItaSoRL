# Balanced confirmation

Date: 2026-10-12
Status: frozen before any cell of `B1` or `B2` existed.
Code: the commit that adds this file. `ITASORL_FOLDS=explicit`.

## Purpose

The published pooled estimand draws the authentic pool from world seeds 800000 and the
surrogate pool from 850000, so a probe can read the world sample instead of the
dynamics. The pellet run `T-touch` gated leakage on the homeostatic reward, not on
pellets consumed. These two runs replace that estimand and that gate. They do not
retrain `C1` or `T-touch` in place. Seeds 10 to 19 have not been used for a survival
agent.

## What is held fixed

`B1` is the `C1` trainer. Drift mode `l3`, surrogate hidden 8, G seed 0, drifts 0 and
0.45, updates 300, 16 episodes per update, 80 steps, trunk 96, ray steps 5, shaping
1.0, pooled episodes 110 by 24 steps, matched pairs 60 with prefix 20 and branch 24.
`--gae-bootstrap successor`. `--adv-norm historical`, the standardization every
recorded run used. The masked alternative exists and is not used here, so `B1` stays a
replication of the corrected trainer under a different estimand. No budget extension.
The corrected-trainer spec records that a 300-update agent equals the 300-update
snapshot of a longer run.

`B2` is the `T-touch` protocol on the same trainer and the same seeds. `--objective
touch --mortal off`. Death is off in the trained arm's training world and in every
evaluation world. The predictor's scripted training episodes stay mortal. Touch
engagement requires the trained return to be at least 1.5 times the better baseline
and at least 0.25 pellets per episode (`TOUCH_ENGAGE_MIN_PELLETS`). One pellet in 64
episodes does not pass.

Both runs are one worker on CUDA, so `G` and the agents train on the same device.
`--save-agents --dump-states`. Each cell records a `stack` block (Python, NumPy,
SciPy, scikit-learn, torch, device). The workstation pin is
`constraints-workstation.txt`. The block in the cell is what the run used.

## Commands

```text
$env:ITASORL_FOLDS = "explicit"
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 10 11 12 13 14 15 16 17 18 19 --gae-bootstrap successor --adv-norm historical --balanced-seed-base 910000 --save-agents --dump-states fullruns/balanced_confirmation_wm/states --out-dir fullruns/balanced_confirmation_wm --device cuda --workers 1
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 --seeds 10 11 12 13 14 15 16 17 18 19 --objective touch --mortal off --gae-bootstrap successor --adv-norm historical --balanced-seed-base 910000 --save-agents --dump-states fullruns/balanced_confirmation_touch/states --out-dir fullruns/balanced_confirmation_touch --device cuda --workers 1
```

Order: `B1`, then `B2`. Outputs stay under `fullruns/` (gitignored) until promotion.

## Seed bases

Training for seed s is `100000 + 10000 s + 16 u`. Seeds 10 to 19 occupy 200000 through
about 298000, short of the Experiment C panels at 320000.

The primary worlds are base 910000, the same base for authentic and surrogate, 110
episodes. 900000 is the engagement metric's worlds (64 episodes) and is not used.
910000 clears every base in `itasorl.post_review.BASES_IN_USE`, including the weather
and ecology streams.

The pooled readout at 800000 and 850000 still runs. It is secondary, for continuity
with `C1`. It does not decide.

## Primary estimand

On each drift-0.45 cell, `paired_pooled_readout` at base 910000. Same world seeds,
both members of a pair in one CV group, a pair kept only if both survive. The number
is the survival arm's mean of fold AUROCs. The interval is the seed-level t-based 90%
interval over the ten seeds, and the seed-paired t-based 90% interval of each margin.

The registered rule adjudicated the bar and the margins on means and reported the
interval beside them. A mean can clear 0.65 while its interval includes values under
the bar. Both clauses here are the lower bound.

## Gates

All of these, on the balanced episodes, or the run is uninformative.

1. Engagement in every cell, under the rule for that run's objective.
2. Speed probe at least 0.75 in every pool.
3. Leakage within 0.1 of 0.5 on the trained reward, on pellets consumed, and on length
   and lifetime.
4. Untrained-arm balanced target within 0.1 of 0.5.
5. Within-pair label-swap shuffled AUROC within 0.1 of 0.5.
6. `identical_first_state` true on every pair.
7. `dropped_pairs` is 0.
8. Every primary target finite.

Under pairing the two members of a pair share the reset, so an authentic-versus-authentic
contrast at the same seeds is 0.5 by construction. That contrast is not a gate. The
pooled L0 at 800000 versus 850000 is reported beside the result and does not decide.

## Decision

| Route | Condition |
|---|---|
| MET | Every gate holds, survival t90 lower bound at least 0.65, and each margin's t90 lower bound at least 0.05 |
| NOT MET | Every gate holds, and the survival lower bound is under 0.65 or either margin's lower bound is under 0.05 |
| UNINFORMATIVE | Any gate fails, a target is non-finite, or a pair is dropped |
| not run | `B2` is not started if `B1` does not finish |

`B1` and `B2` are decided separately. `B2` does not rescue `B1`, and `B1` does not rescue `B2`.

## What a result does not change

The historical pooled numbers stay as recorded, including `C1` at 0.733 with its open
L0. `T-touch` at 0.796 stays as recorded, with the leakage-gate correction named. This
spec does not edit a committed number.

## Result (2026-10-13)

Both runs finished. The decision is UNINFORMATIVE. The speed and leakage gates
recorded on the cells are the independent pooled worlds at 800000 and 850000, not
the retained pairs at 910000. Engagement stays on its own objective-specific
protocol. The untrained matched-world mean is
0.614 [0.599, 0.628], 0.114 from 0.5, outside the 0.1 floor in gate 4. The same untrained
scores appear in `B1` and `B2`. Descriptively the trained arms clear the interval clauses:
`B1` survival 0.800 [0.772, 0.829], `B2` 0.843 [0.825, 0.860], with both margins' lower
bounds above 0.05. FINDINGS 21 records the gates and the numbers. The runner's MET line is
the pooled readout and is not this decision.
