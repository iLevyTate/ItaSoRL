# L3 architecture baseline design (no world-model auxiliary)

Date: 2026-09-26
Status: frozen before the run

## Purpose

Every published result rides on one trunk: a 64-unit embedding into a
96-unit GRU, with the survival arm carrying a next-observation decoder as an
auxiliary loss (`wm_coef = 1.0`). FINDINGS methods note 6 and the design
document's "add at least one architecture baseline before publication" both
name this as the open external-validity gap. The specific reviewer question
is: does the L3 positive depend on the decoder auxiliary, which is itself a
prediction objective? If the survival agent encodes world identity only
because a world-model loss rides along, the "survival objective, uniquely"
wording overclaims.

This run repeats the frozen L3 hidden = 8 protocol with the decoder removed
from the survival arm (`world_model=False`, a model-free recurrent A2C on the
identical GRU trunk). The untrained arm is built with the same flag so the
mechanical floor matches the trunk under test. The predictor arm is unchanged
(it IS the decoder objective and stays as the prediction-only comparator).

## Hypothesis

- **H_arch (primary):** the model-free survival agent's pooled world-identity
  target at L3 hidden = 8 clears the pre-registered bar and both baselines:
  the L3 positive does not depend on the world-model auxiliary.
- **Null:** the model-free survival target falls below the bar while the
  published decoder-carrying survival target (0.752) stands: the positive is
  conditional on the auxiliary prediction loss, and the paper says so.

Either outcome is reportable.

## Configuration (frozen)

Identical to PREREGISTRATION_L3 section 9 and the hidden = 8 n = 10 run,
except `--no-world-model`: drift-mode l3, l3-hidden 8, G seed 0, drifts
[0.0, 0.45], seeds 0..9, updates 300, n_eps 16, max_steps 80, hidden 96,
ray_steps 5, shaping 1.0, pooled n = 110 steps = 24, matched-pair 60/20/24,
Adam 3e-4, gamma 0.99, lambda 0.95, ent 0.01, vf 0.5. `--dump-states` and
`--save-agents` on. The config fingerprint changes (the flag is
science-relevant) so no cell can mix with the published run.

## Gates (all must pass; identical to PREREGISTRATION_L3 section 7)

Gate 0 is inherited (same G, same world; oracle 0.928, floor re-measured by
this run's untrained arm and must satisfy |target - 0.5| < 0.1).
Engagement, L0 equivalence (TOST/ROPE at n = 10), speed positive control
>= 0.75, leakage within 0.1 of 0.5.

## Decision rules (frozen)

- **Positive survives without the auxiliary:** all gates pass AND survival
  pooled target >= 0.65 (t-CI excludes the bar) AND > predictor + 0.05 AND >
  untrained + 0.05.
- **Positive is auxiliary-conditional:** all gates pass AND survival < 0.65.
  Reported as a scope narrowing of the L3 positive, not a retraction.
- **Uninformative:** engagement fails (a model-free forager that does not
  learn cannot test the question).

Secondary: the per-timestep behavior audit (`resid_trace`, seven-channel
basis) is run on the dumps and reported against the 0.65 bar with the same
strengthen / weaken / mediated zones as the 2026-07-12 spec.

## CLI and code touch-points

- `scripts/run_expB2.py`: `--no-world-model` flag; `run_cell` passes
  `world_model` to `train_actor_critic` and `untrained_agent`; the flag joins
  the fingerprinted base config.

## Testing (before launch)

- `config_fingerprint` differs when `world_model` differs.
- `run_cell` builds the survival and untrained agents with
  `world_model=False` under the flag (unit test on a tiny config).
- ruff clean; suite passes.

## Run command

    python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --no-world-model \
        --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 --save-agents \
        --dump-states <repo>/fullruns/l3_h8_nowm/states \
        --out-dir <repo>/fullruns/l3_h8_nowm --device cuda
    python scripts/audit_behavior_mediation.py <repo>/fullruns/l3_h8_nowm/states \
        --json <repo>/fullruns/l3_h8_nowm/behavior_audit.json

## Out of scope

Other cores (LSTM), other widths, and Dreamer-style imagination. One baseline,
one flag.
