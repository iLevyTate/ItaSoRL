# Corrected-trainer confirmation runs

Date: 2026-10-06
Status: frozen before the runs (no confirmation cell existed when this was committed)
Plan: `docs/REVISION_2026-10.md`, step 4. Correction: `docs/CORRECTIONS.md`, 2026-10-06.

## Purpose

Every published survival agent was trained with a GAE bootstrap that read the critic
value before an episode's final transition (`pre_transition_value` in
`docs/RESULTS_MANIFEST.md`). Correcting the algorithm does not tell us whether the
published effect survives. These runs measure it on the comparisons the revised paper
keeps as primary: survival with the next-observation auxiliary at the registered
300-update budget, survival without the auxiliary at 300 updates, and the predictor and
untrained baselines, with the authentic-law (L0) control and every registered gate.

## Why this machine gives a like-for-like comparison

The historical CPU runs `L3-H8-WM-CPU` (device control, 0.730) and `L3-H8-NOWM-CPU`
(no auxiliary, 0.601) ran on a 4-vCPU cloud sandbox with torch 2.14+cpu. This container
(4 vCPU, torch 2.14.1+cpu, numpy 2.4.6, scikit-learn 1.9.1) was checked first: the
original code at `4b6e1f3` reran cell drift 0.45, seed 0 of the device control and
reproduced all 156 recorded values of the committed cell exactly (survival target
0.66570, engagement return -0.65194). So the only difference between a corrected cell here
and its historical CPU cell is the bootstrap. The GPU-published numbers (0.752) come from
a different device and are compared only through the CPU device control.

## Runs (frozen)

Both runs use the L3 protocol of `PREREGISTRATION_L3.md` section 9 unchanged except for
the trainer: drift-mode l3, l3-hidden 8, G seed 0, drifts [0.0, 0.45], agent seeds 0..9,
updates 300, n_eps 16, max_steps 80, hidden 96, ray_steps 5, shaping 1.0, pooled n = 110
by 24 steps, matched pairs 60/20/24, Adam 3e-4, gamma 0.99, lambda 0.95, entropy 0.01,
value 0.5, decoder weight 1.0, `--gae-bootstrap successor` (the default), `--dump-states`,
`--save-agents`, `--device cpu`, 4 workers, `ITASORL_FOLDS=explicit`.

| Run | Flags beyond the protocol | Replaces, like for like |
|---|---|---|
| `C1` corrected, auxiliary on | `--budget-extend 450 --budget-snapshots 100 200` | `L3-H8-WM-CPU` |
| `C2` corrected, auxiliary off | `--no-world-model --budget-extend 450 --budget-snapshots 100 200` | `L3-H8-NOWM-CPU` |

The headline survival arm of each run is the 300-update agent. The budget flags keep
training the same survival agent to 450 updates and evaluate frozen copies at 100, 200
and 450 updates on the drift 0.45 cells (engagement and the pooled readout). A snapshot
after u updates is bit-identical to the agent a u-update run returns
(`tests/test_budget_curve.py`), so the extension cannot change the headline arm. The 450
point of `C2` is the corrected version of the "no auxiliary at 450 updates" budget
comparison; it is reported as a budget point, not as a skill match.

Order: `C1`, then `C2`. Outputs go to `fullruns/corrected_l3_h8_wm` and
`fullruns/corrected_l3_h8_nowm` (gitignored). Cells, results, and the behavior audit are
copied to `artifacts/corrected_runs/<run>/` and promoted with
`scripts/promote_reviewer_gaps_runs.py` to `artifacts/expB2/corrected_*.json`.

## Integrity checks (run before any comparison is read)

1. Every cell records `gae_bootstrap = successor`.
2. The predictor arm trains no actor-critic, so its pooled target must equal the
   historical CPU cell at the same (drift, seed) to the bit. The untrained arm likewise
   (with the decoder on in `C1`, off in `C2`, as in the historical runs). A mismatch stops
   the comparison: it would mean the stack, not the trainer, moved.
3. The fold partition is recorded (`itasorl.folds.class_counts` on a 110 + 110 pool).

## Decision rules (frozen; unchanged from the registered rules)

- **Primary verdict per run** (`decide_h_b2`, PREREGISTRATION_L3 sections 8 and 10): MET if
  every gate passes, the survival mean is at least 0.65 with the t-based 90% CI lower bound
  at least 0.65, and survival exceeds both the predictor and the untrained means by at
  least 0.05. A mean at or above 0.65 with the lower bound below it is inconclusive, not
  met.
- **Gates**: engagement in every cell; L0 equivalence on the ten drift-0 survival targets
  (TOST, margin 0.05, p < 0.05; ROPE [0.45, 0.55] reported beside it); speed control at
  least 0.75; pooled reward leakage within 0.1 of 0.5; untrained floor within 0.1 of 0.5;
  no deaths. A gate that fails or is not shown is reported as failed or open, and the run's
  verdict is then conditional on that gate.
- **Auxiliary comparison** (device-control rule of the 2026-09-27 addendum, same device):
  the auxiliary-conditional reading holds if `C1` meets the primary rule and the `C1` minus
  `C2` difference in survival means exceeds 0.05. The paired-by-seed difference and its
  t-based 90% CI are reported beside it.
- **Correction effect** (reported, not thresholded): `C1` minus `L3-H8-WM-CPU` and `C2`
  minus `L3-H8-NOWM-CPU`, survival pooled target and engagement return, paired by seed with
  t-based 90% CIs. Any change of verdict between the historical and the corrected run is
  reported in FINDINGS, the abstract-level documents, and `docs/CORRECTIONS.md`.

## Secondary and exploratory readouts

- Secondary: the per-timestep behavior control (`scripts/audit_behavior_mediation.py`,
  `resid_trace`) on each run's dumps.
- Exploratory: the budget curve (return, pooled target, environment steps at 100, 200,
  300, 450 updates) for revision step 11, and the readout-only follow-ups of steps 5 to 9
  on the saved corrected agents. None of these changes a primary verdict.

## What is not rerun

Experiments A, B, C, the oracle gates, and the predictor-only arms do not use the
actor-critic. The remaining survival-trained results (L2, B-v3, L1, second capacity,
held-out and cross-recipe transfer, second instances, the skill-matched baseline, the H2
batteries) stay historical and are labeled with the pre-correction trainer.
