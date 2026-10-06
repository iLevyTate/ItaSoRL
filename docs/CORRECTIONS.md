# Corrections and provenance record

Dated entries, newest last. Each entry says what was wrong, which results it touches, what
was changed, and what the change does not establish. The plan these belong to is
`docs/REVISION_2026-10.md`; which implementation produced each committed number is
`docs/RESULTS_MANIFEST.md`.

Earlier corrections are recorded where they were made and are not repeated here: the GAE
padding fix (`679fee6`, `docs/PREREGISTRATION.md` 2026-06-28), the L3 world-params bug
(`docs/PREREGISTRATION_L3.md` 2026-07-10), the Experiment C estimator invalidation
(`docs/FINDINGS.md` 13.C), the matched-pair fold-safety fix (`docs/PREREGISTRATION_L3.md`
2026-07-22), and the explicit fold partition (`docs/FINDINGS.md` methods note 8 and section 16).

## 2026-10-06: results manifest

**What.** No single record said which code, configuration, seeds, fold partition, and device
produced each committed number, so a corrected result could not be told apart from a historical
one by reading the artifacts.

**Change.** `scripts/build_results_manifest.py` assigns every file under `artifacts/` to exactly
one run and records the run's commit, configuration, agent seeds, surrogate seed and capacity,
training budget, evaluation seed bases, fold partition, device, and dependency versions where
those were recorded. `--check` fails when a committed artifact belongs to no run or the
generated pages are stale. Every existing row is marked `historical`.

**Found while building it.** All survival-trained runs from `679fee6` (2026-06-28) to `4b6e1f3`
used the same `compute_gae` bootstrap, the one corrected in the next entry. The predictor arms,
the untrained arms, Experiment A, Experiment B, and Experiment C train no actor-critic and are
unaffected by it. Every readout-only analysis (H2, cross-recipe, sensory echo, observation
localization) reads saved survival agents and inherits the issue. Evaluation worlds are fixed
seed bases shared by every agent seed, and the pooled readout draws its authentic pool
(seed base 800000) and its surrogate pool (850000) from different world samples; both facts
bear on step 5 and step 13.

## 2026-10-06: GAE truncation bootstrap

**What was wrong.** `compute_gae` in `itasorl/experiment_b2.py` bootstrapped every episode
still alive at the 80-step rollout cutoff from `value[:, last]`, the critic value at the
episode's last stored step. That value belongs to the state before the final transition, so
the final residual was r_T + gamma V(h_T) - V(h_T), which is (gamma - 1) V(h_T) + r_T. Neither
treatment of the cutoff justifies it: a continuing task needs V of the successor state, and a
true terminal horizon needs 0. The test `test_compute_gae_truncation_bootstraps_last_value`
asserted the wrong value, so the suite protected the bug.

**Which cutoff this is.** A sampling truncation of a continuing task. `PatchOfEarthV0` has no
horizon (`age_max` is 1e9 and `StepResult.truncated` is always false), death is its only
terminal event, and nothing in the observation counts rollout steps. The corrected bootstrap
is therefore the successor value; deaths keep a zero bootstrap.

**Change.** The collector keeps, for each episode alive at the cutoff, the successor
observation and the final environment action. After `score_actions`, the trainer takes the
recurrent state after the final stored observation, runs one GRU step on the successor
observation with the final action as the previous action, and passes the critic's value of
that state to `compute_gae` through a now-required `bootstrap` argument. No gradient flows
through it and no environment step is taken. Normalization convention: the successor
observation is normalized with the normalizer as it stood for the episode's final stored
observation and is not added to the running statistics. The next-step mask that keeps the
GAE carry from crossing padded steps (`679fee6`) is unchanged.

**Results touched.** Every survival-trained run in `docs/RESULTS_MANIFEST.md` (trainer
`pre_transition_value`) and every readout-only analysis of those agents. The predictor and
untrained arms, Experiments A, B and C, and the oracle gates train no actor-critic and are
unaffected.

**Reproduction of the record.** `train_actor_critic(..., gae_bootstrap="pre_transition")` and
`run_expB2.py --gae-bootstrap pre_transition` keep the historical trainer. On a 12-update
check it produces agent weights and normalizer statistics bit-identical to the original
`4b6e1f3` code, and a reproduction run hashes to the same config fingerprint as the
published runs. The corrected default hashes apart, so corrected cells can never resume into
or mix with a historical run. Each new cell records `gae_bootstrap`.

**What this does not establish.** Whether the published effect sizes or verdicts survive.
That is measured in revision step 4; until then every historical number stays as published,
labeled historical.

## 2026-10-06: the GAE tests

**What was wrong.** `test_compute_gae_truncation_bootstraps_last_value` expected the
pre-transition bootstrap, so it would have failed on a correct implementation and passed on
the bug.

**Change.** `tests/test_gae_bootstrap.py` replaces it. It checks a one-step truncated episode
(reward 1, current value 0.3, successor value 2, gamma 0.99: residual **2.68**, where the old
rule gives 0.997); the same step ending in death, which must ignore any successor value; a
batch of mixed lengths and endings against an independent reference written from the
definition of GAE; padded slots holding 7, 1e30, plus or minus infinity, and NaN; a 12-step
truncated episode against the reference; and, on a real recurrent agent and collector, that
the stored successor observation and final action replay exactly, that the bootstrap equals
one GRU step plus the critic, and that changing only the successor observation changes the
bootstrap and the final advantage. Mutation checks: reverting to the current value, to a
padded slot, to multiplicative masking, or dropping the terminal zero each make the file fail.

**Hardening found by the tests.** Padded slots were masked by multiplying by zero, so an
infinite or NaN padded value would have turned a valid advantage into NaN. `compute_gae` now
masks by selection. On finite inputs the arithmetic is unchanged: the historical trainer is
still bit-identical to `4b6e1f3`, re-checked on a run with deaths and padding.
