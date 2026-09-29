# L3 second fingerprint at fixed capacity (hidden 8, new G seed)

Date: 2026-09-28
Status: FROZEN 2026-09-29 on the owner's instruction ("run whatever is left"),
before any gate-0 or organism run, with one execution amendment recorded here.

Freeze-time amendment (2026-09-29). The run executes on the owner's GPU machine
(RTX 4050, torch 2.7.0+cu126, `--device cuda`, two parallel workers with
`--resume`; `--workers` is outside the config fingerprint), not the CPU cloud
sandbox, because the owner chose local execution. Consequences, fixed before
launch: (1) the direct same-device comparison partner is the published hidden 8,
seed 0 GPU run (0.752 under the legacy split; its explicit-split value comes from
the re-score in `docs/specs/2026-09-28-explicit-cv-folds.md`, run first), and the
CPU device control (0.730) becomes the secondary comparison; (2) the run is scored
under `ITASORL_FOLDS=explicit` as written below; (3) gate 0 also runs on CUDA;
(4) the local chain does not self-commit; results stay in `fullruns/` until the
owner promotes them. Gate-0 order, capacity, seeds, updates, and the decision rule
are unchanged.

## Purpose

FINDINGS 10.9 trained agents against a second, independently trained fingerprint
and got a survival-specific signal under the bar (0.639 [0.610, 0.669] against
predictor 0.534). The frozen fallback of that spec had moved capacity along with
the seed: hidden 8 at G seed 1 failed gate 0 on a leaky untrained floor (0.664),
so the instance ran at hidden 10. The miss therefore has two sources the design
could not separate, the instance and the capacity. The device control (10.8)
removes a third: on the same CPU sandbox the published hidden 8, seed 0 protocol
reads 0.730 [0.668, 0.791].

This run holds capacity at hidden 8 and changes only the G seed, on the same CPU
sandbox, so its result compares directly with 0.730.

## Gate 0 (before any organism run)

`scripts/run_expA_l3.py --g-seed <s> --hiddens 8 --floor-seeds 0 1 2` on world P
at the frozen sigma = 0.02. Pass requires oracle AUROC in [0.85, 0.95],
mechanical leakage clean, and an untrained floor with |target - 0.5| < 0.1.

Frozen order: G seeds 2, 3, 4 at hidden 8, first pass selected, every candidate
recorded in PREREGISTRATION_L3 section 12 before launch. Capacity never moves.
If none of the three passes, no organism run happens and the result is reported
as it stands: counting seed 1, hidden 8 fingerprints beyond seed 0 fail gate 0 in
4 of 4 seeds tried, which would say the seed-0 fingerprint's clean floor is the
exception at this capacity (10.9 already records one oracle-identical but leaky
instance).

## Organism run

Identical to the device control (published hidden 8 protocol, world-model
auxiliary on) except `--l3-seed <selected>`: drift-mode l3, l3-hidden 8, drifts
[0.0, 0.45], seeds 0..9, updates 300, `--dump-states`, `--save-agents`, `--device
cpu`, same cloud sandbox and worker count, explicit CV folds
(`docs/specs/2026-09-28-explicit-cv-folds.md`; the cloud stack already uses that
partition). Then the behavior audit on the dumps.

## Gates and decision rules

PREREGISTRATION_L3 sections 7 and 8, plus the rule of the 2026-09-26
second-instance spec: replication at hidden 8 is claimed only if the pooled
survival target AND `resid_trace` both reach 0.65 with t-CIs above the bar, AND
survival leads predictor and untrained by more than 0.05.

| Outcome | Reading |
|---|---|
| Replication claimed | The full-strength positive holds at two independently trained hidden 8 fingerprints. The 10.9 miss sits with that instance or with capacity 10, not with fingerprint instances in general. |
| Survival-specific (both margins pass) but under the bar | Same pattern as 10.9 at fixed capacity: the survival-specific signal replicates across instances, and its size is instance-dependent even at one capacity. The headline wording keeps "one fingerprint" for the full-strength result. |
| Margins fail | The hidden 7 pattern of 10.5 recurs on a new instance: survival-specificity itself is instance-conditional. |
| Engagement or gate failure | Uninformative, per section 8. |

Secondary, reported but not adjudicated: the paired difference against the
device control's per-seed values (same seeds, same device; the fingerprints
differ).

## Execution

`bash scripts/reviewer_gaps/run_cloud_h8_new_seed.sh` (draft, alongside this
spec). It runs gate 0 in the frozen order, appends the selection to
PREREGISTRATION_L3 section 12, runs the organism and the audit, copies the small
result files under `artifacts/reviewer_gaps_runs/`, and commits and pushes them
itself, as the device-control chain does. About 4 hours of CPU for the organism
run, plus about 8 minutes per gate-0 seed.

## Known limit

State dumps and saved agents stay in the sandbox's gitignored `fullruns/`, as
for every cloud run so far, so readout-only follow-ups on these agents need
retraining.

## Out of scope

Other capacities, other surrogate families, and cross-instance transfer between
the two trained populations.
