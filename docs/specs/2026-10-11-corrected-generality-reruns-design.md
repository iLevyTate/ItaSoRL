# Corrected-trainer reruns of the generality runs, with saved agents

Frozen 2026-10-11, before any of these runs was launched. Nothing in this file changes a
registered rule. It names four runs, their recipes, the checks each must pass before it is
read, and the frozen rule each one is read under.

## Why

The October trainer correction was run on the registered instance only (C1 and C2). The
second capacity (hidden 7) and the two further fingerprint instances (hidden 10 at
surrogate seed 1, hidden 8 at surrogate seed 2) exist only under the pre-correction
trainer, and their L0 gate passes rest on the legacy fold partition. C2's agents were never
saved, so no readout that needs agents (the draw-free balanced readout, nonlinear probe
classes, normalizer-matched replay) can be run on it.

## Runs

All four run in cloud sessions through `scripts/cloud_run_step.sh` from `main`, on the
stack C1 used (torch 2.14.1 CPU wheel, numpy 2.4.6, scikit-learn 1.9.1), with that script's
fixed recipe: drift mode l3, drifts 0 and 0.45, seeds 0 to 9, 300 updates, four workers,
CPU, saved agents, resume, the explicit fold partition, the corrected trainer
(successor-state bootstrap) and the registered 96-unit trunk. The script passes
`--l3-hidden 8 --l3-seed 0` ahead of the extra flags, and a later flag overrides an earlier
one, so only the flags below differ.

| Run | Branch | Extra flags | Config hash | Read under |
|---|---|---|---|---|
| `C2-rerun` | `run/C2-rerun` | `--no-world-model --budget-extend 450 --budget-snapshots 100 200` | `aff3863123e3e3ee` | Integrity only (below). Its purpose is the saved agents. |
| `H7-corrected` | `run/H7-corrected` | `--l3-hidden 7` | `ce2a6fe1ba30cae7` | PREREGISTRATION_L3 sections 7 and 8, as the 2026-07-14 second-capacity entry of section 12 applied them, with every gate counted over every arm and the speed control over every pool (the 2026-10-10 entries), and the behavior audit's `resid_trace` reported beside the primary as that entry did. |
| `H10GS1-corrected` | `run/H10GS1-corrected` | `--l3-hidden 10 --l3-seed 1` | `03346353ac775154` | `docs/specs/2026-09-26-l3-second-fingerprint-instance-design.md`, "Gates and decision rules". |
| `H8GS2-corrected` | `run/H8GS2-corrected` | `--l3-seed 2` | `2823452491c11037` | `docs/specs/2026-09-28-l3-hidden8-second-seed-design.md`, "Gates and decision rules". |

The config hashes were computed from `config_fingerprint` in `scripts/run_expB2.py` before
launch. `C2-rerun`'s equals the hash on every committed C2 cell
(`artifacts/corrected_runs/corrected_l3_h8_nowm/cells`). With the historical trainer
(`--gae-bootstrap pre_transition`), the `H10GS1-corrected` recipe hashes to
`d027bac1ad45f30b`, the hash on every cell of the historical CPU run
(`artifacts/reviewer_gaps_runs/l3_h10_gseed1`), so that rerun differs from its original in
the trainer alone. The historical hidden 7 and hidden 8 seed 2 runs were GPU runs, whose
hash includes the device, so for them the recipe identity rests on the manifest's recorded
config (`docs/RESULTS_MANIFEST.md`, `L3-H7-N10` and `L3-H8-GS2-GPU`: drift mode l3 and
nothing else beyond the standard protocol).

## The surrogate and gate 0

`run_expB2` retrains each fingerprint from its frozen recipe when it starts; it does not
load a file. On CPU that fit matches the exported fingerprints in `artifacts/surrogates`
(their `index.json`). The original hidden 7 and hidden 8 seed 2 runs fit theirs on CUDA,
whose last float bits can differ, and their gate 0 was measured on that fit. So after each
run finishes, gate 0 is measured again on the fingerprint the run used, on the same pinned
stack and on one thread, as the run workers fit it: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/run_expA_l3.py --hiddens <h> --g-seed <s> --device cpu --json
<run>/gate0_cpu.json`. The frozen gate applies unchanged: oracle AUROC in [0.85, 0.95],
mechanical leakage clean, untrained floor within 0.1 of 0.5. A fingerprint that falls
outside the band here makes its run off-design under section 8; it is reported, not
rescued, and no other capacity or seed is substituted.

## Checks before any number is read

1. All 20 cells exist and every cell records `gae_bootstrap = successor`.
2. `C2-rerun`: every value in every cell equals the committed C2 cell to the bit, as the C1
   rerun matched C1. Any mismatch is reported and the rerun's agents are not used in place
   of C2's.
3. State dumps are rebuilt from the saved agents on the same stack with
   `scripts/regenerate_state_dumps.py --report`, and every rebuilt pool must reproduce its
   recorded pooled target. Only then does `scripts/audit_behavior_mediation.py` score
   `resid_trace`. The dumps (about 110 MB per run) are not committed; the rebuild report and
   the audit are.
4. Gate 0 as above.

## How each is read

- The pooled survival target, both margins and every gate are scored as for C1 with
  `scripts/promote_reviewer_gaps_runs.py`, which reads the speed positive control in every
  pool and names a gate-open zone. The L0 gate is read on the explicit partition. The
  registered verdict is the frozen rule's verdict; a failed gate is reported as failed.
- Each result is reported beside its historical row, labeled corrected; the historical rows
  keep their labels. No bar, margin or gate tolerance changes.
- The two further-instance specs and the 2026-07-14 entry use wording ("survival-specific",
  "behavior-independent") that the record withdrew on 2026-10-06. Their decision clauses
  apply as written. The reading is phrased in the corrected record's terms: a margin over
  the prediction-only arm is a difference between training regimes, not a difference of
  objective at matched input.
- Outputs land under `artifacts/goal_stakes/<run>` on each run branch, because the step
  script fixes that path. They are promoted to `artifacts/corrected_runs/` with a manifest
  entry when they are recorded.

## What this does not cover

Readouts on the saved agents of these runs and of the C1 rerun (nonlinear probe classes,
normalizer-matched replay, the policy deviation along visited states, the draw-free
balanced readout, the stream-decoder capacity curve) are frozen in a separate spec before
any of them runs.
