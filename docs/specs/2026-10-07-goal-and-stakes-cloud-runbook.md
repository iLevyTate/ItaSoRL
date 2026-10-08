# Goal and stakes: cloud runbook

Date: 2026-10-07. Spec: `docs/specs/2026-10-07-goal-and-stakes-design.md`.

## Per run

1. Merge `feat/goal-and-stakes` to main first (the cloud session starts from the repo).
2. Create the scheduled routine (claude.ai/code routines, `/schedule` in Claude Code) with
   the prompt below, interval 2 hours, repository ITASORL, branch main.
3. The routine calls `scripts/cloud_run_step.sh <name> 100 <flags>` and exits. The step
   script owns the branch `run/<name>`, the lock, and the commits.
4. When `artifacts/goal_stakes/<name>/DONE` exists, delete the routine, run the readouts
   (step 5), open the PR.
5. Readouts, in a cloud session on the run branch:
   `python scripts/run_mechanism_readouts.py --run-dir artifacts/goal_stakes/<name> --out artifacts/goal_stakes/<name>/mechanism.json --workers 4`
   `python scripts/run_policy_controlled_readouts.py --run-dir artifacts/goal_stakes/<name> --out artifacts/goal_stakes/<name>/policy_controls.json --workers 4`
   `python scripts/run_control_diagnostics.py --run-dir artifacts/goal_stakes/<name> --out artifacts/goal_stakes/<name>/control_diagnostics.json --workers 4`
   Commit the three JSONs. Then `python scripts/decide_goal_stakes.py --runs artifacts/goal_stakes --c1 artifacts/expB2/corrected_l3_h8_wm.json`.

| Run | flags |
|---|---|
| T-touch | `--objective touch --mortal off` |
| S-immortal | `--mortal off` |
| S-scarce | `--n_pellets <chosen>` after `python scripts/calibrate_scarcity.py --out artifacts/goal_stakes/calibration.json` and the dated PREREGISTRATION_L3 amendment recording the chosen value |

Order: T-touch, S-immortal, S-scarce. The C1 mechanism readout runs once, in any session:
`python scripts/run_mechanism_readouts.py --run-dir artifacts/corrected_runs/corrected_l3_h8_wm --out artifacts/goal_stakes/C1/mechanism.json --workers 4`.

## Routine prompt (verbatim)

    You are continuing the ITASORL goal-and-stakes run named <NAME>. Run exactly:
    bash scripts/cloud_run_step.sh <NAME> 100 <FLAGS>
    Report the last five lines of its output and its exit code. Exit code 0 means all
    cells exist: say DONE. Exit code 2 means another step is live: do nothing else.
    Exit code 3 means cells remain: do nothing else. Do not edit any file, do not run
    any other script, do not open a pull request.

## Integrity before reading

Every cell under `cells/` must record `gae_bootstrap = successor`, the run's `objective`
and `mortal`; in T-touch and S-immortal the predictor and untrained pooled targets must
equal `C1`'s cells at the same (drift, seed) to the bit
(`python - <<EOF` comparing `artifacts/corrected_runs/corrected_l3_h8_wm/cells` with the
run's cells on `agents.predictor.pool.target` and `agents.untrained.pool.target`).
