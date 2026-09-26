# L3 second fingerprint instance design

Date: 2026-09-26
Status: frozen before any run

## Purpose

FINDINGS methods note 4: the L3 fingerprint is a single frozen instance,
`G_motion` trained once at seed 0. The held-out probes (10.6, 10.7) show the
world-identity DIRECTION transfers to other fingerprints, but no agent has
been trained against a second instance, so "the headline is instance-specific"
remains open. This run trains a second `G` at hidden = 8 with a different
seed (different authentic training rollouts AND different initialization),
recalibrates gate 0 on it, and repeats the frozen organism protocol.

## Gate 0 (before the organism run)

`scripts/run_expA_l3.py --g-seed 1 --hiddens 8` on world P at the frozen
sigma = 0.02. Pass requires oracle AUROC in [0.85, 0.95], mechanical leakage
clean, and an untrained floor with |target - 0.5| < 0.1. Fallback, frozen
here: if hidden = 8 at seed 1 is out of band or the floor is dirty, sweep
hiddens {7, 9, 10} at seed 1 in that order and freeze the FIRST that passes;
if none passes, sweep seed 2 at hidden 8 then {7, 9, 10}; record every
candidate. No organism run until a candidate passes; the selected (seed,
hidden) is logged in PREREGISTRATION_L3 section 12 before launch.

## Hypothesis

- **H_inst (primary):** at the selected instance, the survival pooled target
  clears the bar and both baselines, and the per-timestep behavior control
  leaves a behavior-independent signal >= 0.65: the L3 positive replicates
  across fingerprint instances.
- **Null:** the survival target falls below the bar at a gate-passing
  instance: the published positive is instance-conditional and the paper says
  so.

## Configuration (frozen)

Identical to the hidden = 8 n = 10 run except `--l3-seed <selected>` (and
`--l3-hidden <selected>` if the fallback moved capacity): seeds 0..9,
updates 300, `--dump-states`, `--save-agents`. The fingerprint changes.

## Gates and decision rules

PREREGISTRATION_L3 sections 7 and 8 verbatim (encoding induced /
strengthened negative / uninformative), plus the behavior-audit zones of the
2026-07-12 spec on `resid_trace`. Replication is claimed only if BOTH the
pooled target and `resid_trace` clear 0.65 with t-CIs excluding the bar.

## CLI and code touch-points

- `scripts/run_expA_l3.py`: `--g-seed` (default 0) passed to
  `train_g_motion`, recorded in the JSON.
- `scripts/run_expB2.py`: `--l3-seed` (default 0) passed to
  `setup_l3_surrogate`; joins the fingerprinted base config. The held-out
  surrogate, if enabled, keeps seed 0.

## Testing (before launch)

- `config_fingerprint` differs when `l3_seed` differs.
- `train_g_motion(seed=1)` produces a net whose weights differ from seed 0
  and whose outputs differ on a fixed input (the seed is live).
- ruff clean; suite passes.

## Run commands

    python scripts/run_expA_l3.py --g-seed 1 --hiddens 8 --floor-seeds 0 1 2 \
        --json <repo>/fullruns/l3_gate0_seed1/calibration.json --device cuda
    # after gate 0 passes and the selection is logged:
    python scripts/run_expB2.py --drift-mode l3 --l3-hidden <h> --l3-seed <s> \
        --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 --save-agents \
        --dump-states <repo>/fullruns/l3_h<h>_gseed<s>/states \
        --out-dir <repo>/fullruns/l3_h<h>_gseed<s> --device cuda
    python scripts/audit_behavior_mediation.py <repo>/fullruns/l3_h<h>_gseed<s>/states \
        --json <repo>/fullruns/l3_h<h>_gseed<s>/behavior_audit.json

## Out of scope

Cross-instance transfer probes between the two trained agent populations
(a natural follow-on, not part of this freeze).
