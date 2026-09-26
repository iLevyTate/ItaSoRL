# L3 sensory-echo control design

Date: 2026-09-26
Status: frozen before any run (readout-only against the saved
`fullruns/l3_h8_heldout` agents; no training anywhere)

## Purpose

FINDINGS methods note 7 states the one scope limit the behavior-mediation
control (10.4, 10.4.1) cannot close: the residualization basis is seven
per-timestep behavior scalars, not the ~146-dim observation, so
"behavior-independent" does not mean "sensory-echo-independent". A recurrent
state that passively mirrors world-dependent inputs (the vision rays under a
different velocity law) would survive the behavior control and read as a
world-identity representation. The untrained and predictor arms bound that
reading at hidden = 8 (0.498 and 0.574 under the same control) but do not
test it directly.

This probe tests it directly. It regresses the exact per-timestep input the
GRU receives (the normalized observation, all 146 dims) out of every h_t,
in-fold, and probes the residual. If the survival signal survives, the
"latent world representation" wording is earned. If it collapses, the paper's
wording changes to "state that echoes world-dependent inputs" before a
reviewer forces the change.

## The control logic, stated once

`h_t = GRU(h_{t-1}, x_t)` where `x_t = norm(obs_t)`. A linear function of
`[x_t, x_{t-1}, cummean(x)_{<=t}]` captures any state component that is a
linear readout of the current input, the previous input, or the running
input average. What survives is state that depends on the input HISTORY
nonlinearly or through integration the running mean does not capture. This is
strictly stronger than the behavior control (438 regressors against 21) and
is deliberately conservative: removing more than is causally available makes
any surviving signal harder to dismiss, exactly as in the 10.4 per-timestep
control.

Three readouts per (drift, seed, arm) cell:

- `obs_trace_only`: can the observation trace alone decode the world?
  (expected high; this is the ceiling, not a claim)
- `resid_obs`: the sensory control (PRIMARY)
- `resid_obs_beh`: sensory + behavior traces jointly (SECONDARY, the strongest
  control the project has; reported, not adjudicated)
- `resid_trace`: the published behavior control recomputed on the regenerated
  seven-channel traces (INTEGRITY: must reproduce 0.723 at hidden = 8)

## Gate (before any number is interpreted)

Integrity gate, same form as the H2 ablations (determinism check #8): the
regenerated pools at drift 0.45 and 0.00 must bit-match the saved
`states_d*_s*_{arm}.npz` `Ha`/`Hs` arrays for every cell, and the drift-0.45
survival pooled target must reproduce the published 0.752 (3 dp). Any
mismatch aborts. The run executes on CUDA because the saved bundles are
GPU-generated (the H2 A1 gate failed on CPU for that reason).

## Decision rules (frozen in advance)

Bar 0.65 and margin 0.05 as everywhere else. n = 10 seeds; t-based 90% CI
adjudicates, seed bootstrap reported.

- **Sensory-independent world representation:** survival `resid_obs` mean
  >= 0.65 AND > untrained `resid_obs` + 0.05, with the t-CI lower bound
  above the untrained mean. Methods note 7 is closed in the headline's favor.
- **Attenuated:** survival `resid_obs` in [0.60, 0.65): a below-bar trace
  survives; the paper reports "partly sensory-mediated" with the number.
- **Largely sensory-mediated:** survival `resid_obs` < 0.60: the paper's
  "behavior-independent world-signal" wording is restricted to
  "behavior-independent but input-mirroring", and the discussion changes.

Honesty checks (not rules): the untrained arm's `resid_obs` must sit near
chance (the control does not manufacture signal); the predictor arm is
reported.

## Reuse protocol (readout-only)

- Agents: `fullruns/l3_h8_heldout/agents/agent_d{0.00,0.45}_s{0..9}_{arm}.pt`
  via `load_agent_bundle`.
- Surrogate: `setup_l3_surrogate(hidden=8, seed=0, params=P)`, bit-identical
  to the organism run.
- Pools: `collect_pool` with the SAME seed bases as `pooled_readout`
  (800_000 authentic, 850_000 surrogate), n_eps = 110, steps = 24, plus the
  new `return_obs=True` which additionally returns the normalized observation
  trace (k, steps, obs_dim). Adding a returned array changes no existing
  number.
- Controls: `itasorl.behavior_audit.sensory_residual_probe_auroc` (new),
  ridge-regularized in-fold regression on the trace basis of
  `_trace_phi(O_t)`, GroupKFold at the episode level, same estimator family
  as the headline probe.

## CLI and code touch-points

- `itasorl/experiment_b2.py::collect_pool`: `return_obs: bool = False`
  appends the normalized observation trace to the returned tuple.
- `itasorl/behavior_audit.py`: `sensory_residual_probe_auroc(H, Ot, y,
  Bt=None, ...)`.
- `scripts/audit_sensory_echo.py`: runner with `--agents-dir --states-dir
  --out-dir --hidden --device --quick`; writes `cells.json` and
  `aggregate.json`.

## Testing (before launch)

- Synthetic: a world signal that reaches the state only through the
  observation trace is removed (no false positive); an orthogonal world
  direction survives (no over-removal); the joint control removes a
  behavior-mediated signal the sensory control alone leaves.
- `collect_pool(return_obs=True)` returns arrays of the right shape and its
  other outputs are unchanged (bit-equal to `return_obs=False`).
- ruff clean; the full suite passes.

## Run command

    python scripts/audit_sensory_echo.py \
        --agents-dir <repo>/fullruns/l3_h8_heldout/agents \
        --states-dir <repo>/fullruns/l3_h8_heldout/states \
        --out-dir <repo>/fullruns/l3_h8_sensory_echo --device cuda

## Out of scope

Nonlinear (MLP) residualization, hidden = 7, and any retraining. If the
primary rule fails, a nonlinear control is not run to rescue it.

## Amendment, 2026-09-26, before any run (synthetic ground truth)

The synthetic control-property tests written before launch showed that the
basis as first written above, `[x_t, x_{t-1}, cummean(x)_{<=t}]`, cannot
distinguish a passive echo from a genuine stored tag whenever the input
stream itself separates the worlds: the cumulative-mean column then linearly
encodes the world label, and the in-fold regression absorbs any persistent
component of `h_t`, genuine or echoed. Measured on the synthetic generator
(orthogonal tag of fixed size, one input channel shifted by 0.3 / 0.75 / 1.5
sd between worlds): the integrated basis leaves 1.000 / 0.964 / 0.797 of a
genuine tag, while the instantaneous basis `[x_t, x_{t-1}]` leaves
1.000 / 1.000 / 0.998 and still removes a pure echo (0.57 / 0.55 / 0.51).
In the real data the observation trace is expected to separate the worlds
strongly (it is the surrogate's own dynamics), so the integrated basis would
be over-strict by construction.

Resolution, frozen before the run: the PRIMARY control `resid_obs` uses the
instantaneous basis `[x_t, x_{t-1}]` (292 columns), which is exactly "state
that mirrors its current input". The integrated basis is reported as
`resid_obs_int` (SECONDARY diagnostic, no rule reads it). The joint control
`resid_obs_beh` uses the instantaneous observation basis plus the full
behavior trace basis. The decision rules above apply to `resid_obs` and are
otherwise unchanged. The runner had only been smoke-tested (`--quick`, seed 0,
tiny pools) before this amendment; no full-scale number existed.
