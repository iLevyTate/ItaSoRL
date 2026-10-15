# Coherence or integration: a correlated-jitter sweep on the saved C1 agents

Frozen 2026-10-15, before any code for it was written or any number below was computed.
Readout-only on saved agents. It changes no registered rule or verdict.

## Why

On the corrected C1 agents, a temporally coherent hand-authored drag is read (fresh probe
0.754, margin over untrained +0.142) and white velocity jitter of the same one-step size is
not (0.488). The learned fingerprint, also coherent (lag-1 autocorrelation 0.83), is read.
Two explanations fit: the state keeps deviations that are coherent in time, or the state
integrates deviation and coherent deviations simply accumulate more. The record notes the
confound: by ten steps the drag's open-loop gap is 2.5 times the jitter's. This sweep
separates the two.

## The family

`G_ar`: the authentic world-P law plus a velocity perturbation that follows a stationary
AR(1) process per component, eta_t = rho * eta_{t-1} + sqrt(1 - rho^2) * sigma * e_t with
e_t standard normal, so the marginal standard deviation of each component is sigma at every
rho. At rho = 0 it is the white-jitter family G_gn. The process state is reset to a draw
from its stationary distribution when the family is reseeded at the start of each pool and
then runs on across episodes, so every episode starts stationary.

World P has velocity decay lambda = 1 - 1.5 * 0.05 = 0.925 per step, so the stationary
standard deviation of the accumulated velocity offset is
sigma * sqrt((1 + lambda * rho) / ((1 - lambda^2) * (1 - lambda * rho))).

## Two sweeps

rho in {0, 0.5, 0.8, 0.9, 0.95, 0.99}.

- Sweep A, matched one-step size: sigma = 0.01 per component at every rho (one-step RMS
  0.01414, the white jitter's).
- Sweep B, matched integrated size: sigma(rho) chosen so the stationary accumulated-offset
  standard deviation equals sweep A's at rho = 0.9. The rho = 0.9 point is shared.

Eleven family settings in all. For each, `scripts/run_texture_fresh_probe.py` with a new
family `ar` scores the drift-0.45 survival, predictor and untrained C1 agents
(`fullruns/corrected_l3_h8_wm/agents`, seeds 0 to 9) with 110 episodes of 24 steps: transfer
of the direction fit on the agent's own learned-law pools, a fresh grouped-CV probe on
authentic against G_ar pools, and the paired-pool fresh readout. Seed bases 2400000
(authentic) and 2450000 (G_ar), disjoint from every base in use. The family's measured
profile (one-step RMS, lag-1 autocorrelation, 24-step open-loop gap on the seed-77
calibration sample) is recorded beside each point.

## Readings, stated before any number

Primary quantity at each point: the seed-paired survival-minus-untrained margin of the fresh
probe (M), with the survival fresh mean beside it.

1. Coherence beyond integration: in sweep B, M at rho = 0.99 minus M at rho = 0, seed-paired,
   has a lower bound above 0.
2. Integration suffices: in sweep B that contrast includes 0 while in sweep A the same
   contrast has a lower bound above 0.
3. Neither: in sweep A the contrast includes 0.

The rho = 0 point of sweep A repeats the white-jitter comparator on new seed bases; its
fresh survival mean is reported beside the recorded 0.488. A point where the survival fresh
mean reaches 0.65 with M at least 0.05 is called kept, as in Table 21. Nothing here is a
registered verdict.

## Output

`artifacts/texture/ar_sweep/` with one JSON per point and `summary.json` holding the profile,
per-seed values, M by point, and the readings.
