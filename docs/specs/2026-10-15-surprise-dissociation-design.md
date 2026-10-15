# Does the agent's world model detect what its state keeps? Matched-input surprise readout

Frozen 2026-10-15, before any code for it was written or any number below was computed.
Readout-only on saved agents. It changes no registered rule or verdict.

## Why

The goal-and-stakes mechanism readout (Appendix I.5) found that the survival agent's own
next-observation prediction error separates the worlds at AUROC 0.846 on C1, near the
oracle's 0.928, while its state reads 0.733. That suggests a dissociation: the agent's model
detects the fake, its state keeps less, and its behavior does not use it. The same artifact
shows the predictor arm's error at about 0.48 and the untrained arm's random decoder at 0.64
to 0.68. Each arm was scored on episodes its own policy drove, and the survival walk visits
states where the two laws differ more. So the existing numbers cannot say whether detection
comes from the survival agent's model or from where its policy goes.

## Design

Agents: the saved C1 agents (`fullruns/corrected_l3_h8_wm/agents`, corrected trainer,
hidden-8 fingerprint), seeds 0 to 9: the drift-0.45 survival, predictor and untrained arms,
and the drift-0 survival arm (raised in the authentic world) as a sign control.

Streams, matched input: the drift-0.45 survival agent's own episodes, recorded with
`eval_protocols.record_trajectories` on the registered pools (110 episodes of 24 steps;
authentic at drift 0, seed base 800000; surrogate at drift 0.45, seed base 850000; ray steps
5). Each arm reads the same recorded observations and actions open loop
(`replay_states`), normalized with its own frozen normalizer. Secondary protocol: the
scripted policy streams of `eval_protocols` (one action sequence for every arm and world).

Per arm and seed, on identical input:

- S, detection by the model: `mechanism_readouts.surprise_auroc` on per-episode summaries
  (mean, max, slope) of the squared next-observation error of the arm's decoder, the same
  function the I.5 readout used. The target at step t is the recorded observation at t+1.
- R, retention by the state: the registered pooled probe on the same replayed states
  (`readout_from_states`).
- Sign: log of mean authentic error over mean surrogate error. Positive means the model is
  more surprised by the authentic world.

L0 analogue: the drift-0.45 survival decoder scored on its own episodes in two authentic
pools (seed bases 800000 and 850000, both drift 0). Its surprise AUROC should sit within
0.1 of 0.5; if it does not, every S here is reported as carrying the world-sample draw.

L1: the same protocol on the corrected L1 agents (`fullruns/corrected_l1_heldout/agents`,
drift 0.023), as a secondary block.

## Readings, stated before any number

All contrasts are seed-paired with t-based 90% intervals, n = 10.

1. The model detects beyond the input floor: S(survival) minus S(untrained) has a lower
   bound above 0.05.
2. Detection exceeds retention: S(survival) minus R(survival) has a lower bound above 0.
   An upper bound below 0 reads the reverse.
3. Prediction training alone gives detection: S(predictor) minus S(untrained) has a lower
   bound above 0.05.
4. The error tracks the law the agent was raised under: the sign is positive for the
   drift-0.45 survival arm and negative for the drift-0 survival arm, each in at least 8 of
   10 seeds.

Each reading is reported as met, not met, or reversed. None is a registered verdict. If
reading 1 is not met, the dissociation is not claimed. Scripted-stream results are reported
beside the matched-input ones and do not decide a reading.

## Output

`artifacts/surprise/corrected_l3_h8_wm.json` and `artifacts/surprise/corrected_l1_heldout.json`,
each with provenance, per-seed values, and the four readings.
