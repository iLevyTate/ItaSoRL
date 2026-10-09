# L1 stream and control readout: frozen design (2026-10-09)

Frozen before the run. Nothing here relaxes a registered margin, changes a gate, or alters the
registered estimand. This is a diagnostic readout on saved agents, not a confirmatory run.

## The question

The L1 negative is a negative **on the state**. Agents raised under the observation-quantization
seam do not meet the registered encoding criterion (survival 0.533, t-based 90% CI [0.506, 0.561],
0 of 10 seeds above the bar), while the oracle reads the same seam at 0.873, in band under its own
grid oracle at the world's sensor noise.

Nobody has ever asked what the L1 **stream** carries. Every stream-only decoder and behavior-trace
readout in this project was scored on the corrected L3 agents. So two very different explanations
of the L1 negative are currently indistinguishable:

1. The seam does not change where the policy goes, so the agent's own observation stream carries
   little about the world and there is nothing for the trunk to keep.
2. The stream carries the seam perfectly well and the trunk does not keep it.

Only the second is an encoding failure in the sense the paper's title claims. The first would mean
the L1 rung is a weak artifact for this agent rather than a case of detectable-but-not-encoded, and
the sufficiency argument would have to lean entirely on the white-jitter case.

## What is scored

The sixty saved agents of the historical L1 held-out run, `fullruns/l1_heldout/agents/`
(`drift_mode = l1`, grid spacing `delta = 0.023`, sensor noise `sigma_o = 0.01`, ten seeds, three
arms, drifts 0 and 0.023). These are **pre-correction trainer** agents and the result is labelled
historical wherever it is reported.

The readout is the existing control battery (`scripts/run_control_diagnostics.py`), run at the L1
rung instead of L3. Nothing about the battery changes; the only new code is the three module
globals that select the rung (`DRIFT_MODE`, `L1_DELTA`, `SENSOR_SIGMA`), exposed as arguments with
defaults that preserve the current L3 behaviour exactly.

## What is read, and what each number answers

Per arm, averaged over the ten seeds with a t-based 90% interval:

- **state probe** (`target`): reproduces the published L1 reading, and is the integrity check.
- **observation-summary decoder** (`obs_summary_only`) and **supervised GRU on the stream**
  (`seq_gru`): what a decoder reads from the agent's own observation and action stream with no
  access to its state. This is the number the question turns on.
- **behavior trace** and the residual controls: what remains of the state after each named basis is
  regressed out, as in the L3 battery, with the held-out R-squared of each basis.

## Integrity gate, checked before the result is read

The state probe at drift 0.023 must reproduce the L1 run's recorded per-seed targets to within
0.01 on every seed and arm, and where the run saved state dumps the regenerated pools must bit-match
them. If the gate fails, the run is void and no number from it is reported.

## The reading, fixed in advance

Let `s` be the survival arm's state probe and `d` the larger of its two stream decoders.

- **S1, THE STREAM CARRIES IT.** `d >= 0.65`. The seam reaches the agent's own stream at a level the
  state does not reflect, so the L1 negative is an encoding failure in the sense the title claims:
  the information was in what the agent saw, and the trunk did not keep it. The paper says so, and
  the L1 case becomes a stronger leg of the sufficiency argument than it is today.
- **S2, THE STREAM DOES NOT CARRY IT.** `d < 0.65`. The seam does not reach the agent's stream at a
  readable level, so the L1 negative does not distinguish "not encoded" from "not expressed in
  behaviour". The paper says that plainly, the L1 case is demoted in the sufficiency argument, and
  the claim leans on the white-jitter comparator-trained run instead.
- In both cases the measured stream numbers are reported beside the state number wherever the L1
  negative appears, and the limitation added on 2026-10-09 stating that the stream was scored on the
  corrected L3 agents only is replaced by the measurement.

No gate, margin, or registered verdict changes on the strength of this readout, and the L1 run's
own adjudication (criterion not met) stands as recorded whatever the stream reads.

## Provenance

Output is a committed artifact under `artifacts/control_diagnostics/`, owned by a run registered in
`scripts/build_results_manifest.py`, with per-seed values recorded. Every number quoted anywhere
from it comes from that artifact, and `scripts/audit_stats_recheck.py` binds the prose to it.
