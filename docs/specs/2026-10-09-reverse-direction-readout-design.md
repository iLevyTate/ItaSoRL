# Reverse-direction readout: frozen design (2026-10-09)

Frozen before the run. Nothing here relaxes a registered margin, changes a gate, or alters the
registered estimand. This is a diagnostic readout, not a confirmatory run, and it carries no power
to overturn a registered verdict.

## The question

Every positive result in this project is in one direction. The agent is raised inside the learned
surrogate and the probe is then asked to separate the law it was raised under from the authentic law
it never lived in. An adversarial review of the manuscript (2026-10-09) made the point that the
paper's own framing, an agent that comes to carry the difference between the world and a copy of it,
describes the other direction, and that the other direction has never been measured.

So: does an agent raised in the **authentic** world carry the difference when probed against a
learned surrogate it never lived in?

This is not a symmetric restatement of the headline. In the reverse direction the surrogate pool is
drawn from dynamics the agent has never experienced, so nothing in training could have shaped the
policy toward the states where the two laws differ, which is the mechanism Section 6.5 of the
manuscript identifies. A reading at chance would therefore say the mechanism is as described. A
reading above the bar would say the state separates the two laws without the rearing, which would
widen the claim and which the paper does not currently make.

## What is scored

The ten saved **drift-0** agents of each arm (survival, predictor, untrained) of the historical
hidden-8 held-out run, `fullruns/l3_h8_heldout/agents/agent_d0.00_s*_{arm}.pt`. These are
**pre-correction trainer** agents. The corrected C1 agents are not in this repository and cannot be
scored (they were produced inside the revision container); this is the only in-band hidden-8 instance
with saved weights, the same constraint that governed the 2026-10-08 world-sample sensitivity run.

The result therefore measures the direction-dependence of this estimator on historical agents. It is
**not** a re-measurement of the corrected headline, and no wording derived from it may present it as
one.

## The readouts

Two, both already in the codebase and unchanged:

1. **Standard pooled readout** (`itasorl.experiment_b2.pooled_readout`) at `drift_sigma = 0.45`: the
   authentic pool from evaluation-world seed base 800000 at drift 0, the surrogate pool from 850000
   at drift 0.45, as in every other L3 cell. Per arm, the pooled target averaged over the ten agent
   seeds, with a t-based 90% interval over seeds.
2. **Balanced pooled readout** (`itasorl.experiment_b2.paired_pooled_readout`) at the same drift,
   which pairs each authentic episode with a surrogate episode from the same world seed and initial
   state. Reported because the standard readout confounds the dynamics with the evaluation-world
   draw, as FINDINGS 17.5 records.

The forward-direction comparison is the same run's recorded drift-0.45 cells (survival 0.752 under
the original partition), which are already committed. No forward cell is recomputed here.

## Integrity gate, checked before the result is read

Rescoring the same drift-0 agents at `drift_sigma = 0` must reproduce this run's recorded
explicit-partition drift-0 targets to within 0.01 on every seed and every arm
(`artifacts/fold_rescore/l3_h8_heldout.json`). These agents were trained on GPU and are rescored on
CPU. If the gate fails, the run is void and no number from it is reported.

## The decision rule, fixed in advance

Let `m` be the survival arm's mean pooled target at drift 0.45 on the drift-0 agents, `u` the
untrained arm's, and the bar 0.65 with the registered margin 0.05. The three outcomes are mutually
exclusive and jointly exhaustive:

- **R1, CARRIED.** `m >= 0.65` AND `m - u >= 0.05`. An agent that never lived in the surrogate
  separates it from the authentic world anyway, so rearing in the surrogate is not necessary for the
  state to carry the difference.
- **R2, NOT CARRIED.** `m < 0.65`. The state of an authentic-raised agent does not meet the registered
  encoding criterion against a surrogate it never lived in.
- **R3, AMBIGUOUS.** `m >= 0.65` AND `m - u < 0.05`. The level is reached but is not attributable to
  training, because the untrained arm is close behind.

## What is written in each case, committed in advance

In all three cases the limitations sentence added on 2026-10-09, which states that the reverse
direction is not tested at drift 0.45, is replaced by the measured result.

- **R1.** The abstract and the Discussion gain one sentence: the direction is not required, and the
  claim is restated as being about agents raised on either side of the gap. The novelty positioning
  against the reality-gap literature is revisited in the same edit, because a reading of this kind
  makes the result closer to generic out-of-distribution detection than the paper currently says.
- **R2.** The limitations sentence is replaced by the measured reading, and the Discussion gains one
  sentence: the mechanism is rearing-dependent, which is the direct evidence for the Section 6.5
  reading that what carries the signal is where the policy learned to take the body.
- **R3.** The limitations sentence is replaced by the measured reading and states plainly that the
  level is reached without the rearing but is not separated from the untrained floor, so the
  direction question is not resolved.

No other wording in the paper changes on the strength of this diagnostic, and in no case does it
alter a gate, a margin, or the registered verdict. The balanced readout is reported beside the
standard one in every case and is never substituted for it.

## Provenance

Output is a committed artifact under `artifacts/l0_audit/`, owned by a run registered in
`scripts/build_results_manifest.py`, with the per-seed values and the sha256 of each agent file
recorded, because the agents themselves are gitignored and not in the repository. Every number quoted
anywhere from this run comes from that artifact, and `scripts/audit_stats_recheck.py` binds the prose
to it.
