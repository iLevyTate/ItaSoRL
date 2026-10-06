# Primary analysis implementation and the L0 gate

Date: 2026-10-06
Status: frozen before any corrected confirmation cell was read
Plan: `docs/REVISION_2026-10.md`, step 5.

## The decision

**Primary implementation for every result produced from this revision on:** the standard
pooled readout of `itasorl/experiment_b2.py` (110 authentic and 110 surrogate episodes of
24 steps, `[mean h, final h]` features, standardized logistic probe, 5-fold grouped CV)
scored on the **explicit-v1** partition (`itasorl.folds`, version string `explicit-v1`). The
partition of the standard designs is serialized in `artifacts/folds/explicit_v1.json`; a test
regenerates it and fails on any difference, so a change of rule must bump the version.

This is the partition already fixed for new runs by the 2026-09-30 comparator convention
(`docs/specs/2026-09-29-fold-gate-clauses-decision.md`). It is chosen because it does not
depend on the installed scikit-learn or numpy, not because of what it does to any number.
It is frozen here before the corrected runs of step 4 are read, and the corrected runs will
be reported on it whatever they show. On the confirmation container the legacy scikit-learn
partition has the same digest as explicit-v1 (`5213ed1294bb...` for 110 + 110), so the
corrected runs have one partition, not two.

**Historical results keep the partition they were published on.** `docs/GATE_TABLE.md` lists
the gates under both partitions wherever the dumps allow, and the comparator convention
still holds: both sides of a comparison come from the same partition.

**Pairs stay together.** Every readout whose design produces pairs (matched pairs, common
garden, the balanced pooled readout below) gives both members one group id, and
`folds.partition_record(...)["groups_intact"]` records that the partition kept them in one
fold. The standard pooled readout has no pairs: its two pools are different world samples.

## L0: what the gate measures

At drift 0 both pools are authentic. The L0 target is the probe's AUROC for telling the
authentic pool (world seeds 800000 + i) from the "surrogate" pool (850000 + i). Two facts
from the audit shape how it is reported.

1. **Pre-intervention decodability is at chance for the standard pair.** From the reset
   observation alone, before any dynamics act, the standard probe reads the standard pair at
   0.453 (out-of-fold 95% interval [0.382, 0.532]; `artifacts/l0_audit/pre_intervention.json`).
   The two world samples are not separable before the intervention.
2. **One world-sample pair is one draw.** Across eight independent pairs of 110 + 110 reset
   observations the same probe reads 0.419 to 0.612 (mean 0.492, sd 0.066). The sd is larger
   than the 0.05 equivalence margin. Every agent seed of a run is scored on the same pair, so
   the ten seeds share that draw, and a TOST across seeds tests equivalence conditional on it.
   It says nothing about variation over world samples.

The agent-based version (`scripts/run_l0_audit.py`) rescoring each drift-0 survival agent on
independent world-sample pairs, and decoding pool membership from h_1, runs on the saved
agents of the corrected run `C1`. The historical agents are not in this repository.

## How the gate is reported

- The registered rule is unchanged: TOST, margin 0.05, alpha 0.05, on the per-seed drift-0
  survival targets of the standard pair. **The margin is not relaxed.** A row that does not
  show equivalence is reported as inconclusive (`docs/GATE_TABLE.md`), and a verdict that
  needs the gate is stated as conditional on it.
- Every L0 statement says that it conditions on the fixed evaluation-world pair, and quotes
  the across-pair spread where it has been measured.
- The ROPE leg is a bootstrap containment share, not a posterior probability (step 13).

## Balanced initial-state sampling

`itasorl.l0_audit.paired_pooled_readout` draws both pools from the same world seeds, keeps a
pair only if both members survive, groups both members together, and resamples whole pairs
for its interval. At drift 0 its twins are identical and its target is exactly 0.5, so it adds
no signal of its own (`tests/test_l0_audit.py`); at drift > 0 any separation comes from the
dynamics. It is run on the corrected agents as a **secondary** readout beside the frozen
primary. It does not replace the primary, because switching estimators after the frozen
protocol would be choosing the analysis by its outcome.
