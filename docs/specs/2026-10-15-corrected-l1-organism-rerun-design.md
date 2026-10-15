# Corrected-trainer rerun of the L1 organism run

Written 2026-10-15. Nothing in this file changes a registered rule. It names one run, its
recipe, the checks it must pass before it is read, and the frozen rule it is read under.

## Chronology, stated first

This run was launched before this file existed. It started at 2026-10-15 09:44:36Z from
`main` at `73da908`. This file was written after the launch and committed before any cell
file, any line of the run log past its configuration header, or any other output of the run
was read. Every report of this run says so: the reading rule was written down after launch
and before any result was seen. The rule below is the one the historical L1 run was already
read under, so no choice in it depends on this run.

## Why

The October trainer correction was rerun for the L3 instances (C1, C2, and the three
further instances of the 2026-10-11 spec). The L1 organism run (`L1-ORGANISM`, survival
0.533, rule not met) exists only under the pre-correction trainer. Its negative supports
the claim that an in-band seam a detector catches need not be encoded, and the measured
effect of the correction elsewhere (+0.003 on the L3 headline, +0.031 on the drift-zero
statistic) has not been tested at L1. This run tests it.

## Run

Local workstation, CPU, launched detached, output under `fullruns/corrected_l1_heldout`
(gitignored):

```
python scripts/run_expB2.py --drift-mode l1 --l1-delta 0.023 --sensor-sigma 0.01 \
  --drifts 0.0 0.023 --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 \
  --gae-bootstrap successor --workers 3 --device cpu --save-agents --resume \
  --dump-states fullruns/corrected_l1_heldout/states --out-dir fullruns/corrected_l1_heldout
```

Stack: the workstation interpreter (Python 3.13.2, the local NumPy 1.26.4 build of
Appendix E item 21, torch 2.7.0+cu126 run on CPU), with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. The historical run used four workers; this one uses three because free
host memory was 2.2 GB at launch. The worker count is not part of the configuration and
each cell is seeded on its own, so it does not change a cell.

## Recipe identity with the historical run

The historical cells carry fingerprint `250cd3eba1d482cf` and record commit `8a71593`. That
commit has no L1 drift mode. The run used the L1 code then in the working tree, which was
committed the next day as `01e88e4`. Checked before any output of this run was read
(`fullruns/corrected_l1_heldout/fingerprint_check.json`):

- Hashing this run's recipe with the fingerprint function and key set of `01e88e4` gives
  `250cd3eba1d482cf`, so every setting shared with the historical run is equal.
- Under the current function the historical recipe hashes to `41b0bdabce1f9dc8` and this
  run's to `d8bb95f746f95e4b`. The two differ in `gae_bootstrap` alone
  (`pre_transition` against `successor`), the trainer correction.

## Gate 0

Gate 0 at L1 is the grid oracle's calibration on the world (oracle 0.873 at
`delta = 0.023`, sensor noise 0.01). It involves no agent and no trainer, so it carries over
unchanged and is not re-measured.

## Checks before any number is read

1. All 20 cells exist, each records `gae_bootstrap = successor` and fingerprint
   `d8bb95f746f95e4b`.
2. The predictor and untrained arms train no actor-critic, so their cells should equal the
   historical cells to the bit on the same partition, as they did in every corrected L3
   rerun. Any difference is reported with its size before anything else.
3. State dumps from this run reproduce each cell's recorded pooled target.

## How it is read

The rule is H_B2 of `docs/PREREGISTRATION.md` sections 6 to 8, as it was applied to the
historical L1 run (rule ledger row "Primary rule (H_B2) at L1"):

- Encoding induced: every gate passes, and the survival pooled target mean is at least
  0.65 and exceeds the predictor and the untrained means by at least 0.05 each.
- Strengthened negative: every gate passes and survival, predictor and untrained all sit
  near 0.5.
- Uninformative: the engagement gate fails.

Gates, as the corrected runs read them: engagement (true return beats random and scripted
by 0.15, lifetime not worse than random by more than 2 steps) in every cell; the L0 gate
(drift-zero survival target equivalent to 0.5 by two one-sided tests, margin 0.05, level
0.05); the speed positive control at least 0.75 in every pool; leakage on summed reward,
episode length and lifetime each within 0.1 of 0.5; survivorship reported per pool. A
failed gate is reported as failed. An outcome the matrix does not name is reported as
such, as for C1.

The decision is read on the explicit (stack-independent) partition, as every corrected run
is; the legacy partition is reported beside it. The t-based 90% interval and the seed-paired
margins are reported and are not part of the rule. No bar, margin or tolerance changes.

The result is reported beside the historical row and labeled corrected; the historical row
keeps its label and numbers. If the corrected run meets the rule, the paper's L1 negative is
withdrawn as a corrected-trainer result and that is stated plainly; nothing is rescued in
either direction.

## Readouts on the saved agents

The stream and control readout of `docs/specs/2026-10-09-l1-stream-readout-design.md` is
run unchanged on this run's saved agents after the checks above, as a diagnostic beside
the historical one. It adjudicates nothing.

## Recording

Promoted to `artifacts/corrected_runs/corrected_l1_heldout/` with a manifest entry
`CORRECTED-L1-ORGANISM`, a rule ledger row, a FINDINGS note under section 14.7, and
manuscript rows beside the historical L1 numbers.
