# Drift-0.45 world-sample sensitivity: frozen design (2026-10-08)

Frozen before the run. Nothing here relaxes a registered margin, changes a gate, or alters the
registered estimand. This is a diagnostic readout, not a confirmatory run, and it carries no
power to overturn a registered verdict.

## The question

The primary pooled readout draws its authentic pool from evaluation-world seed base 800000 and
its surrogate pool from 850000, so the two pools differ both in the velocity law and in the
draw of evaluation worlds. How much the draw alone contributes has been measured at drift zero
only: across eight independent pairs the drift-zero reading has a between-draw standard
deviation of 0.049, against a between-seed standard deviation of 0.017. The larger component is
therefore constant inside every agent seed and contributes nothing to any reported interval.

What has never been measured is the same sensitivity at drift 0.45, which is where the headline
is read. If the reading moves enough across draws that it would fall below the registered 0.65
bar on some draws, the level is a property of the draw as much as of the artifact.

## What is scored

The ten saved survival agents at drift 0.45 of the historical hidden-8 held-out run
(`fullruns/l3_h8_heldout/agents/agent_d0.45_s*_survival.pt`). These are **pre-correction
trainer** agents. The corrected C1 agents are not in the repository and cannot be scored; this
is the only in-band hidden-8 instance with saved weights.

The result therefore measures the structural sensitivity of this estimator to the
evaluation-world draw. It is **not** a re-measurement of the corrected headline, and no wording
derived from it may present it as one.

Nine draws are scored at drift 0.45: the eight independent pairs of `l0_audit.AUDIT_BASES`
(1000000 + 100000k paired with 1050000 + 100000k, k below 8), which are disjoint from the
registered pair, plus the registered pair (800000, 850000) as the reference column.

## The statistic

For each draw, the pooled survival target averaged over the ten agent seeds. The draw is the
unit of inference; nine per-draw means result.

**The means are signed and are never folded.** At drift zero the null is chance and a reading of
0.406 is as much authentic-versus-authentic separation as 0.594, so folding to an absolute
deviation is the conservative choice there and the drift-zero diagnostic uses it. At drift 0.45
there is a real, consistently signed effect, so folding would convert a low draw into a high
number and manufacture a pass. Folding is therefore prohibited here, and this prohibition is
part of the frozen design rather than a later analysis choice.

Reported alongside, none of them decisive: the between-draw standard deviation, a t-based 90%
interval over the eight independent draw means, the per-draw table, and the registered draw's
rank among the nine.

## Integrity gate, checked before the result is read

Rescoring the registered draw must reproduce the recorded per-seed explicit-partition targets
of this run to within 0.01 on every seed. These agents were trained on GPU and are rescored on
CPU, and the project has a recorded CPU-versus-GPU shift of about 0.022 for a full retrain; a
readout is a different operation, and this gate is what establishes that the readout itself
transfers. If the gate fails, the run is void and no number from it is reported.

## The decision rule, fixed in advance

Let `m_1 ... m_8` be the eight independent per-draw means, `m_reg` the registered draw's mean,
and the bar 0.65. Let the rank of `m_reg` among all nine means be 1 for the largest.

The three outcomes are mutually exclusive and jointly exhaustive:

- **R1, SECURE.** `min(m_1..m_8) >= 0.65` AND `rank(m_reg) >= 5`. Every independent draw clears
  the bar and the registered draw sits at or below the median of the nine, so the headline is
  not a product of a favourable draw.
- **R2, DRAW-DEPENDENT.** `min(m_1..m_8) < 0.65`. At least one draw puts the reading under the
  bar, so the level is draw-dependent.
- **R3, INDETERMINATE.** `min(m_1..m_8) >= 0.65` AND `rank(m_reg) <= 4`. Every draw clears the
  bar, but the registered draw sits above the median, so the claim stands while the registered
  draw is on the favourable side of the distribution.

## What is written in each case, committed in advance

In all three cases the limitations sentence stating that the eight-pair rescoring was run at
drift zero only, and that the sensitivity of the headline reading is therefore not measured
directly, is replaced by the measured range.

- **R1.** The abstract is unchanged. The limitations gain the measured range and the statement
  that the registered draw is not favourable.
- **R2.** The abstract gains the across-draw range, and the primary claim is stated as
  conditional on the registered draw. The limitations gain the range and name the draws that
  fall below the bar.
- **R3.** The abstract is unchanged. The limitations gain the range and state plainly that the
  registered draw sits above the median of the nine.

No other wording in the paper changes on the strength of this diagnostic, and in no case does
it alter a gate, a margin, or the registered verdict.

## Provenance

Output is a committed artifact under `artifacts/l0_audit/`, owned by a run registered in
`scripts/build_results_manifest.py`, with the per-seed values and the sha256 of each agent file
recorded, because the agents themselves are gitignored and not in the repository. Every number
quoted anywhere from this run comes from that artifact, and `scripts/audit_stats_recheck.py`
binds the prose to it.
