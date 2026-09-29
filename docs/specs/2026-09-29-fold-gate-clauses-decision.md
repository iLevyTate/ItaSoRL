# Fold-split gate clauses: what to report

Date: 2026-09-29
Status: **DRAFT, awaiting the owner's call.** Nothing here is a new measurement. Every
number is re-read from `artifacts/fold_rescore/*.json` and the promoted run artifacts, and
the arithmetic is reproducible from those files alone.

Two pre-registered gate clauses flip when the published dumps are re-scored under the
stack-independent fold partition (FINDINGS section 16): the hidden 7 gate-0 untrained floor,
and the L0 authentic-vs-authentic equivalence test on the GPU-scored runs. FINDINGS records
both as open. This note argues for how to close them in the write-up, and recommends not
spending GPU time on either.

## 1. One fact explains both flips

The explicit partition is uniformly more optimistic than the legacy one. Re-scoring the
hidden 8 headline dumps (60 cells, `l3_h8_traces.json`) under both schemes moves every
drift/arm/metric mean in the same direction:

| drift | arm | metric | legacy | explicit | delta |
|---|---|---|---|---|---|
| 0.00 | predictor | `resid_trace` | 0.4838 | 0.5133 | +0.0294 |
| 0.00 | predictor | target | 0.4844 | 0.5075 | +0.0231 |
| 0.00 | survival | `resid_trace` | 0.5204 | 0.5498 | +0.0294 |
| 0.00 | survival | target | 0.5166 | 0.5393 | +0.0226 |
| 0.00 | untrained | `resid_trace` | 0.4736 | 0.4790 | +0.0054 |
| 0.00 | untrained | target | 0.4756 | 0.4805 | +0.0049 |
| 0.45 | predictor | `resid_trace` | 0.5745 | 0.6004 | +0.0260 |
| 0.45 | predictor | target | 0.5733 | 0.5883 | +0.0149 |
| 0.45 | survival | `resid_trace` | 0.7259 | 0.7504 | +0.0245 |
| 0.45 | survival | target | 0.7525 | 0.7739 | +0.0214 |
| 0.45 | untrained | `resid_trace` | 0.4982 | 0.5259 | +0.0277 |
| 0.45 | untrained | target | 0.4878 | 0.5130 | +0.0251 |

Twelve of twelve positive, mean +0.0212, range +0.005 to +0.029. The gate-0 floors move the
same way: hidden 8 from 0.482 to 0.508, hidden 7 from 0.566 to 0.615.

The shift lands on signal and null alike, so it is a property of the partition, not evidence
about any agent. The likely mechanism is class balance: the explicit scheme makes every fold
(22,22), while the legacy scheme on this stack leaves two folds at (21,23) and (23,21), and an
imbalanced fold pulls a pooled AUROC down slightly. The re-score does not isolate that
mechanism, so it is stated as the probable cause and not as a measured one. What is measured
is the direction and the size: up, on all twelve reads, by about two points.

## 2. No contrast moves

Because the shift is common, every difference the claims actually rest on survives it
(hidden 8, drift 0.45, pooled target unless noted):

| contrast | legacy | explicit | delta |
|---|---|---|---|
| survival minus the L0 floor (survival at drift 0) | 0.2358 | 0.2347 | -0.0012 |
| survival minus untrained | 0.2646 | 0.2610 | -0.0037 |
| survival minus predictor | 0.1792 | 0.1857 | +0.0065 |
| survival minus untrained, `resid_trace` | 0.2277 | 0.2245 | -0.0032 |

Absolute reads move about 0.021; contrasts move at most 0.007. Every gate that flipped is
written as a comparison against a fixed absolute number (the 0.1 floor tolerance, the
plus-or-minus 0.05 equivalence band). Every clause that held is a contrast. That is the whole
pattern.

## 3. Call A: the L0 equivalence clause

**What flipped.** On the ten drift-0 survival cells shared by every GPU-scored L3 run, the
pooled target reads 0.517 under legacy (TOST p = 0.010, equivalent) and 0.539 under explicit
(TOST p = 0.207, ROPE P = 0.816, not shown). The mean stays inside the equivalence band; the
ten-seed test cannot demonstrate it, because the upper 90% bound clears 0.55.

**More seeds will not rescue it.** Per-seed values are 0.6074, 0.5376, 0.4822, 0.5149,
0.5438, 0.5326, 0.4822, 0.5731, 0.5736, 0.5450 (mean 0.53926, sd 0.03968). Holding the mean
and sd, the one-sided TOST p against the 0.55 bound is:

| n | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| p (upper) | 0.207 | 0.120 | 0.074 | 0.047 | 0.031 | 0.020 |

Acceptance needs roughly 40 drift-0 seeds, so 30 more cells. At this GPU's observed rate,
about an hour per cell with two workers, that is on the order of 15 hours, and it buys
acceptance only if the mean does not drift up. The deficit is not small-sample bad luck; it
is that 0.539 sits 0.011 from a fixed bound.

**Recommendation: report, do not run.** State the clause as open, and state the mechanism
from section 1: the same +0.02 that lifts the null floor lifts the headline, and the margin
between them (0.235) is identical under both partitions. The claim rests on that margin, not
on a binary equivalence verdict. Do not spend the 15 hours.

## 4. Call B: the hidden 7 gate-0 floor

**What flipped.** Gate 0 accepts a capacity only if the untrained pooled target at drift 0.45
sits within 0.1 of 0.5. Hidden 7 reads 0.566 under legacy (pass) and 0.615 under explicit
(fail: three floor seeds 0.624, 0.610, 0.610). Hidden 8 passes under both. The oracle stays in
band either way (0.922, 0.918).

**The larger-sample estimate agrees that it is marginal.** The organism run's ten-seed
untrained arm at hidden 7 reads 0.5856 under legacy and 0.599959 under explicit, inside the
tolerance by 0.00004. So two independent estimates, one on three gate-0 seeds and one on ten
organism seeds, both land at the 0.6 line. This is not a measurement that more seeds will
move off the boundary; the hidden 7 fingerprint's mechanical floor simply sits there. The
2026-09 record already called it marginal (10.5: inside the tolerance, violated per seed).

**Recommendation: keep hidden 7 as the second in-band capacity, with the qualifier, and do
not demote it.** Three reasons. Its survival result is unaffected (0.740, `resid_trace` 0.725,
both above the bar under explicit). Its floor clause is decided by a nuisance variable, and
the same nuisance variable is what raises the headline from 0.752 to 0.774, so discounting
one means discounting the other. And demotion would remove a replication on the strength of a
0.015 excursion past a bright line, which reads as over-correction rather than rigor. What is
owed instead is the sentence FINDINGS 16 already carries plus the mechanism: under the
stack-independent partition this capacity's floor is at the tolerance, and a future hidden 7
organism run would need a spec to justify launching.

## 5. What "primary" should mean, and one inconsistency to fix

Both calls above reduce to one convention, which should be stated once and then held:

- **Published runs stay primary on the legacy partition.** It is what was pre-registered, and
  the re-score reproduces it bit-for-bit on the documented stack (0.752 / 0.726 at hidden 8,
  0.737 / 0.722 at hidden 7, integrity 30/30).
- **Every new run is scored on the explicit partition**, which is already the code default.
- **The re-score is the bridge**: it publishes the +0.02 shift and the invariance of contrasts
  so any two runs can be compared on either scale.

That convention exposes one inconsistency in the current text. FINDINGS 10.9's GPU re-measure
paragraph reports an explicit-scored 0.612 and then says "the gap to 0.752 therefore sits with
the instance." Its like-for-like comparator is the explicit 0.774, which widens the gap from
0.140 to 0.162. The correction runs against our own interest, which is the reason to make it.
The new-seed paragraph already compares correctly (0.676 against 0.774, paired -0.097).

Two ways to settle the comparators, not exclusive:

1. Quote 0.774 in that sentence. Free.
2. Re-score the two new GPU runs' dumps under legacy as well, so both instances can be quoted
   on the published scale. A 60-cell dump set re-scores in about 45 seconds of CPU
   (`q1_fold_rescore.log`, step 1), so this costs a couple of minutes and touches no GPU.
   Readout-only, no training, so it can run while the skill-matched baseline holds the card.

## 6. Recommended actions, each needing a yes

1. Add the section 1 shift table and the section 2 contrast table to FINDINGS 16, as the
   shared mechanism for both flips. No new runs.
2. Close the L0 clause as "open, and not worth 40 seeds", quoting the power table.
3. Keep hidden 7 as the second in-band capacity with the at-tolerance floor qualifier.
4. State the primary-partition convention in FINDINGS methods note 8 and in the paper's
   methods.
5. Fix the 0.752 comparator in the 10.9 GPU paragraph to 0.774, and optionally re-score the
   two new GPU runs under legacy (about two minutes of CPU).

Each of 1, 2, 3, 4 is a text change plus audit assertions. None needs GPU time. Item 5's
optional half is the only run, and it is readout-only.
