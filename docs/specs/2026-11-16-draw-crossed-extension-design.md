# Extension of the draw-crossed readout: the C1 control on fifty-six further draws, and the reading net of its control: frozen design (2026-11-16, UTC)

Frozen before any further draw is scored. This is a readout of saved agents. It trains no agent,
relaxes no registered margin, changes no gate and re-adjudicates no stored run. The registered
verdicts of `C1` and `C2`, the auxiliary rule, every row of the rule ledger and every value of
FINDINGS 25 stay as recorded, whatever this readout shows.

## The question

FINDINGS 25 scored the saved `C1` and `C2` agents on eight fresh draws of evaluation worlds
(`docs/specs/2026-11-12-draw-crossed-readout-design.md`). Two of its readings on the `C1`
survival arm leave questions that more draws can answer and more agent seeds cannot.

1. **The control is not resolved.** With the authentic physics in both pools the `C1` survival
   arm reads 0.530 [0.495, 0.566] across the eight draws, and its label is `straddles the band
   edge`. The standard deviation of the eight draw means is 0.052, against 0.022 for the ten
   seed means. So the width of that interval is set by the number of draws. Eight draws cannot
   say whether the reading lies inside the registered band [0.45, 0.55] or sits above it.
2. **The record has no single number for the reading net of its control.** The reading at
   drift 0.45 (0.776 across the eight draws) and the control at drift zero (0.530) are reported
   side by side. Their difference, paired by agent seed and by draw, has never been computed.

This extension scores `C1` on fifty-six further draws and asks three things:

- where the survival arm's authentic-versus-authentic reading lies, across draws;
- how far the survival arm's reading at drift 0.45 lies above that control, paired by agent
  seed and by draw;
- whether the level at drift 0.45 and the leads over the two baseline arms, on the arms' own
  walks, read on these draws as they did on the first eight.

## What is scored

The saved agents of the `C1` rerun, whose cells equal `C1`'s to the bit
(`artifacts/goal_stakes/C1-rerun/agents` on the origin branch `run/C1-rerun`): the 60 agents
named `agent_d{0.00,0.45}_s{0..9}_{survival,predictor,untrained}.pt`. `C2` is not scored. A
cell is one drift and agent seed, with its three arms. At drift 0.45 the cell's agents were
raised in the surrogate and the second pool is the surrogate. At drift zero they were raised in
the authentic world and both pools are authentic.

The surrogate, the pool size and the episode rule are those of the first readout: the
registered surrogate (hidden 8, G seed 0, world P), 110 attempted episodes of 24 steps with ray
step 5 per pool, an episode that ends in death dropped, and the number kept recorded per pool.

Three feature sets are scored in every cell and draw, each the registered feature
`[mean h, final h]` of a pair of pools: `own/survival`, `own/predictor` and `own/untrained`.
Each arm's own actor drives its episodes, as in the registered readout. The replayed feature
sets of the first readout are not scored. The matched-input contrasts they carry are not the
question here.

## The draws

Fifty-six further pairs of evaluation-world seed bases, `(6000000 + 100000k, 6050000 +
100000k)` for k from 8 to 63. The first base of a pair seeds the authentic pool and the second
seeds the pool at the cell's drift. They continue the sequence of the first readout, whose
eight pairs are k from 0 to 7, and no evaluation-world seed base in the code lies at or above
6800000, so these pairs are disjoint from every draw that has been read.

**Why fifty-six.** The number is fixed here, from FINDINGS 25 alone. With a standard deviation
of 0.0524 between draw means and 0.0222 between seed means at drift zero, fifty-six is the
smallest number of draws for which the draw component of the standard error, 0.0524 divided by
the square root of the number of draws, is no larger than the seed component, 0.0222 divided by
the square root of ten. Beyond that point further draws buy little, because ten agent seeds
are all the record has.

The registered pair (800000, 850000) is scored as well. It carries the integrity gate below and
enters no interval.

## The probes

As in the first readout, and unchanged: the explicit fold partition, five folds, the episode as
the group, the mean of the per-fold AUROCs.

- **Penalty chosen in fold (primary).** `nested_auroc` of
  `scripts/run_probe_penalty_sensitivity.py`.
- **Registered penalty, C = 1.** `cv_auroc` of the same script at C = 1.

The labels below are read on the primary probe. The same summaries and labels are computed for
the registered penalty and reported beside them. If the two probes give different labels, both
are reported with equal prominence.

## The statistic

For each drift, feature set and probe, the readings form a matrix of 10 agent seeds by 56
further draws. **Readings are signed and are never folded about chance.** The summary is the
grand mean with the t-based 90% interval of the first readout, which treats agent seeds and
draws as crossed random factors with one reading per cell
(`itasorl.draw_crossed.crossed_interval`, unchanged). An interval whose variance estimate is not
positive, whose degrees of freedom are below one, or whose matrix has a missing or non-finite
reading is **not computable** and is reported as such.

A contrast is the same summary of a matrix of differences, paired by agent seed and by draw:

- the survival arm minus the predictor arm, and minus the untrained arm, on their own walks at
  drift 0.45, as in the first readout;
- **the reading net of its control**: an arm's reading at drift 0.45 minus the same arm's
  reading at drift zero, for the same agent seed index and the same pair of seed bases. The two
  readings come from two agents, one raised in the surrogate and one raised in the authentic
  world, which share a seed index and were trained apart. The first pool of the pair is seeded
  alike in both; the second pool has the same seed base and differs in its physics. It is
  computed for each arm and labelled for the survival arm.

Reported beside each summary, none of them decisive: the per-draw means, the per-seed means,
the standard deviation of each, the three mean squares, and the registered draw's per-seed
readings and mean.

### Three sets of draws, and which one the labels are read on

- **The fifty-six further draws.** Every summary above. This is the prospective set: no reading
  on these draws exists when this design is frozen. **Every label is read here and nowhere
  else.**
- **The eight draws of 2026-11-12.** Only the reading net of its control, which FINDINGS 25
  did not compute. It is post hoc on those draws, since both of its terms had been read before
  this design, and it is reported as post hoc and carries no label. Its readings are taken
  from `artifacts/draw_crossed/corrected_c1_c2.json` and nothing is rescored.
- **All sixty-four fresh draws together.** The same summaries on the matrix of 10 seeds by 64
  draws, reported as descriptive and without labels, because eight of its columns had been
  read.

## Integrity gate, checked before any label is read

On the registered pair, the registered probe (C = 1) on each arm's own walk must reproduce the
per-seed target recorded for that agent in `artifacts/corrected_verdicts.json` (the `primary`
block at drift 0.45 and the `drift_zero` block at drift zero) to within 0.01, for all 60 agents.
If any agent is missing or outside that tolerance the readout is void: every label reads
`VOID (integrity gate failed)`.

## The labels, fixed in advance

Each label states where a crossed 90% interval lies against a reference value. It is a
description of an interval. It is not a verdict, it meets or fails no rule, and it adds no row
to the rule ledger. The first three are the labels of the first readout, unchanged.

**Level** (the survival arm on its own walk at drift 0.45): `above the bar` when the lower bound
is at or above 0.65; `below the bar` when the upper bound is below 0.65; `straddles the bar`
otherwise.

**Authentic against authentic** (each arm at drift zero): `inside the band` when the interval
lies within [0.45, 0.55]; `outside the band` when it lies wholly below 0.45 or wholly above
0.55; `straddles the band edge` otherwise.

**Lead over a baseline arm** (the two own-walk contrasts at drift 0.45), tested in this order:
`within the margin` when the lower bound is above -0.05 and the upper bound is below +0.05;
`above the margin` when the lower bound is at or above +0.05; `below the margin` when the upper
bound is below +0.05; `straddles the margin` otherwise.

**Net of its control** (the survival arm), tested in this order. The reference value 0.15 is
the distance of the 0.65 bar from chance.

- `at least the bar's distance from chance`: the lower bound is at or above 0.15;
- `above its control`: the lower bound is above zero;
- `not shown above its control`: otherwise.

The baseline arms carry no level label at drift 0.45 and no label on their reading net of its
control. Their summaries are reported all the same.

## What is reported in each case

Every summary and every label is reported, whichever way it falls, in FINDINGS and in the
artifact. No draw, seed, feature set or probe is dropped after the fact. In particular:

- if the drift-zero label of the survival arm is `inside the band`, the record says that across
  fifty-six further draws its authentic-versus-authentic reading lies within the registered
  band, and it also says whether that interval includes 0.5;
- if that label is `straddles the band edge` or `outside the band`, the record says that the
  reading is still not shown to lie within the band, or that it lies outside it;
- if the level label is not `above the bar`, the record says that the level of the `C1`
  reading is not shown above the bar on these draws;
- the label of the reading net of its control is stated as it falls, with its interval.

None of these outcomes changes a registered verdict. `C1`'s registered rule was evaluated on
the registered draw with the registered probe, and it is not met there because its L0 gate
does not pass. That gate is a test on the registered draw. A reading inside the band on other
draws does not pass it, and this readout does not reopen it.

## What this readout does not do

- It does not score `C2`, and it does not score the replayed feature sets, so it says nothing
  new about the auxiliary or about matched input.
- It does not test a second surrogate, core network or world. Every reading is conditional on
  the one surrogate (hidden 8, G seed 0) and on the saved agents.
- It adds draws, not agent seeds. The seed component of every interval stays that of ten seeds.
- It does not match walks by speed or separate speed from goal direction.

## Stack and provenance

The readout runs on CPU with one torch thread per worker, in a container with Python 3.13,
torch 2.14.1, numpy 2.4.6, scipy 1.17.1 and scikit-learn 1.9.1, the stack of the first readout.
A development pass on this stack scored only the registered pair, which is data the record has
already read, to check the gate and the run script. The gate passed on all 60 agents, and all
120 own-walk readings on that pair equal the first readout's to the bit. The same pass printed
the reading net of its control on the eight earlier draws, from the first readout's artifact
(the survival arm, +0.246 [+0.197, +0.295] with the penalty chosen in fold). That is the post
hoc summary named above, and it is why it carries no label. No further draw has been scored.

The output is one committed artifact, `artifacts/draw_crossed/c1_extension.json`, owned by a run
registered in `scripts/build_results_manifest.py`. It holds every per-cell reading, the number
of episodes kept per pool, the penalties chosen in fold, the sha256 of each agent file, the
integrity gate, the summaries on the three sets of draws, and the sha256 of the first readout's
artifact, from which the eight earlier draws are read. `scripts/audit_stats_recheck.py`
recomputes the summaries and labels from the per-cell readings. The agent files stay on their
run branch and are not committed anywhere else.

Run with
`python scripts/run_draw_crossed_extension.py --c1-agents <C1 agents> --checkpoint-dir fullruns/draw_crossed_extension/cells --out artifacts/draw_crossed/c1_extension.json --workers 4`.
