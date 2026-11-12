# Draw-crossed readout of the saved C1 and C2 agents: frozen design (2026-11-12, UTC)

Frozen before any fresh draw is scored. This is a readout of saved agents. It trains no agent,
relaxes no registered margin, changes no gate and re-adjudicates no stored run. The registered
verdicts of `C1` and `C2`, the auxiliary rule and every row of the rule ledger stay as recorded,
whatever this readout shows.

## The question

The registered pooled readout scores every agent seed on one shared pair of evaluation-world
pools (seed bases 800000 and 850000, 110 episodes of 24 steps each). Two properties of that
design limit what its intervals can say.

1. **One shared draw.** An interval over the ten agent seeds is conditional on that one draw.
   The record has measured the draw component twice on the `C1` survival arm: across eight
   further draws the between-draw standard deviation is 0.049 at drift zero and 0.052 at drift
   0.45, against between-seed standard deviations on the registered draw of 0.017 and about
   0.11. At drift zero the draw component is as large as the 0.05 margin of the L0 gate. The baseline arms have never
   been scored on a second draw, so no lead over a baseline has an interval that carries the
   draw.
2. **An untuned probe penalty.** The probe has 192 feature columns and 176 training episodes
   per fold, at the registered penalty C = 1. A later measurement chose C inside each training
   fold and moved the `C1` drift-zero reading on the registered draw from 0.559 to 0.519.

This readout scores every arm on fresh draws with both probes and summarizes each reading over
agent seeds and draws together. It asks three things of the `C1` and `C2` agents:

- the level of the survival arm's reading at drift 0.45, across draws;
- the authentic-versus-authentic reading at drift zero, across draws;
- the survival arm's lead over each baseline arm, on the arms' own walks and at matched input.

## What is scored

The saved agents of the `C1` and `C2` reruns, whose cells equal `C1`'s and `C2`'s to the bit
(`artifacts/goal_stakes/C1-rerun/agents` on the origin branch `run/C1-rerun`, and
`artifacts/goal_stakes/C2-rerun/agents` on `run/C2-rerun`). For each run, the 60 agents named
`agent_d{0.00,0.45}_s{0..9}_{survival,predictor,untrained}.pt` are scored. The budget snapshots
(`_u100`, `_u200`, `_u450`) are not. A cell is one run, drift and agent seed, with its three
arms. At drift 0.45 the cell's agents were raised in the surrogate and the second pool is the
surrogate. At drift zero they were raised in the authentic world and both pools are authentic.

The surrogate is the registered one (hidden 8, G seed 0, world P). Each pool has 110 attempted
episodes of 24 steps with ray step 5, and an episode that ends in death is dropped, as in the
registered readout. The number of episodes kept is recorded per pool.

Seven feature sets are scored in every cell and draw. Each is the registered feature
`[mean h, final h]` of a pair of pools.

- `own/survival`, `own/predictor`, `own/untrained`: each arm's own actor drives its episodes,
  as in the registered readout.
- `replay/predictor`, `replay/untrained`: the survival agent's own episodes on that draw are
  recorded and fed open loop to the other arm's core network, which normalizes them with its
  own frozen normalizer. This is the replay protocol of the record
  (`scripts/run_policy_controlled_readouts.py`).
- `replay_snorm/predictor`, `replay_snorm/untrained`: the same replay with the survival
  agent's normalizer, so that every core network reads the identical normalized stream.

## The draws

Eight fresh pairs of evaluation-world seed bases, `(6000000 + 100000k, 6050000 + 100000k)`
for k from 0 to 7. The first base of a pair seeds the authentic pool and the second seeds the
pool at the cell's drift. No evaluation-world seed base in the code is at or above 3200000, so
these pairs are disjoint from every draw that has been read. The only other seed in that range
is a jitter generator seed near 5000000 in `scripts/run_expB2.py`, which is not a world seed.

The registered pair (800000, 850000) is scored as well. It carries the integrity gate below and
is reported beside the fresh draws. It enters no crossed interval, because it has been read.
The eight audit pairs of `l0_audit.AUDIT_BASES` are not rescored. Their survival readings are
already in `artifacts/l0_audit/` and may be quoted beside this readout as descriptive.

## The probes

Both probes use the explicit fold partition, five folds, with the episode as the group, and
the mean of the per-fold AUROCs, as registered.

- **Penalty chosen in fold (primary).** `nested_auroc` of
  `scripts/run_probe_penalty_sensitivity.py`: inside each outer training fold a grouped inner
  five-fold cross-validation picks C from {0.001, 0.01, 0.1, 1, 10} by the largest mean inner
  AUROC, ties to the smaller C, and the probe is refit at that C and scored on the outer test
  fold.
- **Registered penalty, C = 1.** `cv_auroc` of the same script at C = 1, which is the
  registered probe bit for bit.

The labels below are read on the primary probe. The same summaries and labels are computed for
the registered penalty and reported beside them. If the two probes give different labels, both
are reported with equal prominence.

## The statistic

For each run, drift, feature set and probe, the readings form a matrix of 10 agent seeds by 8
fresh draws. **Readings are signed and are never folded about chance**, at either drift.

The summary is the grand mean with a t-based 90% interval that treats agent seeds and draws as
crossed random factors with one reading per cell. With S seeds and D draws,

- `MS_seed = D * sum_s (mean_s - mean)^2 / (S - 1)`,
- `MS_draw = S * sum_k (mean_k - mean)^2 / (D - 1)`,
- `MS_resid = sum (reading - mean_s - mean_k + mean)^2 / ((S - 1)(D - 1))`,

the variance of the grand mean is estimated by `(MS_seed + MS_draw - MS_resid) / (S * D)`, and
the degrees of freedom are Satterthwaite's for that combination. The interval is approximate.
When the variance estimate is not positive, the degrees of freedom are below one, or any
reading in the matrix is missing or not finite, the interval is **not computable**, and it is
reported as such. A missing interval is never read as a favorable one.

A contrast is the same summary of a matrix of differences, paired by agent seed and by draw:

- the survival arm minus the predictor arm, and minus the untrained arm, on their own walks;
- the survival arm minus the predictor core network, and minus the untrained core network,
  on the survival agent's stream (`replay`), and the same with the survival agent's normalizer
  (`replay_snorm`);
- the `C1` survival arm minus the `C2` survival arm on their own walks, paired by agent seed
  index. The two runs were trained apart, as the record notes for the registered auxiliary
  comparison.

Reported beside each summary, none of them decisive: the eight per-draw means, the ten
per-seed means, the standard deviation of each, the three mean squares, and the registered
draw's per-seed readings and mean.

## Integrity gate, checked before any label is read

On the registered pair, the registered probe (C = 1) on each arm's own walk must reproduce the
per-seed target recorded for that agent in `artifacts/corrected_verdicts.json` (the `primary`
block at drift 0.45 and the `drift_zero` block at drift zero) to within 0.01, for all 60
agents of a run. A run with any agent missing or outside that tolerance is void: every label of
that run reads `VOID (integrity gate failed)`, and the contrast between the runs is void if
either run is. The tolerance is the record's standing one for a readout replayed on another
stack.

## The labels, fixed in advance

Each label states where a crossed 90% interval lies against a registered reference value. The
reference values are the 0.65 bar, the 0.05 margin and the equivalence band [0.45, 0.55]. A
label is a description of an interval. It is not a verdict, it meets or fails no rule, and it
adds no row to the rule ledger.

**Level** (the survival arm on its own walk at drift 0.45, per run):

- `above the bar`: the lower bound is at or above 0.65;
- `below the bar`: the upper bound is below 0.65;
- `straddles the bar`: otherwise.

**Authentic against authentic** (every feature set at drift zero, per run):

- `inside the band`: the interval lies within [0.45, 0.55];
- `outside the band`: the interval lies wholly below 0.45 or wholly above 0.55;
- `straddles the band edge`: otherwise.

**Contrast** (every contrast at drift 0.45), tested in this order:

- `within the margin`: the lower bound is above -0.05 and the upper bound is below +0.05;
- `above the margin`: the lower bound is at or above +0.05;
- `below the margin`: the upper bound is below +0.05;
- `straddles the margin`: otherwise.

The baseline arms carry no level label at drift 0.45, and contrasts carry no label at drift
zero. Their summaries are reported all the same.

## What is reported in each case

Every summary and every label is reported, whichever way it falls, in FINDINGS and in the
artifact. No draw, seed, feature set or probe is dropped after the fact. In particular:

- if the level label of `C1` is not `above the bar`, the record says that the level of the
  `C1` reading is not shown above the bar across draws;
- if the drift-zero label of the `C1` survival arm is not `inside the band`, the record says
  that its authentic-versus-authentic reading is not shown to be at chance across draws;
- if a matched-input contrast is `above the margin`, the record says that the survival core
  network leads that core network on an identical stream by more than the margin, which the
  registered-draw replay did not show for the predictor.

None of these outcomes changes a registered verdict. `C1`'s registered rule was evaluated on
the registered draw with the registered probe, and it is not met there because its L0 gate
does not pass. This readout does not reopen that evaluation.

## What this readout does not do

- It does not score the `predictor-logged` arm of the record's replay table, which needs a
  retraining per seed. The predictor here is the standard predictor arm.
- It does not test a second surrogate, core network or world. Every reading is conditional on
  the one surrogate (hidden 8, G seed 0) and on the saved agents.
- It does not match walks by speed or separate speed from goal direction.
- It estimates the spread over draws from eight draws, so the crossed interval is itself
  uncertain in its width.

## Stack and provenance

The readout runs on CPU with one torch thread per worker, in a container with Python 3.13,
torch 2.14.1, numpy 2.4.6, scipy 1.17.1 and scikit-learn 1.9.1. The `C1` and `C2` runs used
Python 3.11.17 with the same library versions, which is why the integrity gate is part of the
design. A development pass on this stack scored only the registered draw for agent seed 0,
which is data the record has already read, and reproduced the twelve recorded targets to
within 0.000001.

The output is one committed artifact, `artifacts/draw_crossed/corrected_c1_c2.json`, owned by
a run registered in `scripts/build_results_manifest.py`. It holds every per-cell reading, the
number of episodes kept per pool, the penalties chosen in fold, the sha256 of each agent file,
the integrity gate and the summaries. `scripts/audit_stats_recheck.py` recomputes the summaries
and labels from the per-cell readings. The agent files stay on their run branches and are not
committed anywhere else.

Run with
`python scripts/run_draw_crossed_readout.py --c1-agents <C1 agents> --c2-agents <C2 agents> --checkpoint-dir fullruns/draw_crossed/cells --out artifacts/draw_crossed/corrected_c1_c2.json --workers 4`.
