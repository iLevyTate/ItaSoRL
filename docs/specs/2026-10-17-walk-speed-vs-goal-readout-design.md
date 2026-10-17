# Walk speed against goal direction: frozen design (2026-10-17)

Frozen 2026-10-17, before the readout ran on any agent or any number below was computed.
Readout-only on saved agents. It is post hoc, descriptive, and changes no stored value or
verdict. It narrows C1 only. This spec refers to the manuscript's title as "the title" and never
spells it; no outcome below changes it.

## Why

On matched input the C1 reading rides on the walk the survival policy produces: every tested
trunk reads a common scripted walk at only 0.551 to 0.577, and even an untrained trunk reads more
from the survival agent's streams (0.636 on replay) than from scripted ones (FINDINGS 17.6 and
20.5). The survival walk also visits states where the two laws differ more (summed deviation per
24-step episode 0.546 against 0.47 to 0.48 along the other walks, FINDINGS 20.6), and it is the
fastest walk (mean speed 0.158 against 0.147 to 0.150). The manuscript says so and stops there
(main.tex:536, anchor "Whether goal direction is required, rather than walk speed or the larger
law deviation a faster walk visits, was not tested").

Two things separate the survival walk from the scripted one: how fast it moves, and that it
steers, closing the loop between what the agent senses and where it goes. This readout pulls them
apart with two extra walks through the same worlds.

## Design

Agents. The C1 rerun's saved agents (`fullruns/corrected_l3_h8_wm/agents`, corrected trainer,
whose cells equal C1's to the bit), drift 0.45, seeds 0 to 9, three trunks: survival, which
decides, predictor, and untrained, which is the floor.

Worlds. Fresh pools: authentic at drift 0 from seed base 2_800_000 and surrogate at drift 0.45
from 2_850_000, 110 episodes of 24 steps each, ray steps 5. Both bases are registered as "B2-auth"
and "B2-surr" in `REGISTERED_LATER` in `itasorl/post_review.py`. The runner refuses to start when
`post_review.base_collisions` for them against `post_review.all_bases_in_use()` is non-empty, or
when `--n-eps` exceeds their registered footprint. The post-review fresh bases (2100000, 2150000)
are not used, since the readouts that drew them would share these episodes.

Four walks through those worlds. Every trunk is read on every walk.

1. `own`. Each trunk's own deterministic policy, closed loop (`collect_pool`), the registered
   protocol at fresh bases. The two pools are independent, so the probe uses one fold group per
   row. The survival trunk's own walk sets the target speed.
2. `scripted`. The scripted walk of `eval_protocols` (action stream i from
   `default_rng(600000 + i)`, the same stream in both pools), unscaled.
3. `speed_matched`, arm A. The same scripted walk with its thrust multiplied by one per-seed
   `thrust_scale` (`eval_protocols.collect_pool_scripted(thrust_scale=...)`), clipped to the
   world's [0, 1]; headings and draws are unchanged. The scale is found by bisection on the bracket
   [0.5, 1/0.6] (at 1/0.6 the scripted thrust, uniform on [0, 0.6], reaches the world's cap of 1),
   at most 14 steps, so that the walk's mean speed over both pools matches the survival trunk's
   own-walk speed within 0.001. A seed that cannot be matched inside the bracket keeps the nearer
   end, is kept, and is flagged (`summary.speed_match.unmatched_seeds`).
4. `replay`, arm B. The survival agent's recorded env actions, open loop. The actions come from its
   own closed-loop episodes in the registered authentic worlds (800000 + j, through
   `eval_protocols.record_trajectories`, full-length episodes only), so neither B2 pool is the
   world they were chosen in. Sequence j is played in B2-auth + j and in B2-surr + j, and every
   trunk receives each observation and the replayed action as its previous action. The walk keeps
   the survival policy's thrust and turn statistics and loses any steering toward the pellets of
   the world it is played in.

On the `scripted`, `speed_matched` and `replay` walks pool index i carries the same action stream
in both pools, so the probe keeps the two rows of index i in one fold group.

Speed is the per-episode mean of |velocity| after each step, averaged over the full-length
episodes of both pools (the `speed` block of the output). The probe is the registered pooled
readout on the states (`eval_protocols.readout_from_states`: [mean h, h_T], scaler and logistic
regression fit in fold, explicit folds).

## Integrity gate

The base collision and footprint checks pass (the runner refuses to start otherwise), every agent
file is present, and every target is finite (`all_finite` true, exit code 0). Otherwise nothing
from the readout is reported. The speed match is not a gate; its outcome bears on reading 1 below.

## Readings, stated before any number

The output gives, per trunk, seed-paired t-based 90 percent intervals (n = 10) for four contrasts:
`speed_matched - scripted`, `own - speed_matched`, `replay - speed_matched` and `own - replay`,
each labeled ABOVE, BELOW or INCLUDES by where its interval lies relative to 0. Those labels are
reported as written. The readings below are read on the survival trunk.

1. Speed: `own - speed_matched`. "SPEED SUFFICES" if the interval lies inside [-0.05, +0.05]; "SPEED
   DOES NOT SUFFICE" if its lower bound is above 0.05; otherwise "UNRESOLVED". If fewer than 7
   seeds are matched within 0.001, reading 1 is "UNRESOLVED" whatever the interval, and the
   unmatched seeds are named.
2. Closed loop: `own - replay`. "CLOSED LOOP NOT NEEDED" if the interval lies inside
   [-0.05, +0.05]; "CLOSED LOOP ADDS" if its lower bound is above 0.05; otherwise "UNRESOLVED".

Label, from the two readings:

- GOAL DIRECTION: speed does not suffice and the closed loop adds.
- SPEED: speed suffices.
- ACTION STATISTICS: speed does not suffice and the closed loop is not needed. The survival
  policy's thrust and turn sequence carries the reading without steering in the world it is read
  in.
- NOT SEPARATED: anything else.

Reported beside, deciding nothing: the untrained trunk's targets and contrasts (the floor), the
predictor trunk's, `speed_matched - scripted` and `replay - speed_matched` on the survival trunk,
the per-walk speeds, and the per-seed thrust scales.

## Scope and limits

- Seen before this spec was frozen: a development run of the speed match on seed 0 (2026-10-17)
  found the scripted walk's speed spans only 0.1559 to 0.1676 over the whole thrust bracket, under
  the survival walk's 0.1689, so that seed ends at the bracket's upper end and is flagged. Thrust
  barely moves the scripted walk's mean speed (its random turning cancels much of the velocity).
  Arm A is therefore expected to leave several seeds unmatched, and reading 1 may be UNRESOLVED by
  the 7-seed rule above; arm B and reading 2 do not depend on the match. Turn scaling, which would
  change the walk's statistics beyond speed, is not used, and the tolerance is not widened.

The readout narrows the open question for C1 only; the T-touch and S-immortal agents stay "not
tested" here. Arm A changes the heading pattern as well as the speed (the scripted turns are
random), so SPEED DOES NOT SUFFICE does not by itself say what else matters; arm B is the cleaner
test of steering at matched action statistics. The readout records walk speed, not the law
deviation along each walk, so it does not separate speed from the larger deviation a faster walk
visits. No reading here is a verdict, and the registered own-policy protocol and its readings are
unchanged.

## What is written in each case

Under every label: main.tex:536, the sentence "Whether goal direction is required, rather than
walk speed or the larger law deviation a faster walk visits, was not tested; the survival walk is
also the fastest" is replaced by the label and its two contrasts, keeping the clause that the
deviation a faster walk visits is not separated; main.tex:433 (anchor "though it is also the
fastest walk") gains the speed-matched reading; main.tex:1098 (paragraph "Law deviation along each
walk", anchor "The survival walk is also the fastest") gains the speeds of the speed-matched and
replayed walks. FINDINGS gets a dated note beside 20.6 and a pointer in 17.6. A dated follow-up
line goes into `docs/CORRECTIONS.md`.

- GOAL DIRECTION. The Discussion at :536 states that, on C1, the reading needs a walk that steers
  in the world it is read in, beyond its speed. CITATION :35-36 ("the signal rides on the
  trajectories the foraging policy produces") becomes "rides on the goal-directed trajectories the
  foraging policy produces".
- SPEED. :536 states that a scripted walk at the survival walk's speed reads as much, so the reading
  follows walk speed, not steering. CITATION :35-36 says the signal rides on how fast the foraging
  policy walks.
- ACTION STATISTICS. :536 states that the survival policy's action sequence carries the reading even
  when played open loop in another world, so steering is not required and speed alone does not
  account for it. CITATION :35-36 is unchanged.
- NOT SEPARATED. :536 gives the two contrasts and says the readout does not separate speed from
  goal direction. CITATION :35-36 is unchanged.

README, index.template.html and LEARNING are unchanged under every label. The results manifest gets
a new entry, readout only on C1, owning `artifacts/walk_speed/corrected_l3_h8_wm.json`.

## Run

Legend: `$R` is `fullruns/` under the repository root and `$CLOSE` is `fullruns/closing_2026-10/`.

    python scripts/run_walk_speed_readout.py --agents-dir $R/corrected_l3_h8_wm/agents --workers 2 --out $CLOSE/b2/walk_speed_corrected_l3_h8_wm.json

run first with `--dry-run`, from a clean detached worktree with `ITASORL_FOLDS=explicit` exported.
The output carries the provenance block of `scripts/agent_free_common.py` with the commit under
`git_commit_at_promotion`, `tracked_changes_at_generation` (empty), the stack, the agents folder as
a repository-relative path and argv, beside the settings, the per-seed rows and the summary. It
stays under `$CLOSE` until it is promoted to `artifacts/walk_speed/corrected_l3_h8_wm.json`.
Announced in `docs/PREREGISTRATION_L3.md` section 12, 2026-10-17.
