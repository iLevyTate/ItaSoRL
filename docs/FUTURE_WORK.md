# Future work: proposals for later runs

## 1. Status and reading rules

Written 2026-10-19, at the end of the closing pass. The proposal list and the three judges'
verdicts it restates were fixed on 2026-10-17, before any closing-pass readout ran.

Every entry in sections 2, 3, 4, 6 and 7 is a proposal or a design note. None is a finding, and
no verdict, rule or stored value rests on anything here. A proposal becomes a run only through
the record's usual path: a frozen spec, a dated announcement in `docs/PREREGISTRATION_L3.md`
section 12 before launch, and an entry there after it. Section 5 is the only section that points
at readouts that ran; each of those is recorded in its own artifact and section-12 entry, not
here.

The source files, `fullruns/closing_2026-10/handoff/sweep4_extensions.md` and
`sweep4_extensions.json` (21 proposals, E01 to E21, each scored by three judges for reviewer
value, cost honesty and rigor), are local and gitignored, so this file restates what each entry
needs. Costs are the judges' estimates for the owner's laptop and are not measurements. The
local anchor for CPU training cost is the corrected L1 run: 20 three-arm cells at 3 CPU workers
in 1 h 47 min. The future-work paragraph of the manuscript is the public face of the same list.
This file is not shipped in the review archive.

## 2. Next-paper experiments

**E11. Stakes dose on the rich side: a mortality ladder from death-off to the registered
density.** The record says the stakes question is not testable in this world: the frozen
scarcity calibration swept toward scarcer food only (24 down to 4 pellets), and the registered 24
pellets already sat in the target band. A read-only scratch rerun of the same calibration above 24
pellets put the scripted walker's drift-0 death rate near 0.10 at 96 pellets, 0.28 at 48 and 0.45
at 32, so a ladder between S-immortal (no death) and C1 (0.535) exists on the richer side. The
design: rungs at about 0.10 (96 pellets) and 0.28 (48 pellets), seeds 0 to 9, each a 20-cell run
of the C1 recipe with every C1 gate kept as registered, the calibration rerun and committed first
(the scratch rerun is not the record), and a trend rule frozen before launch. Judge 2's two rigor
points bind it. First, the registered stakes rule is the ordering S-immortal < C1 < S-scarce, and
"not testable" stays true of that rule; a ladder is a new rule with its own trend test and must
be announced as one, never as a reading of the registered ordering. Second, a basal-burn rung at
0.2 reads a death rate of 0.00 at drift 0, which is death-off in effect, so rung A is 96 pellets,
not basal 0.2. Judge 0 named the cheaper alternative for the wording: "not testable on the
scarcer side; the richer side was not run", with the scratch rates in a dated note. Cost: about 6
to 7 h per rung on one CUDA worker, or 2.5 to 3 h on CPU at 3 workers. The expected outcome is
flat. The owner parked stakes ladders on 2026-10-15, so this needs an explicit ask.

**E12. Known-use positive control at the headline rung.** Every L3 "not used" reading (the nudge
and the erasure) comes from agents for which knowing the world buys no return, so neither readout
could have come out "used". The design raises agents on the C1 recipe in a two-world mixture with
a world-contingent pellet reward (the sign set by the world, counterbalanced by seed, death off).
Gate U asks that the agent demonstrably act on the world (a pellet contrast with lower bound at
least 0.25). The nudge and erasure readouts and the frozen probe then run unchanged on the new
agents. If both readouts read "used" on an agent that passes gate U, the C1 "not used" readings
stand as sensitive negatives; if neither does, both readouts are labeled insensitive at their
frozen thresholds and every "not used" sentence gains that qualifier. It shares E03's hooks (the
two-world mixture and the signed pellet reward) and is the calibration for every "not used"
sentence in the record. Cost: about 3 to 4 h CPU, after the shared code. The judges add that gate
U is likelier to pass here than at L1, since the two worlds differ in dynamics within an episode,
and that the known-use walks differ from C1's by design, so the probe comparison is between
policies, not on matched input. X01 was the cheaper partial answer for this pass, and it stopped
at its integrity gate (section 5).

**E16. Second world instance: non-uniform drag, with its own gate-0 calibration as the stop
rule.** World P has uniform drag, so the fingerprint is a finite network's approximation error of
a constant; the default world parameters (land drag 0.20, water drag 0.60, gravity 1.0) make drag
depend on terrain. Stage A is readout only (about 8 to 20 min CPU): calibrate the surrogate's gate
0 on the new world, with a stop rule: if no capacity lands in [0.85, 0.95] after the registered
fallback, the instance is recorded NOT CALIBRATED and nothing trains. Stage B, only if Stage A
passes, is one 20-cell run of the C1 recipe with its own registered L0 pair, read under the labels
REPLICATES, WORLD-SPECIFIC, UNINFORMATIVE and NOT CALIBRATED. Judge 1 counted 51 files that
hard-code a land drag of 1.5, including every readout script that would be run on the new agents
and `scripts/run_expA_l3.py`, which Stage A itself needs. A `--world` flag across them is
therefore a prerequisite, not an option: without it each script silently reads the wrong world.
The engagement gate and the basal-burn calibration under gravity 1.0 are open risks, settled
before any freeze.

## 3. Training runs proposed this pass and not launched

The owner's answer to OD20 (2026-10-17) was NO, proposals only. Each of these would be a new
instance on the workstation stack that cannot be bit-checked against C1, every reading would be
descriptive, and none would re-adjudicate a stored run. The costs below carry the judges'
corrections.

- **E03. Known-use positive control at L1** (a world-contingent pellet reward in a two-world
  mixture, then the frozen probe, nudge and erasure on the agent that must use the seam). About a
  day of code (four trainer hooks, a runner, an `--rung l1` option for the nudge and erasure
  scripts) plus about 3 h CPU at 3 workers. The likely outcome is that gate U fails, which
  calibrates nothing, since the stream reader extracts the seam only at about 0.59 to 0.61 and a
  short policy is unlikely to infer the world before its first pellet. Counterbalancing halves
  the seeds per reward sign to 5.
- **E08. Supervised L1 ceiling** (the system-identification head trained on the world label in a
  two-world mixture). It shares E03's mixture hook and runs only together with E03. About 1.5 h
  CPU plus 20 min of readouts. Two loss coefficients, 1.0 and 10.0, are frozen in advance, since
  the L2 ceiling control only partly succeeded.
- **E06. PPO on the registered trunk.** About 1.5 h of code and 3 to 4 h CPU at 3 workers; the GPU
  path is slower on this machine. It inherits C1's open L0 gate. The judges corrected the
  proposal's cost anchor: its "11.5 to 13.3 min per CPU cell" is a retrain-cell time, not a
  three-arm cell.
- **E13. LSTM trunk at the registered width.** About a day of code, not the 2 h proposed: every
  collector stores the carried state, so an [h; c] state widens every stored state to 192 units
  unless each of the 7 or more `step_state` call sites takes an h-only view. About 3 h CPU.
- **E09. The pellet objective with death on.** No new training code; about 2.5 to 3 h CPU at 3
  workers (6 h on one CUDA worker). It must be labeled a "truncating death" cell: under the touch
  objective the -1 death penalty is outside the training reward, so death only truncates episodes
  and the agent is not trained against it.
- **E15. A fine snapshot ladder with the reward and world-model loss curves logged.** A
  default-off `--log-history` hook plus ten full three-arm cells (the runner has no
  arm-selection flag), about 2 to 2.5 h CPU plus 3 to 4 h of readouts. Its snapshot readouts
  needed X02 and X04, which did not run.
- **E18. Trunk width 48 and 192**, with the existing flag and no change to `itasorl`. Two 20-cell
  runs, about 5 to 6 h CPU. The probe's feature count changes with width (96 or 384 columns
  against about 176 training episodes per fold), so a curve across widths mixes trunk capacity
  with probe capacity and cannot be read as a trunk effect. It runs only if E06 or E13 is approved
  first.

## 4. Merged and dropped

E21 (complete removal, one keep vote of three) was merged into E20, which ran as X10. The three
judges described one experiment: grow the erased rank until a refit probe reads near chance, with
an equal-rank sham, on the same agents and thresholds. The merged design took E20's
implementation (flags on `scripts/run_erasure_readout.py`, with the refit done offline on the
stored pools) and kept three elements of E21: the capacity-confounded rule on the sham's
surrogate-world return, the explicit orthogonalization of each refit direction against the
already-erased subspace, and the class-mean swap, two optional rollouts labeled a class-mean
counterfactual, not a twin counterfactual.

E07 and E14 share one script (`scripts/run_mdl_probe.py --mode {mdl,sparse}`) and stayed two
steps, X05 and X06. No proposal was dropped outright: every other one had at least two keep votes
and was placed in Stage 5X, in OD20 or in section 2.

## 5. Readouts run in Stage 5X

Each line gives the step, its proposal, its artifact and what the frozen rule says. The dated
entries under the 2026-10-17 announcements in `docs/PREREGISTRATION_L3.md` section 12 carry the
detail.

- X01, nudge dose-response (E01): STOPPED at gate (I). The untrained arm failed the integrity gate
  on seed 9, so nothing was reported. No artifact.
- X02, development on a fixed walk (E02): not run, by an owner decision of 2026-10-18 made before
  any of its output existed. No artifact.
- X03, head overlap (E04): `artifacts/head_overlap/corrected_l3_h8_wm.json`, label
  "undetermined".
- X04, surprise across snapshots (E05): not run, by the same owner decision. No artifact.
- X05, prequential code length (E07): not run, by the same owner decision. No artifact.
- X06, sparse support (E14): not run, by the same owner decision. No artifact.
- X07, pellet-density dose (E10): `artifacts/pellet_dose/corrected_l3_h8_wm.json`, label
  "locomotion-carried".
- X08, budget-curve dynamics (E17): `artifacts/budget_curve_dynamics/corrected_runs.json`, labels
  "tracks budget" for C1 and "both" for C2; the C2 100-to-200 segment label is not met.
- X09, Bayes factors with an interval null (E19): `artifacts/bayes_factors/interval_null.json`,
  an annotation that relabels nothing; its labels appear only in the manuscript's appendix table.
  The factors are approximate plug-in sensitivity calculations: the raw half-width goes to the
  standardized scale with the observed standard deviation, so they are not exact Bayes factors
  for the fixed raw-scale band.
- X10, full-rank erasure (E20, with E21 folded in): VOID (gate E). Gate E failed, so no use
  reading was made: the in-fold refit reached the 0.55 stop by rank 16 in 1 of 10 seeds (seed 5,
  at rank 2; the rule needs 8); in the other 9 it stayed above 0.55 up to the rank cap of 16. No
  artifact; the per-seed values live in the run output
  (`fullruns/closing_2026-10/x10/corrected_l3_h8_wm.json`).

## 6. Corrections to carry into any future launch

- Anything read against the stored `artifacts/goal_stakes/C1/mechanism.json`, which came from
  the cloud container and records no stack, is checked first against a local rerun of the same
  readout (the B5 rerun played that role for X01), under the B5 NaN rule: a seed that is NaN on
  both sides is excluded, a NaN on one side only fails that arm, and a NaN is never a pass.
- Before X01 is launched again, `scripts/run_nudge_dose_readout.py`:249-253 must stop counting an
  empty pool as a failed seed; as written, the spec's exclusion of seeds empty on both sides
  cannot be reached.
- The actor's emit row is dead (nothing in the world reads it). Any actor-space reading drops it:
  the positive-control direction of X01 and the random expectation 4/96 of X03.
- A world with zero pellets cannot be reset (`itasorl/patch_of_earth.py`:165), so a density ladder
  starts at 2 pellets.
- In an iterative erasure, each refit direction is projected onto the complement of the
  already-erased subspace before it is stacked; judge 1 measured 73 to 80 percent of a refit
  direction's norm inside the erased subspace without it. X10 ran with this in place and its
  gate E still failed, so a relaunch needs a stop rule frozen before it runs, never a looser one
  chosen after.
- Prequential code length is coded in the explicit fold order taken in whole twin groups
  (prefixes of 12, 22, 44, 88 and 176 episodes), so no block boundary splits a world seed's twin
  pair.
- The sealed command in a spec and the chain script that runs it name the same `--workers`; B5
  and X07 ran with 2 where the sealed commands said 4 (values unaffected).

## 7. Relation to the parked list

The DIAMOND scale-up (`docs/specs/2026-10-15-scale-up-learned-world-models-design.md`, design
only), the identity-contingent stakes ladder and the two-laws world stay parked as the owner left
them on 2026-10-15. E11 varies mortality, not the payoff for knowing which world the agent is in,
so it is not the identity-contingent ladder. E16 keeps one law per world, with each agent raised
in either the authentic or the surrogate world, so it is not the two-laws design.
