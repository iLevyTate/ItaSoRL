# Extension readouts X01 to X10: frozen design (2026-10-17)

Frozen 2026-10-17, before any run it governs. Ten readouts extend the record on agents and
artifacts it already holds. Eight read saved agents or stored dumps; X08 and X09 are agent-free
and read committed artifacts only. No agent is trained. Every section is post hoc, descriptive,
and changes no stored value or verdict: no registered rule applies to any reading here, no label
re-adjudicates a stored run, and none changes the title, which this spec refers to as "the
title" and never spells.

## Conventions for every section

Legend: `$R` is `fullruns/` under the repository root and `$CLOSE` is `fullruns/closing_2026-10/`,
the gitignored folder that receives every raw output of this pass.

- Commands run from a clean, detached worktree at the commit that carries this spec, with
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 ITASORL_FOLDS=explicit` exported by the
  run chain. Each command below is the one the chain runs, copied as written. Every output lands
  under `$CLOSE/x0N/`, never under `artifacts/`, `docs/figures/` or a shipped run folder, and is
  promoted later into a new top-level artifact folder with its own results-manifest entry.
- Intervals are seed-paired and t-based at 90 percent over n = 10 seeds unless a section says
  otherwise.
- A failed integrity gate reports nothing: no number from that step is quoted or promoted, and
  the record says which gate failed. A missing or non-finite value never counts as a pass.
- Text maps name the text each label changes in the manuscript (`main.tex`; line numbers of
  the manuscript revision of 2026-10-16, with anchor text), `docs/FINDINGS.md` and the other public documents.
  A surface a label does not name is unchanged by it. No reading here enters the abstract
  without the owner's review.
- Commit key. Each output carries the provenance block of `scripts/agent_free_common.py`. The
  readouts of saved agents or stored dumps (X01 to X07, X10) record the commit under
  `git_commit_at_promotion`; the agent-free X08 and X09 record it under `git_commit`. Each output
  also records `tracked_changes_at_generation` (empty), the stack and argv.
- Order. X08 and X09 run first; X01 runs after the B5 rerun of
  `docs/specs/2026-10-17-closing-readouts-design.md` ended cleanly and `$CLOSE/b5/C1_mech.json`
  exists; X02 runs before X04; X05 and X06 run after the G57 regeneration, with its drop rule for
  `$CLOSE/g57/states_local`; X10 runs last.

## X01. Nudge dose-response

Hypothesis. A nudge along the decoded direction u moves behavior no more than the sham direction
v at 0.5, 1, 2 and 4 s, while a 1 s nudge along the actor-sensitive direction p (the emit row
dropped) does move it.

Command, after `--dry-run` and one seed into `$CLOSE/timing/`:

    python scripts/run_nudge_dose_readout.py --run-dir $R/corrected_l3_h8_wm --magnitudes 0.5 1 2 4 --positive-control actor --integrity-ref $CLOSE/b5/C1_mech.json --workers 2 --ckpt $CLOSE/x01/ckpt --out $CLOSE/x01/corrected_l3_h8_wm.json

Agents, seeds and bases. C1's survival and untrained arms at drift 0.45, seeds 0 to 9. Base
1400000, 64 episodes of 80 steps. Per seed: the unnudged authentic and surrogate rollouts; real
(+a s u in authentic episodes, -a s u in surrogate episodes) and sham (+/-a s v) at each magnitude
a; the positive control at 1 s along p in authentic episodes. p is the top right singular vector
of the actor rows after the dead emit row is dropped (nothing in the world reads it), pushed
through `to_env_action`, so a push that moves only the emit logit never counts. Survivors are
intersected across every condition and `n_common` is written per magnitude. d_u, the between-world
mean shift along u in s units, is recorded so each magnitude is read against the natural scale.

Integrity gate (I). At 1 s the real and sham per-seed scores reproduce the local B5 rerun
(`$CLOSE/b5/C1_mech.json`) within 0.01, else stop and report nothing. The stored
`artifacts/goal_stakes/C1/mechanism.json` records no stack, so a difference from it is a
cross-stack note under the B5 rule of the closing-readouts spec, never this gate. Gate (I) is
applied arm by arm under that spec's NaN rule: the B5 rerun returns NaN on the three
untrained-arm seeds the stored file records as "empty pool" (100 NaN tokens, three "empty pool"
strings). A seed whose B5 value is NaN must come back NaN with the same empty-pool reason and is
excluded; a NaN on one side and a finite value on the other fails that arm. The untrained series
then rests on the finite seeds, and every quoted untrained number states that basis. A NaN is
never a pass.

Gates on the reading.

- (i) Sensitivity: p moves the standardized behavior vector by at least 0.25 (GAP_MIN) in at least
  7 seeds. Otherwise the series is "uninformative: no 1 s nudge moves this policy" and no other
  reading is made.
- (ii) A magnitude with fewer than 7 informative seeds after the survivor intersection is VOID at
  that magnitude, and the attrition is reported per seed.

Rule. At each magnitude, the frozen 0.10 real-minus-sham rule over informative seeds.

Labels.

- "dose-dependent use": real minus sham has a lower bound above 0 at 2 s or 4 s, and the means do
  not decrease with magnitude.
- "not used up to 4 s": every magnitude's interval lies inside [-0.10, +0.10] and gate (i) passed.
- "off-manifold" at 4 s: both |real| and |sham| exceed 0.25; that magnitude is excluded and the
  label is read on the lower three.

The untrained arm's series is the floor and must not pass at any magnitude.

Prediction. Gate (i) passes and the label is "not used up to 4 s".

Stated limit. A structural positive control shows that the policy can be moved. It does not show
that a used world direction would be detected.

Text map.

- "not used up to 4 s": main.tex:554 (item "Stakes and use"), "its use is not shown" gains "(not
  moved up to four times the registered push, with a positive control that does move behavior)",
  and the FINDINGS 18.3 dated note replaces the 2026-10-14 caveat that the nudge test may be
  insensitive.
- "dose-dependent use": main.tex:554, the Discussion paragraph at :536 and the claims-appendix row
  that quotes the nudge change; the abstract gains a clause only after owner review.
- "uninformative": main.tex:554 unchanged; the 18.3 note records that no 1 s push moves this
  policy.

Commit key: `git_commit_at_promotion`.

## X02. Development on a fixed walk

Hypothesis. The residual controls that readings 1 and 2 of the development readout use
(`resid_obs`, `resid_trace`) grow across the 100 to 450-update snapshots on identical input, or
they do not.

Command, after `--dry-run`:

    python scripts/run_development_fixed_walk.py --run wm=$R/corrected_l3_h8_wm/agents --run nowm=$R/corrected_l3_h8_nowm/agents --integrity-ref wm=artifacts/development/corrected_l3_h8_wm.json --integrity-ref nowm=artifacts/development/corrected_l3_h8_nowm.json --out-dir $CLOSE/x02 --ckpt $CLOSE/x02/ckpt --workers 2

Outputs `$CLOSE/x02/fixed_walk_{wm,nowm}.json`.

Agents, seeds and bases. Per run (C1 and C2) and seed 0 to 9, the 300-update agent's recorded
registered-pool episodes (bases 800000 and 850000) are replayed through the 0, 100, 200, 300 and
450-update snapshots. The target, `resid_obs` (`history_basis(Ot, lags = 1)`) and `resid_trace`
(`_trace_phi(Bt)`) come from the battery's own functions.

Integrity gates. `replay_target` per seed and budget equals the stored
`summary.aggregate["<b> replay_target"].per_seed` in
`artifacts/development/corrected_l3_h8_{wm,nowm}.json` within 1e-6 (same stack, same code path;
the stored files record torch 2.7.0+cu126 cpu and Python 3.13.2), else stop. The recorded
observations equal the re-collected episodes one to one, else stop.

Rules and labels.

- Reading 1-fixed: if `resid_obs_fixed(450)` minus `resid_obs_fixed(100)` has a lower bound above
  0.05, "memory beyond the linear senses grows on identical input"; otherwise, if
  `replay_target(450)` minus `replay_target(100)` has a lower bound above 0.05, "only the trace
  sharpens on identical input"; otherwise "no growth on identical input".
- Reading 2-fixed: the same rule on `resid_trace_fixed`.
- Auxiliary-fixed: C1 minus C2 in `resid_obs_fixed` at 450 with a lower bound above 0 reads "the
  auxiliary builds the stored component on identical input".

Predictions. C2 "no growth on identical input" (its fixed-walk target rise is +0.018
[-0.008, +0.044]). C1 is open, with +0.083 as the ceiling for what the residuals can show.

Stated limits. The 0 and 100-update snapshots are read on a walk their own policy would not
produce, as FINDINGS 23.4 already says for the fixed-walk target. The residual probes are the
spec's linear controls, not a memory claim.

Text map, under every outcome: main.tex:1293 (paragraph "Development"), "The control readings
that follow are on own episodes and have no walk-fixed counterpart" is replaced by the fixed-walk
readings beside the own-episode ones; FINDINGS 23.4's "the auxiliary builds the stored component"
gains its identical-input counterpart; one dated CORRECTIONS note. If C1 reads "no growth on
identical input", the :536 paragraph's statement about what a trained agent keeps from an
identical stream is scoped to the 300-update agent.

Commit key: `git_commit_at_promotion`.

## X03. Head overlap

Hypothesis. The decoded direction u and the rank-two probe subspace lie outside the actor's row
space.

Command:

    python scripts/run_head_overlap_readout.py --agents-dir $R/corrected_l3_h8_wm/agents --arms survival predictor untrained --workers 2 --ckpt $CLOSE/x03/ckpt --out $CLOSE/x03/corrected_l3_h8_wm.json

Agents, seeds and bases. C1's three arms at drift 0.45, seeds 0 to 9, on the registered pools.
Quantities per seed and arm:

- (a) r_actor = ||P_rowspace(W_actor) u||^2 divided by the random expectation 4/96 (the emit row
  dropped); the same for the critic row (1/96) and for the decoder's 96-column state block
  (64/96). The decoder comparison is declared weak in advance: a random 64-dimensional subspace of
  96 captures two thirds of any direction.
- (b) The one-step action sensitivity ||d env_action / d h . u|| through `to_env_action` at every
  visited pooled state, divided by the median over 1000 random unit directions.
- (c) The recurrence sensitivity ||d h_{t+1} / d h_t . u||, the same ratio.

Integrity gate. The fitted probe's pooled target equals the cell target within 0.002 for 30 of 30
arm-cells, else stop.

Rule, on the survival arm. "policy-blind" if the upper bound of r_actor is below 0.5 and the upper
bound of the one-step action ratio is below 0.5; "policy-read" if both lower bounds exceed 2;
otherwise "undetermined". The untrained arm's own u is the floor and decides nothing.

Prediction. "policy-blind", with the decoder block reading u above its expectation.

Stated limit. A one-step overlap is not a behavioral test: the recurrence can route a direction
that lies outside the actor row space into later actions, which X01 measures.

Text map. main.tex:554 gains one mechanism sentence under either label. FINDINGS 18.3 gets a dated
note and a table row. Under "policy-read" the note adds that the nudge null is then unexplained
and names X01. The abstract is unchanged.

Commit key: `git_commit_at_promotion`.

## X04. Surprise across snapshots

Hypothesis. The agent's own prediction-error separation of the worlds falls or stays flat from the
100 to the 450-update snapshot while the fixed-walk state target rises.

Command, after X02:

    python scripts/run_surprise_readout.py --agents-dir $R/corrected_l3_h8_wm/agents --drift 0.45 --snapshots 100 200 450 --ckpt $CLOSE/x04/ckpt --out $CLOSE/x04/corrected_l3_h8_wm_snapshots.json

Agents, seeds and bases. C1 only, seeds 0 to 9, on the 300-update agent's own recorded
registered-pool streams. Per snapshot b in 0, 100, 200, 300 and 450: S_b, the prediction-error
AUROC; L_b, the mean squared next-observation error on the authentic pool; R_b, the stored
fixed-walk target.

Integrity gate. S_300, `mean_err_auth` and `mean_err_surr` per seed equal
`artifacts/surprise/corrected_l3_h8_wm.json` `rows[*].own.survival` within 1e-6, and the b = 0
row equals `rows[*].own.untrained`, else stop.

Rule. "detection grows" if S_450 minus S_100 has a lower bound above 0.05; "detection falls" if
its upper bound is below -0.05; otherwise "detection flat". The per-seed Spearman correlation
between L_b and R_b across the five budgets is descriptive and has no threshold.

Prediction. "falls" or "flat".

Stated limits. C1 only: C2 has no decoder, and the corrected L1 run has no snapshots. Early
snapshots are scored on a walk they did not produce. The b = 0 value is the stored 0.764 row, so
the 100-to-450 contrast is the only new number.

Text map. One sentence in FINDINGS 22.2 and one in the manuscript's surprise paragraph that cites
0.626 against 0.764 (main.tex:1089, anchor "a randomly initialized decoder separates the worlds
better than the trained one"). Under "detection falls" the :536 paragraph gains that the model's
error becomes less world-separating as the state becomes more so. Under "detection grows" the
withdrawn 18.3 reading is reopened as a dated note only.

Commit key: `git_commit_at_promotion`.

## X05. Prequential code length (MDL)

Estimands, per (run, drift, seed, arm). The online code length L in bits over the 220 episodes in
the explicit fold order, fixed here before the run and taken in whole twin groups: both episodes
of a world seed sit in the same block, because the pooled probe groups twins
(`experiment_b2.py`:713-714) and a boundary that split a pair would let the probe read the train
twin's label off its test twin. The training prefixes are 12, 22, 44, 88 and 176 episodes (6, 11,
22, 44 and 88 twin pairs, so no boundary splits a pair), and the first block is coded under the
uniform model. Compression is 220 / L. The AUROC-versus-prefix curve is recorded beside it.

Command, after the G57 regeneration (same dirs and drop rule as the G01 readout):

    python scripts/run_mdl_probe.py --mode mdl --states $R/l3_h8_heldout/states $R/l1_heldout/states $R/corrected_l1_heldout/states $R/balanced_confirmation_wm/states $R/balanced_confirmation_touch/states $CLOSE/g57/states_local --prefixes 12 22 44 88 176 --workers 2 --out-dir $CLOSE/x05

Agents, seeds and bases. Stored dumps of the registered pools of each run, seeds 0 to 9, three
arms; drift tags through `experiment_b2.format_drift`.

Integrity gate. The full-n explicit-fold AUROC reproduces the per-dir target of the G01 table in
the closing-readouts spec within 1e-9 (C1 within the G57 tolerance), else that cell is VOID.

How compression is read. With 192 features and the registered C = 1 probe, shuffled labels cost
well above the uniform one bit per episode once the probe starts to fit, so the absolute
compression under a null is about 0.5, not 1. Compression is therefore read only as a seed-paired difference between arms (survival
minus untrained), never as an absolute level. The R2 band and R3 below apply to that difference.

Readings.

- (R1) L3 hidden-8 at drift 0.45 (and C1 if its dumps pass G57): survival minus untrained
  compression has a lower bound above 0, and survival reaches 0.65 AUROC at 88 or fewer training
  episodes per fold.
- (R2) L1 at drift 0.023, historical and corrected: survival minus untrained compression lies
  inside plus or minus 5 percent of the uniform 12 bits (the first block is 12 episodes, 6 twin
  pairs), and the survival curve stays under 0.65 at every prefix: "not extractable at any
  training size tested".
- (R3) Drift 0 of every run: the survival minus untrained difference stays inside the R2 band.

The 12 and 22-episode prefixes code near uniform with 192 features and are reported, not read.

Text map. A code-length column and the curve in the appendix paragraph labeled app:ci95
(main.tex:1314), beside the G01 table; one sentence at main.tex:549 only if R2 holds ("by the
registered probe class at every training size tested"; nonlinear readouts untouched, which H53
speaks to); a FINDINGS dated note; README unchanged.

Commit key: `git_commit_at_promotion`.

## X06. Sparse support

Estimands, per (run, drift, seed, arm). The explicit-fold AUROC of the L1-penalized probe at eight
penalties from 0.001 to 10 (saga, `max_iter` 5000, scaler fit in fold), with the mean nonzero
count per fold; k65, the smallest support at which the fold AUROC reaches 0.65; the AUROC at a
fixed support of 8.

Command, after the G57 regeneration:

    python scripts/run_mdl_probe.py --mode sparse --states <the X05 dirs> --drifts 0.45 0.023 --workers 2 --out-dir $CLOSE/x06

with every dir at its strongest drift, then `--states $R/l3_h8_heldout/states $CLOSE/g57/states_local --drifts 0.00`
into the same out dir (drift 0 for the L3 runs only).

Agents, seeds and bases. As X05.

Integrity gate. The dense C = 1 L2 fit in the same script reproduces the G01 per-dir target within
1e-9, else that cell is VOID. Convergence warnings are counted, and a fit that did not converge is
not read.

Readings.

- (R1) L3 survival at drift 0.45: "concentrated" if k65 is at most 16 of 192 in at least 8 of 10
  seeds, else "distributed".
- (R2) L1 survival at drift 0.023, historical and corrected: "not reached at any support" if no
  penalty reaches 0.65 in any seed.
- (R3) The L3 predictor arm does not reach 0.65 at any support.

Seed-paired survival minus untrained at support 8 is reported with no label.

Stated limit. k65 is a property of the standardized L1-penalized probe family under collinear
units, not an anatomical count.

Text map. One sentence beside the 0.001 variance share in the salience paragraph of
app:fourreadouts (main.tex:1287, anchor "The probe direction carries 0.001 of the variance") saying
whether the direction is also sparse; a column in the G01 appendix table; a clause at main.tex:549
("nor at any support size") only if R2 holds; a "how many units" number for the site and film only
under "concentrated", and then only in the site's index text, never in a film build.

Commit key: `git_commit_at_promotion`.

## X07. Pellet-density dose

Hypothesis. The reading on the C1 survival arm falls when the pellet-directed approach segments
are removed from the walk.

Command, after one density into `$CLOSE/timing/`:

    python scripts/run_pellet_dose_readout.py --run-dir $R/corrected_l3_h8_wm --n-pellets 2 6 12 24 48 96 --arms survival predictor untrained --workers 4 --ckpt $CLOSE/x07/ckpt --out $CLOSE/x07/corrected_l3_h8_wm.json

Agents, seeds and bases. C1's three arms at drift 0.45, seeds 0 to 9. The registered pooled probe
(110 x 24, explicit folds, pools at 800000 and 850000 unchanged) in evaluation worlds with 2, 6,
12, 24, 48 and 96 pellets; the override applies to the evaluation world only and is recorded under
`settings`. Zero pellets is infeasible (`itasorl/patch_of_earth.py`:165 cannot reset an empty
pellet list). The predictor and untrained arms at the same densities are floors, and the
untrained-trunk replay of the trained arm's episodes is the input reference at each density.

Integrity gate. The 24-pellet row reproduces each cell's stored target within 0.002 for 30 of 30
arm-cells (the G57 cross-stack tolerance), else VOID.

Rule.

- (a) "pellet-directed" if and only if at 2 pellets the survival mean is under 0.65, the
  seed-paired 24-minus-2 contrast is at least +0.05 with its interval excluding zero, and the
  untrained arm stays within 0.1 of 0.5 at every density.
- (b) "locomotion-carried" if and only if at 2 pellets the survival mean is at least 0.65 and its
  margin over the untrained arm at 2 pellets is at least +0.05 with the interval excluding zero.
- (c) Otherwise "not separated".

The dose curve is a table, not thresholded, and every row carries "gate-0 oracle not recalibrated
at this density" (`scripts/run_expA_l3.py` has no pellet argument).

Prediction. The survival reading falls toward 0.57 to 0.64 at low density.

Text map. FINDINGS 18.4 and 18.5 gain an ecological row beside the scripted-walk and replay rows;
main.tex:536 ("Whether goal direction is required, rather than walk speed or the larger law
deviation a faster walk visits, was not tested") and :542 (paragraph "One reading deserves a
direct answer") gain the reading beside the B2 walk-speed result. No verdict.

Commit key: `git_commit_at_promotion`.

## X08. Budget-curve dynamics

Disclosed as post hoc: the proposal's numbers were seen in a scratch read before this rule was
written, the same disclosure the within-episode curve (G04) carries.

Command:

    python scripts/read_budget_curve_dynamics.py --budget-curve artifacts/budget_curve.json --run wm=$R/corrected_l3_h8_wm --run nowm=$R/corrected_l3_h8_nowm --out $CLOSE/x08/budget_curve_dynamics.json

Inputs. Agent-free. Per run, the 10 x 4 matrices of target, return and trained_len at 100, 200,
300 and 450 updates: target and return from `artifacts/budget_curve.json`
(`series["decoder on"|"decoder off"][*].target.per_seed` and `.return.per_seed`), trained_len from
the 20 cells (`budget_curve.<u>.eng.trained_len`). Statistics: `paired_t90` per adjacent segment;
the across-seed Spearman correlation per budget with a 10,000-draw permutation p; within-seed
(seed-demeaned) partial slopes of target on return and on update count, with a seed-bootstrap
interval.

Integrity gate. The input arrays equal the committed artifact and cells bit for bit, else nothing
is reported.

Labels, per run. "tracks return" (the return slope's lower bound above 0 and the update slope's
interval includes 0); "tracks budget" (the reverse); "both" or "neither" otherwise. The C2
100-to-200 segment is labeled "decodability moves without return" only if the target interval
excludes 0 while the return and trained_len intervals include 0.

Prediction. The C2 segment label is NOT met.

Stated limits. Four budgets per seed is a thin basis, and return and update count are collinear
within seed, so the partial slopes are unstable. Survival length moves about one step across the
whole ladder.

Text map. One sentence at main.tex:442 (paragraph "The comparison against the training budget")
and one in FINDINGS 17.10, at most. No table or figure.

Commit key: `git_commit` (agent-free).

## X09. Bayes factors with an interval null

Disclosure. The Bayes factors were computed on the committed arrays during script development on
2026-10-17, in a scratch read before this spec was written, the same disclosure X08 carries.
Several of those values differ from the predictions below (for example, the eight independent L0
pairs read about 224, "evidence for the band"). The predictions are kept as they were written in
the closing plan, which predates the scratch read.

Command:

    python scripts/build_bayes_factors.py --out $CLOSE/x09/interval_null.json

Model. A one-sample or paired t-model on each committed per-seed array, JZS prior with Cauchy
scale 0.707 (0.5 and 1.0 reported beside it; the label is read at 0.707 only). H0: the effect lies
inside the region of practical equivalence (ROPE); H1: outside. The ROPE is [0.45, 0.55] for
AUROC-minus-chance readings and plus or minus 0.05 for the sweep C contrast and the L1 margins. The
ROPE is carried to the effect-size scale by a plug-in conversion, c = ROPE half-width divided by the
sample standard deviation.

Inputs, fixed in the script: `artifacts/corrected_verdicts.json`
(`runs.C1.drift_zero.survival.per_seed`), `artifacts/l0_audit/corrected_l3_h8_wm.json`,
`artifacts/texture/ar_sweep_c_followup/summary.json` and `artifacts/texture/ar_sweep_c/C_rho0.00.json`,
`artifacts/expB2/corrected_L1-corrected.json` (`arms["0.023"]`), and the B1 and B2 drift-zero
arrays from `artifacts/balanced_pooled/registered_pair.json`, which is committed before the run.

Readings. (1) The C1 L0 registered pair (0.559). (2) The eight independent pairs, signed and
folded. (3) B1 drift-0 0.561 and B2 0.536. (4) The sweep C stored and follow-up contrasts. (5) The
corrected L1 survival minus untrained and minus predictor.

Labels. BF01 above 3 "evidence for the band", below 1/3 "evidence against the band", otherwise
"indeterminate".

Predictions. (1) against; (2) indeterminate; (3) against and indeterminate; (4) indeterminate;
(5) indeterminate.

Integrity gate. Every input array equals its committed artifact bit for bit, else nothing is
reported.

Stated limits. The result annotates and never relabels: the standing rule forbids re-adjudicating
a stored run with a new estimator. It lives in one appendix table kept visibly apart from the
bootstrap ROPE check (`itasorl/stats.py`:263-264 says that check is not a Bayesian analysis), and
every row says it is conditional on one evaluation-world draw. The labels are reported only inside
that appendix table. Every row that sits beside a registered reading is written as "BF01 = x
(annotation, 0.707 scale)" with no band verdict word, and no X09 label or value enters a
claims-inventory row (row 43 at main.tex:667 carries the registered signed TOST reading and is not
touched).

Text map. One clause each at main.tex:548 and :549 and the appendix table; FINDINGS notes beside
the sections that hold the five readings; the plain-language paper page unchanged.

Commit key: `git_commit` (agent-free).

## X10. Full-rank erasure

Hypothesis. Erasing the state until a refit probe reads near chance leaves the behavior and return
readings "not used", as the rank-two erasure did.

Command, after one seed into `$CLOSE/timing/` (k is unknown in advance):

    python scripts/run_erasure_readout.py --agents-dir $R/corrected_l3_h8_wm/agents --arms survival untrained --max-rank 16 --refit-stop 0.55 --swap --workers 2 --ckpt $CLOSE/x10/ckpt --out $CLOSE/x10/corrected_l3_h8_wm.json

`--swap` is dropped if the owner does not want the counterfactual.

Agents, seeds and bases. C1's survival arm at drift 0.45, seeds 0 to 9, with the untrained arm as
the floor. Registered pools (800000 and 850000) for the subspace and the refit; rollouts at base
1400000, 64 episodes of 80 steps.

Procedure, per seed. Start from the rank-two projector. Refit the registered probe on the erased
stored pools, project each refit direction onto the complement of the erased subspace before it is
stacked, extend the projector, and repeat until the stop statistic is at or under 0.55 or the rank
reaches 16. Record k and the full refit curve. Build a sham of rank k orthogonal to the erased
subspace. Run the four erased and sham rollouts once at rank k. With `--swap`, two more rollouts
(authentic episodes with h <- h - P_k (h - m_surr), surrogate episodes with m_auth), labeled a
class-mean counterfactual, not a twin counterfactual.

The stop statistic is computed in fold. For each outer fold of the explicit twin-grouped
partition, the erasure subspace is grown from the rank-two projector refit on that fold's training
rows only, and the refit probe trained on erased training rows is scored on erased test rows; the
out-of-fold scores are pooled per rank (`fullrank.refit_curve_infold`, a list of {rank,
refit_auroc, n_scored} per seed). k is the smallest rank whose pooled in-fold refit is at or under
0.55, else 16 (`k`, `stop_reason`); the summary's `final_refit_per_seed` is the in-fold value at k.
The rollout projector is then grown on all episodes to that k. The reason: an erasure fit on every
episode, test folds included, pushes the refit AUROC toward and below chance once the real signal
is erased (synthetic tests read 0.27 to 0.45), which would trip the stop early and understate k.
The all-episode curve (`fullrank.refit_curve_alldata`) is reported beside, labeled as including the
test folds and biased low, and is never read.

Integrity gates.

- (I) The rank-two per-seed `frozen_probe_unerased`, `frozen_probe_erased` and `refit_probe_erased`
  reproduce `artifacts/mechanism/erasure/corrected_l3_h8_wm.json` within 0.01 (same stack, the
  stored code path), else stop.
- (E) The in-fold refit is at or under 0.55 in at least 8 of 10 seeds at rank at most 16, else
  VOID and only k is reported.
- (C) At least 5 common survivors per seed, with the every-attempted-episode return and
  termination lines reported beside.
- (K) Capacity-confounded: if the sham's surrogate-world return drops by more than 10 percent of
  the unerased return (the upper bound of the paired difference below -0.10 times the unerased
  mean), no use reading is made.

Readings, with the erasure spec's frozen thresholds (`docs/specs/2026-10-15-online-erasure-design.md`,
readings 1 to 3).

- "used under removal": real minus sham at least +0.10, interval excluding zero.
- "costly": erased minus sham surrogate-world return with an upper bound below 0.
- "not used under removal": both worlds' scores inside [-0.10, +0.10].
- "swap moves behavior": the swap's gap-closed score minus the sham's is at least +0.10 with the
  interval excluding zero in the authentic world.

Predictions. Gate E passes with k between 4 and 10; "not used" at full rank; the swap does not
move behavior.

New thresholds and stated limits. The 0.55 stop and the rank cap of 16 are new thresholds, stated
here and in the dated pre-run amendment of the erasure spec before the run, with the refit curve
reported so a reader can apply another stop. Pooled out-of-fold refit scores become unreliable as
the erased rank approaches the state width (the in-fold scaler blows up on near-null residual
dimensions), which is why the cap is 16 of 96 units. A fold that runs out of new directions keeps
its last subspace at higher ranks.

Text map. FINDINGS 23.3's label scope and main.tex:1291 (paragraph "Removal while acting") gain the
full-rank reading with k and the equal-rank sham; the main.tex:554 sentence on use; a CORRECTIONS
follow-up line inside the erasure entry. "costly" or "used" at full rank makes :554 a positive
statement about the state's world information, and the abstract gains a clause only after owner
review.

Commit key: `git_commit_at_promotion`.
