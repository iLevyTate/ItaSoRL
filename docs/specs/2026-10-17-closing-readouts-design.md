# Closing readouts on saved agents and stored dumps: frozen design (2026-10-17)

Frozen 2026-10-17, before any run it governs. Seventeen readouts close gaps the record already
discloses. All but three read saved agents or stored state dumps; G47 and H31 are agent-free
oracle computations, and C1 retrains only the small Experiment B world models (no actor-critic,
no saved agent). Nothing here relaxes a registered margin, changes a gate, or alters a
registered estimand. Every section is post hoc, descriptive, and changes no stored value or
verdict, except where it names a frozen spec that governs its design, and even there no
registered verdict is re-read. This spec refers to the manuscript's title as "the title" and
never spells it; no outcome below changes the title.

## Conventions for every section

Legend: `$R` is `fullruns/` under the repository root and `$CLOSE` is `fullruns/closing_2026-10/`,
the gitignored folder that receives every raw output of this pass.

- Commands run from a clean, detached worktree at the commit that carries this spec. The run
  chain exports `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 ITASORL_FOLDS=explicit`;
  a command that sets `ITASORL_FOLDS` itself overrides it. Each command below is the one the
  chain runs, copied as written.
- No command writes under `artifacts/`, `docs/figures/` or a shipped run folder (the corrected
  C1, C2, B1, B2 and L1 run folders, the hidden-8 held-out agents and the seed-2 agents). Raw
  outputs stay under `$CLOSE`. After every result is read they are promoted into new artifact
  paths, each owned by its own results-manifest entry; no committed artifact is rewritten.
- Intervals are seed-paired and t-based at 90 percent over n = 10 seeds unless a section says
  otherwise.
- Integrity comes first. Each section names a gate. A gate that fails stops that readout: no
  number from it is reported, quoted or promoted, and the record says which gate failed and that
  nothing was read. A missing or non-finite value never counts as a pass.
- Text maps. For each label, a section names the text it changes in the manuscript (`main.tex`;
  line numbers are those of the manuscript revision of 2026-10-16, given with anchor text because lines
  drift), `docs/FINDINGS.md`, `README.md`, `index.template.html`, `docs/LEARNING.md`, the
  results manifest (changed only through `scripts/build_results_manifest.py` and regenerated,
  never hand-edited) and the `CITATION.cff` abstract. A surface a label does not name is unchanged
  by that label. Every result also gets a dated FINDINGS note citing this spec and its section,
  a dated follow-up line in `docs/CORRECTIONS.md` where the section says so, and a result entry
  in section 12 of `docs/PREREGISTRATION_L3.md`.
- Commit key. Each output carries the provenance block of `scripts/agent_free_common.py`
  (`provenance()`): the commit, `tracked_changes_at_generation` (which must be empty), the stack
  (with scipy and the device), the agents or states folder as a repository-relative path, and
  argv. Agent-free computations (G47, C1, H31) record the commit under `git_commit`. Every
  readout of a saved agent or a stored dump (B1, B3 to B8, G01 to G04, G07, the G57 report, H53)
  records it under `git_commit_at_promotion`. A recorded commit is compared with the run commit
  by prefix, since the recorder writes a short hash.
- World-seed bases. B2 has its own spec,
  `docs/specs/2026-10-17-walk-speed-vs-goal-readout-design.md`. Its bases, 2_800_000 and
  2_850_000, B6's, 2_700_000 and 2_750_000, and G03's fresh read base 2_250_000 ("G03-fresh") are
  listed in `REGISTERED_LATER` in `itasorl/post_review.py`. Each of those runners refuses to start
  when `post_review.base_collisions` against `BASES_IN_USE + NEW_BASES` is non-empty for its own
  bases. 2_200_000 belongs to the post-review balanced pairs (`BALANCED_FRESH_BASE`) and is
  never reused here.

## B1. The L1 state at a supra-band grid

Why. The L1 negative is read at the in-band grid Delta = 0.023, where the state reads 0.551 and
the oracle 0.873. main.tex:549 says the readout was never shown to register a quantization seam
at any grid, so the negative cannot tell absence of the seam from the state apart from
insensitivity of this readout. At Delta = 0.06, with the same sensor noise 0.01, the oracle reads
1.000.

Command (R06):

    python scripts/run_l1_h2_ablations.py --agents-dir $R/corrected_l1_heldout/agents --states-dir $R/corrected_l1_heldout/states --out-dir $CLOSE/b1_corrected --channels a1 --deltas 0.023 0.06 --device cpu

then the same on `$R/l1_heldout/{agents,states}` to `$CLOSE/b1_historical`.

Agents, seeds and bases. The A1 channel reads the drift-0.023 agents of each run, three arms
(survival, predictor, untrained), seeds 0 to 9, 30 agents per run: the corrected L1 rerun
(`fullruns/corrected_l1_heldout/agents`), which decides, and the historical L1 held-out run
(`fullruns/l1_heldout/agents`), the like-for-like companion reported beside it, which decides
nothing. Pools are the registered pair, authentic 800000 + i at drift 0 and surrogate 850000 + i
under the L1 grid, 110 episodes of 24 steps, ray steps 5, CPU. The ladder is Delta in {0, 0.023,
0.06}; the script always adds 0.

Integrity gate. At Delta = 0.023 every regenerated pool must equal its saved dump in the run's
`states/` folder bit for bit (`a1_aggregate.json`: `integrity.checked` true and
`integrity.pools_bit_match` true). A mismatch aborts the script; a missing dump sets the match to
false. Either way that run reports nothing. If only the historical half fails, the corrected half
is still read.

Labels, read on the corrected run at Delta = 0.06 (`<arm>_d0.0600_mean`, `_tci90`, `_per_seed`):

- (a) SENSITIVE: the untrained arm reads above 0.5, its mean above 0.5 and its 90 percent lower
  bound above 0.5. The readout registers this class of seam at a supra-band grid.
- (b) BOTH AT CHANCE: the survival and untrained intervals at Delta = 0.06 both contain 0.5.
- (c) OTHER: neither (a) nor (b), for example survival above 0.5 with the untrained interval
  containing 0.5. This label is not in the outcome map the closing plan froze; it is written here
  so the map is exhaustive.

Under every label, survival minus untrained at Delta = 0.06 is reported descriptively as the
encoding contrast, with the predictor arm beside it. The `l0_anchor` and
`survival_monotonicity_rho` keys the script writes are not reported or quoted anywhere.

Text map.

- (a) SENSITIVE. main.tex:112: the clause "that the state readout can register a seam of this
  class at any grid was not shown (Section~\ref{sec:limitations})" is replaced by the measured
  reading at Delta = 0.06. main.tex:549: its last two sentences ("The state readout was never
  shown to register an observation-quantization seam at any grid: ..." and "The L1 negative
  therefore does not separate absence of the seam from the state from insensitivity of this
  readout to this class of artifact.") are replaced by the readings at 0.06 and the statement
  that the readout registers this class of seam at a supra-band grid, so the in-band negative is
  a negative at Delta = 0.023, not an insensitivity of the readout to the class. main.tex:88
  (anchor "yet the state reads 0.551"), :170 (anchor "an in-band L1 seam did not meet the
  encoding criterion"), :197 (anchor "Oracle-detectable hand-authored seams did not meet the
  encoding criterion") and :538 (anchor "In-band membership is not sufficient for the tested
  state readout") each gain a qualifier that at a supra-band grid the state reads the seam, so
  the L1 negative is specific to the in-band grid. Survival minus untrained is quoted beside as a
  description. FINDINGS: dated note beside the corrected L1 section, and the TL;DR's L1
  sensitivity line written from this label. README: the L1 sensitivity sentence says the readout
  registers the seam at 0.06. index.template.html: the L1 sentence of the "The matched negative"
  entry (:807) likewise. CITATION: the sentence at :24-25 ("Rerun on the corrected trainer, the L1
  state reads 0.551, as before the correction.") gains that the same readout reads the seam at a
  coarser, supra-band grid. LEARNING: unchanged. Manifest: a new entry, readout only on
  L1-CORRECTED and L1.
- (b) BOTH AT CHANCE. main.tex:112 and :549 keep their caveats; in :549 the clause "no agent
  state was read at a coarser grid such as the supra-band $\Delta = 0.06$ of the calibration"
  becomes "a supra-band grid ($\Delta = 0.06$) was also not registered" with the survival and
  untrained values. :88, :170, :197 and :538 are unchanged. FINDINGS: dated note and the TL;DR
  line say the readout did not register the seam at 0.06 either. README and index.template.html:
  the L1 sensitivity sentence says the same. CITATION and LEARNING: unchanged. Manifest: new
  entry as in (a).
- (c) OTHER. main.tex:549: the clause on the unread coarser grid is replaced by the per-arm
  values at 0.06; no other manuscript sentence changes. FINDINGS: dated note and TL;DR line with
  the values. README and index.template.html: the L1 sensitivity sentence gains the values.
  CITATION and LEARNING: unchanged. Manifest: new entry as in (a).

The title stays under every label. The historical companion's values appear only in the FINDINGS
note.

Commit key: `git_commit_at_promotion`, in `provenance.json` in each out dir.

## B3. The registered authentic-raised readout on the corrected C1 drift-0 agents

A frozen spec governs the design: `docs/specs/2026-10-09-reverse-direction-readout-design.md`,
with its dated amendment of 2026-10-17 (the C1 drift-0 agents are local, which corrects that
spec's line on the corrected agents). The readout, the bar, the margin and the three labels are
the frozen ones. The label on C1 is descriptive: it enters no rule-ledger row, and the ledger's
"about" field gains one sentence saying so.

Command (R04):

    python scripts/reverse_direction.py --agents-dir $R/corrected_l3_h8_wm/agents --out $CLOSE/b3/reverse_direction_corrected_l3_h8_wm.json --integrity-ref artifacts/fold_rescore/corrected_l3_h8_wm_integrity_ref.json --checkpoint-dir $CLOSE/b3/ckpt

Agents, seeds and bases. The 30 drift-0 agents of C1 (`agent_d0.00_s*_{survival,predictor,untrained}.pt`
in `fullruns/corrected_l3_h8_wm/agents`, seeds 0 to 9), read at drift 0.45 on the registered pair
(authentic 800000, surrogate 850000), with the balanced same-world-seed readout beside.

Integrity gate. Rescored at drift 0, every arm and seed must reproduce the recorded
explicit-partition drift-0 target in `artifacts/fold_rescore/corrected_l3_h8_wm_integrity_ref.json`
(built from `fullruns/corrected_l3_h8_wm/cells/cell_d0.00_s*.json`) within 0.01: PASS with
`n_checked == 30` and worst deviation at most 0.01. Otherwise stop.

Labels, frozen in the reverse-direction spec, with m and u the survival and untrained means at
drift 0.45: R1 CARRIED (m >= 0.65 and m - u >= 0.05), R2 NOT CARRIED (m < 0.65), R3 AMBIGUOUS
(m >= 0.65 and m - u < 0.05).

Text map. Under every label: main.tex:550, the clause "on the corrected drift-zero agents of C1
only a balanced diagnostic of the authentic-raised direction exists (0.787 against an untrained
0.540, ...)" is replaced by the registered pooled reading on C1 with its label, the balanced
value kept beside it; main.tex:497 (paragraph "Raised in the authentic world", anchor "On the
corrected C1 agents raised at drift zero and read at 0.45") gains the pooled C1 reading before the
balanced one; PAPER_OUTLINE row 51 gains the C1 values; FINDINGS 17.5.2 and 20.7 get a dated
note; the manifest gets a new entry with claims [51].

- R1. main.tex:497 and the Discussion at :536 (anchor "On historical agents, rearing in the
  surrogate is not required") say the direction is not required on historical and corrected
  agents alike.
- R2. main.tex:497 says the corrected agents do not reach the bar under the registered readout,
  so the historical CARRIED outcome is not repeated on the corrected trainer; :536 keeps "On
  historical agents" and adds the corrected reading.
- R3. main.tex:497 says the level is reached on the corrected agents but is not separated from
  the untrained arm; :536 adds the same.

README, index.template.html, LEARNING and CITATION are unchanged under every label.

Commit key: `git_commit_at_promotion`.

## B4. The C1 pooled population probe, regrouped by world seed

Why. The pooled population probe (`cg_probe`) groups its folds by pooled row, while every
individual runs its common-garden pairs on the same world seeds (base 930000), so one world
seed's pairs from two individuals can land in different folds (main.tex:805, corr:pooledrows).
The C1 readings 0.609 (survival) and 0.536 (untrained) were not regrouped.

Command (R07):

    python scripts/validate_population_readout.py --run-dir $R/corrected_l3_h8_wm --out $CLOSE/b4/corrected_l3_h8_wm_world_grouped.json --group-by-world-seed

Agents, seeds and bases. The ten drift-0.45 survival agents and the ten untrained agents of C1
(seeds 0 to 9), the common-garden tail at drift 0.45, 110 pairs per individual at base 930000,
prefix 20, tail 24, ray steps 5.

Integrity gate. Both values come from the same tails in one pass: `pooled_probe` (row-grouped,
the stored estimator) and `pooled_probe_world_grouped` with its `_ci` and `n_world_groups`. The
output must be finite and carry both. The row-grouped values are compared with the stored
0.608710562414266 (survival) and 0.5362701774429925 (untrained) in
`artifacts/population_readout/corrected_l3_h8_wm.json`. Equality is the expected outcome. Any
difference is a platform note (the stored run used torch 2.14.1+cpu in the revision container,
this machine 2.7.0+cu126), and the contrast is then read between the two local values, never
against the stored ones.

Reading rule, written before the run. Under the explicit fold scheme, grouping the pooled rows by
world seed gives the same folds as grouping by row whenever no pair dies and the number of pairs
is a multiple of five. The scheme assigns equal-size groups to folds round robin in reverse
order, and row i = individual x n_pairs + p lands in the fold of world seed p, so both
groupings put the same rows together. The world-grouped value can therefore differ from the
row-grouped one only through pair deaths. An equal value is the expected outcome and does not
confirm that grouping does not matter. A `--quick` smoke on 2026-10-17 gave equal values for both
populations (0.47622 and 0.53511 at quick scale), as this rule predicts.

Labels.

- EQUAL: the world-grouped value equals the row-grouped value for both populations (difference
  under 1e-12). Reported with the number of pairs that died.
- DIFFERS: any other outcome. The difference per population and the number of dead pairs are
  reported.

Beside either label, claim row 49 ("attenuated, not blind") is checked on the world-grouped
values: it holds when the world-grouped survival value stays above the world-grouped untrained
value and below the per-individual survival mean (0.649). If it does not hold, the row is
qualified as below.

Text map. Under every label: main.tex:805 (corr:pooledrows), the sentences "The code was not
changed. The C1 population readings (pooled 0.609 for survival and 0.536 untrained) are not
regrouped" are rewritten: an opt-in world-seed grouping was added with the default unchanged,
and regrouped by world seed the C1 readings are the measured values; the pilot's tails were not
saved, so its panels still cannot be rescored. main.tex:818, the sentence "It was not changed in
code (item~\ref{corr:pooledrows})" is rewritten the same way.

- EQUAL. main.tex:675 (claims row 49) gains "(the same under world-seed grouping)". FINDINGS:
  the TL;DR line at :150 ("reads 0.609 where independent known-decodable individuals average
  0.649") gains the same parenthesis, and the dated note of 17.11 records the rerun and the
  reading rule above. PAPER_OUTLINE row 49 gains the world-grouped value. index.template.html
  (:717, "a little low (0.609 against 0.649 one by one)"): unchanged.
- DIFFERS. main.tex:675 and :818 quote both values; the FINDINGS TL;DR line at :150 and 17.11
  quote both; PAPER_OUTLINE row 49 gains the world-grouped value; index.template.html :717 quotes
  the world-grouped value beside 0.609. If claim 49 fails on the world-grouped values, row 49's
  "not blind" is qualified to the row-grouped estimator in main.tex:675, PAPER_OUTLINE and the
  FINDINGS TL;DR.

README, LEARNING and CITATION are unchanged under both labels. Manifest: a new entry, claims [49].

Commit key: `git_commit_at_promotion`.

## B5. Surprise AUROC with twin groups for C1, T-touch and S-immortal

Why. The mechanism readouts' prediction-error AUROC (readout 8(a)) gave each episode its own
fold group, while each authentic rollout and its surrogate twin share a world seed and a first
state. The stored 0.846 (C1), 0.912 (T-touch) and 0.848 (S-immortal) are lower bounds
(main.tex:799, corr:surprisetwins). The code now groups twins; this run measures the grouped
values.

Command (R08):

    python scripts/run_mechanism_readouts.py --run-dir $R/corrected_l3_h8_wm --out $CLOSE/b5/C1_mech.json --workers 4 --also-ungrouped-surprise

T-touch and S-immortal use `--run-dir $R/branch_preserve/run/<X>/artifacts/goal_stakes/<X>`,
with the output at `$CLOSE/b5/<X>_mech.json`.

Agents, seeds and bases. The drift-0.45 survival and untrained agents of each run, seeds 0 to 9:
C1 from `fullruns/corrected_l3_h8_wm/agents`, T-touch and S-immortal from the preserved copies of
their run branches. Fresh rollouts at base 1400000 (64 episodes of 80 steps), as stored. The
grouped and ungrouped surprise AUROCs come from the same rollouts in one pass
(`surprise.auroc_ungrouped` beside the grouped value).

Tolerance gate, written before the run. Per seed, |delta| <= 0.01 against the stored
`artifacts/goal_stakes/<run>/mechanism.json` (the C1 file for C1) for four quantities: the
intervention real-minus-sham score, adaptation, surprise_corr, and the stored ungrouped surprise
AUROC (compared with the local ungrouped value). The stored files came from the cloud stack and
record no stack, so a miss is a cross-stack note: it is reported per seed and does not void the
grouped-versus-ungrouped comparison, which is read between two local values from one pass.

NaN rule (H18), written before the run. The stored C1 file records NaN with the reason "empty
pool" on three untrained-arm seeds (100 NaN tokens and three "empty pool" strings; T-touch and
S-immortal record none), so the stored untrained floor rests on 7 of 10 seeds and its surprise on
6. The tolerance applies arm by arm. A seed whose stored value is NaN must come back NaN with the
same empty-pool reason, and it is excluded from the mean. A seed that is NaN on one side and
finite on the other is a FAIL for that arm: it is reported, and that arm's grouped value is not
quoted. The seed basis (for example "7 of 10 seeds") is stated beside every untrained C1 number
the result quotes. A NaN is never a pass.

Labels, per run, on the survival arm's grouped surprise mean (the quantity the frozen surprise
rule reads, threshold 0.65):

- GROUPED ABOVE 0.65: the grouped mean is at or above 0.65.
- GROUPED BELOW 0.65: the grouped mean is under 0.65. The stored verdict is not re-read; the
  record states that the grouped value falls under the threshold the stored verdict read.

The grouping effect, grouped minus ungrouped on the same rollouts, is reported per run with its
interval, and so is the untrained arm's grouped value on its stated seed basis.

Text map. Under every label: main.tex:754 (anchor "whose code now groups pairs but whose stored
values were computed with one group per episode and are kept as lower bounds"), :799
(corr:surprisetwins, anchor "No stored run is rescored"), :818 (anchor "The surprise readout
among the mechanism readouts was fixed with a regression test and not rescored") and :1089
(anchor "lower bounds because the readout split same-world twins across folds") gain the grouped
follow-up values; the stored 0.912, 0.848 and 0.846 stay as recorded. The manifest strings at
`scripts/build_results_manifest.py`:141-148 (the comment "the T-touch and S-immortal agents are
not on this machine; no rescore" and the `SURPRISE_SPLIT_TWINS` text ending "not rescored") are
rewritten to name the grouped follow-up artifact, and the manifest is regenerated. CORRECTIONS:
the dated erratum on the 2026-10-16 surprise entry folds in the result.

- GROUPED ABOVE 0.65: the added clause says the grouped value is above the threshold, so the
  stored lower bound and the verdict agree.
- GROUPED BELOW 0.65: the added clause says the grouped value falls under 0.65 on that run and
  that the stored verdict, which read the ungrouped mean, is not re-read.

README, index.template.html, LEARNING and CITATION are unchanged under both labels.

Commit key: `git_commit_at_promotion`, under a "provenance" key in each output.

## B6. Development: held-out transfer of the 300-update probe

Why. The development readout's transfer column scored the frozen 300-update probe on the
450-update states of the same world seeds it was fit on, so its 0.785 (C1) and 0.689 (C2) are
upper bounds, not held-out transfer (main.tex:797, corr:devtransfer).

Command (R14), after the same command with `--dry-run`:

    python scripts/run_development_heldout_transfer.py --run wm=$R/corrected_l3_h8_wm/agents --run nowm=$R/corrected_l3_h8_nowm/agents --integrity-ref artifacts/development/corrected_l3_h8_wm.json --out-dir $CLOSE/b6 --ckpt $CLOSE/b6/ckpt --workers 2

Output: `$CLOSE/b6/heldout_transfer.json`.

Agents, seeds and bases. The C1 and C2 survival snapshots after 300 and 450 updates, seeds 0 to
9. The probe is fit on the 300-update agent's states on the registered pools, as stored, and
scored on fresh pools at bases 2_700_000 (authentic) and 2_750_000 (surrogate), registered in
`REGISTERED_LATER` in `itasorl/post_review.py` as "B6-auth" and "B6-surr". The runner refuses to
start when those bases collide with `BASES_IN_USE + NEW_BASES`, and the `transfer_probe` overlap
guard refuses a train and test world set that share a seed.

Integrity gate. The runner first recomputes the stored in-sample column: `transfer_from_300` per
seed and budget must equal `summary.aggregate["<b> transfer_from_300"].per_seed` in
`artifacts/development/corrected_l3_h8_wm.json` and `corrected_l3_h8_nowm.json` within 1e-9.
Otherwise stop. On a failure the fallback the owner chose applies: the 0.785 / 0.689 clause is
dropped from main.tex:1293 and FINDINGS 23.4 rather than replaced, and nothing from this run is
quoted.

Labels. One, REPORTED: per run, the held-out transfer at 450 updates with its interval, and the
b = 300 held-out self-transfer row (the 300-update probe on fresh 300-update states), beside the
stored in-sample 0.971 and 0.949 and the cross-validated 0.733 and 0.613. Held-out minus stored
in-sample, seed-paired, is reported descriptively. No threshold reads these values; the
reorganization label rests on the cosine between the 300 and 450-update directions and does not
read this column.

Text map (REPORTED). main.tex:797 (corr:devtransfer), after "are upper bounds and not held-out
transfer", gains the held-out values on fresh worlds. main.tex:1293 (paragraph "Development"):
the sentence that quotes the transfer column gains the held-out values beside the in-sample ones.
FINDINGS 23.4: dated note with both. CITATION: the reorganization sentence at :66-67 ("while the
probe direction reorganizes") stays, since it does not rest on this column; nothing is added
unless the abstract quotes the transfer. README, index.template.html and LEARNING: unchanged.
Manifest: a new entry, readout only on the development readout.

Commit key: `git_commit_at_promotion`.

## B7. L1 noise knockout with fresh noise per episode

Why. The A2 noise knockout reseeded every surrogate world's noise stream to one key per agent
seed, so every surrogate episode of a seed carried the same noise sequence (main.tex:804,
corr:l1noise). The stored survival 0.525, untrained 0.509 and predictor 0.516 were not rescored.
The knockout is already read as vacuous given the null primary encoding, so no reading depends on
this run.

Command (R03):

    python scripts/run_l1_h2_ablations.py --agents-dir $R/l1_heldout/agents --states-dir $R/l1_heldout/states --noise-json $R/l1_noise_calib.json --channels noise --device cpu --noise-reseed per-episode --out-dir $CLOSE/b7

`scripts/promote_h2_batteries.py` is never run on this output.

Agents, seeds and bases. The 30 drift-0.023 agents of the historical L1 held-out run, seeds 0 to
9, three arms. Training pools at 800000 + seed (drift 0) and 850000 + seed (Delta = 0.023), the
transfer pools at 960000 + seed and 970000 + seed with the calibrated noise sigma from
`fullruns/l1_noise_calib.json`; under `per-episode` the k-th surrogate episode of a seed draws
its own Philox stream (`noise_episode_seed`).

Integrity gate. The noise channel has no bit check of its own. It is read only if the same
agents' registered pools bit-match their saved dumps: the historical half of B1
(`$CLOSE/b1_historical/a1_aggregate.json`, `integrity.checked` and `integrity.pools_bit_match`
both true). If that check failed or did not run, B7 reports nothing. The output must also be
finite with n = 10 per arm.

Labels, from the frozen L3-style rule the script applies to the noise channel (`gn_verdict`:
bar 0.65, margin 0.05 over the untrained arm), whose stored call is H2_SUPPORTED (both clauses
fail):

- SAME CALL: the per-episode reading also fails both clauses (H2_SUPPORTED).
- CALL CHANGES: any other call (PARTIAL, H2_NEGATIVE or UNINFORMATIVE).

Per-episode minus stored per-pool, seed-paired per arm, is reported descriptively under both.

Text map. Under every label: main.tex:804 (corr:l1noise), the final clause "is kept as recorded
and is not rescored" becomes "is kept as recorded; rescored with fresh noise per episode, it reads
x (untrained y)" with the call. Per label:

- SAME CALL: the FINDINGS note beside the A2 noise knockout (the dated 2026-10-16 note) and the
  CORRECTIONS follow-up line say the shared draw does not change the call.
- CALL CHANGES: the same two places say the per-episode call differs and that the stored reading
  stays as recorded, the knockout still being vacuous given the null primary encoding.

README, index.template.html, LEARNING and CITATION are unchanged under both labels. Manifest: a
new entry, readout only on L1 (historical).

Commit key: `git_commit_at_promotion`, in `provenance.json` in the out dir.

## B8. Sweep C, a second jitter realization

Why. The sweep C intervals are conditional on one realization of the per-episode jitter. This
draws a second realization at the two points the stored contrast reads.

Command (R02):

    python scripts/run_texture_fresh_probe.py --agents-dir $R/corrected_l3_h8_wm/agents --family arc --param 0.031593406593406590 --rho 0.0 --jitter-key agent --workers 2 --out $CLOSE/b8/C_rho0.0.json

then `--param 0.007631397040054837 --rho 0.99 ... C_rho0.99.json`.

Agents, seeds and bases. The C1 survival and untrained agents at drift 0.45, seeds 0 to 9. The
two points are the stored rho = 0 point (sigma 0.031593406593406590) and the follow-up's
rho = 0.99 point at the shared point's exact gap (sigma 0.007631397040054837), on the sweep C
bases 2500000 (authentic) and 2550000 (the per-episode draw). `--jitter-key agent` keys the draw
by agent seed instead of the pool key, so the realization differs from the stored one.

Integrity gate. Both outputs are finite with n = 10 seeds per arm. Otherwise stop.

Labels, the three readings of `docs/specs/2026-10-15-coherence-finite-horizon-companion-design.md`
applied descriptively to the second-realization contrast M(0.99) minus M(0), where M is the
seed-paired survival-minus-untrained margin of the fresh probe: (1) coherence raises the reading
(lower bound above 0); (2) within the margin unit (interval inside [-0.05, +0.05]); (3) neither.
Beside them, the second-realization contrast is compared with the stored follow-up contrast,
-0.017 [-0.051, +0.016]: CONSISTENT if its mean lies inside that interval, SHIFTED otherwise.

Text map. main.tex:806 (corr:sweepc, anchor "the contrast reads $-0.017$") gains the
second-realization contrast and its reading under every label. FINDINGS 22.3 gets a dated note.
The claim that coherence did not raise the margin stands under readings (2) and (3); under reading
(1) on the second realization, :806 says the two realizations disagree on that claim and the
stored contrast stays as recorded. README, index.template.html, LEARNING and CITATION: unchanged
(their sweep C wording is set by the A3 docs fix, not by this run). Manifest: a new entry,
readout only on C1.

Commit key: `git_commit_at_promotion`.

## C1. Experiment B, the L2 prediction-only arm with correct action timing

Why. The L2 world model is fed (O_t, A_t) and scored against O_{t+1}, which A_{t+1} produces, so
it predicts every next observation without the action that causes it (main.tex:796,
corr:l2timing). The effect on decodability was not measured. This retrains the small GRU world
models under both alignments on this machine. No actor-critic is trained and no agent is saved.

Command (R09). For align in `legacy`, then `obs_then_action`:

    python scripts/run_expB_{full,kstep,nonlinear,gap,surprise}.py --align <a> --out-dir $CLOSE/expB_<a>_local

with the `full` run first, since `run_expB_surprise.py` reads `<out-dir>/expB_full.json` and
fails without it.

Agents, seeds and bases. Three seeds, the stored Experiment B configuration, the drift sweep of
each script (drift 0.45 is the reading). No saved agent is used. Figures go only into the out
dir and are never promoted; the `docs/figures/expB_*.png` files stay as stored.

Integrity gate. Every script writes its JSON, and `git status --short docs/figures` in the run
worktree is empty afterwards. Legacy reproduces when its local drift-0.45 prediction-only target
lies inside the recorded row's 90 percent interval. The recorded row 4 is 0.510 with a population
standard deviation of 0.039 over three seeds (`artifacts/expB/summary.json`), so its t-based
interval is 0.510 plus or minus t(0.95, 2) x 0.039 x sqrt(3/2) / sqrt(3), that is [0.430, 0.591].

Contrast. Aligned minus legacy, both measured on this machine in this pass, per readout. Aligned
against the stored values is not used as a contrast.

Labels, decided on the drift-0.45 prediction-only state target (row 4), stated in advance:

- RESTORE: legacy reproduces and the aligned value lies inside the recorded row's interval
  [0.430, 0.591]. The negative stands with correct timing.
- NARROW: legacy reproduces and the aligned value reaches the 0.65 bar. The prediction-only
  negative then holds only for the one-step-late timing.
- DISCLOSURE STANDS: anything else, including a legacy run that does not reproduce.

(The closing plan worded NARROW as the aligned value dropping below 0.65. The stored value, 0.510,
is already below the bar, so that wording cannot separate any outcome; the label is read as the
aligned arm reaching the bar, the case that would narrow the claim.)

Text map. Rewritten under every label: main.tex:796 (corr:l2timing), "and the effect on
decodability was not measured" is replaced by the aligned and legacy values; README.md:189-192
("and the effect on the state reading has not been measured"); PAPER_OUTLINE.md:69 ("the effect
on the state reading has not been measured"); the L2 timing note in
`scripts/build_results_manifest.py`:191 (ending "was not measured, and the stored numbers stand"),
regenerated. Changed per label:

- RESTORE. main.tex:197 (anchor "under prediction or survival objectives") stays and gains the
  aligned value. main.tex:331 (main-text L2 paragraph, anchor "Its action timing differs from the
  L3 predictor's") states that with correct timing the prediction-only arm reads the aligned value.
  Claims rows 4 to 7 (main.tex:619-622) and PAPER_OUTLINE rows 4 to 7 gain the aligned values
  beside the stored ones. :862 (anchor "its effect on decodability was not measured"), Table
  tab:expB (:866-880) with its caption, the fig:expB caption (:888), the prediction-error
  paragraph (:891), the horizon paragraph (:895, anchor "which reaches it one step late") and the
  fig:kstep caption (:902) gain the aligned values. README.md:168 (Experiment B L2 status row):
  the parenthetical says the corrected timing gives the same result. index.template.html:813
  ("Detectable does not mean learned"): the timing clause says the corrected timing reads the
  same. docs/LEARNING.md:475 (Act 2, anchor "The prediction-only creature, living in a world whose
  ink smears"): one sentence that the rerun with correct timing reads the same. CITATION :20-22
  (the parenthetical "the L2 prediction-only world model received each action one step late, so
  that arm was a weaker predictor than designed"): gains that the corrected timing reads the same.
- NARROW. main.tex:197 becomes "under survival objectives, and under prediction only with the
  one-step-late timing"; :331 and claims rows 4 to 7 say the prediction-only negative holds only
  for that timing; the appendix sites above (:862, tab:expB, :888, :891, :895, :902) quote the
  aligned values; PAPER_OUTLINE rows 4 to 7 say the same. README.md:168, index.template.html:813,
  LEARNING:475 and CITATION :20-22 say the prediction-only negative does not survive correct
  timing. The title is not changed; that decision stays with the owner.
- DISCLOSURE STANDS. main.tex:197 and rows 4 to 7 stay as written; :331, :862 and the appendix
  sites quote the measured aligned and legacy values as a disclosure. README.md:168,
  index.template.html:813, LEARNING:475 and CITATION :20-22 keep their timing caveat with the
  measured values.

Under OD17 the stored figures stay and only their captions change. Manifest: entries for the
legacy and aligned outputs (survival trainer none), claims [4, 5, 6, 7].

Commit key: `git_commit` (agent-free: no saved agent is read).

## G01. Probe-penalty sensitivity

Why. Every probe value on record uses C = 1, and the open L0 gate, the one reason the L3 rule is
not met, is read at that penalty. With 192 feature columns and about 176 training episodes per
fold the penalty sets the probe's effective capacity.

Command (R11, after R10):

    python scripts/run_probe_penalty_sensitivity.py --states $R/l3_h8_heldout/states $R/l1_heldout/states $R/corrected_l1_heldout/states $R/balanced_confirmation_wm/states $R/balanced_confirmation_touch/states $CLOSE/g57/states_local --out-dir $CLOSE/g01

The last dir is dropped if the G57 check fails (R10 ended with a non-zero code, the report is
missing, or `states_local` does not hold one dump per agent as G57 below counts them); the chain
then logs "C1 rows omitted (G57 gate)".

Agents, seeds and bases. Stored dumps only, the registered pools of each run (authentic 800000,
surrogate 850000; the matched-world runs' pooled cells likewise). Grid C in {0.001, 0.01, 0.1, 1,
10}, plus a nested in-fold C chosen by twin-grouped inner cross-validation on the outer training
rows only. Explicit folds, `max_iter` 2000, convergence warnings counted, a shuffled-label
baseline at each C. Drift tags are formatted with `experiment_b2.format_drift` (0.023 is
`0.0230`); a states dir with no file for a listed drift is a hard error.

Integrity gate. At C = 1 the script must reproduce, within 1e-9, one (drift, survival) target per
states dir, read from the record:

| states dir | strongest drift | drift 0 | source |
|---|---|---|---|
| `l3_h8_heldout` | d0.45 = 0.7739256198347108 | 0.5392561983471074 | `artifacts/fold_rescore/l3_h8_heldout.json`, explicit aggregate target means |
| `l1_heldout` | d0.023 = 0.5521487603305786 | 0.5463636363636364 | `artifacts/fold_rescore/l1_heldout.json`, explicit (the cell targets 0.5334 / 0.5221 are the legacy partition and are not the target) |
| `corrected_l1_heldout` | d0.023 = 0.5506 | 0.5482 | cell targets in `fullruns/corrected_l1_heldout/cells`, n = 10 each; the script prints and gates on the full-precision value it reads |
| `balanced_confirmation_wm` | d0.45 = 0.7582 | 0.561 | `fullruns/balanced_confirmation_wm/cells` |
| `balanced_confirmation_touch` | d0.45 = 0.8349 | 0.5359 | `fullruns/balanced_confirmation_touch/cells` |
| C1 (`states_local`) | regenerated targets | regenerated targets | within the G57 tolerance, labeled "cross-stack regenerated, max abs target delta = x" |

The script fails closed on any dir whose target is missing.

Labels, descriptive, per states dir and arm. No TOST and no gate is computed at any C other than
1.

- PENALTY-STABLE: at the strongest drift and at drift 0, the survival target at every C in the
  grid lies within 0.02 of its C = 1 value.
- PENALTY-SENSITIVE: otherwise; the range across C and the C that gives the extreme are
  reported. The drift-0 survival reading of C1 (the open L0 gate) is reported by name under
  either label.

Text map. Under every label: a paragraph and table in the appendix that carries the t-based
intervals (label app:ci95); main.tex:284, the clause "and no probe hyperparameter is tuned"
gains a pointer to that table; FINDINGS gets a dated methods note.

- PENALTY-STABLE: the limitation at main.tex:548 (item "The L0 gate does not pass, on one
  evaluation draw") gains that the drift-0 reading moves by at most the measured amount across C.
- PENALTY-SENSITIVE: :548 gains that the L0 reading depends on the fixed penalty, with its range
  across C.

README, index.template.html, LEARNING and CITATION: unchanged. Manifest: a new methods entry.

Commit key: `git_commit_at_promotion`.

## G02. The drift-0.45 world-sample readout on the saved C1 agents

A frozen spec governs the design: `docs/specs/2026-10-08-drift-045-world-sample-sensitivity-design.md`,
with its dated amendment of 2026-10-17. Its procedure, statistic (signed per-draw means, never
folded) and labels are reused on the corrected agents. This run is post hoc: its R1, R2 and R3
labels are computed and reported but trigger none of the frozen spec's wording, and the 0.042
propagation in the record stays.

Command (R05):

    python scripts/run_world_sample_sensitivity.py --agents-dir $R/corrected_l3_h8_wm/agents --out $CLOSE/g02/d045_world_samples_corrected_l3_h8_wm.json --arms survival --integrity-ref artifacts/fold_rescore/corrected_l3_h8_wm_integrity_ref.json --checkpoint-dir $CLOSE/g02/ckpt

Agents, seeds and bases. The ten drift-0.45 survival agents of C1, seeds 0 to 9. Nine draws: the
eight independent pairs of `l0_audit.AUDIT_BASES` (1000000 + 100000k with 1050000 + 100000k, k
below 8) and the registered pair (800000, 850000) as the reference column.

Integrity gate. The registered draw reproduces the recorded explicit drift-0.45 survival targets
of the integrity reference within 0.01 on every seed: exit code 0, `n_checked` 10, pass true.
Otherwise the run is VOID and no number is reported.

Labels. The frozen R1 SECURE, R2 DRAW-DEPENDENT and R3 INDETERMINATE, computed on C1 and reported
as descriptions. Beside them: C1's between-draw standard deviation, the per-draw table, the
registered draw's rank, and the propagated interval sqrt(SE^2 + SD_C1^2) for the corrected
primary.

Text map, the same under every label. The appendix paragraph app:drawprop (main.tex:1376-1405,
anchor "Combining each pooled-world cell's conditional standard error in quadrature") gains C1's
own between-draw standard deviation and the propagated lower bound beside 0.633, which stays as
the record's propagation with the historical 0.042. main.tex:548 (anchor "the corrected primary's
bar clause would be inconclusive (lower bound 0.633") gains C1's own value and the C1 per-draw
range beside "eight independent draws read 0.728 to 0.860". main.tex:1094 (anchor "a world-sample
shift of the size Section~\ref{sec:l0audit} describes") is made consistent with the C1 spread.
FINDINGS: dated note beside 17.5. README, index.template.html, LEARNING and CITATION: unchanged.
Manifest: a new entry modeled on the historical world-sample entry, claims [50].

Commit key: `git_commit_at_promotion`.

## G03. The balanced sensory-echo control on a second evaluation draw

A frozen spec governs the procedure: `docs/specs/2026-10-13-balanced-sensory-echo-design.md`,
with its dated amendment of 2026-10-17. The verdict at base 910000 is unchanged; the second draw
is descriptive. The untrained floor clause there is met by 0.004 on one draw.

Command (G1, GPU):

    python scripts/audit_sensory_echo.py --agents-dir $R/balanced_confirmation_wm/agents --cells-dir $R/balanced_confirmation_wm/cells --balanced-seed-base 2250000 --integrity-seed-base 910000 --drifts 0.45 --device cuda --out-dir $CLOSE/g03/wm

then

    python scripts/audit_sensory_echo.py --agents-dir $R/balanced_confirmation_touch/agents --cells-dir $R/balanced_confirmation_touch/cells --objective touch --mortal off --balanced-seed-base 2250000 --integrity-seed-base 910000 --drifts 0.45 --device cuda --out-dir $CLOSE/g03/touch

Distinct out dirs, no `--resume`, never inside the shipped run folders.

Agents, seeds and bases. The B1 and B2 agents (seeds 10 to 19, three arms each), drift 0.45, on
CUDA as recorded. Integrity is read from the 910000 rollout; the readings come from the fresh
read base 2_250_000 ("G03-fresh" in `REGISTERED_LATER`), 110 pairs, the same twin-pair folds.
2_200_000 is the post-review balanced pairs' base and is refused by the script.

Integrity gate. At 910000 every cell's regenerated balanced target matches the recorded
`agents[arm].balanced.target` within 5e-4, the first states match and no pair is dropped, as the
frozen spec requires. The base collision check at 2_250_000 must be empty. Every
`adjudication_blockers()` check stays. If pairs drop at the fresh base, `n_pairs` or a WITHHELD
zone is reported as it comes out and nothing is filled in.

Labels, the frozen floor and trained clauses computed descriptively on the second draw: floor
explained (untrained `resid_obs` mean within 0.1 of 0.5) or not explained; per run,
sensory-independent under matched worlds, attenuated or largely sensory-mediated. The raw
untrained balanced mean on the second draw is reported beside them.

Text map. Under every label: main.tex:354 (anchor "Both floor verdicts, the raw failure at 0.614
and the controlled pass at 0.596, rest on the single evaluation draw at base 910000") gains the
second draw's raw and controlled untrained values and the trained readings.

- Floor met on the second draw too: main.tex:198, :202 and :554 (anchor "whose untrained floor
  clause is met by 0.004 on one evaluation draw" and its variants) say the floor clause is met on
  two draws, with both distances.
- Floor not met on the second draw: :198, :202 and :554 keep "met by 0.004 on one evaluation
  draw" and add that a second draw does not meet it, with its distance from 0.5.

FINDINGS 21: dated note. CORRECTIONS: follow-up line. README, index.template.html, LEARNING and
CITATION: unchanged. Manifest: new files `balanced_confirmation/sensory_echo_b{1,2}_fresh.json`,
readout only on B1 and B2; the existing B1 and B2 files are not touched.

Commit key: `git_commit_at_promotion`.

## G04. The within-episode decoding curve

Post hoc: a within-episode curve on the historical dumps was looked at in a scratch read before
this spec was written (recorded in the dated 2026-10-17 note of
`docs/specs/2026-10-11-post-review-readouts-design.md`).

Command (R12, after R11):

    python scripts/run_step_resolved_readout.py --states $R/l3_h8_heldout/states $R/l1_heldout/states $R/corrected_l1_heldout/states $R/balanced_confirmation_wm/states $R/balanced_confirmation_touch/states $CLOSE/g57/states_local --out-dir $CLOSE/g04

The same dirs and the same drop rule as G01.

Agents, seeds and bases. Stored dumps of the registered pools. The registered linear probe on h_t
alone at steps 1, 2, 4, 8, 12, 16 and 24, for every arm, at each dir's strongest drift (0.45, or
0.023 for the two L1 runs) and at drift 0. C1 reads the G57 regenerated dumps.

Integrity gate. Before any curve is read the script reproduces each dir's registered pooled
target (C = 1, explicit folds, the concatenated [mean h, h_T] feature) against the per-dir target
table of G01, within 1e-9 (C1 within the G57 tolerance), and the step-24 column, the probe on
h_T alone, must equal the h_T component of the recorded target where one is recorded (R205's
"final" feature set for C1). A unit test pins the step-24 column. Failure stops that dir.

Labels, per dir and arm at the strongest drift, on the seed-paired step-24 minus step-1 target:

- BUILDS WITHIN THE EPISODE: lower bound above 0.05.
- FLAT: interval inside [-0.05, +0.05].
- UNRESOLVED: otherwise.

The first step at which the arm's mean reaches 0.65 is reported descriptively.

Text map, the same under every label: one sentence beside the integrated-deviation lead saying
that the curve does not by itself separate coherence from integrated deviation. The closing plan
points at main.tex:760; that line is the gate-calibration paragraph, so the sentence goes with
the limitation that names integrated deviation (main.tex:551, item "Comparators", anchor "not on
detectability or integrated deviation"). FINDINGS: dated methods note. README,
index.template.html, LEARNING and CITATION: unchanged. Manifest: a new methods entry.

Commit key: `git_commit_at_promotion`.

## G07(a). World features on each arm's own episodes

A frozen spec governs the design: `docs/specs/2026-10-15-world-draw-features-design.md`, whose
secondary protocol, each arm on its own episodes, was never run. This is that protocol, run after
the replay numbers were seen, so it is post hoc and its labels are descriptive. The stored replay
artifacts are not rewritten.

Command (R15):

    python scripts/run_world_features_readout.py --agents-dir $R/corrected_l3_h8_wm/agents --protocol own --workers 2 --ckpt $CLOSE/g07/ckpt_wm --out $CLOSE/g07/corrected_l3_h8_wm.json

then the same with nowm (`$R/corrected_l3_h8_nowm/agents`, `$CLOSE/g07/ckpt_nowm`,
`$CLOSE/g07/corrected_l3_h8_nowm.json`). The output never goes to
`artifacts/world_features/corrected_l3_h8_*.json`.

Agents, seeds and bases. As the frozen spec and its amendments: C1's drift-0.45 survival,
predictor and untrained arms and drift-0 survival arm, and C2's survival arm, seeds 0 to 9; 1100
episodes of 24 steps per seed at base 2600000, ray steps 5; the reach radius calibrated at base
2650000; Euclidean distances; ridge strength chosen in fold. Each arm now runs its own policy on
those world seeds.

Integrity gate. Every value is finite, and the output carries the per-episode feature values and
the per-seed values for every arm and feature that the spec's Output section names. Otherwise
stop. Reported beside, not gated: the drift-0.45 survival arm's own-episode values against its
stored replay values (the same agent on the same episodes).

Labels: the frozen readings 1 to 4 and the labels "follows usefulness", "keeps everything",
"keeps nothing beyond the senses" and "mixed", computed on the own-episode protocol and reported
as descriptions beside the stored replay labels.

Text map, the same under every label. main.tex:1289 (paragraph "World features"), the sentence
"the specification's secondary protocol, each arm on its own episodes, was not run" is replaced
by the reason recorded at FINDINGS:5832-5834 (an own-episode pass run now is a new readout chosen
after the replay numbers were seen, so it can change no reading or label) and the own-episode
residual R^2 range. FINDINGS 23.2: dated note with the own-episode readings. README,
index.template.html, LEARNING and CITATION: unchanged. Manifest: new files under
`artifacts/world_features/own_episode_posthoc/`, label-neutral.

Commit key: `git_commit_at_promotion`.

## G47. The L1 gate-0 oracle under both fold partitions

Why. README's first number, the detector AUROC 0.873 at Delta = 0.023, reproduces under the
legacy partition; the stack-independent explicit partition gives 0.871, and no document gives the
command.

Command (R01):

    ITASORL_FOLDS=legacy python scripts/run_expA_l1.py --out $CLOSE/g47/gate0_l1_legacy.json --deltas 0.022 0.023 0.024

then the same with `ITASORL_FOLDS=explicit` to `gate0_l1_explicit.json`.

Agents, seeds and bases. Agent-free: the oracle on matched pairs at the recorded calibration
settings (sensor noise 0.01), at grids 0.022, 0.023 and 0.024.

Integrity gate. Legacy gives 0.872875 at 0.023, bit-equal to `artifacts/expA/l1_calib.json`;
explicit gives 0.871125. Otherwise stop and promote nothing.

Labels. One, REPRODUCED, when the gate passes. The neighbouring grids are reported beside.

Text map. FINDINGS methods note 8 gains the row "gate 0 L1 (delta 0.023): oracle 0.873 legacy /
0.871 explicit". main.tex:298 (anchor "the organism's finer grid, $\Delta = 0.023$, at 0.873")
and :524 (anchor "(oracle 0.873)") gain "(legacy partition; 0.871 on the stack-independent
one)". README :8 and :196 may gain "(0.871 on the stack-independent partition)". docs/REPRODUCE.md
gains an agent-free calibration block with the two commands. index.template.html, LEARNING and
CITATION: unchanged. Manifest: a new entry, experiment A, claims []. If the gate fails, the
wording written for the no-run path stays and cites no new artifact.

Commit key: `git_commit` (agent-free).

## G57. Regenerating the C1 state dumps on this machine

Why. The documented retrain-free route to the 0.733 headline regenerates C1's dumps from the saved
agents and checks each pool against its recorded target bit for bit, which holds only on the
original container. On this machine the regeneration is a cross-stack replay.

Command (R10):

    python scripts/regenerate_state_dumps.py --run-dir $R/corrected_l3_h8_wm --out-dir $CLOSE/g57/states_local --report $CLOSE/g57/c1_regen.json --tolerance 0.002 --workers 2

Never inside the shipped run folder.

Agents, seeds and bases. Every saved agent of C1 that the script's task list matches
(`agent_d{0.00,0.45}_s{0..9}_{survival,predictor,untrained}.pt`; the `_u100`, `_u200` and `_u450`
snapshots are not matched), 60 agents, each with the knobs and recorded target of its cell
(`fullruns/corrected_l3_h8_wm/cells/cell_d*_s*.json`), on the registered pools.

Integrity gate. Exit code 0, one dump written per agent (60; the closing plan's count of 30 is
the drift-0.45 half, which is what X07's "30 of 30 arm-cells" counts), and every |regenerated
target minus recorded target| at most 0.002 (the report's per-cell `reproduced_exact` and
`within_tolerance`, `mode: cross_stack_replay`, and the stack versions).

Labels.

- WITHIN TOLERANCE: every cell within 0.002. The maximum absolute delta and the count of cells
  equal to the bit are reported. C1 rows downstream (G01, G04, X05, X06) are labeled
  "cross-stack regenerated, max abs target delta = x".
- TOLERANCE EXCEEDED: any cell beyond 0.002. The C1 rows of G01 and G04 are omitted, the miss is
  reported, and the run chain logs "C1 rows omitted (G57 gate)".

Text map. docs/REPRODUCE.md: the command at :86 gains the `ITASORL_FOLDS=explicit` prefix and
`--report`; the bit-equality sentence at :81-83 ("every pool is checked against its recorded
target") says the check is bit-equal on the recording stack and within a stated tolerance
elsewhere; the replay table gains a row for the C1 regeneration on this machine with the count of
equal cells and the maximum delta under either label, naming `artifacts/repro_check_c1_regen/`.
The bundle text in `scripts/reproduce.py` gets the same prefix. main.tex, FINDINGS (beyond a dated
note), README, index.template.html, LEARNING and CITATION: unchanged. Manifest: a new entry
"C1-REGEN-LOCAL" for the report only, in a top-level folder of its own.

Commit key: `git_commit_at_promotion`.

## H31. The drag comparator's oracle at the setting used

Why. The quadratic drag was run at eps = 6.0 (main.tex:1230; FINDINGS:852, :873, :3990), but the
gate-0 sweep has rows only at eps 0.5, 1, 2, 4, 8 and 16 (`artifacts/texture/gate0_qd.json`), and
the paper's no-in-band argument quotes eps 4 and 8.

Command (R16), timed first with
`python scripts/run_expA_l3.py --family qd --qd-epss 6.0 --floor-seeds 0 --n-pairs 6 --device cpu --json $CLOSE/timing/h31.json`:

    python scripts/run_expA_l3.py --family qd --qd-epss 4.0 6.0 8.0 --device cpu --json $CLOSE/h31/qd_eps6_oracle.json

No `ITASORL_FOLDS` prefix: the script uses no fold scheme.

Agents, seeds and bases. Agent-free: the gate-0 oracle on 60 matched pairs and the script's own
untrained floor seeds, on world P, at eps 4, 6 and 8.

Integrity gate. In the same output, the eps 4 and eps 8 oracle readings equal the stored
`oracle_auroc` of `gate0_qd.json` bit for bit (0.6555555555555556 and 0.6972222222222222).
Otherwise VOID. The stored floors (0.5601928374655648 and 0.6816804407713498) are compared within
0.01, and a miss there is a cross-stack note, not a void, since the stored file records no stack.

The result is descriptive, enters no gate and changes no verdict: the drag's gate 0 stays failed
whatever eps 6 reads, and "eps 6.0 was not calibrated through gate 0" stays true. The three-seed
floor at eps 6 is labeled the gate-0 floor and is kept apart from the C1 untrained arm's 0.612 on
the drag pools.

Labels, with the registered gate-0 band [0.85, 0.95] (the `band` field of `gate0_qd.json` and
main.tex:112):

- (a) IN BAND: the eps 6 oracle lies inside [0.85, 0.95]. main.tex:1230's "Its gate-0 sweep found
  no in-band $\epsilon$" is rewritten to quote the reading at the setting used.
- (b) OUTSIDE: anything else. :1230's sentence gains "(the $\epsilon = 6.0$ setting used reads x)".

(The closing plan gave the window as "0.60 to 0.90, the gate-0 window the record uses". The record
uses [0.85, 0.95], so that is the window read here.)

Text map, beyond the per-label sentence: FINDINGS 14.5 gets the number under both labels;
main.tex:538 (anchor "its oracle reads 0.656 at $\epsilon = 4$ and 0.697 at $\epsilon = 8$")
may gain the eps 6 value between them. README, index.template.html, LEARNING and CITATION:
unchanged. Manifest: the agent-free entry that owns the file.

Commit key: `git_commit` (agent-free).

## H53. A nonlinear state probe on the corrected L1 agents

Why. L2 and L3 have nonlinear probe readings; the L1 negative is read by the linear probe class
alone.

Command (R17, after R06 ended with exit code 0 and its bit check passed; dry run first):

    python scripts/run_l1_nonlinear_probe.py --states-dir $R/corrected_l1_heldout/states --arms survival predictor untrained --drifts 0.00 0.023 --out $CLOSE/h53/l1_nonlinear_corrected.json

If B1's corrected bit check did not pass, the chain logs "H53 skipped (B1 bit-check)" and nothing
runs.

Agents, seeds and bases. The stored dumps of the corrected L1 rerun (registered pools), seeds 0
to 9, three arms, drifts 0 and 0.023 (file tag `0.0230`). An MLP (the battery's 64-unit recipe)
and a random forest (200 trees, `random_state` 0) on the pooled feature, explicit folds.

Integrity gate. The linear C = 1 survival target at drift 0.023 reproduces 0.5506 (the cell
targets of `fullruns/corrected_l1_heldout/cells`, full precision as read) within 1e-9. Otherwise
nothing is reported.

Labels, on the survival arm at drift 0.023:

- (a) Either nonlinear probe reads at or above 0.65.
- (b) Both read under 0.65.

Text map.

- (a) The L1 negative is qualified as "under the linear probe" at main.tex:88 (anchor "yet the
  state reads 0.551"), :170 (anchor "an in-band L1 seam did not meet the encoding criterion") and
  :549 (anchor "The L1 negative therefore does not separate absence of the seam").
- (b) main.tex:344, "Nonlinear probes read the C1 survival state at about the linear value",
  gains the L1 clause with the two values.

Under both: FINDINGS gets a dated note beside the corrected L1 section; README (the H04 L1
sentence) gains the qualifier only under (a); index.template.html, LEARNING and CITATION:
unchanged. Manifest: a new entry, readout only on L1-CORRECTED. The result is descriptive and
changes no verdict.

Commit key: `git_commit_at_promotion`.

## Text map for shipping the C1 and C2 agents (H12)

If the review supplement carries the C1 and C2 saved agents and rerun cells, three sentences of
`docs/REPRODUCE.md` become false. Each is rewritten from the decision the scratch size build
makes, with the wording fixed here.

- Both shipped. :80 ("The raw state dumps of `C1` and `C2` are not in git or in the supplement.")
  becomes "The raw state dumps of `C1` and `C2` are not in git or in the supplement, but the
  review build carries both runs' saved agents and rerun cells (`fullruns/corrected_l3_h8_wm/` and
  `fullruns/corrected_l3_h8_nowm/`, per-file digests in `SHA256SUMS`), so the dumps rebuild from
  them without retraining." :89-90 ("The review build carries no `C1` or `C2` agents, so from it
  the dumps need a retrain") becomes "From the review build the dumps rebuild with the command
  above; a full retrain (`reproduce.py retrain C1 --execute`, section 2) is needed only to
  reproduce the training itself." :220 ("only `fullruns/l3_h8_heldout/agents` and
  `fullruns/l3_h8_gseed2_gpu/agents`, the saved agents") becomes a list that adds the C1 and C2
  agents and rerun cells.
- C1 only. The same three sentences name C1 alone, and :89-90 keeps the retrain route for C2:
  "The review build carries the `C1` agents and rerun cells, so C1's dumps rebuild with the command
  above; C2's need a retrain (`reproduce.py retrain C2 --execute`, section 2)."
- Neither. The three sentences stay as written.
