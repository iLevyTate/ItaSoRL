# Revision 2026-10: correcting the trainer and bringing the claims in line

Started 2026-10-06 on branch `claude/affectionate-carson-azhxt2`, from the reviewed commit
`4b6e1f3`. This file holds the plan, the claim decisions, and the status of each step.
Dated corrections go in `docs/CORRECTIONS.md`. Which implementation produced each committed
number is in `docs/RESULTS_MANIFEST.md`, generated from `scripts/build_results_manifest.py`.

## Ground rules

- Every number published before this revision is **historical**. It stays in its artifact
  and its document, labeled as produced by the pre-correction trainer, until a corrected run
  replaces it. A corrected run adds a new artifact; it never overwrites an old one.
- The reviewed commit `4b6e1f3` is the retained original. It is reachable in git history, and
  the corrected trainer keeps the historical bootstrap behind an explicit flag so the original
  training can be rerun bit for bit (step 2).
- Frozen preregistrations are not edited. Corrections and new protocols are appended to their
  section 12 deviation logs as dated amendments, and new designs get their own spec under
  `docs/specs/`.
- A gate that fails or cannot be shown to pass is reported as failed or open. No margin is
  relaxed after the fact.

## Claim decisions

The paper is re-centered on one question: under a stated training protocol, is the condition
(authentic or surrogate dynamics) decodable from the agent's recurrent state, beyond the
named baselines and controls? Everything else is secondary or exploratory, and is worded to
match the estimator that measured it. Claim numbers are the rows of the claims inventory in
`docs/PAPER_OUTLINE.md`.

| Claims | Topic | Decision | Needs a corrected run? |
|---|---|---|---|
| 1, 2, 3, 11, 36 | Oracle ceilings (agent free) | Retain. Reword as privileged detector scores at a stated detector-side noise (step 10). | No: no actor-critic involved |
| 4, 5, 6, 7 | Experiment B, prediction-only L2 null | Retain as historical. Reword "not encoded" as "did not meet the registered encoding criterion" (step 14). | No: no actor-critic involved |
| 12, 13, 14, 15, 16 | L3 headline: survival with the next-observation auxiliary at 300 updates, against predictor and untrained arms, with its gates | **Primary.** Replace with the corrected ten-seed run (step 4). | Yes |
| 29, 30 | Auxiliary removed, and the device control | **Primary comparison.** Replace with corrected no-auxiliary and with-auxiliary runs on one device (step 4). Bound by budget (step 11). | Yes |
| 17, 18, 19, 20, 28 | Behavior and sensory residualization | Secondary. Reword to name the features and the residualization model (step 8). Re-run on the corrected agents' dumps. | Readout on corrected agents |
| 24 | Common garden ("persistent") | Narrow to "prefix condition remains decodable after restoring authentic dynamics" unless the controlled persistence test of step 7 is run. | Optional (step 7) |
| 21, 22, 23, 25, 26, 31 | Second capacity, held-out and cross-recipe transfer, second instances | Secondary, historical, labeled with the pre-correction trainer. | Not in this revision |
| 32, 33, 34 | H2 graded seam, texture knockout, observation localization | Exploratory, historical. Texture result narrowed to "the original direction does not transfer" (step 9). | Not in this revision |
| 8, 9, 10, 35 | L2 survival arcs and the L1 organism | Historical, labeled with the pre-correction trainer; the registered-criterion wording applies. | Not in this revision |
| 27 | Experiment C | Exploratory, out of the abstract. Restricted to the pooled population readout (step 12). | No: no actor-critic involved |
| 37 | Engagement-margin sweep | Historical gate audit. | No |
| (skill-matched baseline) | Return matching at 450 updates | Stays a **failed** match. No mediation inference either way (step 11). | No |

### Decisions after the corrected readouts (2026-10-06, FINDINGS 17)

| Claims | What the corrected agents showed | Wording from here on |
|---|---|---|
| 12 to 16 | `C1` replaces the device control: 0.733 [0.669, 0.797], margins clear 0.05, L0 open on the registered pair and equivalent across independent pairs (17.2, 17.5) | "Under the stated protocol, with each arm's own policy driving the episodes, the condition is decodable from the survival agent's state above the predictor and untrained arms; MET on the decodability clauses, L0 open." |
| 12 to 14, every "survival-specific" phrase | On replayed survival streams the predictor reads 0.721 and the exposure-matched predictor 0.738 against 0.733; under one scripted policy every arm reads 0.55 to 0.58 (17.6) | The margin over the predictor is a difference between training regimes. At matched input it is not a difference of objective; the signal rides on the trajectories the survival policy generates. "Survival-specific" and "encoded by the survival objective" are withdrawn. |
| 17 to 20, 28 | The best-fitting nonlinear control leaves 0.620, under the bar; a decoder of the observation stream alone reads 0.724 (17.8) | Name the basis and model; say that the strongest well-fit control leaves a signal under the bar. |
| 24 | Retention under identical input not shown; the tail is decodable mostly from the physical footprint (17.7) | "Prefix condition remains decodable after restoring authentic dynamics, mostly through the physical footprint of the prefix." |
| 32, 33 | White jitter is read by no arm; a hand-authored coherent drag at matched one-step RMS is read (17.9) | Decoding tracks temporally coherent, state-dependent deviations among the tested classes; nothing specific to learned dynamics. |
| 29, 30 | `C2` 0.613 at 300 updates against `C1` 0.733 on one device, +0.120 [+0.065, +0.174]; 0.685 without the decoder at 450 updates (17.4, 17.10) | "At the registered 300-update budget the auxiliary-conditional reading holds on the decodability clauses; without the decoder the arm approaches the bar with more training, and at 450 updates the decoder arm still reads higher." |
| 27 | The pooled probe reads 0.609 where independent individuals average 0.649 (17.11) | Restricted to the pooled readout; informative about individual signals of that size. |

## Steps and status

| Step | What | Status |
|---|---|---|
| 1 | Results manifest, provenance, claim decisions | Done 2026-10-06: `docs/RESULTS_MANIFEST.md`, `artifacts/results_manifest.json`, this file, `docs/CORRECTIONS.md` |
| 2 | Successor-state bootstrap in `itasorl/experiment_b2.py` | Done 2026-10-06; historical trainer kept as `--gae-bootstrap pre_transition`, verified bit-identical to `4b6e1f3` |
| 3 | Tests for the intended bootstrap semantics | Done 2026-10-06: `tests/test_gae_bootstrap.py`; mutation-checked |
| 4 | Measure the correction: diagnostic, frozen protocol, corrected confirmation runs | Done 2026-10-06 (FINDINGS 17.1 to 17.4, `docs/CORRECTIONS.md`): `C1` 0.733 [0.669, 0.797], MET on the decodability clauses conditional on the open L0 gate; `C2` 0.613, NOT MET; `C1` minus `C2` +0.120 [+0.065, +0.174]; correction effect +0.003 and +0.012 at drift 0.45 |
| 5 | L0 and the fold partition | Done 2026-10-06: primary analysis frozen; folds versioned and serialized; `docs/GATE_TABLE.md`; pre-intervention audit; agent-based audit on the `C1` agents (FINDINGS 17.5: independent world-sample pairs average 0.493, sd 0.049; balanced readout 0.767) |
| 6 | Arm-by-arm training and evaluation table; predictor evaluated under its own policy | Done 2026-10-06: `docs/METHODS_ARMS.md`; readouts on the `C1` agents (FINDINGS 17.6): on replayed survival streams the predictor reads 0.721 and the exposure-matched predictor 0.738 against survival 0.733, so the survival-over-predictor margin is not a difference of objective at matched input |
| 7 | Memory: controlled persistence test or narrower claim | Done 2026-10-06 (FINDINGS 17.7): retention under identical input not shown (replay 0.561); with memory zeroed the tail still reads 0.662, so the common-garden signal is mostly the prefix's physical footprint; narrow wording kept |
| 8 | Behavioral and sensory controls | Done 2026-10-06 (FINDINGS 17.8): the best-fitting control (MLP, held-out R^2 0.89) leaves 0.620, under the bar; a decoder of the observation stream alone reads 0.724; "behavior-independent" retired |
| 9 | Texture knockout interpretation | Done 2026-10-06 (FINDINGS 17.9): white jitter is read by no arm; a hand-authored coherent drag at matched one-step RMS is read (transfer 0.721, fresh 0.754); the tested classes separate on temporal coherence, not learnedness. Comparator-trained runs not run (`qd` failed gate 0) |
| 10 | Oracle and surrogate descriptions | Done 2026-10-06: FINDINGS 10.1.1 (surrogate as implemented, linear-fit control, held-out one-step error vs rollout divergence, detector-side sigma, privileged detector score, agent-accessible detector at 0.991); PREREGISTRATION_L3 amendment on fixed epochs; audited |
| 11 | Auxiliary-loss conclusion bounded by budget | Done 2026-10-06 (FINDINGS 17.10): without the decoder 0.613 at 300 and 0.685 at 450 updates; at 450 the decoder arm still reads +0.136 higher at about equal return; skill-matched baseline stays a failed match; exploratory cross-replay splits the effect between trunk and trajectories |
| 12 | Evolutionary readout | Done 2026-10-06: per-individual readout, lineage summary, value of world information; validation on the `C1` agents (FINDINGS 17.11): the pooled probe reads 0.609 where individuals average 0.649; the evolution is not rerun, so the claim stays restricted to the pooled readout |
| 13 | Statistical terminology and estimators | Done 2026-10-06: ROPE relabeled as a bootstrap check (FINDINGS methods note 9, `itasorl/stats.py`); per-cell intervals tied to their estimator, pair-level resampling for paired readouts, `fold_mean_auroc_ci`; seed-paired margin intervals for every run (`docs/CONTRAST_INTERVALS.md`); conditioning stated; analysis tiers fixed |
| 14 | Claims and methods rewritten consistently | Done 2026-10-06: FINDINGS (banner, TL;DR, narrowed sections, section 17), README, CITATION.cff, the site (numbers generated from the corrected artifacts), PAPER_OUTLINE (status column, rows 38 to 49), LEARNING (Act 10), ITASORL.md, METHODS_ARMS, the world spec; retired phrasings guarded by the audit on the public pages and flagged by the manuscript checker |
| 15 | Related work and the reproducibility package | Done 2026-10-06. Meta-RL comparison (RL^2, Mikulik et al., VariBAD, PEARL, RMA) in `docs/ITASORL.md`; `docs/REPRODUCE.md`; `scripts/reproduce.py` (tables without training, retrain recipes, anonymized supplement verified to pass the tests and the full audit when extracted); frozen surrogates in `artifacts/surrogates/`; manuscript tables and check (`scripts/build_paper_tables.py`); the audit's stated purpose corrected |

## Where the manuscript lives

The LaTeX manuscript (`docs/paper`) is local to the owner's checkout and is not in git, and
everything generated for it stays local too: `scripts/build_paper_tables.py` writes its
tables to the gitignored `docs/paper_tables/`, beside the manuscript directory, never inside it. This
revision edits what is in the repository: `docs/FINDINGS.md`, `docs/PAPER_OUTLINE.md`,
`README.md`, `CITATION.cff`, the site template, and the plain-language pages. Step 15 adds a
check that the manuscript's tables match the committed results when the manuscript source is
present (the audit renders the tables in memory, so CI needs no paper files); the owner
applies the same claim table to the LaTeX source.
