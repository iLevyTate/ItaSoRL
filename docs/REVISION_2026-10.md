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

## Steps and status

| Step | What | Status |
|---|---|---|
| 1 | Results manifest, provenance, claim decisions | Done 2026-10-06: `docs/RESULTS_MANIFEST.md`, `artifacts/results_manifest.json`, this file, `docs/CORRECTIONS.md` |
| 2 | Successor-state bootstrap in `itasorl/experiment_b2.py` | Done 2026-10-06; historical trainer kept as `--gae-bootstrap pre_transition`, verified bit-identical to `4b6e1f3` |
| 3 | Tests for the intended bootstrap semantics | Done 2026-10-06: `tests/test_gae_bootstrap.py`; mutation-checked |
| 4 | Measure the correction: diagnostic, frozen protocol, corrected confirmation runs | In progress: historical cell reproduced bit for bit on this container; quick-scale diagnostic passed end to end; protocol frozen (`docs/specs/2026-10-06-corrected-trainer-confirmation-design.md`, PREREGISTRATION_L3 2026-10-06 entry); runs `C1` and `C2` launched |
| 5 | L0 and the fold partition | Pending |
| 6 | Arm-by-arm training and evaluation table; predictor evaluated under its own policy | Pending |
| 7 | Memory: controlled persistence test or narrower claim | Pending |
| 8 | Behavioral and sensory controls | Pending |
| 9 | Texture knockout interpretation | Pending |
| 10 | Oracle and surrogate descriptions | Pending |
| 11 | Auxiliary-loss conclusion bounded by budget | Pending |
| 12 | Evolutionary readout | Pending |
| 13 | Statistical terminology and estimators | Pending |
| 14 | Claims and methods rewritten consistently | Pending |
| 15 | Related work and the reproducibility package | Pending |

## Where the manuscript lives

The LaTeX manuscript (`docs/paper`) is local to the owner's checkout and is not in git. This
revision edits what is in the repository: `docs/FINDINGS.md`, `docs/PAPER_OUTLINE.md`,
`README.md`, `CITATION.cff`, the site template, and the plain-language pages. Step 15 adds a
check that the manuscript's tables match the committed results when the manuscript source is
present; the owner applies the same claim table to the LaTeX source.
