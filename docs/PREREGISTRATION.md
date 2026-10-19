# Pre-Registration - ITASORL Experiment B-v2 (Survival-Coupled Incidental Detection)

**Status:** pre-registered *before* the confirmatory run. **Date:** 2026-06-28.
**Branch:** `expB2-survival-coupling`. Configuration below is frozen at commit time;
any later change is logged in §12.

## 1. Background and the contrast

Experiment B trained an agent *only* to predict its sensory stream and asked whether
world identity is incidentally decodable from its recurrent state. It is not
(pooled target AUROC ≈ 0.50). A privileged oracle detects the same L2 artifact at
≈ 0.99 (Experiment A). What the objective does not require, the agent does not
represent. B-v2 changes one thing: the agent now **acts to stay alive in a world
whose dynamics drift**. Survival depends on coping with the drifting drag, so
modelling those dynamics is useful for dinner.

## 2. Hypothesis

- **H_B2 (primary):** a survival actor-critic encodes world identity in its recurrent
  state *above* the prediction-only and untrained baselines and *above chance*.
- **Null:** pooled target ≈ 0.50, no better than the baselines - readout-not-reward
  bites even under survival pressure.

Either outcome is a substantive, reportable result.

## 3. Agents - identical trunk and identical readout; only the objective differs

| agent | objective | role |
|-------|-----------|------|
| `untrained` | random init (normalizer warmup only) | mechanical floor - drift perturbs inputs, so any recurrent code separates the worlds somewhat |
| `predictor` | next-step prediction, scripted policy | Experiment B's objective on this trunk (prediction, no survival) |
| `survival` | recurrent A2C + GAE (+ world-model decoder aux) | acts to stay alive under drifting dynamics |

"Survival induces encoding" requires `survival > predictor > untrained`.

## 4. World and the survival coupling (readout-not-reward preserved)

- `PatchOfEarthV0` with **harsh metabolism** (`E0=1.0, basal_E=0.4`): a non-forager
  starves in ≈ 50 steps, so staying alive *requires* foraging - and foraging
  efficiency depends on the L2-drifting drag that governs movement.
- **Denser food** (`n_pellets=24, reach=0.08, pellet_r=0.03`) so the eat-reward is not
  impossibly sparse for a from-scratch policy.
- **World identity is never in the observation or the reward.** Reward is the native
  homeostatic survival signal only.
- **Potential-based food-approach shaping** (`Φ = −dist to nearest pellet`) aids
  learning only; it provably preserves the optimal policy, is identical in authentic
  and surrogate worlds, and is *not* world identity. It enters the training reward
  only; reported returns/lifetimes use the TRUE reward.

## 5. Readouts (episode = unit of inference; GroupKFold)

- **PRIMARY - pooled (Experiment B frame):** independent authentic (`drift=0`) vs
  surrogate (`drift=d`) episodes, fixed length, drop early deaths. Probe a *persistent*
  world-identity direction in `episode_features(h_t) = [mean h, final h]`, reusing
  `experiment_b.episode_features` / `probe_auroc` verbatim. ≈ 0.50 ⇒ no incidental
  encoding (directly comparable to Exp B).
- **SECONDARY - matched-pair recurrent:** shared authentic prefix, branch authentic vs
  drift from a bit-identical state. This measures *detectability* of the artifact in
  the agent state, not persistent encoding; reported, but not the headline.

## 6. Estimand and smallest effect of interest

- **Estimand:** per-episode AUROC of the linear world-identity probe (pooled readout),
  averaged over seeds.
- **SESOI:** claim incidental encoding only if the survival pooled target is
  **≥ 0.65 AND exceeds both `predictor` and `untrained` by ≥ 0.05**.

## 7. Gates - ALL must pass before the survival target is interpreted

1. **Engagement:** the trained agent forages meaningfully better than random *and*
   scripted policies - true return higher by ≥ `ENGAGE_MARGIN` (0.15), with lifetime no
   worse than random by `LIFE_TOL` (2 steps). Otherwise the run is **uninformative**.
   (Return is the discriminating signal; lifetime saturates at the frozen food density -
   see §12. Margin calibrated on the de-risk and frozen for the confirmatory run.)
2. **L0 control:** at `drift=0` the pooled target is equivalent to 0.5 by TOST
   (margin ±0.05) - proves the readout manufactures no signal.
3. **Positive control:** the speed probe is high (≥ 0.75) - the state is probeable.
4. **Leakage audit:** world identity is NOT decodable from reward-sum / episode-length
   / lifetime (each within 0.1 of 0.5) - the target reads the artifact, not "I lived
   longer in world X."

## 8. Decision matrix

- **Encoding induced:** all gates pass AND survival target ≥ 0.65 AND > predictor+0.05
  AND > untrained+0.05.
- **Strengthened negative:** all gates pass (incl. engagement) AND survival ≈ 0.50
  ≈ predictor ≈ untrained.
- **Uninformative:** engagement gate fails.

## 9. Fixed configuration (frozen at commit)

drifts `[0.0, 0.45]`; seeds `[0,1,2]`; updates `300`; `n_eps=16`; `max_steps=80`;
`hidden=96`; `ray_steps=5`; `shaping_coef=1.0`; pooled `n=110, steps=24`; matched-pair
`pairs=60, prefix=20, branch=24`. Optimizer Adam `lr=3e-4`, `gamma=0.99`, `lam=0.95`,
`ent_coef=0.01`, `vf_coef=0.5`, `wm_coef=1.0`. Hyperparameters were tuned on a
`drift=0.45` de-risk and are frozen for the confirmatory run.

## 10. Analysis plan

Mean ± std over seeds. Primary contrast: survival vs predictor vs untrained pooled
target across drift. L0 equivalence by TOST. Report every gate. Figure: |target−0.5|
vs drift per agent (`expB2_survival.png` in the run output dir, default repo root or `--out-dir`);
raw metrics in `expB2_results.json`. Published confirmatory copies live in
`artifacts/expB2/`.

## 11. Key interpretive caveat

- The matched-pair readout measures **detectability** (the artifact perturbs inputs of
  any recurrent net); the pooled readout measures a **persistent direction** =
  incidental encoding. Only the latter answers H_B2.
- **Reactive vs. representational:** the agent may adapt to *felt* drag reactively
  without *classifying* the world. The pooled probe across independent episodes is the
  conservative test - a consistent direction implies more than momentary reaction. A
  common-garden / held-out fixed-dynamics probe is the cleaner separation and is future
  work.
- The result is conditional on this architecture, this world family, and L2 only.

## 12. Deviations from pre-registration

- **2026-07-18 - retroactive log entry (methodology audit): the replication
  verdict was rendered with the L0 equivalence gate formally inconclusive.**
  Section 7 requires all gates to pass before the survival target is
  interpreted. The full-scale replication's L0 TOST was inconclusive at n = 3
  (p = 0.20; underpowered, as FINDINGS section 9 discloses in its caveats), yet
  the pre-registered "strengthened negative" verdict was rendered. The gap was
  disclosed in FINDINGS but never logged here as a deviation; it is now. The
  gate later passed cleanly at n = 10 (TOST and ROPE), so the verdict stands;
  the mechanical lesson - the pipeline prints gates and verdicts independently
  rather than conditioning one on the other - is recorded in the audit notes.
- **2026-06-28 - GAE training bug found and fixed; sweep re-run.** A code-hardening
  pass found a real correctness bug in `compute_gae`: the advantage accumulator gated
  its carry with the *current* step's mask instead of the *next* step's, leaking the
  padded value slot into the last step of any episode shorter than `max_steps`. Under
  the harsh metabolism most episodes die early, so the **survival** arm of the first
  confirmatory run was trained with corrupted advantages (the `predictor` and
  `untrained` baselines do not use GAE and were unaffected). Fixed (commit `679fee6`,
  verified against an independent textbook GAE) and the confirmatory sweep was **re-run**
  with the corrected code. The first run's artifacts/numbers are superseded.
- **2026-06-28 - Engagement gate recalibrated on de-risk data.** The fixed-GAE de-risk
  revealed that lifetime **saturates** at the frozen food density (24 pellets, 80-step
  cap): a random agent already survives ~68/80, so the originally-specified "strictly
  longer lifetime" criterion flips on ~1 step of noise even though the trained agent
  forages far better on return (−0.55 vs −0.98). The gate was changed to a **true-return
  margin** (≥ 0.15 over both baselines) with a lifetime "not-worse" check, calibrated on
  de-risk (pilot) data and frozen before the confirmatory re-run - consistent with §9's
  "tuned on the de-risk, frozen for the run" policy. This cleanly separates the engaged
  forager (+0.43) from a non-learner (+0.04).
- **2026-10-06 - CORRECTION: the GAE truncation bootstrap used the pre-transition value
  (revision step 2; `docs/CORRECTIONS.md`).** An episode alive at the 80-step rollout cutoff
  bootstrapped its final residual from V(h_T), the critic value at its last stored step,
  so the final term read r_T + gamma V(h_T) - V(h_T) instead of r_T + gamma V(h_{T+1}) -
  V(h_T). The 80-step cutoff is a sampling truncation of a continuing task (the world has
  no horizon; death is its only terminal event), so the correct bootstrap is the value of
  the successor state. `itasorl/experiment_b2.py` now computes it from one extra GRU step on
  the successor observation with the final action as the previous action, without
  gradients and without another environment step; deaths still bootstrap from 0, and the
  679fee6 padding guard is unchanged. Every survival result from 679fee6 to 4b6e1f3 used
  the old bootstrap and is now labeled historical (`docs/RESULTS_MANIFEST.md`); the
  predictor and untrained arms do not use GAE and are unaffected. The historical trainer
  stays available as `--gae-bootstrap pre_transition` for exact reproduction only. Whether
  the published effect survives the correction is measured, not assumed (revision step 4).
- **2026-10-10 - IN-PLACE EDITS TO EXISTING LINES OF THIS DOCUMENT (a record; nothing is
  changed by this entry).** Three commits changed existing lines here instead of appending.
  `75c49a2` (2026-07-20) changed "The ralph hardening pass" to "A code-hardening pass" in the
  2026-06-28 entry. `5e9e84b` (2026-07-20 local, 2026-07-21 UTC) replaced long dashes with
  hyphens in twenty lines. `8eb9174` (2026-08-23), after the B-v2 runs, rewrote the
  background paragraph of section 1: it dropped the "FINDINGS conjectures" framing and the
  "asks whether incidental encoding now emerges" clause and states the conjecture directly.
  None of these changed a number, a rule, a gate or a verdict. The same commits are listed
  for `docs/PREREGISTRATION_L3.md` in its 2026-10-10 entry.
- **2026-10-16 - H_B2 IS APPLIED TO L1 RUNS IT WAS NOT REGISTERED FOR (a record; no rule,
  number or verdict changes).** This document registers H_B2 for the L2 run only (section 11),
  yet three rule-ledger rows read L1 runs under rules frozen for other runs. `L1-PRIMARY`
  applies H_B2 (sections 6 and 8) to the historical L1 organism run. No L1 rule existed before
  that run: the 2026-07-21 H2 spec lists it only as the deferred A3 positive control, with no
  estimand, gate-0 rule or decision rule, and the L1 code, the Δ = 0.023 gate-0 selection, the
  σ_o noise comparator and the result were first committed together in `01e88e4`
  (2026-07-25), while the historical cells record `8a71593`, which has no L1 drift mode.
  `L1-CORRECTED` names H_B2 as the rule the historical run was read under, so the corrected
  rerun inherits a rule applied after the fact. `NOISE-KNOCKOUT-L1` applies the channel-1
  positive rule of the 2026-07-22 spec, whose only mention of L1 is that it does not collide
  with it. Each borrowed rule was applied with its thresholds unchanged: H_B2 for `L1-PRIMARY`
  and `L1-CORRECTED`, the channel-1 rule (0.65 bar, untrained floor plus 0.05) for
  `NOISE-KNOCKOUT-L1`. `docs/CORRECTIONS.md`, 2026-10-16, and FINDINGS 14.7.
- **2026-10-19 - TIMING OF THE TUNING RUN AND THE ENGAGEMENT MARGIN, STATED IN ONE PLACE (a
  record; no rule, number or verdict changes).** The section 9 configuration was tuned on a
  drift-0.45 de-risk run before this registration was committed (`51e086e`, 2026-06-28); its
  outputs were not committed, and section 9 names no selection criterion. The 0.65 bar and the
  0.05 margins first appear in `51e086e`. The 2026-06-28 engagement recalibration (`5c3aca1`)
  was made after the first confirmatory run (`07ef113`) had been read and the GAE fix
  (`679fee6`) had landed, and before the rerun (`83e9297`, 2026-06-29 UTC). It was calibrated
  on a de-risk run under the fixed trainer; the non-learner reference of the entry above (+0.04)
  is the one the `5c3aca1` code comment calls "the v2 pilot". The 2026-06-28 entry stays as
  written. The manuscript's section 4.3 tabulates this with the later registrations;
  `docs/CORRECTIONS.md`, 2026-10-19, and FINDINGS 11 note 2.
