# Pre-Registration - ITASORL Experiment L3 (Learned-World-Model Surrogate, Incidental Detection)

**Status:** pre-registered *before* the confirmatory run. **Date:** 2026-07-10.
**Gate:** the L2/pooled line is a confirmed negative under scale - B-v3 regime n=10
survival pooled 0.610 (90% CI [0.585, 0.634]) and the sysid-aux capacity ceiling
n=10 pooled 0.596 (90% CI [0.577, 0.616]) both sit entirely below the 0.65 bar, so
the pre-registered "proceed to L3 only if L2 holds under scale" condition is met.
Configuration below is frozen at commit time; any later change is logged in section 12.

## 1. Background and the contrast

Experiments B-v2 (L2 drift, AR(1)) and B-v3 (L2 regime, per-episode constant offset)
asked whether a survival actor-critic incidentally encodes world identity when the
surrogate differs from the authentic world by a *hand-specified dynamics parameter*
(the drag). It does not, and the negative is now near-architectural: even a trunk
supervised directly on drag (the sysid-aux ceiling) reads the pooled
persistent-direction probe at only ~0.60, while the matched-pair detectability channel
reaches ~0.70 - identity is decodable when forced in, but the pooled readout saturates
below the bar (`docs/FINDINGS.md` section 7.1, section 9).

L3 changes *what makes the surrogate a surrogate*. Instead of a hand-tuned drag
parameter, the surrogate world is produced by a **separate neural network trained to
imitate the authentic dynamics**. Its differences from the authentic world are the
**generative fingerprint** of that imitation: smoothing / blur, missing fine detail, a
characteristic learned noise texture, and mode collapse (rare events that should occur
never do) (`docs/ITASORL.md` L3, ladder section). This is a structurally different, more
realistic surrogate than a single dynamics knob, and it is the project's headline target
(`docs/ITASORL.md`).

## 2. Hypothesis

- **H_L3 (primary):** a from-scratch survival agent living across authentic and
  L3-surrogate worlds incidentally encodes world identity in its recurrent state *above*
  the prediction-only and untrained baselines and *above* the SESOI, read out but never
  rewarded.
- **Null:** pooled target ~0.50, no better than baselines - even a rich generative
  fingerprint is not incidentally encoded under readout-not-reward.

Either outcome is substantive. The result is *conditional on this surrogate's difficulty*
(section 7 oracle gate): it says what kind of substrate artifact survival will or will not
encode, not that no fingerprint could ever be encoded.

## 3. Agents - identical trunk and identical readout; only the objective differs

Unchanged from B-v2/B-v3: `untrained` (mechanical floor), `predictor` (next-step
prediction on this trunk), `survival` (recurrent A2C + GAE + world-model decoder aux).
"L3 fingerprint induces encoding" requires `survival > predictor > untrained` and
survival >= SESOI. The probe is identical across all three.

## 4. World and the L3 surrogate (readout-not-reward preserved)

**Signal targeted (decision, section 11):** the organism test targets **Signal A**
(latent geometry) - the agent lives in both authentic and L3-surrogate worlds and the
probe reads world identity from its recurrent state - keeping the "incidental,
readout-not-reward" framing identical to B-v2/B-v3. Signal B (prediction-residual of a
world model frozen on authentic then dropped into the surrogate) is used only inside the
section 7 oracle check, not as the organism headline.

**Surrogate construction:**

1. **Two-stage construction (C -> B), resolved after a code-level integration review
   (2026-07-10).** `G` is a small predictive world-model (`RecurrentWorldModel`) trained
   on authentic `PatchOfEarthV0` rollouts and then frozen. Because `G` predicts
   OBSERVATIONS while the survival world needs full internal STATE (energy, food, death),
   "wrap `G` as the World" is realized in two stages:
   - **Stage 1 - oracle gate (Option A, observation-channel).** Authentic physics is kept
     intact (survival stays well-defined); an L3-surrogate trajectory is an authentic
     rollout whose OBSERVATIONS are passed through frozen `G`, so the tell is `G`'s
     generative reconstruction fingerprint (blur, missing fine detail, mode collapse).
     This is what the section-7 Experiment-A oracle detects. Cheap; no change to the
     survival dynamics.
   - **Stage 2 - organism run (Option B, dynamics-level), CONTINGENT on the gate.** Only
     if Stage 1's oracle lands in [0.85, 0.95] do we build the dynamics-level surrogate:
     `G` (trained on logged STATE transitions) drives the world's dynamics, with
     energy/food/death computed from `G`'s predicted state, plugged into
     `itasorl/experiment_b2.py::make_world` as the new `l3` drift-mode where the L2 path
     lives. If `G`'s learned dynamics cannot sustain survival (engagement gate fails),
     that is logged in section 12 and the organism construction is re-scoped before any
     multi-seed run - a stated empirical risk, not assumed away.
   The authentic world (drift off) is unchanged and byte-identical to B-v2/B-v3 authentic
   in both stages.
2. **Difficulty is calibrated by outcome, not architecture (decision).** We choose the
   **smallest-capacity** `G` whose Experiment-A oracle detectability (section 7) lands in
   the pre-set band **AUROC in [0.85, 0.95]**: clearly detectable (the fingerprint exists)
   but not trivially separable like L2's ~0.99. Capacity is the *single* difficulty knob.
3. **Fixed data budget (decision).** `G` trains on a fixed budget of authentic rollouts
   (frozen in section 9); we do NOT trade data against capacity - capacity alone sets
   difficulty, so the surrogate's fidelity is reproducible.
4. Same harsh metabolism, denser food, and potential-based food shaping as B-v2/B-v3
   (`SURVIVAL_METAB` / `SURVIVAL_FOOD`); reported returns use the TRUE reward.
5. World identity (authentic vs G-surrogate) is never in the observation or the reward;
   only the probe ever sees it.

## 5. Readouts (episode = unit of inference; GroupKFold)

Identical harness to B-v3:
- **PRIMARY - pooled (Experiment B frame):** `target` AUROC of a linear probe on
  `episode_features(h_t) = [mean h, final h]`, plus additive `target_var` / `target_full`
  and per-probe `selectivity = target - shuffled`.
- **SECONDARY - matched-pair recurrent:** detectability of the fingerprint in the agent
  state from a bit-identical shared prefix; reported, not the headline.

## 6. Estimand and smallest effect of interest

- **Estimand:** per-episode selectivity-corrected AUROC of the pooled linear
  world-identity probe, averaged over seeds.
- **SESOI (decision - reuse 0.65):** claim incidental encoding only if the survival
  pooled target is **>= 0.65 AND exceeds both `predictor` and `untrained` by >= 0.05** -
  the same bar as B-v2/B-v3, so L3 is directly comparable to the earlier rungs.

## 7. Gates - ALL must pass before the survival target is interpreted

The four B-v3 gates, PLUS a new L3-specific manipulation gate that must pass FIRST:

0. **Oracle-detectability gate (L3-specific, decision - oracle-first).** BEFORE any
   organism run, an Experiment-A oracle (a privileged discriminator with direct access to
   observations / prediction residuals) must detect authentic vs G-surrogate at
   **AUROC in [0.85, 0.95]**. Below 0.85: the surrogate is too good, there is no
   fingerprint to encode - fix `G`, do not run the organism. Above 0.95: too easy
   (approaching L2's trivial regime) - increase `G`'s capacity. This gate both confirms the
   tell exists and freezes the difficulty.
1. **Engagement:** trained TRUE return beats random and scripted by ENGAGE_MARGIN with
   lifetime not worse by LIFE_TOL.
2. **L0 control:** authentic-vs-authentic pooled target equivalent to 0.5 by TOST.
3. **Positive control:** speed probe >= 0.75.
4. **Leakage audit:** reward-sum / length / lifetime each within 0.1 of 0.5.

## 8. Decision matrix

- **Encoding induced (conditional on this fingerprint):** gate 0 passes AND all gates pass
  AND survival target >= 0.65 AND > predictor+0.05 AND > untrained+0.05.
- **Strengthened negative:** gate 0 passes AND all gates pass AND survival ~= 0.50 ~=
  predictor ~= untrained. (The strongest form of the project's core claim: even a rich
  learned-surrogate fingerprint, verified detectable by an oracle, is not incidentally
  encoded under survival pressure.)
- **Uninformative:** gate 0 fails (surrogate mis-calibrated) or engagement fails.

## 9. Fixed configuration (frozen at commit)

- **Surrogate net `G`:** family = small recurrent predictor (GRU core). Fixed training
  config, frozen here: next-state MSE loss, Adam `lr=1e-3`, a fixed number of epochs to
  convergence (early-stop on held-out authentic MSE, patience 5), seed 0. The ONLY free
  knob is hidden size, swept ascending over `{8, 16, 24, 32, 48, 64}`; the FIRST size
  whose section-7 oracle AUROC lands in [0.85, 0.95] is FROZEN (selected value + oracle
  AUROC recorded in section 12 before any organism run).
- **Capacity-sweep fallback (no valid capacity in the grid).** Because AUROC decreases
  with capacity, the sweep can step OVER the band (e.g. hidden=16 gives 0.97, hidden=24
  gives 0.82). If no grid point lands in [0.85, 0.95]: (a) bisect hidden size between the
  two bracketing points; if still no fit, (b) hold the smallest capacity that is >= 0.85
  and REDUCE the authentic-data budget (a secondary difficulty knob) until oracle AUROC
  enters the band. Whichever path is used, the final `G` config is frozen in section 12
  and the organism run does not start until gate 0 passes. This guarantees gate 0 is
  reachable.
- **`G` training data budget:** fixed authentic-rollout budget frozen here (initial
  proposal: the same collection scale as B-v2 `collect_pool`, i.e. on the order of a few
  hundred authentic episodes); no data/capacity trade.
- **Organism config (mirrors B-v3):** drifts `[0.0, surrogate]`; seeds `[0,1,2]`
  confirmatory, power extension `0..9`; updates `300`; `n_eps=16`; `max_steps=80`;
  `hidden=96`; `ray_steps=5`; `shaping_coef=1.0`; pooled `n=110, steps=24`; matched-pair
  `pairs=60, prefix=20, branch=24`. Adam `lr=3e-4`, `gamma=0.99`, `lam=0.95`,
  `ent_coef=0.01`, `vf_coef=0.5`, `wm_coef=1.0`.

## 10. Analysis plan

Mean +/- std over seeds. Primary contrast: survival vs predictor vs untrained pooled
target across authentic vs G-surrogate. Report every gate including the oracle gate.
Same figure/JSON schema as B-v2/B-v3. Also report the sysid-aux CEILING for L3 as the
capacity reference, and the matched-pair detectability as the "decodable-when-forced"
comparison, exactly as for L2.

Rigor carried from the B-v3 audit (2026-07-10):
- **Honest seed-level CI.** The primary "clears / misses 0.65" adjudication uses a
  **t-based (or BCa) 90% CI on the per-seed AUROCs**, not the percentile-of-mean
  bootstrap alone (at n=10 the percentile-of-mean interval under-covers at the boundary).
  Report both; the decision follows the t/BCa interval.
- **Per-world survivorship.** Report authentic vs surrogate survivor counts / death rates
  per cell, so the pooled survivorship asymmetry (section 11) is quantified, not assumed.
- **Multiplicity.** The volatility readouts (`target_var`, `target_full`) are **secondary
  / exploratory**; only the level `target` enters the pre-registered SESOI decision. Any
  volatility-based claim is flagged exploratory and not multiplicity-corrected.

## 11. Key interpretive caveat

- **This is a different, higher rung, not the L2 claim.** A positive L3 result answers
  "does survival encode a *learned-world-model generative fingerprint*", not "does
  survival encode L2 drift". A negative is the strongest form of *detectable != learned*:
  the fingerprint is oracle-verified present (gate 0), yet the survival organism does not
  form a persistent world-identity direction from it.
- **Signal A vs Signal B.** The organism headline is Signal A (latent geometry from living
  in both worlds); Signal B (frozen-world-model prediction-residual) is the oracle-side
  check only. Conflating them would confuse detectability with incidental encoding.
- **Difficulty is a design choice, made explicit.** The oracle band [0.85, 0.95] fixes how
  hard the tell is; the result is conditional on that band, this architecture, this world
  family, and this surrogate net.
- **Fingerprint character is confounded with capacity, and we bound it rather than assume
  it away.** A low-capacity `G` and a high-capacity `G` can both land in the oracle band
  while producing *qualitatively different* artifacts (heavy blur vs mode collapse vs a
  learned noise texture). The band fixes *difficulty*, not artifact *type*. We therefore
  (a) record which fingerprint dominates at the frozen capacity (via the oracle's most
  informative feature channel) in section 12, and (b) run the organism test at TWO
  in-band capacities when the sweep yields two candidates, reporting whether the
  encoding verdict is stable across artifact type. A single-capacity result is reported
  as conditional on that artifact.
- **Pooled readout is Exp-B-comparable, not confound-clean.** As in B-v2/B-v3, the pooled
  probe drops early deaths per world, so a harder surrogate that kills more episodes
  yields a survivorship-selected surrogate pool. The matched-pair channel (bit-identical
  shared prefix, equal-length truncation) is the confound-clean detectability guard; the
  pooled target is read as directly comparable to Experiment B, not as the
  confound-controlled estimate. This run additionally reports **per-world survivor counts
  / death rates** (section 10) so the asymmetry is visibly bounded.

## 12. Deviations from pre-registration

- **2026-07-10, milestone-1 empirical finding (Stage-1 oracle calibration, GPU).** The
  Stage-1 observation-channel construction (Option A: authentic physics, observations passed
  through frozen `G`) was built and its oracle calibrated on a from-scratch `G` capacity sweep
  (hidden 8-128; one-step reconstruction AND open-loop imagination; n=200-300 authentic
  episodes; local GPU). Result: the artifact is **trivially detectable (oracle AUROC ~1.0) at
  every capacity and every imagination horizon**, and a matched sensor-noise / variance floor
  did NOT restore a calibratable gradient. Reason: a strong privileged oracle separates "real
  sensor data" from "any neural-net reconstruction of it" via `G`'s systematic per-dimension
  manifold, not via `G`'s fidelity - so capacity is not a working difficulty knob and the
  [0.85, 0.95] band is unreachable without deliberately weakening the oracle (which would game
  the gate). Per the pre-agreed escalation, the Stage-1 obs-channel construction is **RETIRED**
  and the dynamics-level construction becomes primary: the surrogate's observations are
  produced by the REAL sensor model applied to `G`'s predicted STATE, so observations live on
  the authentic manifold and the sole tell is `G`'s dynamics error, which capacity controls.
  Section 4 Stage 1 is superseded by this entry; the section-7 oracle gate now runs against the
  dynamics-level surrogate, with capacity re-validated as the difficulty knob before any
  organism run.
- **2026-07-10, GATE 0 MET (dynamics-level oracle calibration, GPU).** Built the dynamics-level
  construction: `G_motion` (learned velocity law, `(vel,a)->vel_next`, drag withheld) via
  `itasorl/surrogate_l3.py`, the `l3` world hook, and the L2-style residual oracle
  (`itasorl/experiment_a_l3.py`, matched pairs, exact per-step authentic law from the logged
  transitions). Capacity is a clean MONOTONE difficulty knob; the section-7 [0.85, 0.95] band is
  reachable. **Frozen gate-0 configuration:** sensor-noise floor **sigma=0.05**, capacity
  **hidden=8** -> oracle AUROC **0.890** (centered in-band; hidden=4 -> 0.940 also in-band;
  hidden>=16 -> <0.75). Leakage: MECHANICAL channels (length, metadata) clean at 0.5; `reward`
  legitimately leaks (~0.86 at hidden=4; different dynamics -> different movement cost ->
  different reward, the documented dynamics-rung consequence, docs/FINDINGS.md section 2.2). The
  residual oracle uses ONLY the velocity residual, so it does not exploit reward - BUT the reward
  coupling is a readout-not-reward CONSIDERATION for the organism run: the B-v2/B-v3 leakage
  audit + matched-pair equal-length truncation must be re-checked at the frozen difficulty
  (smaller dynamics error at hidden=8 should shrink the reward leak vs hidden=4). Gate 0 is met;
  the organism run may proceed with these controls reported.
- **2026-07-10, ORGANISM RUN n=3 (confirmatory) + an unresolved measurement discrepancy.** Ran
  `run_expB2 --drift-mode l3 --l3-hidden 8` (n=3 seeds, 300 updates, local GPU, `fullruns/l3_n3`).
  At drift 0.45 the pooled target was **survival 0.801 (90% CI [0.760, 0.842]), predictor 0.779,
  untrained 0.706**; L0 control clean (0.514); manipulation check: artifact survival-relevant.
  Primary H_B2 NOT met by the SESOI (survival only +0.022 over predictor, < 0.05). TWO problems
  make this verdict NOT trustworthy yet: (1) **high organism mechanical floor** - the untrained
  (mechanical) agent already reads 0.706, so there is little headroom to isolate incidental
  encoding; (2) a **re-calibration sweep** (untrained floor vs oracle AUROC across `G` capacity)
  shows the tension is structural: larger `G` drops the untrained floor toward chance (hidden 32
  -> 0.47, 128 -> 0.46) but the oracle signal COLLAPSES below the band (hidden 32 -> 0.70, 128 ->
  0.63), so a clean-floor-AND-detectable capacity does not robustly exist above hidden=8.
  UNRESOLVED: an isolated untrained-floor probe at hidden=8 gives **0.548 reproducibly** (identical
  on CPU and GPU training, so `G` is not the noise source), which CONTRADICTS the full run's 0.706
  and lies OUTSIDE the run's CI [0.66, 0.80]. The full pipeline and the isolated measurement
  disagree for an unidentified reason. **The L3 organism verdict is ON HOLD** pending
  reconciliation of that discrepancy (a focused debugging task, not a re-run); only then is a
  clean-floor difficulty and a real H_L3 adjudication meaningful.
- **2026-07-10, DISCREPANCY RESOLVED - a world-params bug; the confound was an artifact.** Root
  cause of the 0.548-vs-0.706 divergence: `run_expB2` runs in
  `P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)`, but `setup_l3_surrogate` trained
  `G_motion` on bare `WorldParams()` DEFAULTS - so `G` learned the WRONG world's dynamics and was
  deployed in another, inflating its systematic error and faking the high mechanical floor.
  Confirmed on GPU: `G`-trained-on-default read on `P` reproduces the run's 0.734 exactly;
  `G`-trained-on-`P` drops the untrained floor to 0.483 (chance). Fixed by passing `params=P` at
  both setup sites (regression test `test_g_motion_wrong_world_diverges_more`). **The n=3 organism
  result above (untrained 0.706, survival 0.801) is INVALID - a bug artifact - and is retracted.**
  Re-calibrated GATE 0 on the CORRECT world `P`: at hidden=8 the dumb/untrained floor is ~0.48
  (chance) at every capacity 8-32, and the oracle lands in-band at **sigma=0.02 -> AUROC 0.928**
  (leakage clean). So a clean L3 difficulty DOES exist - oracle-detectable fingerprint AND a
  chance-level organism mechanical floor. Frozen gate 0 (world `P`): **hidden=8, sigma=0.02,
  oracle 0.928, untrained floor 0.483.** Next: re-run the organism with the fix (`G` on `P`) for
  a real H_L3 adjudication - this time the mechanical floor is at chance, so survival-specific
  encoding can be isolated.
- **2026-07-10, CORRECTED ORGANISM RUN n=3 - a PRELIMINARY POSITIVE (not yet decisive).** Re-ran
  `run_expB2 --drift-mode l3 --l3-hidden 8` (n=3, 300 updates, local GPU, `fullruns/l3_n3_fixed`,
  commit `eeee512`) with `G` trained on the correct world. At drift 0.45 the pooled target is
  **untrained 0.482 +/- 0.055** (mechanical floor AT CHANCE - the fix holds; per-seed
  0.483/0.414/0.549), **predictor 0.573 +/- 0.057**, **survival 0.777 +/- 0.122** (per-seed
  0.853/0.636/0.841). This is the project's FIRST genuine survival-specific dissociation:
  survival beats predictor by +0.204 and untrained by +0.295 (both >> the 0.05 SESOI), the
  shuffled negative control is at chance for every arm (survival 0.556, so it is real signal not
  probe overfitting), matched-pair 0.906, speed positive control 0.966, engagement 100%,
  manipulation check survival-relevant. The pipeline verdict prints **H MET (encoding induced,
  conditional on gates)**. HONEST CAVEATS (do not oversell): the mean 0.777 clears 0.65, but the
  t-based 90% CI **[0.648, 0.906]** just STRADDLES the bar (the printed percentile bootstrap
  [0.705, 0.849] excludes it, but that interval under-covers at the boundary - same issue flagged
  for the ceiling); 1/3 seeds (0.636) is below 0.65; variance is high (sd 0.122); the L0 TOST is
  underpowered at n=3 (mean 0.514, not shown equivalent); and the dynamics-rung reward-coupling is
  not yet ruled out for the organism (needs the leakage audit + a held-out probe). VERDICT: a
  strong, real, encouraging POSITIVE SIGNAL - the first sign that L3 (a learned-dynamics
  fingerprint) reverses the L2 nulls - but NOT a settled result at n=3. The **n=10 power
  extension is running** (`fullruns/l3_n10`) to test whether the CI clears 0.65 decisively; the
  reward/held-out controls remain to be reported before any headline claim.
- **2026-07-12, AUDITED ORGANISM RUN n=10 + POST-HOC CONFOUND AUDITS - POSITIVE, PARTLY
  BEHAVIOR-MEDIATED.** Re-ran `run_expB2 --drift-mode l3 --l3-hidden 8` (n=10, seeds 0..9, 300
  updates, local GPU, `fullruns/l3_n10_audited`) with the pooled-path leakage audit added (PR #39). At
  drift 0.45 the survival pooled target is **0.752**, honest t-based 90% CI **[0.698, 0.807]** (EXCLUDES
  the 0.65 bar; per-seed 0.853/0.636/0.841/0.823/0.830/0.573/0.705/0.782/0.759/0.723; 8/10 seeds >=
  0.65), with **predictor 0.573** and **untrained 0.488** (mechanical floor at chance) - survival beats
  both baselines by well over the 0.05 SESOI, and the L0 authentic-vs-authentic control is **0.517**
  (chance). This is a determinism reproduction of the `fullruns/l3_n10` headline (identical 0.752).
  CONTROLS THAT PASS: (1) POOLED reward-leakage audit - world identity is NOT decodable from summed
  reward: **reward-leak AUROC mean 0.541, range [0.517, 0.559], clean 10/10 seeds** (peer-review item 1,
  RESOLVED). (2) Per-world survivorship - **0 early deaths in every pool (110/110 authentic and surrogate,
  all seeds)**, so dropping early deaths introduces no asymmetry (peer-review item 2, RESOLVED). (3)
  Nonlinear-probe check (post-hoc, on the dumped states): a random-forest probe on the UNTRAINED agent
  reads only **0.517** (its shuffled baseline 0.506; L0 untrained 0.483), so the survival-vs-untrained
  dissociation is NOT a linear-probe artifact - the world is genuinely near-chance-decodable in the
  untrained state even nonlinearly.
  CONTROL THAT PARTLY MEDIATES - BEHAVIORAL MEDIATION (post-hoc, on the dumped states): the agent moves and
  forages differently in the two worlds, so per-episode behavior alone (speed/energy/food/drag) decodes the
  world at **~0.69** (0.689 linear, 0.704 nonlinear). Controlling for behavior CLEANLY (behavior->state
  regression fit IN-FOLD, no leakage) leaves a behavior-independent world-signal of **~0.66** (0.676 linear
  basis, 0.659 quadratic; per-seed quad 0.71/0.58/0.71/0.69/0.73/0.54/0.62/0.70/0.67/0.64, 6/10 >= 0.65) -
  clearly above the 0.488 untrained floor but only AT the 0.65 bar. So behavior mediates ~0.09 of the 0.752
  headline; a real behavior-independent component survives, but it is weak-to-moderate, not the abstract
  world-identity direction the raw 0.752 suggests. (An in-SAMPLE residualization gave a deflated 0.56-0.63
  by over-removing via leakage; the in-fold ~0.66 is the defensible figure. Still a soft UPPER bound: only
  4 coarse per-episode MEANS were controlled, so per-timestep behavior would likely lower it further.) The
  reward audit passed because reward is one scalar; behavior is multivariate and a much stronger separator.
  VERDICT: a real, reward- and survivorship-controlled, nonlinear-robust POSITIVE - L3 (a learned-dynamics
  fingerprint) reverses the L2 nulls and the survival agent (uniquely) carries world-discriminative state,
  read out and never rewarded. But it is only PARTLY behavior-independent (~0.66 after clean behavior
  control, right at the 0.65 bar), so the strong "abstract world-identity direction at 0.752" reading is
  NOT supported; the honest headline is "survival-relevant state that differs by world, plus a modest
  (~0.66) behavior-independent world component." STILL OWED: (a) a
  richer per-timestep behavior control to tighten the behavior-independent estimate; (b) the pre-registered
  hidden=4 second-capacity replication (section 11); (c) the held-out/common-garden probe (a world-signal
  that transfers to an UNSEEN fingerprint is much harder to dismiss as behavior).
- **2026-07-12 - BEHAVIOR AUDIT MADE REPRODUCIBLE (code, no new runs).** The ad hoc mediation audit above
  is now canonical committed code (`itasorl/behavior_audit.py` + `scripts/audit_behavior_mediation.py`,
  in-fold controls, control properties unit-tested on synthetic ground truth). Re-run on the same dumps it
  reproduces the published survival d=0.45 figures exactly: target 0.752, behavior-only 0.689 linear /
  0.705 nonlinear (prose said 0.704; random-forest seed wiggle), controlled 0.676 linear / 0.659 quadratic
  (artifact: `artifacts/expB2/behavior_audit_l3_n10.json`). The dump now also persists PER-TIMESTEP
  behavior traces (`bta`/`bts`), and the script runs the strictly stronger per-timestep control when
  traces are present, with a decision rule fixed in advance
  (`docs/specs/2026-07-12-l3-behavior-audit-design.md`): survival per-timestep-controlled
  mean >= 0.65 strengthens the claim; [0.60, 0.65) weakens it to a below-bar trace; < 0.60 means largely
  behavior-mediated. Owed items (a) and (b) reduce to two human-launched runs
  (`scripts/README.md`, "L3 owed runs"): hidden=8 with traces (headline capacity) and hidden=4 n=10.
- **2026-07-13 - PER-TIMESTEP BEHAVIOR CONTROL: CLAIM STRENGTHENED (hidden=8 re-run with traces).** Re-ran
  the frozen hidden=8 n=10 protocol with trace-extended dumps (`fullruns/l3_h8_traces`, local GPU). The
  pipeline is deterministic end-to-end: every published figure reproduced exactly (survival target 0.752
  [0.704, 0.797]; behavior-only 0.689/0.705; per-episode controlled 0.676/0.659; L0 mean 0.517, ROPE
  equivalence accepted; reward-leak clean 10/10; 0 deaths, all pools 110/110). The pre-registered decision
  rule then evaluated the strictly stronger per-timestep control: survival resid_trace = **0.726** (90% CI
  [0.685, 0.765], 9/10 seeds >= 0.65; quadratic variant 0.721 [0.678, 0.760]) -> >= 0.65, STRENGTHENS, and
  unlike the per-episode estimate the CI EXCLUDES the 0.65 bar. Honesty checks on real data: the untrained
  agent's resid_trace is 0.498 (exact chance) even though untrained BEHAVIOR alone decodes the world at
  0.645, so the control neither manufactures nor spares signal; predictor resid_trace 0.574 (the
  survival-only dissociation is preserved under the control). Two notable structural facts: (1) the full
  behavior trace alone decodes the world at 0.803, BETTER than the state probe itself - behavior is highly
  world-discriminative; (2) the per-episode-mean control had been OVER-removing (0.676), exactly the
  attenuation the synthetic tests predicted, while the surgical per-timestep control leaves ~0.73. The
  honest headline improves to: reward- and survivorship-controlled, nonlinear-robust, with a
  behavior-independent world-signal of **~0.73** whose CI clears the pre-registered bar. Caveats: the
  residualization is linear/quadratic over phi = [b_t, b_(t-1), cummean(b)]; a full-history or nonlinear
  control could in principle remove more. Artifact: `artifacts/expB2/behavior_audit_l3_h8_traces.json`.
  STILL OWED: the hidden=4 second-capacity replication (section 11) and the held-out/common-garden probe.
- **2026-07-13 - HIDDEN=4 RUN UNINFORMATIVE (gate failure); GATE 0 RE-VALIDATED PER CAPACITY AND THE
  SECOND CAPACITY RE-FROZEN AT HIDDEN=7.** The owed hidden=4 n=10 run (`fullruns/l3_h4_traces`, local
  GPU) completed and FAILS its gates at drift 0.45: untrained mechanical floor **0.891** (not
  chance), pooled reward-leak 0.637 with the clean gate passing in **0/10 seeds**, engagement
  passing in only **30% of seeds** (the drift-trained policy barely recovers: train@0.45 eval@0.45
  return -1.069, vs -0.219 at hidden=8). Per the section-8 decision matrix the run is
  **UNINFORMATIVE** (gate-0 / engagement failure), NOT a negative; the L0 control (0.517, TOST and
  ROPE both accept equivalence) confirms the apparatus itself is fine. ROOT CAUSE: hidden=4's
  in-band status (oracle 0.940) was frozen from the 2026-07-10 calibration made on the WRONG world
  at sigma=0.05, BEFORE the world-params fix; the post-fix re-calibration froze gate 0 only at
  hidden=8 and swept the untrained floor over capacities 8-32, so hidden=4 was never re-validated
  on world `P` (its reward leak was itself anticipated in the gate-0 entry above). FIX: gate 0 is
  now a committed, runnable check (`scripts/run_expA_l3.py`) validating BOTH halves per capacity on
  world `P` at the frozen sigma=0.02 - the residual oracle (band [0.85, 0.95], mechanical leakage)
  AND the organism-side untrained floor (|target - 0.5| < 0.1 at drift 0.45, arm built exactly as
  `run_expB2` builds it) - with a hidden=8 regression check and a selection rule frozen in advance:
  the smallest in-band capacity below 8 with a clean floor (section-9 fallback (a), bisection
  between the bracketing capacities 4 and 8). RECALIBRATION RESULT
  (`fullruns/l3_gate0_recal/calibration.json`): hidden=8 regression EXACT (oracle 0.928, floor
  0.482); hidden=4 oracle 0.932 in-band but floor **0.896** (independently reproducing the organism
  run's 0.891 from 3 seeds); hidden=5 oracle 0.972 (OUT of band); hidden=6 oracle 0.946 in-band but
  floor 0.647; **hidden=7 oracle 0.922 in-band, mechanical leakage clean, floor 0.566** -> second
  capacity RE-FROZEN at **hidden=7**. Honest note: the hidden=7 floor (0.566) is elevated relative
  to hidden=8 (0.482) though within the frozen tolerance; the SESOI already requires survival >
  untrained + 0.05, so the elevated floor is priced into adjudication. The hidden=7 n=10 organism
  run with trace dumps is running (`fullruns/l3_h7_traces`); its result and the per-timestep
  behavior audit will be recorded here when complete.
- **2026-07-14 - SECOND-CAPACITY REPLICATION AT HIDDEN=7 (n=10): THE BEHAVIOR-INDEPENDENT SURVIVAL
  SIGNAL REPLICATES (~0.72, CI clears the bar) BUT THE SURVIVAL-VS-PREDICTOR DISSOCIATION DOES NOT -
  THE FULL "ENCODING INDUCED" VERDICT IS ARTIFACT-CONDITIONAL.** The re-frozen hidden=7 n=10 run
  (`fullruns/l3_h7_traces`, local GPU, frozen protocol, trace dumps) completed with ALL gates passing:
  engagement 10/10 seeds at drift 0.45 (pooled train@0.45 eval@0.45 return -0.734, a partial recovery
  sitting between hidden=8's -0.219 and hidden=4's failing -1.069; the per-seed engagement criterion
  passes in every seed), L0 control 0.517 (TOST p=0.010 and ROPE P=0.999 both accept equivalence),
  speed positive control 0.959, pooled reward-leak 0.567 clean in 10/10 seeds, 0 deaths in every pool
  (110/110 both worlds, all seeds), and the untrained mechanical floor pooled at **0.586** (dev 0.086,
  inside the frozen <0.1 tolerance and matching the recalibration's 0.566, though per-seed it is
  violated in s3 0.732 and s9 0.683 with s6 0.605 marginal). PRIMARY at drift 0.45: survival pooled
  target **0.737**, 90% CI [0.688, 0.780], 8/10 seeds >= 0.65 (per-seed
  0.540/0.748/0.820/0.799/0.696/0.623/0.749/0.768/0.770/0.852) - above the 0.65 bar and clearing
  untrained (0.586) by +0.151, BUT **predictor reads 0.714** [0.687, 0.740], so survival leads the
  predictor by only +0.023 and the section-8 requirement (> predictor + 0.05) is NOT met: the clean
  hidden=8 dissociation (survival 0.752 vs predictor 0.573) does not replicate at hidden=7. BEHAVIOR
  AUDIT (`artifacts/expB2/behavior_audit_l3_h7_traces.json`, frozen decision rule): survival
  resid_trace = **0.722** (90% CI [0.678, 0.763], 8/10 seeds >= 0.65; quadratic 0.704) -> >= 0.65
  with the CI excluding the bar, an almost exact replication of hidden=8's 0.726 - the
  behavior-independent survival world-signal is STABLE across the two in-band artifacts. The
  dissociation is not: predictor resid_trace 0.691 (vs 0.574 at hidden=8) and untrained resid_trace
  0.579 (vs 0.498 = exact chance at hidden=8). READING (per the section-11 two-capacity clause): the
  hidden=7 artifact is qualitatively coarser - mechanically leakier (untrained floor 0.586/resid
  0.579) and far more behaviorally salient (the behavior trace alone decodes the world at 0.762-0.796
  in ALL arms, including untrained, vs 0.645 untrained at hidden=8) - so at this capacity every
  trained agent picks the fingerprint up and the survival-SPECIFIC part of the claim is conditional
  on the subtler hidden=8 artifact. The cross-capacity finding that survives both runs is: a
  reward-clean, survivorship-clean, behavior-independent world-signal of ~0.72 in the survival
  agent's state at both frozen capacities. Secondary notes: matched-pair survival mp_target 0.814;
  volatility readout target_var 0.706 / target_full 0.763 (above the 0.65 bar, consistent with the
  volatility-encoding observation on earlier rungs). LAST OWED ITEM, the held-out/common-garden probe
  (a world-signal that transfers to an UNSEEN fingerprint), is now run and recorded in the entry below.
- **2026-07-14 - HELD-OUT FINGERPRINT + COMMON-GARDEN PROBE (n=10): THE WORLD-SIGNAL GENERALIZES TO AN
  UNSEEN SAME-RECIPE FINGERPRINT (TRANSFER POSITIVE) BUT DOES NOT SURVIVE A COMMON-GARDEN CONTROL
  (REACTIVE, NOT A PERSISTENT REPRESENTATION).** Two readout-only evaluation channels on one frozen
  hidden=8 training run (`fullruns/l3_h8_heldout`, local GPU, frozen protocol, spec
  `docs/specs/2026-07-14-l3-heldout-common-garden-probe-design.md`); no change to training,
  the surrogate family, or the pre-registered headline probe. GATES REPRODUCED (third independent
  determinism check): survival pooled target **0.752** (90% CI [0.704, 0.797]), byte-matching the
  published hidden=8 headline; L0 control 0.517 (TOST p=0.010, ROPE P=0.999 both accept equivalence);
  pooled reward-leak 0.541 clean in 10/10 seeds; 0 deaths in every pool (110/110 both worlds, all
  seeds). CHANNEL 1, UNSEEN-FINGERPRINT TRANSFER: the world-identity direction fit against the trained
  hidden=8 fingerprint, frozen, then scored on a FRESH authentic pool vs the held-out hidden=7
  fingerprint the agent never trained against. Survival `transfer_target` = **0.773** (90% CI
  [0.722, 0.824], 9/10 seeds >= 0.65), predictor 0.633 (3/10), untrained mechanical floor 0.569
  (0/10). Frozen decision rule (survival >= 0.65 AND > untrained + 0.05 = 0.619) PASSES on both
  clauses -> GENERALIZES beyond the single trained fingerprint instance. Scope caveat, stated in the
  spec and honored here: hidden=7 is the SAME surrogate recipe at a different capacity, not a
  different surrogate family; cross-recipe transfer is out of scope for this run. CHANNEL 2, COMMON
  GARDEN: 20-step prefix in either the authentic or the hidden=8 surrogate world, then both groups
  continue under IDENTICAL authentic dynamics for a 24-step tail; the probe reads tail-only state,
  labels = prefix world, GroupKFold. Survival `cg_tail_target` = **0.557** (90% CI [0.492, 0.622],
  1/10 seeds >= 0.65), predictor 0.409, untrained 0.377. Frozen decision rule (survival >= 0.65 AND >
  untrained + 0.05) FAILS on the first clause. DECAY CHECK on the last 8 tail steps: survival
  `cg_latetail_target` = **0.492** (90% CI [0.431, 0.553]), at chance -> the prefix-world signal
  washes out along the shared tail. READING (frozen rules applied): the L3 world-signal is NOT an
  overfit to the one artifact instance `G_0` (transfer generalizes), but once the felt dynamics are
  made identical it does not persist in tail-only state - so the signal reads as REACTIVE tracking of
  the currently-felt dynamics, not a persistent stored world-identity representation. This is an
  informative negative for the representational reading (spec: "not a gate failure"), and it resolves
  the long-standing reactive-vs-representational ambiguity (section-11 caveats) toward reactive. This
  was the last owed post-hoc item; the L3 arc is complete. Provenance addendum (2026-07-16): the
  behavior audit re-run on this bundle's fresh `--save-agents` dumps reproduces resid_trace 0.726
  (90% CI [0.685, 0.765], 9/10 seeds) identically; per-seed values committed as
  `artifacts/expB2/behavior_audit_l3_h8_heldout.json`. Tooling: `audit_behavior_mediation.py` skips
  the heldout sibling dumps (`*_h7transfer.npz`, `*_cg.npz`) it does not parse.

- **2026-07-14 - REVERSE-TRANSFER RUN ANNOUNCED (frozen BEFORE launch).** The spec's staged
  follow-up condition is met (forward transfer positive), so the reverse direction runs next:
  train at hidden=7, hold out the SUBTLER hidden=8 fingerprint
  (`fullruns/l3_h7_heldout`, n=10, frozen protocol, `--save-agents`, trace dumps). Decision
  rules are IDENTICAL in form to the forward spec and frozen here: transfer - survival
  transfer_target >= 0.65 AND > untrained transfer + 0.05 -> the world-signal generalizes to a
  subtler unseen fingerprint (the stricter version of the 0.773 claim); common garden - same
  form, second data point on persistence at the coarser training artifact. INTERPRETATION
  LIMIT, stated in advance: at hidden=7 the survival-vs-predictor dissociation did not hold
  (two-capacity entry above), so this run reads the survival arm against the untrained floor
  only and cannot support any survival-SPECIFICITY claim. Determinism expectation: the
  standard-probe half must reproduce the `fullruns/l3_h7_traces` table exactly (survival 0.737
  [0.688, 0.780]); any deviation invalidates the run. Result to be recorded here when complete.

- **2026-07-16 - REVERSE-TRANSFER RUN RECORDED (completed 2026-07-15): THE COARSE-TRAINED
  DIRECTION READS THE SUBTLER UNSEEN FINGERPRINT ONLY PARTIALLY - THE FROZEN RULE FAILS
  (INFORMATIVE NEGATIVE); HELD-OUT TRANSFER IS DIRECTION-DEPENDENT.** Executes the 2026-07-14
  freeze above (`fullruns/l3_h7_heldout`, n=10, `--save-agents`; per-seed summary committed as
  `artifacts/expB2/heldout_l3_h7_reverse_summary.json`). DETERMINISM GATE PASSES: the
  standard-probe half reproduces the `fullruns/l3_h7_traces` table exactly - survival pooled
  target 0.737 (boot 90% CI [0.688, 0.780]), L0 0.517 (TOST p=0.010 / ROPE accept equivalence),
  reward-leak 0.567 clean 10/10, engagement 10/10, 0 deaths per pool - so the run is valid per
  the freeze. TRANSFER (train hidden=7, hold out the subtler hidden=8): survival
  `transfer_target` = **0.638** (t 90% CI [0.600, 0.676], 4/10 seeds >= 0.65) vs untrained
  mechanical floor **0.525** and predictor 0.603. The frozen rule (>= 0.65 AND > untrained +
  0.05 = 0.575) FAILS on the absolute bar; the floor-margin clause alone passes (+0.063).
  Adjudicated NEGATIVE per the frozen rule. COMMON GARDEN: survival `cg_tail_target` = **0.598**
  (t 90% CI [0.547, 0.649], 4/10 seeds >= 0.65), decaying to 0.489 on the last-8 window;
  predictor 0.504, untrained 0.456. Rule FAILS -> REACTIVE, a second independent data point
  matching the forward run. READING (frozen rules applied; the freeze's interpretation limit
  honored - no survival-specificity claim at hidden=7, and predictor transfer 0.603 indeed sits
  near survival 0.638): held-out transfer is DIRECTION-DEPENDENT. Fit on the subtle hidden=8
  fingerprint the direction reads coarser unseen artifacts (same-recipe 0.773, cross-recipe
  0.684); fit on the coarse hidden=7 fingerprint it reads the subtler one only partially (0.638,
  above the floor but below the bar). The published "fingerprint-GENERAL" wording is qualified
  accordingly in FINDINGS 10.6 and README.

- **2026-07-16 - RESEARCH-INTEGRITY AUDIT: FREEZE-TIMING NOTE FOR THE HIDDEN=7 SELECTION.** A
  commit-level audit of this log found: the capacity-fallback RULE was frozen in the original
  pre-registration commit (`0217263`, 2026-07-10), strictly before the hidden=4 gate failure it was
  later applied to. The specific "RE-FROZEN at hidden=7" statement, however, landed in the same
  commit as the hidden=7 n=10 result artifact (`85229c2`, 2026-07-14), so the capacity selection and
  its result are commit-simultaneous rather than sequential. The honest reading: hidden=7 was selected
  by a rule frozen before the failure, but the selection itself has no committed timestamp preceding
  the run. The per-timestep behavior-control spec has clean sequential timing (spec `1b5985e`
  2026-07-12, results `bf8a74e` 2026-07-13). Every published number was re-verified against committed
  per-seed artifacts by `scripts/audit_stats_recheck.py` (117 checks, all passing); all 15 cited arXiv
  IDs were resolved against arxiv.org with matching titles and authors. See `docs/AUDIT_2026-07.md`.

- **2026-07-16 - CROSS-RECIPE TRANSFER PROBE (n=10, readout-only): THE WORLD-SIGNAL READS A
  DIFFERENT SURROGATE FAMILY (RULE PASSES); THE CONSTANT-DRAG FAMILY IS UNCALIBRATABLE AND DROPPED.**
  Executes the pre-registered spec `docs/specs/2026-07-15-l3-crossrecipe-transfer-probe-design.md`
  (frozen before any run; PR #51) against the SAVED `fullruns/l3_h8_heldout` agents; no training
  anywhere. Committed artifact: `artifacts/l3_crossrecipe/summary.json`. GATE-0 CALIBRATION: the
  `G_rff` family's round-1 D sweep {8..128} stepped over the band (0.975 at D=64, 0.624 at D=128);
  the pre-registered bisection fallback landed **D=80** in-band (oracle 0.887, leakage pass,
  untrained floor 0.538). The `G_cd` family has an EMPTY calibration window: by the eps where its
  oracle reaches the band (eps=3.2: 0.854; eps=6.4: 0.907) the untrained mechanical floor exceeds
  0.6 (0.859, 0.913) - a drag-coefficient bias large enough to detect is felt so grossly that any
  recurrent state separates the worlds. DROPPED per the pre-stated rule (recorded, no transfer
  claim, no penalty to the primary). Both cd calibration double-runs produced byte-identical JSONs.
  INTEGRITY GATE (fourth determinism check): all 60 reloaded agents regenerated their standard
  pools bit-identically against the saved dumps and the drift-0.45 pooled survival mean reproduced
  the published **0.752** exactly. CHANNEL (PRIMARY, `G_rff` D=80): survival `transfer_rff_target`
  = **0.684** (boot 90% CI [0.657, 0.710]; t 90% CI [0.654, 0.715], lower bound above the bar;
  7/10 seeds >= 0.65, per-seed min 0.584), predictor 0.574 [0.554, 0.593] (0/10), untrained
  mechanical floor 0.548 [0.538, 0.557] (0/10). Frozen rule (survival >= 0.65 AND > untrained +
  0.05 = 0.598) PASSES on both clauses, machine-checked in the runner aggregate (`rff_rule_pass`
  true, margin +0.034). READING: the world-identity direction fit against the trained MLP
  fingerprint reads a different-recipe fingerprint (cosine-basis ridge texture) the agent never
  lived with, survival-specifically; attenuated versus same-recipe transfer (0.773 -> 0.684), as
  expected for a farther family. Scope, stated honestly: the reactive interpretation of the
  common-garden entry above is UNCHANGED - this extends the generality of the reactive world-signal
  across surrogate recipes, it does not revisit persistence. The cd drop is itself informative:
  coefficient-bias artifacts cannot be made subtle-but-detectable in world P, independently
  motivating the learned-texture construction of the L3 rung.

- **2026-07-18 - CORRECTION (OUTCOME-BLIND): COMMON-GARDEN CHANNEL RE-SCORING UNDER THE
  FIXED ESTIMATOR.** The 2026-07-14/16 held-out and reverse-run adjudications' common-garden
  channel (cg_tail 0.557 / late-tail 0.492; reverse 0.598/0.489) was computed with the
  pre-fix pair-splitting `cg_probe` (per-episode CV groups; bias toward AUROC 0 whenever the
  surviving pair count is not a multiple of 5 - fixed in commit `1633bca`; see FINDINGS
  13.C). The committed artifacts carry the bias signature (drift-0 cg floors 0.001-0.27;
  pair counts 96-110). The cg channel is being re-scored from the saved `_cg.npz` tail dumps
  with the fixed estimator (`scripts/reanalyze_cg_states.py`); the frozen decision rules are
  UNCHANGED and will re-adjudicate the corrected numbers. The transfer channel (frozen-fit
  train/test, no CV) and the mp channel (pair counts 60/25, fold-safe; no pair drops) are
  unaffected and stand. This entry is recorded BEFORE the re-score's outcome is known; the
  "REACTIVE" adjudication is provisional until it lands.

- **2026-07-18 - OPERATIVE-ESTIMAND ERRATA.** Section 6 registers the "selectivity-corrected
  AUROC" as the estimand; every adjudication (0.752, 0.737, 0.726, 0.722 vs the 0.65 bar)
  used the raw pooled/resid target with the shuffled probe as a gate-style control. No
  verdict could have flipped (shuffled at chance throughout); the operative definition is
  the raw target adjudicated at the bar, selectivity reported as a control.

- **2026-07-22 - CORRECTION: THE MP CHANNEL WAS NOT FOLD-SAFE.** The 2026-07-18 entry above
  asserted the matched-pair channel "(pair counts 60/25, fold-safe; no pair drops)" was
  unaffected by the `1633bca` estimator fix. That assertion was checked (per FINDINGS 10.6's
  "confirm, do not assume" banner) and is FALSE: under the sklearn in use (1.5.2), GroupKFold
  with the pre-fix singleton groups splits matched-pair twins at both pair counts (33/60,
  24/25), and the recorded drift-0.00 mp floors carry the 13.C bias signature (0.134-0.137
  instead of ~0.5). Both held-out bundles' mp channels were re-scored from rollouts
  regenerated off the saved agent bundles (`scripts/reanalyze_mp_readout.py`; deterministic -
  all 120 cells reproduce their stored pre-fix values bit-for-bit under singleton-group
  scoring). Under the fixed pair-grouped estimator all drift-0.00 floors return to 0.500
  exactly; corrected drift-0.45 survival means are 0.893 (forward, was 0.883) and 0.855
  (reverse, was 0.814 - the value quoted in the 2026-07-16 entry). The bias deflated, never
  inflated. The mp channel is secondary/demoted (no frozen rule reads it), so no adjudication
  changes; the corrected aggregates are committed as
  `artifacts/expB2/heldout_l3_h8_mp_rescore.json` and
  `artifacts/expB2/heldout_l3_h7_reverse_mp_rescore.json`. The transfer channel's exemption
  in the 2026-07-18 entry stands (frozen-fit train/test, no CV grouping).

- **2026-07-24 - A2 OBSERVATION-CHANNEL LOCALIZATION (n=10, readout-only):
  WORLD-IDENTITY SIGNAL IS VISUAL, NOT INTEROCEPTIVE at hidden=8.** Executes
  `scripts/run_l3_obs_localization.py` against the saved
  `fullruns/l3_h8_heldout` agents; no training. The frozen agent's observations
  are channel-masked before the running norm and new drift-0.45 pools are
  collected under the hidden=8 fingerprint. Baseline (no mask) survival mean
  **0.753** reproduces the headline. Masking **vision** (120/146 dims) drops the
  signal to **0.686** (7/10 seeds >= 0.65). Masking **interoception** (14/146
  dims) leaves it at **0.756** (8/10). Masking **all** observations collapses it
  to **0.500**. READING: the world-identity signal is carried by the visual
  stream / visually-shaped behavior, not by the explicit velocity/energy
  interoceptive channels. This is consistent with section 10.4's behavior-
  mediation finding; it does not distinguish passive visual echo from a visually-
  guided behavior plan. The probe's positive controls (speed, energy, food) remain
  readable in every mask condition, so the chance collapse is not a dead-probe
  artifact. Local artifact: `fullruns/l3_h8_obs_localization/aggregate.json`.

- **2026-07-24 - A2 OBSERVATION-CHANNEL LOCALIZATION (n=10, readout-only):
  HIDDEN=7 SIGNAL IS ROBUST TO SINGLE-CHANNEL MASKING.** Executes
  `scripts/run_l3_obs_localization.py` with `--hidden 7` against the saved
  `fullruns/l3_h7_heldout` agents; no training. The frozen agent's observations
  are channel-masked before the running norm and new drift-0.45 pools are
  collected under the hidden=7 fingerprint. Baseline (no mask) survival mean
  **0.737** reproduces the headline. Masking **vision** (120/146 dims) leaves the
  signal at **0.764** (9/10 seeds >= 0.65). Masking **interoception** (14/146
  dims) leaves it at **0.742** (8/10). Masking **all** observations collapses it
  to **0.500**. Unlike hidden=8, where vision masking causes a clear drop, the
  hidden=7 signal is robust to single-channel lesion: the coarser surrogate's
  artifacts propagate into both visual and interoceptive streams. The probe's
  positive controls (speed, energy, food) remain readable in every mask condition,
  so the chance collapse is not a dead-probe artifact. Local artifact:
  `fullruns/l3_h7_obs_localization/aggregate.json`.

- **2026-07-25 - A3 L1 H2 SUBSTRATE-GROUNDING BATTERY (n=10): ORGANISM ENCODING
  NEGATIVE at matched in-band discretization.** Extends the L3 H2 battery to the
  L1 observation-quantization rung. GATE 0: fine Δ sweep under B-v2 world params
  freezes **Δ = 0.023** (oracle AUROC **0.873**, leakage clean;
  `fullruns/l1_calib.json`). Noise comparator gate freezes **σ_o = 0.01**
  (oracle AUROC **0.873**, leakage clean; variance/high-frequency oracle, not
  the L1 grid residual; `fullruns/l1_noise_calib.json`). ORGANISM RUN: B-v2
  protocol with `--drift-mode l1 --l1-delta 0.023 --sensor-sigma 0.01`, drifts
  {0.0, 0.023}, seeds 0..9, three arms, `--save-agents` + state dumps
  (`fullruns/l1_heldout`). Survival pooled target **0.533** (90% CI [0.509,
  0.556], 0/10 ≥ 0.65); predictor 0.489; untrained 0.494. L0 ROPE accepts
  chance; engagement/leakage/survivorship clean; ceilings alive. Primary H_B2
  NOT met - oracle-detectable L1 leaves no incidental world-identity encoding.
  A1 GRADED-SEAM: integrity gate bit-matches dumps and reproduces 0.533; Δ ladder
  stays near chance (0.522 / 0.519 / 0.528 / 0.547 / 0.533). A2 NOISE KNOCKOUT:
  transfer onto matched-band iid obs noise gives survival 0.525 / untrained
  0.509 / predictor 0.516 (frozen rule fails both clauses). A2 CHANNEL
  LOCALIZATION: baseline 0.533; intero mask ~0.51; all-mask 0.500; vision mask
  elevates survival and untrained alike (~0.70) - non-specific lesion artifact
  given the null baseline. READING: L1 does not reproduce the L3 positive; H2
  substrate-grounding remains L3-specific. Detectability of an observation-level
  seam is not sufficient for incidental encoding. Artifacts:
  `fullruns/l1_heldout/`, `fullruns/l1_h2_ablations/`,
  `fullruns/l1_obs_localization/`.

- **2026-07-23 - H2 TEXTURE-KNOCKOUT: STRUCTURE + DOSE-RESPONSE ABLATIONS (n=10,
  readout-only): H2 CONFIRMED IN TEXTURE-SPECIFIC FORM at hidden=8 and hidden=7.**
  Executes the frozen spec
  `docs/specs/2026-07-22-h2-substrate-grounding-ablations-design.md` (PR #99)
  against the saved `fullruns/l3_h8_heldout` and `fullruns/l3_h7_heldout` agents;
  no training. GATE-0 CALIBRATION: `G_gn` (Gaussian-jitter velocity law) freezes at
  `sigma_v=0.01` with oracle AUROC **0.865** (band pass), leakage pass, and
  untrained floor **0.448** (within tolerance). The same-recipe MLP capacity
  ladder {h16, h32, h64} is calibrated sub-band with clean floors and leakage:
  oracle AUROCs **0.788, 0.656, 0.603**. INTEGRITY GATE (sixth determinism
  check at hidden=8, seventh at hidden=7): all 60 reloaded agents per capacity
  regenerated their standard pools bit-identically and the drift-0.45 pooled
  survival means reproduced the published **0.752** (hidden=8) and **0.737**
  (hidden=7) exactly. CHANNEL 1 (structure knockout, PRIMARY): at hidden=8
  survival `transfer_gn_target` = **0.539** (0/10 seeds >= 0.65), predictor 0.556,
  untrained floor 0.542; at hidden=7 survival **0.510**, predictor 0.539,
  untrained floor 0.520. The frozen positive rule fails both clauses at both
  capacities, so the direction does NOT read matched-band unstructured jitter.
  CHANNEL 2 (dose-response, SECONDARY): at hidden=8 survival transfer means
  **h16 0.701, h32 0.622, h64 0.541** (seeds >= 0.65: 8/10, 3/10, 0/10); at
  hidden=7 **h16 0.574, h32 0.516, h64 0.513** (1/10, 0/10, 0/10). VERDICT: H2
  is supported in texture-specific form across both in-band capacities; the
  world-signal loads on the learned structure of the substrate velocity-law
  surrogate, not on generic dynamics perturbation. The survival-specificity claim
  remains conditional on the subtler hidden=8 artifact (section 10.5).

- **2026-09-26 - SENSORY-ECHO CONTROL (n = 10, readout-only): THE BEHAVIOR-INDEPENDENT
  WORLD-SIGNAL IS NOT A PASSIVE ECHO OF THE INPUTS (rule passes).** Executes the frozen
  spec `docs/specs/2026-09-26-l3-sensory-echo-control-design.md` against the saved
  `fullruns/l3_h8_heldout` agents (no training). PRE-RUN AMENDMENT, logged in the spec
  before launch: the synthetic control-property tests showed the basis as first written
  (`[x_t, x_{t-1}, cummean(x)]`) absorbs any persistent tag once the inputs carry the
  label; the primary basis was frozen as the instantaneous `[x_t, x_{t-1}]`, the
  integrated basis demoted to a secondary diagnostic. INTEGRITY GATE (determinism check
  #8): all 60 regenerated pools bit-match the saved dumps; drift-0.45 survival target
  reproduces **0.752** and the seven-channel behavior control reproduces **0.723**
  exactly. RESULT at drift 0.45: survival `resid_obs` = **0.731** (t 90% CI
  [0.690, 0.772]; boot [0.695, 0.765]; 8/10 seeds >= 0.65) vs untrained 0.534 and
  predictor 0.542; the frozen rule (>= 0.65 AND > untrained + 0.05) PASSES on both
  clauses, t-CI lower bound above the untrained mean. Secondary: `resid_obs_int` 0.758,
  `resid_obs_beh` (sensory + behavior jointly) 0.670 [0.638, 0.702], `obs_trace_only`
  0.709. Drift-0.00 floors near chance under every control. Committed artifact:
  `artifacts/expB2/sensory_echo_l3_h8.json`; FINDINGS 10.4.2. The first full run
  completed all 60 cells and then failed while printing the summary (a NumPy bool in the
  JSON encoder); the aggregate was rebuilt from the saved `cells.json` with a
  `--reaggregate` path added to the runner, no cell recomputed.

- **2026-09-27 - SECOND FINGERPRINT INSTANCE, GATE 0 (G seed 1; spec `docs/specs/2026-09-26-l3-second-fingerprint-instance-design.md`; recorded by the run chain mechanically BEFORE launch).** Calibration on world P at the frozen sigma=0.02: hidden=8: oracle 0.928 (in band True), mech leak pass, floor 0.664 (dirty) -> fail; hidden=7: oracle 0.957 (in band False), mech leak pass, floor 0.640 (dirty) -> fail; hidden=9: oracle 0.982 (in band False), mech leak pass, floor 0.487 (ok) -> fail; hidden=10: oracle 0.893 (in band True), mech leak pass, floor 0.484 (ok) -> PASS. Selected hidden=10 at G seed 1 (first passing candidate in the frozen order 8, 7, 9, 10). The organism run launches with this instance; its result is recorded in a later entry.

- **2026-09-27 - ARCHITECTURE BASELINE (no world-model auxiliary; n = 10; CPU cloud
  sandbox): THE L3 POSITIVE IS AUXILIARY-CONDITIONAL.** Executes the frozen spec
  `docs/specs/2026-09-26-l3-architecture-baseline-design.md` (`--no-world-model`; survival
  and untrained arms without the next-observation decoder, predictor unchanged; otherwise
  the frozen hidden = 8 protocol). All gates pass (engagement 20/20, L0 0.514 TOST p = 0.006
  / ROPE accept, speed >= 0.835, pooled leak clean 20/20, floor 0.529, 0 deaths). Drift 0.45:
  survival **0.601** (t 90% CI [0.549, 0.654]; boot [0.559, 0.649]; 1/10 seeds >= 0.65),
  predictor 0.589, untrained 0.529; behavior-controlled survival 0.646 [0.593, 0.698].
  Frozen rule: survival < 0.65 -> AUXILIARY-CONDITIONAL cell (lead over predictor +0.013,
  under the 0.05 margin; over untrained +0.072); section 8 matrix: intermediate zone.
  Execution differs from the published GPU run (CPU sandbox, 4 vCPU, 3 workers, torch
  2.14+cpu); a device control was pre-registered as a dated addendum to the spec BEFORE
  its launch and is running. Committed: `artifacts/reviewer_gaps_runs/l3_h8_nowm/`,
  `artifacts/expB2/arch_baseline_l3_h8_nowm.json`; FINDINGS 10.8.

- **2026-09-27 - SECOND FINGERPRINT INSTANCE (G seed 1, hidden 10; n = 10; CPU cloud
  sandbox): SURVIVAL-SPECIFIC DISSOCIATION REPRODUCES, ABSOLUTE BAR MISSED (replication
  not claimed).** Executes the frozen spec
  `docs/specs/2026-09-26-l3-second-fingerprint-instance-design.md` at the instance the
  gate-0 entry above selected mechanically (hidden 8 at seed 1 failed on a dirty floor
  0.664; hidden 10 passed: oracle 0.893, floor 0.484). All gates pass (engagement 20/20,
  L0 0.529 TOST p = 0.039 / ROPE accept, speed >= 0.830, pooled leak clean 20/20, floor
  0.514, 0 deaths; the demoted matched-pair leak channel is clean in 7/10 untrained cells).
  Drift 0.45: survival **0.639** (t 90% CI [0.610, 0.669]; boot [0.614, 0.664]; 5/10 seeds
  >= 0.65), predictor 0.534, untrained 0.514; behavior-controlled survival 0.658
  [0.625, 0.691] (5/10). Frozen rule: replication requires both the pooled target and
  `resid_trace` >= 0.65 with t-CIs excluding the bar -> NOT MET; the two margin clauses
  pass (+0.105 over predictor, +0.126 over untrained; t-CI excludes both baseline means).
  Section 8 matrix: intermediate zone. Committed: `artifacts/reviewer_gaps_runs/
  l3_gate0_seed1/`, `l3_h10_gseed1/`, `artifacts/expB2/second_instance_l3_h10_gseed1.json`;
  FINDINGS 10.9.

- **2026-09-28 - DEVICE CONTROL (decoder-carrying hidden = 8 protocol, unchanged, on the
  same CPU cloud sandbox; n = 10): DEVICE IS NOT THE CAUSE; THE AUXILIARY-CONDITIONAL
  VERDICT STANDS.** Executes the dated addendum (2026-09-27) to
  `docs/specs/2026-09-26-l3-architecture-baseline-design.md`, frozen before launch
  (`scripts/reviewer_gaps/run_cloud_device_control.sh`; launched 2026-09-27 22:33 UTC,
  self-committed 02:44 UTC in `f746469`; an earlier 17:06 UTC launch committed no
  results). All gates pass (engagement 20/20, L0 0.529 TOST p = 0.039 / ROPE accept,
  speed >= 0.830, pooled leak clean 20/20, floor 0.523, 0 deaths). Drift 0.45: survival
  **0.730** (t 90% CI [0.668, 0.791]; boot [0.677, 0.782]; 8/10 seeds >= 0.65), predictor
  0.589, untrained 0.523; behavior-controlled survival 0.710 [0.656, 0.764] (7/10).
  Frozen rule: >= 0.65 with the t-CI excluding the bar AND a lead over the no-auxiliary
  0.601 of more than 0.05 (lead +0.128; paired by seed +0.128 [+0.089, +0.168]) -> DEVICE
  NOT THE CAUSE. The predictor arm is bit-identical per seed to the architecture-baseline
  run's, a determinism check across sessions. Committed:
  `artifacts/reviewer_gaps_runs/l3_h8_wm_cpu/`,
  `artifacts/expB2/device_control_l3_h8_wm_cpu.json`; FINDINGS 10.8.

- **2026-09-28 - NONLINEAR JOINT CONTROL (readout-only; n = 10; owner's GPU machine): THE
  WORLD-SIGNAL SURVIVES AN MLP INPUT-PLUS-BEHAVIOR CONTROL AT THE BAR.** Executes probe B
  of the frozen spec `docs/specs/2026-09-27-local-strengthening-probes-design.md` on the
  regenerated hidden = 8 pools (integrity 30/30 bit-match, 0.752 reproduced; legacy
  GroupKFold split of the owner's stack, FINDINGS methods note 8). Survival
  `resid_obs_beh_mlp` = **0.654** (t 90% CI [0.621, 0.687]; 6/10 seeds), `resid_obs_mlp`
  0.653 [0.633, 0.674]; predictor 0.542 / 0.530, untrained 0.509 / 0.497. Frozen rule
  (>= 0.65 AND > untrained + 0.05) PASSES at the mean; the t-CI lower bound is below the
  bar, stated as such. Committed: `artifacts/expB2/sensory_echo_l3_h8_mlp.json`; FINDINGS
  10.4.2 addendum. Process note: the first local attempt died at cell 8 on an
  out-of-memory error; the runner gained a `--resume` path and the remaining cells were
  computed without recomputing the first seven.

- **2026-09-29 - SECOND INSTANCE, GPU RE-MEASURE (hidden 10, G seed 1, the 10.9 protocol
  unchanged, on the owner's GPU machine; n = 10): RULE NOT MET; THE DEVICE IS NOT THE
  SOURCE OF THE GAP.** Re-executes `docs/specs/2026-09-26-l3-second-fingerprint-instance-
  design.md` at the already-selected instance on CUDA (gate 0 re-run first: oracle 0.893,
  floor 0.521, pass), explicit fold split (the cloud run's partition). Launched 2026-09-28
  21:13 UTC; the single-worker start paged and was resumed at 22:03 UTC with two CUDA
  workers (`--workers` is outside the config fingerprint; `--resume` kept cell 1); organism
  run done 02:58 UTC, audit 03:00 UTC. Gates: engagement 20/20, speed >= 0.833, pooled leak
  clean 20/20, floor 0.521, 0 deaths; L0 0.539 NOT shown equivalent (TOST p = 0.207, ROPE
  P = 0.816), reported as an open gate on this run; the verdict below does not depend on it.
  Drift 0.45: survival **0.612** (t 90% CI [0.582, 0.642]; 3/10 seeds >= 0.65), predictor
  0.529, untrained 0.521; behavior-controlled survival 0.638 [0.612, 0.663] (4/10). Frozen
  rule (both the pooled target and `resid_trace` >= 0.65 with t-CIs excluding the bar) ->
  NOT MET; margin clauses pass (+0.083 over predictor, +0.091 over untrained). Paired by
  seed against the cloud CPU run: survival -0.027 [-0.060, +0.005], predictor -0.005.
  Committed: `artifacts/reviewer_gaps_runs/l3_gate0_seed1_gpu/`, `l3_h10_gseed1_gpu/`,
  `artifacts/expB2/second_instance_l3_h10_gseed1_gpu.json`; FINDINGS 10.9 addendum.

- **2026-09-29 - SENSORY-ECHO CONTROL AT HIDDEN = 7 (readout-only on the saved hidden = 7
  agents; n = 10; owner's GPU; legacy fold split): RULE PASSES.** Executes the frozen
  `docs/specs/2026-09-26-l3-sensory-echo-control-design.md` at the second in-band capacity
  (`scripts/audit_sensory_echo.py --hidden 7`). Integrity 60/60 bit-match, 0.737 reproduced.
  Drift 0.45: survival `resid_obs` **0.684** (t 90% CI [0.627, 0.741]; 7/10 seeds >= 0.65)
  vs untrained 0.549 and predictor 0.589; `resid_obs_int` 0.695; `resid_obs_beh` 0.671
  [0.618, 0.725]; seven-channel `resid_trace` 0.739. Frozen rule (>= 0.65 AND > untrained
  + 0.05) -> PASSES on both clauses; the t-CI lower bound is below the bar, stated. The
  survival-minus-predictor lead under the sensory control (+0.095) is reported, not
  adjudicated. Process note: the run was interrupted once by an operator restart and
  resumed from its cell file; the 60 cells are unique. Committed:
  `artifacts/expB2/sensory_echo_l3_h7.json`; FINDINGS 10.4.2 hidden = 7 addendum.

- **2026-09-29 - EXPLICIT-FOLD RE-SCORE OF THE PUBLISHED DUMPS (readout-only; owner's
  stack, numpy 1.26.4 / scikit-learn 1.5.2): RECORD REPRODUCED; ONE GATE CLAUSE FLIPS.**
  Executes `docs/specs/2026-09-28-explicit-cv-folds.md`, frozen the same day before the
  run. Legacy column reproduces 0.752 / 0.726 (hidden 8), 0.737 / 0.722 (hidden 7),
  common garden 0.666 / 0.684, every stored matched-pair value, gate 0 (0.928 / 0.482,
  0.922 / 0.566). Explicit: hidden 8 survival **0.774** [0.727, 0.821], `resid_trace` 0.750;
  hidden 7 0.740 [0.688, 0.791], 0.725; common garden 0.677 / 0.677; no decision or margin
  clause on these moves. FLIP: gate 0 at hidden 7 FAILS under explicit (untrained floor
  0.615 against the |target - 0.5| < 0.1 tolerance; the n = 10 organism-run floor reads
  0.599959, inside by 0.00004) -> FINDINGS 16; hidden 8 passes under both. Partition
  correction: the legacy split on this stack is (22,22) x 3, (21,23), (23,21). Not
  re-scored: B-v3 (dumps absent) and the sensory-echo joint controls (pool-regenerating).
  `REFERENCE_SURVIVAL_TARGET` explicit entries recorded (8: 0.774, 7: 0.740). Committed:
  `artifacts/fold_rescore/` (17 files); FINDINGS methods note 8 and section 16.

- **2026-09-29 - HIDDEN-8 NEW-SEED INSTANCE, GATE 0 (G seed 2; spec `docs/specs/2026-09-28-l3-hidden8-second-seed-design.md`; recorded by the run chain mechanically BEFORE launch).** Calibration on world P at the frozen sigma=0.02: hidden=8: oracle 0.939 (in band True), mech leak pass, floor 0.540 (ok) -> PASS. Selected hidden=8 at G seed 2 (first passing candidate in the frozen order G seeds 2, 3, 4 at hidden 8). The organism run launches with this instance; its result is recorded in a later entry.
- **2026-09-29 - HIDDEN-8 NEW-SEED INSTANCE, RESULT (G seed 2; spec `docs/specs/2026-09-28-l3-hidden8-second-seed-design.md`; executed on the owner's GPU machine, RTX 4050, torch 2.7.0+cu126, explicit folds, n = 10; FINDINGS 10.9).** Drift 0.45: survival 0.676 t90 [0.636, 0.717], predictor 0.627, untrained 0.550; `resid_trace` 0.660 [0.631, 0.689]. Frozen rule: replication NOT claimed (t-CI lower bounds under 0.65 on both readouts); margin over untrained passes (+0.126); margin over predictor misses by 0.0009 (+0.0491 against 0.05), so the strict table row is "margins fail". Gates: engagement 20/20, pooled leak clean, floor 0.550, speed at least 0.814, 0 deaths; the L0 TOST reads 0.539, p = 0.207 (open; see the entry below). No deviation from the frozen protocol; the spec's secondary device comparison is confounded (GPU against the CPU control) and is reported, not adjudicated.
- **2026-09-29 - L0 EQUIVALENCE GATE UNDER THE EXPLICIT FOLD SPLIT (found while promoting the entry above; FINDINGS section 16 addendum).** The re-score of the explicit-folds spec did not compute the L0 TOST. On the GPU machine's ten drift-0 survival cells (one set, shared by every GPU-scored L3 run because the surrogate is off at drift 0) the legacy split gave 0.517, TOST p = 0.010 (gate 2 passed as published) and the explicit split gives 0.539, TOST p = 0.207, ROPE P = 0.816 (not shown equivalent at n = 10). The L1 organism run moves 0.522 (p = 0.029) to 0.546 (p = 0.374). The cloud CPU runs pass under explicit (0.514 no-auxiliary, 0.529 world-model). Recorded as an open gate-2 clause on the GPU-side L3 and L1 runs; no verdict is re-adjudicated and no run is repeated to chase the number.

- **2026-09-30 - SKILL-MATCHED BASELINE, STAGE 1 SELECTION AND A BLIND CLARIFICATION OF THE
  MATCH CLAUSE (spec `docs/specs/2026-09-29-l3-skill-matched-baseline-design.md`, amendment
  of 2026-09-30 03:17 UTC).** Stage 1 ran the frozen ascending ladder and stopped at the
  first budget: at `--survival-updates 450` the model-free survival arm returned +0.0067 over
  three seeds at train 0.45 / eval 0.45 (per seed +0.2156, -0.3497, +0.1543; sd 0.3102),
  against the published GPU decoder-carrying reference -0.219. Selection is correct as
  written (the stage-1 rule glosses "within 0.05" as "at least -0.269"), but the arm
  overshot the reference by 0.2257, 1.26 calibration standard errors, so the stage-2 clause
  "within 0.05 of -0.219", which carries no such gloss, became load-bearing. The reading was
  fixed asymmetrically BEFORE any stage 2 cell existed (stage 2 launched 02:09:15 UTC; the
  cells directory was empty at 03:17 UTC when the amendment was committed): an overshoot with
  the primary probe below the bar strengthens the decoder-direct reading, an overshoot with
  the primary probe at or above the bar yields no skill-mediation verdict and is reported as
  confounded by excess skill, and an undershoot is "skill not matched" as originally
  specified. Two design limitations are recorded in the same amendment and not fixed
  retroactively: the ladder ascends only, so an overshoot cannot be corrected within the
  design, and the plus-or-minus 0.05 point window is about one standard error of the quantity
  it gates. Stage 2 (n = 10, `--survival-updates 450`, two CUDA workers, explicit folds) is
  running; its result is recorded in a later entry.

- **2026-09-30 - SKILL-MATCHED BASELINE, STAGE 2 RESULT: NO SKILL-MEDIATION VERDICT (spec
  `docs/specs/2026-09-29-l3-skill-matched-baseline-design.md` as amended 2026-09-30 03:17 UTC;
  executed on the owner's GPU machine, RTX 4050, torch 2.7.0+cu126, explicit folds, n = 10,
  20/20 cells complete 23:56 UTC; FINDINGS 10.8.1).** Match statistic: the stage-2 survival
  arm's mean eval@0.45 return is R = -0.1559 (sd 0.304, se 0.096) against the reference
  -0.219 and the window [-0.269, -0.169], so R sits 0.0131 above the window, about 0.14 of
  one standard error. Primary probe at drift 0.45: survival 0.717 t90 [0.679, 0.755] (8/10),
  predictor 0.588, untrained 0.513; `resid_trace` 0.724 [0.690, 0.757]. Both margin clauses
  and the bar clause pass, so the probe reads at or above the bar. Frozen table row 4
  therefore governs: **no skill-mediation verdict**, the positive reported as confounded by
  excess skill. The 10.8 auxiliary-conditional verdict stands unchanged, neither narrowed to
  a skill effect nor strengthened to decoder-direct; adjudicating the confound would need a
  budget between 300 and 450, which the ascending-only ladder cannot reach within this
  design. Gates: engagement 20/20, pooled leak clean in every cell, untrained floor 0.513,
  speed at least 0.837, 0 deaths; the L0 TOST reads 0.539, p = 0.285, ROPE P = 0.736, the
  same open gate-2 clause recorded above for GPU runs under the explicit split, and no
  verdict here rests on it. No deviation from the frozen protocol. Recorded alongside the
  readout and not adjudicated: this is the first above-bar L3 reading from an arm carrying no
  next-observation decoder. Committed: raw
  `artifacts/reviewer_gaps_runs/l3_h8_nowm_skill_u450/`, summary
  `artifacts/expB2/skill_matched_l3_h8_nowm_u450.json` (`skill_match` block).

- **2026-10-06 - CORRECTED-TRAINER CONFIRMATION RUNS ANNOUNCED (frozen BEFORE launch; spec
  `docs/specs/2026-10-06-corrected-trainer-confirmation-design.md`; correction
  `docs/CORRECTIONS.md` 2026-10-06).** The GAE truncation bootstrap of every survival run in
  this log read the critic value before an episode's final transition; it now reads the
  value of the successor state. The trainer is the only change: the section 9 configuration,
  the section 7 gates, and the section 8 and 10 decision rules are unchanged. Two n = 10 runs
  on a CPU container shown to reproduce a committed device-control cell bit for bit:
  `C1` (auxiliary on, like for like with the 2026-09-28 device control) and `C2` (auxiliary
  off, like for like with the 2026-09-27 architecture baseline), each extended to 450 survival
  updates with frozen snapshots at 100, 200 and 450 for the budget curve; the headline arm is
  the 300-update agent. Every historical number above stays as recorded and is labeled
  historical in `docs/RESULTS_MANIFEST.md`; the results are recorded in a later entry.

- **2026-10-06 - DESCRIPTION CORRECTION: THE SURROGATE AS IMPLEMENTED (revision step 10;
  `docs/CORRECTIONS.md`).** Section 9 describes `G` as a "small recurrent predictor (GRU core)"
  trained "to convergence (early-stop on held-out authentic MSE, patience 5)" over hidden sizes
  {8, 16, 24, 32, 48, 64}. Every L3 result in this log used `itasorl.surrogate_l3.train_g_motion`
  instead: a feed-forward 4 -> h -> h -> 2 ReLU MLP on (vel, a), trained full batch with Adam
  (lr 1e-3) for a FIXED 300 epochs on 250 x 40 authentic scripted-policy transitions, with no
  held-out set and no early stopping, at the capacities recorded in the entries above (8, 7, 4,
  10, and the 16 / 32 / 64 ladder). Section 9 is left as frozen; this entry records that the
  executed configuration differs from it. Introducing early stopping now would produce a
  different fingerprint and would need its own gate 0. In world P drag is the constant 1.5, so the
  authentic law is linear in (vel, a) and withholding drag from `G` withholds a constant; a
  least-squares linear fit recovers the law to rounding, so the fingerprint is the approximation
  error of this finite-trained network (`scripts/run_surrogate_diagnostics.py`,
  `artifacts/surrogate_diagnostics.json`).

- **2026-10-06 - WORDING AMENDMENT: THE 2026-07-23 H2 TEXTURE ENTRY (revision step 9; spec
  `docs/specs/2026-10-06-texture-comparator-design.md`).** The 2026-07-23 entry headlines H2
  as "confirmed in texture-specific form". Its measurements stand. What they show is
  narrower: the direction fit on the learned law does not transfer to matched-band
  Gaussian jitter, and the graded blend lowers decoding as the dynamics difference
  shrinks. No fresh probe was fit on the jitter comparator and no agents were trained under
  it, so a learned-texture mechanism is not established. The entry is left as written; the
  frozen comparator spec separates the three questions and fixes the wording each result
  earns. The hand-authored quadratic-drag comparator failed its gate 0 (amendment in that
  spec), so the learned-texture rule cannot be met in this revision.

- **2026-10-06 - CORRECTED-TRAINER CONFIRMATION RUNS: RESULTS (spec
  `docs/specs/2026-10-06-corrected-trainer-confirmation-design.md`; FINDINGS 17).** Integrity
  passes on both runs: every cell records the successor bootstrap, and the predictor and
  untrained arms equal the historical CPU cells to the bit. `C1` (auxiliary on): survival
  0.733 [0.669, 0.797], predictor 0.589, untrained 0.523, both margins with intervals above
  0.05; every gate passes except L0, which reads 0.559 on the registered pair (TOST
  p = 0.939); verdict MET on the decodability clauses, conditional on the open L0 gate. `C2`
  (auxiliary off): survival 0.613 [0.552, 0.675], predictor 0.589, untrained 0.529, every gate
  passes; verdict NOT MET. Auxiliary comparison: `C1` minus `C2` +0.120 [+0.065, +0.174], so
  the auxiliary-conditional reading holds on the decodability clauses, conditional on `C1`'s
  L0 gate. Correction effect against the historical CPU runs: +0.003 and +0.012 at drift
  0.45, +0.031 and +0.001 at drift 0. The margins were not relaxed and nothing in this entry
  changes a frozen rule. The secondary and exploratory readouts on the corrected agents
  (FINDINGS 17.5 to 17.11) are reported there, each against its own frozen spec where one
  exists; the cross-run replay of 17.10 was written after these results were read and is
  labeled exploratory.

- **2026-10-07 - GOAL AND STAKES RUNS: PREREGISTERED (spec
  `docs/specs/2026-10-07-goal-and-stakes-design.md`).** Three runs on the corrected trainer,
  cell for cell against `C1`, frozen before any cell exists. `T-touch`: the trained arm is
  rewarded +1 per pellet eaten with death off (a goal without stakes). `S-immortal`: the
  survival reward with death off. `S-scarce`: the survival reward at a pellet count chosen by
  a frozen calibration rule (scripted-walker 80-step death rate in [0.40, 0.60] at drift 0,
  within 0.10 of the drift 0.45 rate, zero deaths inside the 24-step pooled window), to be
  recorded here as a dated amendment before launch. Two runner flags (`--objective`,
  `--mortal`) default to the registered behavior and are excluded from the config fingerprint
  at their defaults, so no historical checkpoint changes hash. Frozen wording: `T-touch`
  meeting the primary rule reads "a pellet-directed goal without survival stakes produces
  walks from which the condition is decodable; survival is not required"; the stakes claim
  needs `S-immortal` < `C1` < `S-scarce` with the paired `S-scarce` minus `S-immortal` contrast
  at least +0.05 and its 90% interval excluding zero; no survival-specific wording returns
  on any outcome beyond "pellet-seeking alone was not enough in this world". Engagement for
  the touch objective is a 1.5x ratio over the better baseline, since the registered
  absolute margin was calibrated on the survival reward's scale. Everything else is the
  registered rule set. Four mechanism readouts are frozen in the same spec and applied to
  every run and to `C1` on the saved trained arm: an intervention test (state nudged by one
  standard deviation along the decoded direction against a sham direction; passes if it
  closes at least 0.10 more of the authentic-to-surrogate behavior gap than the sham, paired
  interval excluding zero), a descriptive between-world behavior comparison, a
  prediction-error channel (surprise probe at least 0.65 and a within-world correlation with
  the decoded direction of at least 0.20), and a within-lifetime foraging-recovery test that
  is reported as uninformative unless the surrogate reduces intake in the first half of an
  episode. None of these changes the primary verdict or the stakes rule.
- **2026-10-07 - COMPARATOR-TRAINED `qd` RUN RECORDED AS EXECUTED AND SET ASIDE (FINDINGS
  17.9 addendum).** The 2026-10-06 entries above and the comparator spec say the
  comparator-trained run `T-qd` is not run. One was in fact executed on 2026-10-06 (commit
  `2606e7f`, trainer and `itasorl/` identical to main; family `qd` at eps 6.0, hidden 8,
  n = 10, successor bootstrap; output `fullruns/T_qd_l3_h8_wm/`, gitignored). Its gate 0
  had already failed (untrained floor deviation 0.108 outside the 0.1 tolerance; drag
  ceiling undefined), so it bears on no rule in this document. Descriptive outcome at
  drift 0.45: survival 0.580 (90% CI [0.542, 0.615]), predictor 0.573, untrained 0.608,
  engagement in 6 of 10 seeds. It is not promoted and is not counted as a negative. No
  `gn`-trained run was executed. The learned-texture rule remains unmet in this revision.

- **2026-10-08 - DRIFT-0.45 WORLD-SAMPLE SENSITIVITY, ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-08-drift-045-world-sample-sensitivity-design.md`).** The pooled readout
  draws its authentic and surrogate pools from different evaluation-world seed bases, and how
  much that draw contributes has been measured at drift zero only, where it moves the reading
  with a between-draw standard deviation of 0.049 against a between-seed deviation of 0.017.
  This diagnostic carries the same eight independent pairs to drift 0.45, on the ten saved
  survival agents of the historical hidden-8 held-out run, because the corrected C1 agents are
  not in the repository. It is a readout on pre-correction agents and measures the estimator's
  sensitivity to the draw, not the corrected headline. The statistic is the signed per-draw
  mean with the draw as the unit of inference; folding to an absolute deviation is conservative
  at drift zero but anti-conservative where a real signed effect exists, so it is prohibited
  here. The decision rule, the three outcome labels, the integrity gate against the recorded
  per-seed values, and the exact wording written in each of the three cases are all fixed in
  the spec before the run. Section 5's readout, section 6's bar and margins, and every gate in
  section 7 are unchanged, and this diagnostic cannot overturn a registered verdict.

- **2026-10-09 - REVERSE-DIRECTION READOUT, ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-09-reverse-direction-readout-design.md`).** Every positive on record is in
  one direction: the agent is raised inside the surrogate and the probe separates the law it was
  raised under from the authentic law. The other direction, an agent raised in the authentic world
  probed against a learned surrogate it never lived in, has never been measured, and it is the
  direction the project's own framing describes. This diagnostic scores the ten saved drift-0
  agents of each arm of the historical hidden-8 held-out run with the standard pooled readout at
  drift 0.45 and with the balanced same-world-seed readout, because the corrected C1 agents are not
  in the repository. It is a readout on pre-correction agents and measures the direction-dependence
  of the estimator, not the corrected headline. The integrity gate is that the same agents rescored
  at drift 0 reproduce this run's recorded explicit-partition drift-0 targets within 0.01 on every
  seed and arm, or the run is void. The three outcome labels (CARRIED, NOT CARRIED, AMBIGUOUS),
  their thresholds, and the exact wording written in each case are fixed in the spec before the run.
  Section 5's readout, section 6's bar and margins, and every gate in section 7 are unchanged, and
  this diagnostic cannot overturn a registered verdict.

- **2026-10-09 - COMPARATOR-TRAINED RUN ON THE WHITE-JITTER FAMILY, ANNOUNCED BEFORE THE RUN
  (spec `docs/specs/2026-10-09-gn-comparator-trained-design.md`).** The 2026-10-08 texture entry
  above records that no `gn`-trained run was executed and that the learned-texture rule remains
  unmet. This is that run. `G_gn` at `sigma_v = 0.01` already passed gate 0 on world `P` before
  this entry (oracle 0.8653, in band; mechanical leakage clean; untrained floor 0.4479, inside
  tolerance; `artifacts/expH2/texture_knockout_h8.json`), which is what distinguishes it from the
  hand-authored quadratic drag whose gate 0 failed. Unlike the two diagnostics announced on
  2026-10-08 and earlier on 2026-10-09, this run is **confirmatory**: it applies section 6's rule
  and section 7's gates verbatim, with nothing relaxed, to agents raised inside an in-band
  unstructured perturbation. The configuration is the registered L3 one with the surrogate family
  switched (ten seeds, drifts 0 and 0.45, 300 updates, three arms, corrected successor-value
  trainer, agents saved). The wording written in each of the three cases, including the case where
  the rule is MET and the paper's sufficiency claim must therefore be withdrawn in its current
  form, is fixed in the spec before the run.

- **2026-10-09 - L1 STREAM AND CONTROL READOUT, ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-09-l1-stream-readout-design.md`).** The L1 negative is a negative on the
  state: agents raised under the observation-quantization seam read 0.533 while the oracle reads
  the seam at 0.873 in band under its own grid oracle. Every stream-only decoder and behavior-trace
  readout in this project was scored on the corrected L3 agents, so two explanations of that
  negative are currently indistinguishable: the seam may not change where the policy goes, leaving
  nothing in the agent's own stream for the trunk to keep, or the stream may carry it and the trunk
  not keep it. Only the second is an encoding failure in the sense the sufficiency claim needs.
  This diagnostic runs the existing control battery on the sixty saved agents of the historical L1
  held-out run at that run's own rung (grid spacing 0.023, sensor noise 0.01). The battery does not
  change; the only new code makes the rung selectable, with the L3 default unchanged. The integrity
  gate is that the state probe reproduces the L1 run's recorded per-seed targets within 0.01 and
  that regenerated pools bit-match the saved dumps, or the run is void. The two readings (S1, the
  stream carries it; S2, it does not) and the wording written in each case are fixed in the spec
  before the run. The L1 run's own adjudication stands as recorded whatever the stream reads, and
  no gate, margin, or registered verdict changes.

- **2026-10-09 - OUTCOME OF THE WHITE-JITTER COMPARATOR-TRAINED RUN: UNINFORMATIVE.** The run
  announced above executed at the registered configuration (`artifacts/texture/T_gn_l3_h8_wm.json`,
  FINDINGS 14.5.1). Section 7 gate 3, the speed positive control at 0.75, reads
  0.74545 on the survival arm at drift 0.45, seed 2, missing by
  0.00455 on one of twenty cells. Section 8's two verdicts both
  require every gate to pass, so neither is reached, and the matrix names no cell for a
  positive-control failure. The margin is not relaxed. The run is uninformative, is not counted as
  a negative, and strengthens no claim; the learned-texture rule remains unmet. Descriptive
  outcome at drift 0.45, recorded because this log and the manuscript quote it: survival
  0.522 (90% CI [0.502, 0.542]),
  predictor 0.511, untrained 0.552,
  0 of 10 seeds at the bar, survival minus untrained
  -0.030 [-0.051, -0.008]. Every other gate passes, including
  the drift-zero L0 equivalence the corrected headline run does not.

- **2026-10-09 - CORRECTION TO THE WHITE-JITTER GATE SCORING (the outcome entry above is left
  as written, per the append-only rule).** The positive control reported in that entry was scored
  on the **survival arm only**, which understated the failure. Section 7 gate 3 carries no arm
  restriction and the project's own convention, written into `scripts/promote_bv3_gates.py`, is
  "speed probe at least 0.75 in every pool". Rescored that way across both drift slices and all
  three arms, the control is short in **9 of 60** arm-by-cell pools, worst
  **0.6862** (short by 0.0638), all-pool mean
  0.8179; the shortfall is 3 of 10 untrained cells at drift 0, 5 of 10 untrained
  cells at drift 0.45, and 1 of 10 survival cells at drift 0.45. The survival-arm figures quoted in
  the entry above (0.74545, 19 of 20) are correct for that arm and are retained as
  a recorded field. The promoter (`scripts/promote_texture_trained.py`) and the audit are corrected,
  and the promoter now refuses to write an artifact with any registered gate unscored, which also
  closed a second gap: engagement had not been recorded at all and is now read from the run's cells
  (20 of 20 engaged). **The verdict is unchanged, UNINFORMATIVE, and no margin was relaxed in either
  direction.** The correction makes the failure larger, not smaller: the shortfall sits mainly in
  the untrained arm, the baseline the registered margins are measured against, so the run does not
  establish that its own comparator is probeable in this family.

- **2026-10-09 - SECOND CORRECTION TO THE WHITE-JITTER RUN: IT WAS NOT THE REGISTERED
  CONFIGURATION (the two entries above stand as written, per the append-only rule).** The frozen
  spec said the run would execute "the registered L3 configuration with the surrogate family
  switched". It did not. The agent trunk was **8 units wide, not the registered 96**: the launch
  command carried `--hidden 8`, which sets the agent's GRU width, conflated with `--l3-hidden 8`,
  which sets the surrogate's and is the correct gate-0 value. The same conflation was in the
  Colab notebook's extras cell, which ships in the supplement; it is fixed there now. Verified
  from the saved weights, `cell.weight_hh` is `(24, 8)` on all three arms against `(288, 96)` for
  every other run in the project. Found on 2026-10-09 while investigating why the positive
  control failed.

  This is a second and more fundamental reason the run is uninformative, independent of the
  positive control recorded above, and it is the likely cause of that failure: the control is low
  in every arm at both drifts, including the drift-0 cells where no surrogate is installed, which
  no property of the jitter could produce. A trunk a twelfth of the registered width also makes
  the descriptive numbers incomparable to every other cell in this project, so they should not be
  read as a reading of the white jitter at all. **Nothing changes in the verdict or anywhere else:
  the run was already UNINFORMATIVE, already supported no claim, and still does. The
  learned-texture rule remains unmet, and a usable trained case still requires a new run under a
  new frozen specification, at the registered trunk width.**

- **2026-10-10 - S-SCARCE CALIBRATION: THE FROZEN RULE CHOOSES THE REGISTERED DENSITY, SO
  `S-scarce` IS NOT RUN (spec `docs/specs/2026-10-07-goal-and-stakes-design.md`, "Calibration of
  the scarce rung"; output `artifacts/goal_stakes/calibration.json`).** The calibration ran as
  frozen: `python scripts/calibrate_scarcity.py --out artifacts/goal_stakes/calibration.json`,
  basal burn 0.4, 200 scripted and 200 random episodes of 80 steps per world per setting, on a
  local Windows CPU (the rule fixes no device; it trains nothing). Scripted-walker 80-step death
  rates at drift 0 and drift 0.45, by pellet count: 24 pellets 0.535 and 0.515; 16 pellets 0.68
  and 0.625; 12 pellets 0.72 and 0.73; 8 pellets 0.815 and 0.825; 6 pellets 0.885 and 0.87; 4
  pellets 0.885 and 0.895. No setting had a death inside the 24-step pooled window in either
  world. The largest pellet count in the [0.40, 0.60] band is 24, its drift-0.45 rate is within
  0.10 of its drift-0 rate, and it has no early deaths, so the rule chooses 24. That is the
  registered density `C1` already uses. The spec assumed the registered density carried low
  stakes and that the rule would choose a scarcer one; for the scripted walker it already sits
  in the band, and every scarcer setting overshoots it. The basal-0.5 fallback applies only when
  no setting lands in the band, so it does not apply. **`S-scarce` at the chosen setting would
  be a rerun of `C1`'s configuration, so it is not run, and no other pellet count is substituted
  after seeing these rates.** The stakes rule needs `S-immortal` < `C1` < `S-scarce`, so it
  cannot be tested in this world. The decision output records the stakes rule as "not testable:
  the frozen calibration chose the registered density, so no scarcer rung exists", writes neither
  stakes wording, and reports the `S-immortal` and `C1` survival means descriptively with no
  threshold. The survival-specificity wording stays unavailable, because it needs the stakes rule
  met. `T-touch`, `S-immortal`, every gate, and every other frozen rule are unchanged.

- **2026-10-10 - GOAL AND STAKES: THE PREDICTOR ARM INHERITED `--mortal off`, SO INTEGRITY
  CHECK 2 COULD NOT PASS; FIXED, AND BOTH RUNS RESTART (spec
  `docs/specs/2026-10-07-goal-and-stakes-design.md`, integrity check 2).** The spec states
  that the predictor and untrained arms are objective-free and, in `T-touch` and `S-immortal`,
  equal `C1`'s arms to the bit. The implementation applied the `--mortal` switch to every world
  `make_world` builds, including the 80-step scripted episodes the predictor arm trains on. In
  `C1` the scripted walker dies in about half of those episodes (0.535 at the registered density,
  the calibration above), so with death off the predictor trained on longer episodes and
  different normalizer statistics. Both cloud runs completed all 20 cells this way. In every cell
  of both runs the untrained target equals `C1`'s to the bit (20 of 20 per run) and the predictor
  target does not (0 of 20; on the four drift-0 seeds `C1` reads 0.431, 0.554, 0.607, 0.483 and
  both runs read 0.470, 0.497, 0.492, 0.476). `T-touch` and `S-immortal` agree with each other
  to the bit, and a first step on an unpinned torch gave the same predictor values as the
  pinned step, so neither the device nor the library version is the cause. Under check 2 a
  mismatch stops the comparison, so **no cell of either run is read.** The predictor's scripted
  collector now always builds mortal worlds, whatever `--mortal` says; the switch acts on the
  trained arm's world only, which is what the spec describes. A test trains the predictor with
  the switch on and off and requires identical weights, on a batch shown to contain deaths
  inside 80 steps, and it fails on the old code. At the default (`--mortal on`) nothing changes,
  so no recorded run and no fingerprint moves. Both runs' cells move to
  `artifacts/goal_stakes/quarantine/` on their run branches, and both runs restart from zero
  cells on the fixed code, pinned to `C1`'s torch 2.14.1, numpy 2.4.6 and scikit-learn 1.9.1.
  Every gate, margin, and frozen wording is unchanged.

- **2026-10-10 - WHITE-JITTER COMPARATOR-TRAINED RUN AT THE REGISTERED TRUNK, ANNOUNCED BEFORE THE
  RUN (spec `docs/specs/2026-10-10-gn-comparator-trained-h96-design.md`).** The 2026-10-09
  white-jitter run trained an 8-unit trunk instead of the registered 96 and is recorded above as
  uninformative; that record stands. This entry announces the run the 2026-10-09 spec actually
  described: the same question, gate 0, decision rule, gates, and pre-written outcome text, with the
  agent trunk at its registered width of 96. It executes in the Claude Code cloud container pinned
  to `C1`'s stack (torch 2.14.1+cpu, numpy 2.4.6, scikit-learn 1.9.1, 4 workers), which reproduces
  `C1`'s cells to the bit, on branch `run/T-gn-h96`. The integrity check before reading requires
  every cell to record the successor bootstrap, the survival objective, death on, and a 96-unit
  trunk. Section 5's readout, section 6's bar and margins, and every gate in section 7 are unchanged,
  and this run cannot overturn a registered verdict.

- **2026-10-10 - BEHAVIOR-MEDIATION AUDIT OF `T-touch` AND `S-immortal` FROM REBUILT STATE DUMPS.**
  The two runs saved their agents but not their pooled-state dumps, which the behavior-mediation
  audit reads. `scripts/regenerate_state_dumps.py` replays the standard pooled readout on each
  saved agent (world seed bases 800000 and 850000, 110 episodes by 24 steps) and writes the dumps in
  the run's own layout. A rebuilt dump is used only if its pool target equals the target the run
  recorded in that cell to the bit; one failure stops the audit. The audit is descriptive, as it is
  for every other run, and is not a gate.

- **2026-10-10 - THIRD CORRECTION TO THE WHITE-JITTER RUN, AND ONE FOR THE QUADRATIC-DRAG RUN:
  BOTH COMPARATOR-TRAINED RUNS HAD AN 8-UNIT TRUNK, AND THE FROZEN SPECS CARRIED THE ERROR (the
  2026-10-09 entries above stand as written, per the append-only rule).** The second correction
  above places the error in "the launch command". It was in the frozen design. The command block
  of `docs/specs/2026-10-09-gn-comparator-trained-design.md` (line 44, committed in `d341091`
  before the run, rebased with the same text as `caa2ce0`, the parent of the run commit
  `60eb355`, and unchanged since) reads `--l3-hidden 8 --hidden 8`, and the launch copied
  it, so the spec's own sentence that the run is "the registered L3 configuration ... and
  nothing else changed" contradicted its command. The conflation began in the 2026-10-06 texture
  comparator spec, which describes the comparator-trained protocol as "corrected trainer,
  auxiliary on, hidden 8 trunk". The same entry's "`(288, 96)` for every other run in the
  project" is also wrong. The quadratic-drag comparator-trained run (`T-qd`, executed on
  2026-10-06 against that spec's "not run" line and set aside in FINDINGS 17.9) has
  `cell.weight_hh` `(24, 8)` on all 60 saved agents, and its recorded config fingerprint
  `23a620e51d4e26fd` is reproduced by its flags with `--hidden 8` only (`--hidden 96` gives
  `f947a5bce5be281b`). Its speed positive control is short in 8 of 60 pools, worst 0.7018
  (untrained, drift 0, seed 2), which no entry reported, and its three short untrained drift-0
  pools equal the white-jitter run's to every digit; at drift 0 no surrogate is installed, so
  that fits the trunk-width reading. The other saved agent sets checked from their weights on
  2026-10-10 all have `(288, 96)`: `l1_heldout`, `l3_h7_heldout`, `l3_h8_heldout`,
  `l3_h8_gseed2_gpu`, `l3_h10_gseed1_gpu`, `l3_h8_nowm` and `l3_h8_nowm_skill_u450`. Runs whose
  agents were not kept cannot be checked this way. **No verdict changes**: the white-jitter run
  was already UNINFORMATIVE, `T-qd` was already excluded from the learned-texture rule, and
  neither supports any claim. Any later comparator-trained run needs a new frozen spec whose
  command states `--hidden 96`. FINDINGS 14.5.1 and 17.9 carry dated corrections; both specs
  carry dated notes at their end; `docs/CORRECTIONS.md`, 2026-10-10.

- **2026-10-10 - FREEZE TIMING THAT GIT DOES NOT SUPPORT: FOUR RUNS WHOSE SPEC WAS NOT
  COMMITTED BEFORE THE RUN (no earlier entry is edited).** Several entries and specs say a
  design was "frozen before the run". For four runs no committed timestamp precedes the run.
  (1) The hidden = 7 selection, already recorded in the 2026-07-16 freeze-timing entry above as
  commit-simultaneous with its result. (2) The sensory-echo control of 2026-09-26: its spec,
  runner, the 2026-09-26 entry above and the promoted artifact all first appear in one commit,
  `a641ca0` at 17:45:13, while the run's cells were written at 16:37:16 and its aggregate at
  16:38:53 (local time, UTC-4, as are all times here). The spec's "frozen before any run" and
  its "Amendment, 2026-09-26, before any run" are statements in a file first committed about an
  hour after the run ended; no ref holds an earlier version. (3) The nonlinear joint control
  (probe B of `docs/specs/2026-09-27-local-strengthening-probes-design.md`, recorded in the
  2026-09-28 entry above): its spec was first committed in `791a30a` at 2026-09-27 19:23:54,
  about seven minutes after the run started. The first predictor trace was written at 19:20:45
  and the first survival trace at 19:24:28, 34 seconds after the commit, so the spec landed
  mid-run, before any survival cell had finished. (4) The A1 graded-seam ablation: its spec,
  which reads "FROZEN for the A1 launch", was first committed in `c4e4417` at 2026-07-21
  23:55:04, 87 seconds after the run wrote its aggregate (23:53:37). The written-before-launch
  statements may be true; git cannot show it. These are the cases the 2026-10-10 review checked
  and found; it did not audit every other spec's timing. No number, rule or verdict changes.
  FINDINGS 10.4.2 now describes the sensory-echo spec this way.

- **2026-10-10 - PROVENANCE OF THE THREE ENTRIES AND SPECS HEADED 2026-10-09 (dates only; the
  headers and entries stay as written).** The reverse-direction, white-jitter comparator-trained
  and L1 stream entries above, and their specs, are headed 2026-10-09, while their commits and
  run logs read 2026-10-08. The headers carry the UTC date; the commits and logs carry local time,
  UTC-4. Each was committed before its run started. Reverse direction: spec and entry
  `8aa2b9f` at 2026-10-08 21:39:03, run started 21:39:23 (`fullruns/reverse_direction/run.log`).
  White jitter: spec and entry `d341091` at 21:41:15; the spec's device paragraph `60eb355` at
  21:57:27; the launcher queued the run at 21:57:09, 18 seconds before that paragraph was
  committed, and the run started at 22:00:55 (`fullruns/T_gn_l3_h8_wm/run.log`). L1 stream:
  spec and entry `e4686f2` at 22:35:15, run started 22:39:14
  (`fullruns/l1_stream_readout/run.log`). The run logs are gitignored local files. `8aa2b9f`
  and `d341091` live on `origin/feat/final-review-fixes`; the runs' own lineage,
  `origin/feat/direction-and-comparator-runs`, holds their rebased copies `13f60bc` and
  `caa2ce0` with the same text and timestamps (`caa2ce0` is the parent of `60eb355`), together
  with `60eb355` and `e4686f2`, and reached `main` in the squash merge `c05d791` (#130).

- **2026-10-10 - IN-PLACE EDITS TO EXISTING LINES OF THIS DOCUMENT (a record; nothing is changed
  by this entry).** The deviation log is meant to grow only by appended entries. Six commits
  changed existing lines of this file instead (`git log --numstat`). `ba254ec` (2026-07-10)
  rewrote section 4's description of how `G` becomes a world into the two-stage construction, 81
  minutes after the registration commit `0217263` and before the first entry in this section was
  written. `c594c3d` (2026-07-10) replaced this section's "(none yet)" placeholder with the
  first entry. `15b1fc7` (2026-07-14) replaced the closing "STILL OWED" sentence of the
  2026-07-14 hidden = 7 entry when the owed probe was recorded the same day. Three
  repository-wide passes followed: `75c49a2` (2026-07-20) changed the hardware string "RTX 4050"
  to "local GPU" and moved spec paths from `docs/superpowers/specs/` to `docs/specs/` in twelve
  lines, `5e9e84b` (2026-07-20 local, 2026-07-21 UTC) replaced a long dash with a hyphen in one
  line, and `8eb9174` (2026-08-23) did the same in two lines. The passes also touched one to
  twenty lines each of `docs/PREREGISTRATION.md`, `docs/PREREGISTRATION_Bv3.md` and
  `docs/PREREGISTRATION_C.md`, including a reworded background paragraph and an added reader
  note. Apart from the pre-run section 4 resolution, none of these edits changed a number, a
  rule, a gate or a verdict.

- **2026-10-10 - CLARIFICATION: WHERE THE L0 GATE'S MARGIN IS REGISTERED (nothing is
  re-adjudicated).** Section 7 gate 2 above reads "equivalent to 0.5 by TOST" with no margin. It
  inherits the B-v3 gates ("The four B-v3 gates"), `docs/PREREGISTRATION_Bv3.md` section 7 says
  its gates are "Identical to B-v2", and the margin, plus or minus 0.05, is written in
  `docs/PREREGISTRATION.md` section 7 gate 2. The test level of 0.05, which every L0 adjudication
  on record uses and which the 2026-10-06 primary-analysis spec states, and the [0.45, 0.55] ROPE
  band appear in no preregistration. They are the defaults of `itasorl.stats.equivalence_test`
  and `itasorl.stats.rope_test`, applied as the project's reporting convention.

- **2026-10-10 - OBSERVATION-CHANNEL LOCALIZATION: WHAT THE TWO 2026-07-24 ENTRIES CAN SHOW
  (a record; those entries stay as written and no verdict is re-adjudicated).** Both entries
  say the positive controls (speed, energy, food) "remain readable in every mask condition, so
  the chance collapse is not a dead-probe artifact". That holds for the single-channel masks
  only. With all 146 dimensions masked the trunk receives a constant input and, under the
  deterministic policy, its own constant previous action, so every episode in both pools
  produces the same state sequence and every probe reads exactly 0.500: speed and energy read
  0.500 in all 30 all-mask cells of `fullruns/l3_h8_obs_localization/cells.json` and of the
  hidden-7 counterpart. The all-mask collapse is true by construction and tests nothing. A
  masked channel is zeroed before the frozen normalizer, so it enters the trunk as the constant
  -mean/sd: for the hidden-8 seed-0 survival agent at drift 0.45 the interoceptive channels sit
  as far as 31.8 sd below their training mean, the vision channels at a median of 1.0 sd. The
  single-channel masks are constant-input lesions, so the hidden-8 entry's "VISUAL, NOT
  INTEROCEPTIVE" reading is narrowed to what such a lesion shows. At hidden 7 the masks also
  lift the untrained arm (0.724 under the vision mask with 8 of 10 seeds at the bar, 0.646
  under the interoception mask with 5 of 10); paired by seed, survival minus untrained is
  +0.041 [-0.017, +0.098] and +0.096 [+0.024, +0.169] against +0.151 [+0.099, +0.202] unmasked
  (`artifacts/expH2/obs_localization_h7.json`). The hidden-7 entry's "ROBUST TO SINGLE-CHANNEL
  MASKING" therefore describes a reading that is not specific to training, and its clause
  that the coarser surrogate's artifacts propagate into both streams is withdrawn. FINDINGS
  14.6 carries the same note.

- **2026-10-10 - THE HIDDEN = 7 GATES, COUNTED OVER EVERY ARM, AND THE HIDDEN = 8 REGRESSION
  CHECK (a record; the 2026-07-13 and 2026-07-14 entries stay as written).** The 2026-07-14
  hidden = 7 entry reports "ALL gates passing" with "speed positive control 0.959, pooled
  reward-leak 0.567 clean in 10/10 seeds". Those two figures are the survival arm's at drift
  0.45. Section 7 gate 4 ("reward-sum / length / lifetime each within 0.1 of 0.5") names no
  arm, and the comparator-trained runs are scored over every arm
  (`scripts/promote_texture_trained.py`). Counted that way, one pool of the 60 is outside the
  tolerance: predictor, drift 0.45, seed 8, reward-sum AUROC 0.616
  (`fullruns/l3_h7_traces`; the same value in `artifacts/expB2/heldout_l3_h7_reverse_summary.json`).
  The other 59 pools are clean. The speed control's minimum over every pool is 0.784 (untrained,
  drift 0, seed 0), which passes. The hidden = 7 claims are historical and secondary and none
  rests on the predictor pool; the exception is recorded rather than rounded away, as the
  B-v3 gate entry of 2026-09-26 does for its one leakage cell. Separately, the 2026-07-13 entry
  calls the hidden = 8 regression "EXACT (oracle 0.928, floor 0.482)". `scripts/run_expA_l3.py`
  passes that check when the oracle is within 0.05 of 0.928 and the floor within 0.1 of 0.483.
  The recalibration's floor, 0.482, is the mean of three floor seeds (0.4831, 0.4140, 0.5487),
  while the 0.483 frozen on 2026-07-10 matches the seed-0 floor; the oracle reads 0.9278. The
  check passed within its tolerances, which is what "EXACT" should have said. FINDINGS 10.5
  carries both notes.

- **2026-10-10 - THE SPEED POSITIVE CONTROL OF SIX ENTRIES WAS THE LOWEST ARM MEAN, NOT THE
  LOWEST POOL (a record; the entries stay as written and the threshold is not relaxed).** Gate 3
  asks for the speed probe at or above 0.75 in every pool. The architecture-baseline,
  second-instance, device-control, GPU re-measure, hidden-8 new-seed and skill-matched entries
  above (2026-09-27 to 2026-09-30) quote a minimum that `scripts/promote_reviewer_gaps_runs.py`
  computed over arm means. Read over every drift, arm and seed, the lowest pools are:
  architecture baseline 0.835 -> 0.787; second instance on CPU 0.830 -> 0.753; device control
  0.830 -> 0.772; GPU re-measure 0.833 -> 0.748; hidden-8 new seed 0.814 -> 0.758; skill-matched
  baseline 0.837 -> 0.790. The corrected runs `C1` and `C2` read 0.772 and 0.787, which their
  verdicts already used. Every run but the GPU re-measure still passes. There, one pool of 60
  (untrained, drift 0.45, seed index 6) reads 0.748, so the positive control is not met on that
  run; by the rule applied to the white-jitter run (2026-10-09 entries above), its readings are
  descriptive. Its rule was not met in any case, so no verdict moves. The artifacts record the
  per-pool value with its location and keep the arm-mean figure beside it
  (`gates.speed_min`, `speed_min_at`, `speed_min_of_arm_means`); FINDINGS 10.9 carries the
  table, and `docs/CORRECTIONS.md` the correction. Runs promoted after this change, the
  goal-and-stakes runs included, record the per-pool minimum and are read under the per-pool
  rule.

- **2026-10-10 - TWO RECORDED VERDICT WORDINGS RESTATED (a record; the entries stay as written,
  no reading moves and no margin changes).** (1) The 2026-10-06 results entry gives the `C1`
  verdict as "MET on the decodability clauses, conditional on the open L0 gate" and says the
  auxiliary-conditional reading "holds on the decodability clauses, conditional on `C1`'s L0
  gate". The frozen rule counts every gate, so with L0 open the registered verdict is NOT MET;
  the decodability clauses are met. The auxiliary rule asks for `C1` to meet its primary rule,
  so it is NOT MET as registered, while its lead, +0.120 [+0.065, +0.174], holds on the
  decodability clauses. `artifacts/corrected_verdicts.json` now labels the run "NOT MET as
  registered (open gate(s): l0); decodability clauses MET", and FINDINGS 17.2, 17.4 and 17.12
  lead with the registered outcome. (2) The 2026-07-25 L1 entry says the oracle-detectable L1
  artifact "leaves no incidental world-identity encoding". Primary H_B2 is not met, as
  recorded. The survival mean, 0.533, sits above chance: the t-based 90% interval of its ten
  seeds is [0.506, 0.561], and the TOST against 0.5 at the 0.05 margin does not accept
  equivalence (p = 0.147). The reading is "below the bar", not "no encoding"; FINDINGS 14.7
  and section 6 carry the same wording, and `docs/CORRECTIONS.md` both corrections.

- **2026-10-11 - GOAL AND STAKES: RESULTS (spec `docs/specs/2026-10-07-goal-and-stakes-design.md`;
  FINDINGS 18; decision output `artifacts/goal_stakes/decision.json`).** Both runs executed in the
  Claude Code cloud container pinned to the `C1` stack, on the predictor-mortal fix recorded above.
  Integrity: every cell records the successor bootstrap and its run's objective and mortality, and
  the predictor and untrained pooled targets equal `C1`'s to the bit in 20 of 20 cells of each run.
  `T-touch` meets the primary rule with every gate passing: survival 0.796 [0.762, 0.830], ahead
  of the predictor by +0.207 and of the untrained arm by +0.273, ten of ten seeds at or above
  0.65, touch engagement 1.30 pellets against the better baseline's 0.52 (the 1.5x ratio passes),
  and the L0 TOST at drift zero accepts equivalence (0.522). The frozen wording applies: "a
  pellet-directed goal without survival stakes produces walks from which the condition is
  decodable; survival is not required." `S-immortal` meets the decodability clauses (survival
  0.744 [0.696, 0.793], +0.156 and +0.222 over the two control arms) with the L0 gate open (0.537,
  not equivalent), as in `C1`. The stakes rule is not testable, as the 2026-10-10 calibration entry
  records; the descriptive survival means are `S-immortal` 0.744 and `C1` 0.733 on the same seeds.
  Mechanism readouts, frozen wording: intervention, in both runs "the decoded direction did not
  move behavior under the tested nudge" (real minus sham, `T-touch` -0.001 [-0.014, +0.011],
  `S-immortal` -0.029 [-0.057, -0.002], ten informative seeds each); surprise, `T-touch` "the
  surrogate is surprising to the agent, but the decoded direction is not explained by that
  surprise" (AUROC 0.912, correlation -0.146) and `S-immortal` "the decoded direction tracks the
  agent's own prediction error" (AUROC 0.848, correlation -0.240); adaptation, `T-touch` "foraging
  in the fake world recovers within a lifetime" and `S-immortal` "no within-lifetime recovery was
  detected". No survival-specific wording is earned: it needs `T-touch` to fail and the stakes rule
  to be met, and neither holds. The predictor and untrained floors of the intervention readout are
  recorded beside it. Two earlier attempts of each run (an unpinned torch; the predictor inheriting
  `--mortal off`) are quarantined on the run branches and enter no result. `C1` reran on the same
  stack with its agents saved: pool targets equal `C1`'s to the bit in 60 of 60 arm-cells. The
  mechanism readout on those agents, a descriptive reference that `decide_goal_stakes.py` does not
  read, gives `C1` the same intervention wording (real minus sham +0.017 [-0.019, +0.054]), "the
  surrogate is surprising to the agent, but the decoded direction is not explained by that
  surprise" (AUROC 0.846, correlation -0.126), and no within-lifetime recovery.

- **2026-10-11 - WHITE-JITTER COMPARATOR-TRAINED RUN AT THE REGISTERED TRUNK: RESULT (spec
  `docs/specs/2026-10-10-gn-comparator-trained-h96-design.md`; FINDINGS 18.6).** Integrity passes:
  20 cells, successor bootstrap, survival objective, death on, and a 96-unit trunk in all 60 saved
  agents. Under the registered matrix the run is **uninformative**: the L0 gate fails (the drift-0
  survival target reads 0.559, TOST p = 0.94, not equivalent to chance), and the matrix requires
  every gate for either verdict. The pre-written text for a gate failure applies: the run is
  reported as uninformative, it is not promoted, it neither strengthens nor weakens the sufficiency
  claim, and the limitation that no comparator-trained run supports that claim stays. Recorded
  descriptively and not as a verdict: the survival arm reads 0.541 [0.523, 0.559] at drift 0.45,
  zero of ten seeds at the bar, +0.040 over the predictor and +0.044 over the untrained arm.
  Engagement passes in 20 of 20 cells and the speed positive control in 60 of 60 pools (minimum
  0.772).

- **2026-10-11 - CORRECTED-TRAINER RERUNS OF THE GENERALITY RUNS ANNOUNCED (frozen BEFORE
  launch; no rule changes).** Four runs, specified in
  `docs/specs/2026-10-11-corrected-generality-reruns-design.md`: the hidden = 7 second
  capacity, the hidden = 10 seed 1 and hidden = 8 seed 2 fingerprint instances, each on the
  corrected trainer with saved agents, and a rerun of `C2` whose purpose is its saved agents.
  Each runs in the cloud on the stack `C1` used, from `main`, with the explicit fold partition.
  Each instance is read under the rule that adjudicated its historical run, every gate counted
  over every arm, and its gate 0 is measured again on the fingerprint the run fits; a gate-0
  miss makes that run off-design and is reported. The `C2` rerun must match the committed `C2`
  cells to the bit before its agents stand in for `C2`'s. Results are recorded beside the
  historical rows, which keep their labels.

- **2026-10-11 - CORRECTED-TRAINER RERUNS OF THE GENERALITY RUNS: RESULTS (spec
  `docs/specs/2026-10-11-corrected-generality-reruns-design.md`; FINDINGS 20; summaries
  `artifacts/expB2/corrected_{C2-rerun,H7-corrected,H10GS1-corrected,H8GS2-corrected}.json`, cells
  and post-run checks under `artifacts/corrected_runs/`).** All four ran in the cloud on the `C1`
  stack from `main` at `3036eb3`. Every cell records the successor bootstrap and the config hash
  the spec computed before launch. `C2-rerun`: its cells share 5000 values with the committed `C2`
  cells and all 5000 are bit-equal, so its saved agents stand in for `C2`'s; it is read under no
  rule. The drift-0 cells of the three instance runs are `C1`'s agents (1590 of 1600 shared
  values bit-equal; the ten that differ are the cross-evaluation return in each run's own
  surrogate), so all three carry `C1`'s open L0 gate (0.559, TOST p = 0.939). Rebuilt state dumps
  reproduced 60 of 60 pools in each instance run before `resid_trace` was scored. Hidden 7: gate 0
  measured on the run's own CPU fit fails on the untrained floor (oracle 0.918 in band, mechanical
  leakage clean, floor 0.601 outside the 0.1 tolerance), so the run is off-design under section 8
  and is reported, not rescued, and no other capacity is substituted. Recorded descriptively and
  not as a verdict: survival 0.703 [0.650, 0.756] with a lower bound of 0.6496, predictor 0.703,
  untrained 0.593, survival `resid_trace` 0.717, and two pools at drift 0.45, seed 8, outside the
  reward-leak tolerance. Hidden 10 at G seed 1: gate 0 passes (oracle 0.893, equal to the recorded
  value, floor 0.485); survival 0.626 [0.598, 0.654] and `resid_trace` 0.637 [0.610, 0.664], so
  replication is not claimed under the 2026-09-26 rule; both margin clauses pass (+0.092 over the
  predictor, +0.112 over the untrained arm); the L0 gate is open, so no verdict that needs every
  gate is reached either way. Hidden 8 at G seed 2: gate 0 passes (oracle 0.939, equal to the
  recorded value, floor 0.538); the speed positive control falls under 0.75 in one pool of 60
  (untrained, drift 0.45, seed 0, 0.725) and the L0 gate is open, so the run is uninformative under
  its spec's outcome table; descriptively survival 0.659 [0.619, 0.699] leads the predictor by
  +0.031, under the 0.05 margin. Each result is reported beside its historical row, which keeps its
  label, and a margin over the predictor is read as a difference between training regimes. No bar,
  margin or tolerance changed. The post-review readouts (`docs/specs/2026-10-11-post-review-readouts-design.md`
  and its amendment) on the agents of all five reruns, and the L1 stream-decoder capacity curve,
  are recorded in FINDINGS 20 as diagnostics and change no registered verdict; those on the hidden
  7 and hidden 8 seed 2 reruns are descriptive, because those runs are off-design and
  uninformative. The draw-free
  `C1` minus `C2` contrast on balanced pairs reads +0.121 [+0.074, +0.168] on a fresh world base
  and +0.124 [+0.073, +0.175] on the registered one, both above the auxiliary rule's 0.05; the
  auxiliary rule stays NOT MET as registered, because `C1` does not meet its primary rule.

- **2026-10-13 - balanced confirmation, result.** Seeds 10 to 19 were run under
  `docs/specs/2026-10-12-balanced-confirmation-design.md`, which was committed before the
  runs. The deciding readout pairs the worlds on base 910000. Both runs are UNINFORMATIVE:
  the untrained matched-world mean is 0.614, 0.114 from 0.5, outside the frozen 0.1 floor.
  The trained-arm intervals clear the bar and the margins descriptively (`B1` 0.800
  [0.772, 0.829], `B2` 0.843 [0.825, 0.860]). FINDINGS 21. No registered threshold moved.

- **2026-10-13 - balanced sensory-echo control, result.**
  `docs/specs/2026-10-13-balanced-sensory-echo-design.md` was committed at `3d85f11` before
  any trained arm was scored under the control. On the same base-910000 pairs, regressing
  out `[x_t, x_{t-1}]` puts the untrained arm at 0.596, 0.096 from 0.5, inside the 0.1 floor.
  `B1` survival `resid_obs` is 0.761 [0.727, 0.795] and `B2` is 0.780 [0.757, 0.804]. Both
  meet the frozen rule (sensory-independent under matched worlds). The 2026-10-12 routes stay
  UNINFORMATIVE. FINDINGS 21. No registered threshold moved.

- **2026-10-12 - the capacity grid.** Section 9 registers the surrogate capacity grid
  {8, 16, 24, 32, 48, 64}. `itasorl/experiment_a_l3.py` sweeps (4, 8, 16, 32, 64), and the
  organism runs used hidden 7 and hidden 10 as well. The entries of 2026-07-10, 2026-07-14,
  2026-07-16 and 2026-09-27 record the sweeps that ran. None of them named the grid change
  itself. No recorded verdict moves. The grid in section 9 is what was registered; the
  sweeps above are what was run.

- **2026-10-16 - THE `T-touch` OUTCOME OF 2026-10-11, QUALIFIED (a later record of the
  2026-10-12 correction; the 2026-10-11 entry stays as written).** That entry reads `T-touch`
  as meeting the primary rule with every gate passing. Its leakage gate summed the homeostatic
  reward, and the pellet objective trains on pellets consumed, so the reward that run trained on
  was never gated (`docs/CORRECTIONS.md`, "2026-10-12: the pellet run's leakage gate summed the
  homeostatic reward"). Full-rule validity of `T-touch` is therefore not established. The 0.796
  [0.762, 0.830] stays as recorded, and so do the other gate values. The matched-world pellet run,
  `B2`, is UNINFORMATIVE on the untrained floor; under the sensory-echo control it reads 0.780
  [0.757, 0.804] with that rule met, under a control frozen after the raw outcomes and untrained
  readings were known; the untrained interval [0.577, 0.615] crosses 0.60 (FINDINGS 21). The
  frozen wording "survival is not required"
  is supported only in that narrower form, for the controlled decodability reading of the
  pellet-seeking, death-off agents. No registered threshold moved and no verdict is re-read.

- **2026-10-17 - L1 STATE AT A SUPRA-BAND GRID (B1), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B1).** The L1 negative is read at
  the in-band grid 0.023, and the record says the state readout was never shown to register a
  quantization seam at any grid. This readout reads the saved agents' states at the supra-band
  grid 0.06 as well: the 30 drift-0.023 agents of the corrected L1 rerun, which decide, and those
  of the historical L1 run beside them. Integrity gate: the regenerated pools at 0.023 equal the
  saved dumps bit for bit, or that run reports nothing. The labels (readout sensitive, both arms
  at chance, other) and the text each one changes are fixed in the spec; the title stays under
  every label. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-18 - L1 STATE AT A SUPRA-BAND GRID (B1), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B1; artifacts
  `artifacts/expL1/l1_coarse_grid_{corrected,historical}.json`).** Ran 2026-10-18 (UTC) at
  `75ce098`; read 2026-10-19. Integrity passes for both runs: at grid 0.023 every regenerated pool
  equals its saved dump bit for bit. Label, decided by the corrected run: (a) SENSITIVE. At grid
  0.06 the untrained arm reads 0.672 [0.640, 0.704], so the state readout registers this class of
  seam at a supra-band grid, and the L1 negative is a negative at grid 0.023, not an insensitivity
  of the readout to the class. Described beside the label, which does not read it: the survival arm
  reads 0.643 [0.619, 0.667] at 0.06, below the untrained arm (survival minus untrained -0.029
  [-0.062, +0.005]), so what registers the seam at 0.06 is the readout on an untrained trunk, not a
  survival-specific encoding. At 0.023 the reading reproduces (survival 0.551 [0.531, 0.570]). The
  historical companion decides nothing; its survival arm reads 0.638 at 0.06, and its untrained and
  predictor values equal the corrected run's (one floor read twice). No prediction was stated. The
  title stays. Changes no stored value or verdict.
  *(Note 2026-10-20: label (a) reads the readout's sensitivity only. At grid 0.06 the survival arm
  also misses the encoding criterion: under the 0.65 bar, survival minus untrained -0.029
  [-0.062, +0.005] and survival minus predictor +0.032 [-0.005, +0.069], both under the 0.05
  margin, so the criterion is not met at either grid (`docs/CORRECTIONS.md`, entry of
  2026-10-20). Changes no stored value or verdict.)*

- **2026-10-17 - WALK SPEED AGAINST GOAL DIRECTION (B2), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-walk-speed-vs-goal-readout-design.md`, the whole spec).** Whether the C1
  reading needs goal direction, rather than the speed of the survival walk or the larger law
  deviation a faster walk visits, was not tested. This readout reads C1's saved drift-0.45 agents
  on the survival walk, a speed-matched scripted walk and an open-loop replay of the survival
  actions, on fresh pools at bases 2800000 and 2850000 registered for it. Integrity gate: the base
  collision check is empty and every value is finite. The labels and text map are fixed in the
  spec. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-18 - WALK SPEED AGAINST GOAL DIRECTION (B2), RESULT (spec
  `docs/specs/2026-10-17-walk-speed-vs-goal-readout-design.md`; artifact
  `artifacts/walk_speed/corrected_l3_h8_wm.json`).** Ran 2026-10-18 (UTC) at `75ce098`. Integrity
  passes: the base collision and footprint checks pass, every agent file is present and every
  target is finite. Reading 1 (speed) is UNRESOLVED: only seed 5 of 10 was speed-matched within
  0.001 and the rule needs 7 (unmatched 0, 1, 2, 3, 4, 6, 7, 8, 9); the survival arm's own walk
  minus the speed-matched walk reads +0.104 [+0.050, +0.159], lower bound 0.0497. Reading 2 is
  CLOSED LOOP ADDS: own walk minus open-loop replay reads +0.073 [+0.051, +0.095], lower bound
  0.0512. Label: NOT SEPARATED, since GOAL DIRECTION needs speed not to suffice. The spec's stated
  expectation, that the speed match would leave several seeds unmatched and reading 1 might be
  UNRESOLVED, was borne out. The replay walk is still slower than the own walk (0.174 against
  0.185). Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-17 - REVERSE DIRECTION ON THE CORRECTED C1 DRIFT-0 AGENTS (B3), ANNOUNCED BEFORE THE
  RUN (spec `docs/specs/2026-10-17-closing-readouts-design.md`, section B3).** The design of the
  2026-10-09 reverse-direction entry above, with its spec's dated 2026-10-17 amendment, applied to
  the 30 drift-0 agents of the C1 rerun, now on this machine. Integrity gate: the drift-0 rescore
  reproduces the C1 drift-0 cell targets within 0.01 on all 30 agent-cells, or the run is void.
  The CARRIED, NOT CARRIED and AMBIGUOUS labels are descriptive on C1 and enter no rule-ledger
  row. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-18 - REVERSE DIRECTION ON THE CORRECTED C1 DRIFT-0 AGENTS (B3), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B3; artifact
  `artifacts/l0_audit/reverse_direction_corrected_l3_h8_wm.json`).** Ran 2026-10-18 (UTC) at
  `75ce098`. Integrity passes: the drift-0 rescore reproduces the recorded targets on 30 of 30
  agent-cells, worst deviation 0.0008, a cross-stack rescore within 0.01 and not bit-equal. Label:
  R1 CARRIED, descriptive on C1 with no rule-ledger row: survival 0.754 [0.721, 0.787], untrained
  0.523 [0.491, 0.555], predictor 0.572 [0.545, 0.599], margin over untrained +0.231. The stored
  balanced value, 0.787 [0.761, 0.814] against predictor 0.616 and untrained 0.540, stays beside
  it. No prediction was stated. Changes no stored value or verdict.

- **2026-10-17 - C1 POPULATION PROBE REGROUPED BY WORLD SEED (B4), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B4).** The pooled population probe
  on C1's survival and untrained agents (0.609 and 0.536) grouped its folds by pooled row. This
  rerun computes the row-grouped and world-seed-grouped values on the same tails. Integrity gate:
  both values are finite and present; the row-grouped value is compared with the stored one, and a
  difference is a platform note. The spec states before the run that under the explicit folds the
  two groupings can differ only through pair deaths, so an equal value is the expected outcome.
  Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-19 - C1 POPULATION PROBE REGROUPED BY WORLD SEED (B4), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B4; artifact
  `artifacts/population_readout/corrected_l3_h8_wm_world_grouped.json`).** Ran 2026-10-19 (UTC) at
  `75ce098` as a fresh rerun; the 2026-10-18 attempt was interrupted by the owner's stop and
  recorded no reading. Integrity passes: both values are finite and come from the same tails in one
  pass. Platform note: the local row-grouped values differ from the stored ones by +1.2e-4
  (survival) and -2.5e-4 (untrained), so the contrast is read between the two local values. Label:
  DIFFERS. Survival reads 0.609 grouped by row and 0.617 grouped by world seed (+0.008, 110 world
  groups); untrained 0.536 and 0.537 (+0.001, 108 world groups). The spec's expected outcome,
  EQUAL, was not borne out. The label is to be reported with the number of dead pairs, and the
  output did not record it; only the world-group counts survive. Claim row 49 holds on the
  world-grouped values (0.617 above 0.537 and below the per-individual 0.649). Changes no stored
  value or verdict.

- **2026-10-17 - SURPRISE AUROC WITH TWIN GROUPS (B5), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B5).** The stored surprise AUROCs of
  C1, T-touch and S-immortal split same-world twins across folds and are lower bounds. This rerun
  computes the grouped and ungrouped values in one pass on the saved agents. Tolerance gate: per
  seed within 0.01 of the stored values for four quantities, a miss being a cross-stack note; NaN
  rule: a stored NaN must come back NaN with the same reason and is excluded, a NaN on one side
  only fails that arm, and a NaN is never a pass. Post hoc, descriptive, changes no stored value
  or verdict.

- **2026-10-18 - SURPRISE AUROC WITH TWIN GROUPS (B5), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B5; artifacts
  `artifacts/goal_stakes_grouped_surprise/{C1,T-touch,S-immortal}.json`).** Ran 2026-10-18 (UTC) at
  `75ce098` on the saved agents of the three runs. The NaN rule is met on every arm of every run:
  no seed is NaN on one side only. Tolerance gate: T-touch and S-immortal reproduce their stored
  values within 0.01 on all four quantities, and their ungrouped surprise exactly (worst 0.0000);
  C1's ungrouped surprise misses 0.01 on five survival
  seeds (worst 0.051) and five untrained seeds (worst 0.035), a cross-stack note that does not void
  the comparison. Label: GROUPED ABOVE 0.65 in all three runs. Grouped surprise AUROC: C1 0.856
  [0.791, 0.922], T-touch 0.917 [0.897, 0.937], S-immortal 0.859 [0.814, 0.904]; grouping effect
  -0.0002 [-0.008, +0.007], +0.005 [-0.001, +0.010] and +0.011 [+0.007, +0.015], only the last
  excluding zero. The stored 0.846, 0.912 and 0.848 stay as recorded lower bounds. The three runs
  used `--workers 2` where the sealed command says 4; values are unaffected. No prediction was
  stated. Changes no stored value or verdict.

- **2026-10-17 - DEVELOPMENT HELD-OUT TRANSFER (B6), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B6).** The development readout's
  transfer column is in-sample. This readout scores the frozen 300-update probe of C1 and C2 on
  fresh 450-update and 300-update states at bases 2700000 and 2750000, registered for it.
  Integrity gate: the stored in-sample transfer column is reproduced within 1e-9, or the 0.785 and
  0.689 clause is dropped instead and nothing from the run is quoted. Post hoc, descriptive,
  changes no stored value or verdict.

- **2026-10-18 - DEVELOPMENT HELD-OUT TRANSFER (B6), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B6; artifact
  `artifacts/development/heldout_transfer.json`).** Ran 2026-10-18 (UTC) at `75ce098` on fresh
  pools at bases 2700000 and 2750000. Integrity passes: the stored in-sample transfer column is
  reproduced within 1e-9 (40 of 40 values for C1 and 40 of 40 for C2), so the 0.785 and 0.689
  clause stays. Label: REPORTED. The frozen 300-update probe reads the held-out 450-update states
  at 0.736 [0.680, 0.791] on C1 and 0.632 [0.594, 0.670] on C2, beside the in-sample 0.785 and
  0.689; held-out minus in-sample -0.049 [-0.068, -0.030] and -0.057 [-0.084, -0.029]. No threshold
  reads these values, and the reorganization label does not read this column. No prediction was
  stated. Changes no stored value or verdict.

- **2026-10-17 - L1 NOISE KNOCKOUT WITH FRESH NOISE PER EPISODE (B7), ANNOUNCED BEFORE THE RUN
  (spec `docs/specs/2026-10-17-closing-readouts-design.md`, section B7).** The historical L1 noise
  knockout gave every surrogate episode of a seed one noise sequence. This rerun draws fresh noise
  per episode on the 30 drift-0.023 agents of the historical L1 run. Integrity gate: the same
  agents' registered pools bit-match their dumps (the historical half of B1), and every value is
  finite. Labels: the same call as stored, or a changed call. Post hoc, descriptive, changes no
  stored value or verdict.

- **2026-10-18 - L1 NOISE KNOCKOUT WITH FRESH NOISE PER EPISODE (B7), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B7; artifact
  `artifacts/expL1/l1_h2_noise_per_episode.json`).** Ran 2026-10-18 (UTC) at `75ce098` on the 30
  drift-0.023 agents of the historical L1 run. Integrity passes: the same agents' registered pools
  bit-match their dumps (the historical half of B1), and every per-seed value is finite, n = 10 per
  arm. Label: SAME CALL. With fresh noise per episode the survival arm reads 0.508 [0.485, 0.531],
  untrained 0.513 [0.488, 0.538] and predictor 0.513 [0.488, 0.537], margin -0.005, and the call is
  H2_SUPPORTED, as stored (0.525, 0.509, 0.516). No prediction was stated; the knockout stays read
  as vacuous. Changes no stored value or verdict.

- **2026-10-17 - SWEEP C SECOND JITTER REALIZATION (B8), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B8).** The sweep C intervals rest on
  one realization of the per-episode jitter. This readout draws a second realization at the two
  points the stored contrast reads, on C1's saved agents. Integrity gate: finite values, n = 10
  per arm. The companion spec's three readings are applied to the new contrast as descriptions.
  Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-18 - SWEEP C SECOND JITTER REALIZATION (B8), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section B8; artifacts
  `artifacts/texture/ar_sweep_c_realization2/C_rho{0.0,0.99}.json`).** Ran 2026-10-17 to 2026-10-18
  (UTC) at `75ce098`: one second realization, keyed by agent seed, at rho 0.0 and 0.99. Integrity
  passes: every value finite, n = 10 per arm at both points. Reading (2), within the margin unit:
  M(0.99) minus M(0) reads -0.0002 [-0.027, +0.027], with M(0) +0.053 [+0.005, +0.101] and M(0.99)
  +0.053 [+0.011, +0.096]. Beside it: CONSISTENT, since the new mean lies inside the stored
  follow-up interval -0.017 [-0.051, +0.016]. No prediction was stated. Changes no stored value or
  reading.

- **2026-10-17 - EXPERIMENT B WITH CORRECT ACTION TIMING (C1), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section C1).** The L2 prediction-only world
  model received each action one step late. This rerun trains the small Experiment B world models
  under the legacy and the corrected alignment on this machine (no actor-critic, no saved agent).
  Gate: every output exists and the stored figures are untouched; legacy must reproduce the
  recorded row's interval. The RESTORE, NARROW and DISCLOSURE STANDS labels and the text each one
  changes are fixed in the spec. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-18 - EXPERIMENT B WITH CORRECT ACTION TIMING (C1), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section C1; artifacts
  `artifacts/expB_legacy_local_2026-10-18/` and `artifacts/expB_aligned_2026-10-18/`, JSON only).**
  Ran 2026-10-18 (UTC) at `75ce098`, both alignments retrained on this machine, three seeds each.
  Integrity passes: all ten outputs written, the stored figures untouched, and the legacy
  drift-0.45 prediction-only target, 0.517 ± 0.056, lies inside the recorded row's interval [0.429,
  0.591] (recorded 0.510 ± 0.039). Label: RESTORE. The aligned value, 0.544 ± 0.037, lies inside
  the same interval, so the L2 negative stands with correct timing. Prediction error reads
  0.599 ± 0.007 aligned against 0.596 ± 0.004 legacy (recorded 0.596 ± 0.007); the nonlinear probe
  0.459 ± 0.015 against 0.486 ± 0.034. The frozen spec printed the recorded interval as [0.430, 0.591], a
  rounding slip: its own formula, 0.510 plus or minus t(0.95, 2) x 0.039 x sqrt(3/2) / sqrt(3),
  gives [0.429, 0.591]. No label moves, since 0.517 and 0.544 lie inside either way. No prediction
  was stated. Changes no stored value or verdict.

- **2026-10-17 - PROBE-PENALTY SENSITIVITY (G01), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G01).** Every probe value on record
  uses C = 1. This readout scores the stored dumps of six runs at C in {0.001, 0.01, 0.1, 1, 10}
  and a nested in-fold C. Integrity gate: C = 1 reproduces one recorded target per states dir
  within 1e-9 (C1 within the G57 tolerance), or that dir reports nothing. No TOST or gate is
  computed at any other C. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-17 - PROBE-PENALTY SENSITIVITY (G01), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G01; artifacts
  `artifacts/probe_sensitivity/*.json`, six states dirs).** Ran 2026-10-17 (UTC) at `75ce098`,
  after G57 passed; read 2026-10-19. Integrity passes on all six dirs: at C = 1 each reproduces its
  recorded target with worst difference 0.0, the C1 rows on the regenerated dumps, labeled
  "cross-stack regenerated, max abs target delta = 0.00165". Label: PENALTY-SENSITIVE for every
  states dir, the extreme at C = 0.001 and below the C = 1 value in each. The open L0 reading of C1
  at drift 0 ranges 0.507 to 0.559 across C (nested in-fold C 0.519 [0.498, 0.540]). At C = 0.1 and
  C = 10 the survival target stays within 0.02 of its C = 1 value in every dir at both drifts; the
  sensitivity comes from C = 0.001 and 0.01. Descriptive: no TOST or gate is computed at any other
  C. No prediction was stated. Changes no stored value or verdict.

- **2026-10-17 - DRIFT-0.45 WORLD-SAMPLE READOUT ON THE C1 AGENTS (G02), ANNOUNCED BEFORE THE RUN
  (spec `docs/specs/2026-10-17-closing-readouts-design.md`, section G02).** The design of the
  2026-10-08 world-sample entry above, with its spec's dated 2026-10-17 amendment, applied to C1's
  ten drift-0.45 survival agents. Integrity gate: the registered draw reproduces the recorded C1
  targets within 0.01 on all ten seeds, or the run is void. The R1, R2 and R3 labels are computed
  as descriptions and trigger no frozen wording; the recorded 0.042 propagation stays. Post hoc,
  descriptive, changes no stored value or verdict.

- **2026-10-18 - DRIFT-0.45 WORLD-SAMPLE READOUT ON THE C1 AGENTS (G02), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G02; artifact
  `artifacts/l0_audit/d045_world_samples_corrected_l3_h8_wm.json`).** Ran 2026-10-18 (UTC) at
  `75ce098`. Integrity passes: the registered draw reproduces the recorded targets on 10 of 10
  seeds, worst deviation 0.00165. Label, computed as a description that triggers no frozen wording:
  R1 SECURE. The eight independent draws read 0.685 to 0.829, 8 of 8 at or above 0.65; the
  registered draw, 0.733, ranks 6 of 9; C1's own between-draw SD is 0.052. Propagated with that SD,
  the corrected primary reads 0.733 [0.618, 0.848], a lower bound below the 0.65 bar; the record's
  propagation with the historical 0.042 stays. No prediction was stated. Changes no stored value or
  verdict.

- **2026-10-17 - BALANCED SENSORY-ECHO CONTROL ON A SECOND DRAW (G03), ANNOUNCED BEFORE THE RUN
  (spec `docs/specs/2026-10-17-closing-readouts-design.md`, section G03).** The 2026-10-13
  sensory-echo result rests on one evaluation draw at base 910000. This readout reads the B1 and
  B2 agents on fresh pairs at base 2250000, registered for it, with integrity still checked at
  910000 as the frozen spec requires. The 910000 result is unchanged, and the frozen clauses are
  computed on the second draw as descriptions. Post hoc, descriptive, changes no stored value or
  verdict.

- **2026-10-19 - BALANCED SENSORY-ECHO CONTROL ON A SECOND DRAW (G03), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G03; artifacts
  `artifacts/balanced_confirmation/sensory_echo_b{1,2}_fresh.json`).** Ran 2026-10-19 (UTC) at
  `75ce098` on CUDA, B1 then B2. Integrity passes for both: at 910000 every cell matches its
  recorded balanced target (largest difference 0.0), no pair is dropped, the first states match,
  the 2250000 base collision check is empty and no adjudication blocker is raised. On the second
  draw the floor is explained (untrained `resid_obs` 0.577, 0.077 from 0.5, inside the 0.1 floor by
  0.023; the first draw's 0.596 was 0.096 from 0.5, inside by 0.004), and the reading is
  sensory-independent under matched worlds for B1 (0.724 [0.689, 0.759]) and for B2 (0.761 [0.746,
  0.777]). B1 and B2 share their untrained and predictor agents, so this is one floor read twice.
  The 910000 result is unchanged. No prediction was stated. Changes no stored value or verdict.

- **2026-10-17 - WITHIN-EPISODE DECODING CURVE (G04), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G04).** The registered probe on h_t
  alone at steps 1 to 24 of the registered pools, from the stored dumps of six runs. Integrity
  gate: each dir's registered target reproduces within 1e-9 (C1 within the G57 tolerance) and the
  step-24 column matches the recorded final-state component where one exists. A curve on the
  historical dumps was looked at in a scratch read before the spec, which the spec records. Post
  hoc, descriptive, changes no stored value or verdict.

- **2026-10-17 - WITHIN-EPISODE DECODING CURVE (G04), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G04; artifacts
  `artifacts/step_resolved/*.json`, six states dirs).** Ran 2026-10-17 (UTC) at `75ce098`, after a
  first attempt was interrupted; read 2026-10-19. Integrity passes on all six dirs: each registered
  target reproduces with worst difference 0.0, and the step-24 column equals the recorded final
  reading on 60 of 60 cells. Labels on the step-24 minus step-1 target at the strongest drift: the
  survival arm BUILDS WITHIN THE EPISODE in the four L3 dirs (C1 regenerated +0.272 [+0.204,
  +0.339]) and is UNRESOLVED in the two L1 dirs; the label of every dir and arm is in the
  artifacts. Post hoc: a curve on the historical dumps was seen in a scratch read before the spec.
  No prediction was stated. Changes no stored value or verdict.

- **2026-10-17 - WORLD FEATURES ON EACH ARM'S OWN EPISODES (G07), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G07(a)).** The world-feature spec's
  secondary protocol was not run. It runs now on the C1 and C2 agents, after the replay numbers
  were seen, under that spec's dated 2026-10-17 amendment. Integrity gate: finite values, with the
  per-episode features and the per-seed values per arm and feature present. The frozen readings
  are computed as descriptions; the stored artifacts are not rewritten. Post hoc, descriptive,
  changes no stored value or verdict.

- **2026-10-18 - WORLD FEATURES ON EACH ARM'S OWN EPISODES (G07), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G07(a); artifacts
  `artifacts/world_features/own_episode_posthoc/corrected_l3_h8_{wm,nowm}.json`).** Ran 2026-10-18
  (UTC) at `75ce098`, C1 then C2. Integrity passes for both: every value finite, with the
  per-episode features and the per-seed values present. Label on both runs: "keeps nothing beyond
  the senses", as on the stored replay protocol; it means that no near feature met both clauses of
  reading 1. The own-episode survival residual R squared ranges -0.0053 to +0.0018 across the two
  runs. The label reads the drift-0.45 survival arm, whose own episodes are the replay episodes, so
  it could not move under this protocol; what moves is the other arms: survival minus untrained
  r2_state rises on every instrumental feature, and I1 changes sign on both runs. Run after the
  replay numbers were seen. No prediction was stated. Changes no stored value or verdict.

- **2026-10-17 - L1 GATE-0 ORACLE UNDER BOTH PARTITIONS (G47), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G47).** Agent-free. The L1 gate-0
  oracle at grids 0.022, 0.023 and 0.024 under the legacy and the explicit fold partition.
  Integrity gate: legacy gives 0.872875 at 0.023, bit-equal to the stored calibration, and
  explicit gives 0.871125, or nothing is promoted. Post hoc, descriptive, changes no stored value
  or verdict.

- **2026-10-17 - L1 GATE-0 ORACLE UNDER BOTH PARTITIONS (G47), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G47; artifacts
  `artifacts/fold_rescore/gate0_l1_{legacy,explicit}.json`).** Agent-free, ran 2026-10-17 (UTC) at
  `75ce098`; read 2026-10-19. Integrity passes: legacy gives 0.872875 at grid 0.023, bit-equal to
  `artifacts/expA/l1_calib.json`, and explicit gives 0.871125. Label: REPRODUCED. The oracle at
  0.023 reads 0.873 on the legacy partition and 0.871 on the explicit one. No prediction was
  stated. Changes no stored value or verdict.

- **2026-10-17 - C1 STATE DUMPS REGENERATED ON THIS MACHINE (G57), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G57).** C1's state dumps are
  regenerated from the saved agents of the C1 rerun as a cross-stack replay. Integrity gate: every
  regenerated target lies within 0.002 of its recorded value; otherwise the C1 rows of the
  probe-penalty and within-episode readouts are omitted and the miss is reported. Post hoc,
  descriptive, changes no stored value or verdict.

- **2026-10-17 - C1 STATE DUMPS REGENERATED ON THIS MACHINE (G57), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section G57; report
  `artifacts/repro_check_c1_regen/c1_regen_local_stack.json`; the dumps are not shipped).** Ran
  2026-10-17 (UTC) at `75ce098`. Integrity passes: 60 dumps written and every regenerated target
  within 0.002 of its recorded value, 60 of 60, max abs target delta 0.00165; 22 of 60 equal to the
  bit. Label: WITHIN TOLERANCE, so the C1 rows of G01 and G04 are kept, labeled "cross-stack
  regenerated, max abs target delta = 0.00165". No prediction was stated. Changes no stored value
  or verdict.

- **2026-10-17 - THE DRAG COMPARATOR'S ORACLE AT EPS 6.0 (H31), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section H31).** Agent-free. The gate-0
  oracle of the hand-authored quadratic drag at eps 4, 6 and 8, eps 6 being the setting the
  comparator runs used. Integrity gate: the eps 4 and 8 oracle readings equal the stored sweep bit
  for bit, or the run is void. The reading enters no gate; the drag's gate 0 stays failed whatever
  eps 6 reads. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-17 - THE DRAG COMPARATOR'S ORACLE AT EPS 6.0 (H31), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section H31; artifact
  `artifacts/agent_free/qd_eps6_oracle.json`).** Agent-free, ran 2026-10-17 (UTC) at `75ce098`; the
  script exits 1 by design when no eps passes gate 0, and the second of two runs is the output.
  Integrity passes: the eps 4 and 8 oracle readings equal the stored sweep bit for bit (0.656 and
  0.697), and the floors agree within 0.01, so no cross-stack note is triggered. Label: (b)
  OUTSIDE. The eps 6 oracle reads 0.671, outside [0.85, 0.95]. It enters no gate, and the drag's
  gate 0 stays failed. No prediction was stated. Changes no stored value or verdict.

- **2026-10-17 - L1 NONLINEAR STATE PROBE (H53), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section H53).** An MLP and a random forest
  on the corrected L1 rerun's stored dumps, the rung whose negative has been read by the linear
  probe alone. Run only after the B1 bit check passed. Integrity gate: the linear C = 1 survival
  target at drift 0.023 reproduces 0.5506 within 1e-9, or nothing is reported. Post hoc,
  descriptive, changes no stored value or verdict.

- **2026-10-18 - L1 NONLINEAR STATE PROBE (H53), RESULT (spec
  `docs/specs/2026-10-17-closing-readouts-design.md`, section H53; artifact
  `artifacts/expL1/l1_nonlinear_corrected.json`).** Ran 2026-10-18 (UTC) at `75ce098`, after the B1
  bit check passed. Integrity passes: the linear C = 1 survival target at drift 0.023 reproduces
  0.5506 with mean absolute difference 0.0. Label: (b), both nonlinear probes under 0.65 on
  survival at drift 0.023: random forest 0.529 [0.516, 0.543] and MLP 0.534 [0.512, 0.557], against
  the linear 0.551 [0.531, 0.570]. No prediction was stated. Descriptive, changes no stored value
  or verdict.

- **2026-10-17 - NUDGE DOSE-RESPONSE (X01), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X01).** C1's survival and
  untrained agents, nudged along the decoded direction and a sham at 0.5, 1, 2 and 4 standard
  deviations, with a positive control along the actor-sensitive direction. Integrity gate: the 1 s
  real and sham scores reproduce the local B5 rerun within 0.01, arm by arm under the NaN rule, or
  nothing is reported. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-19 - NUDGE DOSE-RESPONSE (X01), RESULT: STOPPED at gate (I) (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X01; no artifact).** Ran 2026-10-18
  to 2026-10-19 (UTC) at `75ce098` and stopped at its integrity gate, as the spec requires. The
  owner decided on 2026-10-19 (00:19 UTC) to record the step as stopped, with no survival-only
  reading. Gate (I) passed on the survival arm (10 of 10 seeds, 1 s real and sham equal to the B5
  rerun) and failed on the untrained arm: seed 9 has an empty pool in the nudge run and a finite
  value in the B5 rerun, a NaN on one side only. Nothing was read or reported, and no summary was
  computed from the per-seed checkpoints. Disclosures: the positive control moved behavior in 10 of
  10 survival seeds; `scripts/run_nudge_dose_readout.py`:249-253 writes no 1 s block for an empty
  pool and counts it as a failure, so the spec's exclusion of seeds empty on both sides (0, 6 and 7
  here) is unreachable in the script, and seed 9 fails regardless of that defect; while diagnosing,
  seed 0's survival per-magnitude scores were printed to the session (one seed, raw), and no other
  seed or summary was looked at. The prediction (gate (i) passes and the label is "not used up to 4
  s") was not tested.

- **2026-10-17 - DEVELOPMENT ON A FIXED WALK (X02), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X02).** The residual controls of
  the development readout on identical input, replaying the 300-update agent's recorded episodes
  through every snapshot of C1 and C2. Integrity gates: the stored replay targets are reproduced
  within 1e-6, and the recorded observations equal the collected episodes, or nothing is reported.
  Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-19 - DEVELOPMENT ON A FIXED WALK (X02), RESULT: NOT RUN (owner decision 2026-10-18,
  before any result was seen).** The owner's final run list of 2026-10-18 (22:47 UTC) left this
  step out, and it was never launched. No artifact. Every disclosure it would have replaced stands:
  the development readout's control readings are on own episodes and have no walk-fixed
  counterpart.

- **2026-10-17 - HEAD OVERLAP (X03), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X03).** The overlap of the decoded
  direction with the actor, critic and decoder weights of C1's three arms, and the one-step action
  and recurrence sensitivities along it. Integrity gate: the fitted probe reproduces each cell
  target within 0.002, 30 of 30, or nothing is reported. Post hoc, descriptive, changes no stored
  value or verdict.

- **2026-10-19 - HEAD OVERLAP (X03), RESULT (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X03; artifact
  `artifacts/head_overlap/corrected_l3_h8_wm.json`).** Ran 2026-10-19 (UTC) at `75ce098`. Integrity
  passes: the fitted probe reproduces the cell target within 0.002 on 30 of 30 arm-cells. Label:
  undetermined. The survival arm's r_actor reads 0.905 [0.495, 1.315] and its one-step action ratio
  1.242 [0.899, 1.585] (ratios to the random expectation), so neither "policy-blind" (both upper
  bounds under 0.5) nor "policy-read" (both lower bounds above 2) holds. The prediction,
  "policy-blind" with the decoder block reading u above its expectation, was not borne out: the
  decoder state block reads 1.005 [0.944, 1.065], at its expectation (declared weak in advance). A
  one-step overlap is not a behavioral test, and X01, which would have measured that, stopped at
  its gate. Changes no stored value or verdict.

- **2026-10-17 - SURPRISE ACROSS SNAPSHOTS (X04), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X04).** C1's prediction-error
  separation of the worlds at the 0, 100, 200, 300 and 450-update snapshots on the 300-update
  agent's own streams. Integrity gate: the 300-update and zero-update rows reproduce the stored
  surprise artifact within 1e-6, or nothing is reported. Post hoc, descriptive, changes no stored
  value or verdict.

- **2026-10-19 - SURPRISE ACROSS SNAPSHOTS (X04), RESULT: NOT RUN (owner decision 2026-10-18,
  before any result was seen).** The owner's final run list of 2026-10-18 (22:47 UTC) left this
  step out, and it was never launched. No artifact. Every disclosure it would have replaced stands:
  the surprise readout on identical input (0.626 against 0.764) and the withdrawn reading of
  FINDINGS 18.3 stay as written.

- **2026-10-17 - PREQUENTIAL CODE LENGTH (X05), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X05).** The online code length of
  the registered probe over training prefixes taken in whole twin groups, on the stored dumps of
  six runs, read only as a seed-paired difference between arms. Integrity gate: the full-n AUROC
  reproduces each dir's recorded target within 1e-9 (C1 within the G57 tolerance), or that cell is
  void. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-19 - PREQUENTIAL CODE LENGTH (X05), RESULT: NOT RUN (owner decision 2026-10-18, before
  any result was seen).** The owner's final run list of 2026-10-18 (22:47 UTC) left this step out,
  and it was never launched. No artifact. Every disclosure it would have replaced stands: no
  code-length column sits beside the probe-penalty table.

- **2026-10-17 - SPARSE SUPPORT (X06), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X06).** The L1-penalized probe at
  eight penalties on the same dumps, with the smallest support that reaches 0.65. Integrity gate:
  the dense C = 1 fit reproduces each dir's recorded target within 1e-9, or that cell is void; a
  fit that did not converge is not read. Post hoc, descriptive, changes no stored value or
  verdict.

- **2026-10-19 - SPARSE SUPPORT (X06), RESULT: NOT RUN (owner decision 2026-10-18, before any
  result was seen).** The owner's final run list of 2026-10-18 (22:47 UTC) left this step out, and
  it was never launched. No artifact. Every disclosure it would have replaced stands: the salience
  reading gains no support-size number.

- **2026-10-17 - PELLET-DENSITY DOSE (X07), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X07).** The registered pooled
  probe on C1's three arms in evaluation worlds with 2 to 96 pellets. Integrity gate: the
  24-pellet row reproduces every cell target within 0.002, 30 of 30, or the run is void. Post hoc,
  descriptive, changes no stored value or verdict.

- **2026-10-19 - PELLET-DENSITY DOSE (X07), RESULT (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X07; artifact
  `artifacts/pellet_dose/corrected_l3_h8_wm.json`).** Ran 2026-10-19 (UTC) at `75ce098`. Integrity
  passes: the 24-pellet row reproduces every cell target within 0.002, 30 of 30. Label:
  locomotion-carried. At 2 pellets the survival arm reads 0.726 [0.655, 0.796], +0.197 [+0.117,
  +0.278] over the untrained arm. The prediction, that the survival reading falls toward 0.57 to
  0.64 at low density, was not borne out: the survival curve is flat at 0.72 to 0.73 from 2 to 96
  pellets. The untrained trunk replaying the survival arm's own episodes reads 0.620 to 0.660
  across densities, so the walk's input carries much of the reading. Every row carries "gate-0
  oracle not recalibrated at this density". The run used `--workers 2` where the sealed command
  says 4; values are unaffected. No verdict; changes no stored value.

- **2026-10-17 - BUDGET-CURVE DYNAMICS (X08), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X08).** Agent-free, on the
  committed budget curve and cells of C1 and C2. Disclosed as post hoc: its numbers were seen in a
  scratch read before the rule was written. Integrity gate: the inputs equal the committed
  artifact and cells bit for bit. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-17 - BUDGET-CURVE DYNAMICS (X08), RESULT (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X08; artifact
  `artifacts/budget_curve_dynamics/corrected_runs.json`).** Agent-free, ran 2026-10-17 (UTC) at
  `75ce098`; read 2026-10-19. Integrity passes: the input arrays equal the committed budget curve
  and cells bit for bit. Labels: C1 "tracks budget" (target per 100 updates +0.064 [+0.047,
  +0.080], on return +0.090 [-0.026, +0.195]); C2 "both" (on return +0.177 [+0.001, +0.316], whose
  lower bound is +0.0005 to four decimals; per 100 updates +0.021 [+0.002, +0.049]); the C2
  100-to-200 segment label "decodability moves without return" is NOT met (target +0.037 [-0.006,
  +0.079]). The prediction that the segment label is not met was borne out. The slope intervals are
  seed-bootstrap 90 percent intervals. Post hoc: its numbers were seen in a scratch read before the
  rule was written. Changes no stored value or verdict.

- **2026-10-17 - BAYES FACTORS WITH AN INTERVAL NULL (X09), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X09).** Agent-free, on committed
  per-seed arrays. Disclosed as post hoc: the values were computed in a scratch read during script
  development, and the spec keeps the predictions written before it. Integrity gate: every input
  equals its committed artifact bit for bit. The result annotates and never relabels, and no label
  enters a claims row. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-17 - BAYES FACTORS WITH AN INTERVAL NULL (X09), RESULT (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X09; artifact
  `artifacts/bayes_factors/interval_null.json`).** Agent-free, ran 2026-10-17 (UTC) at `75ce098`;
  read 2026-10-19. Integrity passes: every input equals its committed artifact. The spec reports the
  labels only in the manuscript's appendix table; here each reading is given at the 0.707 scale, to 3
  significant digits: (1) the C1 L0 registered pair, BF01 = 0.133 (annotation, 0.707 scale); (2) the
  eight independent L0 pairs, signed BF01 = 224 (annotation, 0.707 scale) and folded BF01 = 3.73
  (annotation, 0.707 scale); (3) B1 drift-0 survival BF01 = 0.747 (annotation, 0.707 scale) and B2
  BF01 = 4.99 (annotation, 0.707 scale); (4) the sweep C stored contrast BF01 = 89.1 (annotation,
  0.707 scale) and the follow-up contrast BF01 = 38.4 (annotation, 0.707 scale); (5) corrected L1
  survival minus untrained BF01 = 4.13 (annotation, 0.707 scale) and minus predictor BF01 = 21.2
  (annotation, 0.707 scale). Prediction (1) was borne out; predictions (2) to (5) were not, in any of
  their eight rows. The predictions predate a scratch read during script development that saw several
  of these values, which the spec discloses. The factors annotate and never relabel, and none enters
  a claims row. Changes no stored value or verdict.

- **2026-10-17 - FULL-RANK ERASURE (X10), ANNOUNCED BEFORE THE RUN (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X10).** The rank-two erasure of
  the C1 survival agents extended until an in-fold refit probe reads 0.55 or less, or the rank
  reaches 16, with an equal-rank sham, under the erasure spec's dated 2026-10-17 pre-run
  amendment. Integrity gates: the rank-two values reproduce the stored erasure artifact within
  0.01, or nothing is reported; the refit reaches the stop in at least 8 of 10 seeds, or only k is
  reported. Post hoc, descriptive, changes no stored value or verdict.

- **2026-10-18 - FULL-RANK ERASURE (X10), RESULT: VOID (gate E) (spec
  `docs/specs/2026-10-17-extension-readouts-design.md`, section X10; no artifact).** Ran 2026-10-18
  (UTC) at `75ce098` with `--max-rank 16 --refit-stop 0.55 --swap`. Gate (I) passed: the rank-two
  values reproduced the stored erasure artifact within 0.01. Gate E failed, so no use reading was
  made: the in-fold refit reached the 0.55 stop by rank 16 in 1 of 10 seeds (seed 5, at rank 2; the
  rule needs 8); in the other 9 it stayed above 0.55 up to the rank cap of 16. Nothing was
  promoted; the per-seed values live in the run output
  (`fullruns/closing_2026-10/x10/corrected_l3_h8_wm.json`), not in a committed artifact. The
  prediction (gate E passes with k between 4 and 10) was not borne out. The stored rank-two reading
  stands as recorded.

- **2026-10-19 - TIMING OF THE EXECUTED SURROGATE AND OF THE GATE-0 HANDICAP, STATED IN ONE PLACE
  (a record; the entries above stay as written and no rule, number or verdict changes).** This
  registration was committed at `0217263` (2026-07-10, 10:23 UTC). The MLP
  `itasorl.surrogate_l3.train_g_motion` that every L3 result used, in place of the section 9
  recurrent predictor (2026-10-06 entry above), was first committed in `489a772` (14:00 UTC the
  same day), after this registration and before the first organism run. Section 9 names hidden
  size as the only free knob and no detector-side handicap. Gate 0 was frozen at sigma_d = 0.05,
  hidden 8, oracle 0.890 in `9e64ffa` (16:32 UTC). The first organism run (n = 3) was recorded in
  `ced27f4` (21:23 UTC), then traced to the world-parameters bug and retracted, and gate 0 was
  re-frozen at sigma_d = 0.02, oracle 0.928, in `6150a15` (22:42 UTC), before the corrected
  organism run. The sigma_d that every later L3 result uses was therefore frozen after an
  organism result had been read. Separately, the X07 result entry above says "the walk's input
  carries much of the reading"; it is read as "the reading depends on the walk's input", since
  the input reference is a separate reading and not a share. The manuscript's section 4.3
  tabulates these dates; `docs/CORRECTIONS.md`, 2026-10-19.

- **2026-10-19 - THE B5 SURPRISE VALUES CALLED LOWER BOUNDS (a record; the entries above stay as
  written and no rule, number or verdict changes).** The B5 entries above (the 2026-10-17
  announcement and its 2026-10-18 result) say the stored surprise AUROCs "are lower bounds" and
  "stay as recorded lower bounds"; read both as "are values from a biased estimator". A bias in
  expectation does not make each stored value a numerical bound, and `C1`'s local grouped minus
  ungrouped, -0.0002 [-0.008, +0.007], includes zero. The stored values and the GROUPED ABOVE
  0.65 label are unchanged. `docs/CORRECTIONS.md`, 2026-10-19.

- **2026-10-19 - THE 2026-10-06 SURROGATE ENTRY IS A DEVIATION, NOT A DESCRIPTION CORRECTION (a
  record; the entries above stay as written and no rule, number or verdict changes).** The
  2026-10-06 entry headed "DESCRIPTION CORRECTION: THE SURROGATE AS IMPLEMENTED" is reclassified
  as a deviation from this registration. Section 9 registered a recurrent predictor trained with
  held-out early stopping; every L3 result used the feed-forward MLP `train_g_motion` trained for
  a fixed 300 epochs, first committed after this registration (2026-10-19 timing entry above).
  The registered configuration is therefore not the one executed, and no L3 rule was read on the
  registered surrogate. The heading stays as written. The manuscript revision of 2026-10-19
  (full-review pass) calls it a deviation; `docs/CORRECTIONS.md`, 2026-10-19.

- **2026-10-19 - THE MATCHED-WORLD SENSORY-CONTROL LABEL AND THE STAKES WORDING, READ AS
  CONDITIONAL RESULTS (a record; the entries above stay as written and no rule, number or verdict
  changes).** The 2026-10-13 result entry and the 2026-10-19 G03 result entry record `B1` and
  `B2` as "sensory-independent under matched worlds", the frozen label of
  `docs/specs/2026-10-13-balanced-sensory-echo-design.md`. That procedure was frozen after the
  raw `B1` and `B2` outcomes and the untrained diagnostics were known, so the label names a
  conditional result under it: not a confirmation under the original complete rule, on which
  both runs stay UNINFORMATIVE, and not independence from nonlinear or longer-history sensory
  information. The second draw at base 2250000 was read under that already fixed procedure and
  is not a prospective test of it. The frozen wording "survival is not required", quoted in the
  2026-10-07 and 2026-10-11 entries, stands only in the narrower form the 2026-10-16 entry gives
  it, with this conditional status. The manuscript revision of 2026-10-19 (full-review pass)
  words both so; `docs/CORRECTIONS.md`, 2026-10-19.

- **2026-10-20 - THE CONSTANT-DRAG DROP, THE COHERENCE CONTRAST, THE INTERVAL-NULL FACTORS AND
  THE HIDDEN-7 LOWER BOUND, READ IN THEIR TESTED SCOPE (a record; the entries above and the frozen
  specs stay as written and no rule, number or verdict changes).** The 2026-07-16 cross-recipe
  entry says a detectable drag-coefficient bias is felt so grossly "that any recurrent state
  separates the worlds" and that "coefficient-bias artifacts cannot be made subtle-but-detectable
  in world P". Read the first as holding for the tested untrained trunks (floors 0.859 and 0.913)
  and the second as a calibration that was unsuccessful under the tested settings (the `G_cd`
  family, its eps grid, the detector handicap of gate 0 and the 0.1 floor rule); neither is a
  general claim about coefficient-bias artifacts. The B8 text of
  `docs/specs/2026-10-17-closing-readouts-design.md` says the claim that coherence did not raise
  the margin stands under readings (2) and (3); read that claim as "the tested contrasts did not
  demonstrate an increase". The frozen sweep C contrast is +0.009 [-0.027, +0.045] and the
  follow-up -0.017 [-0.051, +0.016], and the follow-up does not establish equivalence. The X09
  result entry's factors are approximate plug-in sensitivity calculations: the raw-scale null
  region goes to the standardized scale as c = h / s with the observed standard deviation, so they
  are not exact Bayes factors for the fixed raw-scale band. The 2026-10-11 hidden-7 entry gives
  its lower bound as 0.6496, under the bar; generated tables now print that digit. The manuscript
  revision for the second blind review words these so; `docs/CORRECTIONS.md`, 2026-10-20.

- **2026-10-20 - THE BEHAVIOR CONTROL'S UNTRAINED CHECK AND THE PELLET-DENSITY CURVE, READ AS
  OBSERVED (third blind review; a record; the entries above stay as written and no rule, number
  or verdict changes).** The 2026-07-13 per-timestep-control entry says the untrained agent's
  resid_trace is 0.498 "(exact chance)" and that "the control neither manufactures nor spares
  signal"; the 2026-07-14 hidden-7 entry repeats "0.498 = exact chance". Read 0.498 as near
  chance, and the second phrase as "under this control an untrained trunk keeps little residual
  signal": a diagnostic relative to the basis and the regressor the control uses, which does not
  show that the control removes every behavior-related contribution in the trained agents. The
  2026-10-19 X07 result entry says "the survival curve is flat at 0.72 to 0.73 from 2 to 96
  pellets". Read it as the observed means, not as invariance: paired by seed, the survival
  reading at 2 pellets minus 24 pellets is -0.007 [-0.022, +0.008] (t-based 90%, n = 10), and
  every other density's paired 90% interval against 24 lies within plus or minus 0.05, recomputed from
  the per-seed rows of `artifacts/pellet_dose/corrected_l3_h8_wm.json`; the reading changes
  little across the tested evaluation densities. The manuscript revision for the third blind
  review words these so; `docs/CORRECTIONS.md`, 2026-10-20, third blind review.

## 13. How to run (milestones, in order)

1. **Build + calibrate the surrogate.** Train `G` on authentic rollouts; wrap as a World;
   run the Experiment-A oracle; ascend capacity until oracle AUROC in [0.85, 0.95]; FREEZE
   and record in section 12. (CPU-friendly; no GPU sweep.)
2. **Gate check.** Only if gate 0 passes, proceed. Otherwise re-tune `G`.
3. **Organism run** (GPU, mirrors B-v3), dumping states for offline re-probing:

       python scripts/run_expB2.py --drift-mode l3 --dump-states runs/l3/states
       python scripts/run_e2e.py --b2-drift-mode l3 --b2-dump-states runs/l3/states
       # capacity ceiling reference:
       python scripts/run_expB2.py --drift-mode l3 --sysid-aux
       # power extension (n=10) via the seeds 0..9 profile

   (The `l3` drift-mode and the `G` World are new code, gated behind this preregistration;
   no multi-hour organism sweep before milestone 1's oracle gate passes.)
