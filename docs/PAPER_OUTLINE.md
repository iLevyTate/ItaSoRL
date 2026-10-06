# Writeup outline and claims inventory

*Structured summary of the project's results. Every claim must trace to a row in
the claims inventory below; every row traces to a committed artifact. Claim decisions
for the 2026-10 revision (which claims stay primary, which become secondary or
exploratory, which need a corrected run) are in `docs/REVISION_2026-10.md`; which
implementation produced each number is in `docs/RESULTS_MANIFEST.md`.*

## Structure

1. **Introduction** (from ITASORL.md Part I)
   - The question, as one measurement: under a stated training protocol, is the
     condition (authentic or surrogate dynamics) decodable from the agent's recurrent
     state, beyond named baselines and controls, set against how detectable the flaw
     is to a privileged detector at a stated detector-side noise.
   - The white space: evaluation-awareness work imports the concept of a test;
     probing work reads facts inside one world; mismatch detectors are built to
     detect; meta-RL trains across conditions, so inferring the condition is
     instrumental. Here each agent trains in one condition and nothing names it.
   - Contributions: (a) the detectability-versus-decodability gap as a measurable
     object, with detectors scored at one shared handicap; (b) hand-authored flaws
     (L1, L2) that a detector finds and that did not meet the registered encoding
     criterion; (c) a learned-dynamics flaw (L3) that met it for survival plus
     next-observation prediction at the registered budget, bounded by that budget,
     with behavior and sensory controls that name their basis; (d) a measurement
     apparatus with its failure modes documented: fold-partition dependence,
     world-sample conditioning of the L0 gate, and a trainer defect found and
     corrected (`docs/CORRECTIONS.md`).
2. **The world and the surrogate ladder** (from ITASORL_world_spec.md; FINDINGS 1,
   methods note 10)
   - "A Patch of Earth" v0 as implemented: no weather or PDE fields, a closed unit
     square with clipping walls, constant drag in world P (so the authentic law is
     linear), basal drain 0.02 per step; ~146-dim observation in which interoception
     carries velocity and every vision ray carries a radial velocity.
   - Ladder L0-L4; difficulty calibrated by a detector band, not by fiat. The L3
     surrogate as implemented (FINDINGS 10.1.1): an MLP velocity law trained for a
     fixed 300 epochs, a linear-fit control, held-out one-step error against rollout
     divergence, a privileged detector score at detector-side σ = 0.02, and an
     agent-accessible detector on the agent's own observations.
3. **Methods** (from PREREGISTRATION*.md; FINDINGS 2, 9, 10.1, 11)
   - Experiment A detectors, matched pairs, leakage-audit battery.
   - Experiment B probes; three arms on an identical trunk that differ in objective,
     training policy, and data at once (`docs/METHODS_ARMS.md`); pooled and
     matched-pair readouts; the fold partition frozen and versioned
     (`docs/specs/2026-10-06-primary-analysis-and-l0.md`).
   - Trainer: A2C with GAE; the truncation bootstrap reads the successor-state value
     (corrected 2026-10-06); historical runs used the pre-transition value.
   - Gates: detector band, engagement, L0 equivalence (TOST; the bootstrap ROPE
     check is a percentile-interval check, not a Bayesian analysis), leakage.
   - Prospective protocols, dated amendments, exploration, and corrections kept
     separate; analysis tiers in FINDINGS methods note 9.
4. **Results**
   - 4.1 Detectability ceilings (FINDINGS 2, 10.1, 15): L1 1.000, L2 0.993 with
     nearly noiseless inputs; at the L3 gate's σ = 0.02, L2 0.646 and 0.618 against
     L3 0.928. Privileged detector scores, not agent capabilities.
   - 4.2 Hand-authored flaws (FINDINGS 3, 4, 9, 14.7): the prediction-only state stays
     near chance across channels, horizons, and probes at L2; survival coupling 0.610
     [0.585, 0.634]; capacity ceiling 0.596; L1 organism 0.533 at the in-band grid.
     Each "did not meet the registered encoding criterion".
   - 4.3 The L3 primary result (FINDINGS 10, 17): the corrected ten-seed run of
     survival with the next-observation auxiliary at 300 updates, against the
     predictor and untrained arms, with its gates under the frozen partition. The
     historical value 0.752 [0.698, 0.807] is reported beside it, labeled with the
     pre-correction trainer.
   - 4.4 The auxiliary contrast, bounded by budget (FINDINGS 10.8, 10.8.1, 17): the
     corrected decoder-on minus decoder-off difference on one device with its
     seed-paired interval, and the budget curve to 450 updates. Historical: 0.601
     without the auxiliary against 0.730 with it on the same CPU; 0.717 without it at
     450 updates with a failed skill match, so no mediation inference either way.
   - 4.5 Controls, named by basis (FINDINGS 10.4, 10.4.1, 10.4.2, 17): what remains
     after per-timestep linear residualization on speed, energy, food, and drag
     (historical 0.726), with position and heading (0.723), and after regressing out
     the current and previous observation (0.731); the held-out nuisance R² of each
     control and the readouts under scripted and replayed policies on the corrected
     agents.
   - 4.6 Secondary and exploratory (historical trainer unless stated): second capacity
     (10.5), held-out and cross-recipe transfer (10.6, 10.7), second and third
     fingerprint instances (10.9); the common garden, where prefix condition remains
     decodable after restoring authentic dynamics (0.666 forward, 0.684 reverse;
     FINDINGS 10.6.1), with the controlled persistence test on the corrected agents
     (17); the graded seam (14) and the texture knockout read as non-transfer of the
     original direction (14.5), with the fresh-probe comparators on the corrected
     agents (17); observation localization (14.6).
   - 4.7 Experiment C (FINDINGS 13.D, 13.E): selection did not raise world
     decodability under the pooled population readout (contrast -0.002 [-0.013,
     +0.009]); the readout is not a per-individual test. Out of the abstract.
5. **Limitations** (FINDINGS 11; `docs/CORRECTIONS.md`)
   - One surrogate per run and one set of evaluation worlds: intervals over agent
     seeds condition on both. L0 depends on the world sample (`artifacts/l0_audit/`).
   - The arms differ in objective, training policy, and data at once.
   - Controls remove what their basis and model can express.
6. **Related work** (ITASORL.md Part I literature sections, incl. the 2026-07
   axis-by-axis cross-reference: evaluation awareness behavioral and probe-based
   (arXiv:2309.00667, 2407.04694, 2505.23836, 2507.01786, 2509.13333),
   incidental-encoding probes (AtariARI arXiv:1906.08226; Othello-GPT
   arXiv:2210.13382), agent-side mismatch detection (GalilAI arXiv:2110.15489;
   RAPT arXiv:2602.01515), latent-context inference and meta-RL (RL²
   arXiv:1611.02779, Mikulik et al. arXiv:2010.11223, VariBAD arXiv:1910.08348, PEARL
   arXiv:1903.08254, RMA arXiv:2107.04034), and necessity-of-world-models theory
   (Richens et al. arXiv:2506.01622). State the novelty claim exactly as worded in
   ITASORL.md's novelty posture; list the preempted overclaim phrasings as what we do
   *not* claim.)
7. **Reproducibility statement** (`docs/REPRODUCE.md`: tables regenerated from
   committed artifacts without training, retraining recipes, frozen surrogates, the
   anonymized supplement; FINDINGS 12; `docs/CORRECTIONS.md`; CITATION.cff)

## Claims inventory

| # | Claim | Status | Number | Doc section | Artifact | Figure |
|---|-------|--------|--------|-------------|----------|--------|
| 1 | L0 control at chance (oracle) | agent-free | AUROC 0.523 | FINDINGS 2.1 | artifacts/expA/summary.json | expA_ceiling.png |
| 2 | L1 detectable (oracle) | agent-free | AUROC 1.000 | FINDINGS 2.1 | artifacts/expA/summary.json | expA_ceiling.png |
| 3 | L2 detectable (oracle) | agent-free | AUROC 0.993 | FINDINGS 2.2 | artifacts/expA/summary.json | expA_L2_ceiling.png |
| 4 | L2 did not meet the registered encoding criterion (recurrent state) | no actor-critic | 0.510 ± 0.039 @ drift 0.45 | FINDINGS 3.1 | artifacts/expB/summary.json | expB_incidental.png |
| 5 | L2 surprise channel weak | no actor-critic | 0.596 ± 0.007 @ drift 0.45 | FINDINGS 3.2 | artifacts/expB/summary.json | expB_channels.png |
| 6 | No liftoff with horizon | no actor-critic | 0.48-0.51 across 0/8/16 | FINDINGS 3.3 | artifacts/expB/summary.json (recorded rerun; see 3.3 correction) | expB_kstep.png |
| 7 | L2 null holds under a nonlinear probe | no actor-critic | 0.482 ± 0.031 | FINDINGS 3.4 | artifacts/expB/summary.json | - |
| 8 | Survival coupling does not rescue L2 | historical | 0.523 ± 0.045 (replication) | FINDINGS 9 | artifacts/expB2/expB2_results.json | expB2_survival.png |
| 9 | B-v3 regime negative at scale | historical | 0.610, 90% CI [0.585, 0.634], n=10 | FINDINGS 7.1 | artifacts/expB2/bv3_n10_summary.json | - |
| 10 | L2 capacity ceiling below bar | historical | 0.596, 90% CI [0.577, 0.616], n=10 | FINDINGS 7.1 | artifacts/expB2/sysid_ceiling_n10_summary.json | - |
| 11 | L3 gate frozen in-band | agent-free | oracle 0.928, floor 0.483 | FINDINGS 10.1 | PREREGISTRATION_L3.md sec. 12 | - |
| 12 | L3 decodable from survival-trained agents carrying the next-observation auxiliary at 300 updates (neither objective alone reached the bar at that budget; FINDINGS 10.8) | historical; primary, replaced by C1 | 0.752, t 90% CI [0.698, 0.807], 8/10 seeds | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 13 | L3 predictor baseline near chance | historical; primary, replaced by C1 | 0.573 [0.546, 0.599] | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 14 | L3 untrained floor at chance | historical; primary, replaced by C1 | 0.488 [0.461, 0.514] | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 15 | Reward leak clean | historical; primary, replaced by C1 | 0.541, clean 10/10 | FINDINGS 10.3 | PREREGISTRATION_L3.md sec. 12 (n=10 audited entry) | - |
| 16 | No survivorship asymmetry | historical; primary, replaced by C1 | 0 early deaths, 110/110 all pools | FINDINGS 10.3 | PREREGISTRATION_L3.md sec. 12 | - |
| 17 | Behavior alone decodes world | historical; secondary | trace 0.803 [0.763, 0.840] | FINDINGS 10.4 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 18 | Remains after per-timestep linear behavior control (four-channel; position/heading added in 10.4.1) | historical; secondary | 0.726 [0.685, 0.765], 9/10; quad 0.721; position/heading-controlled 0.723 [0.682, 0.760], 8/10, quad 0.700 | FINDINGS 10.4, 10.4.1 | artifacts/expB2/behavior_audit_l3_h8_traces.json; artifacts/expB2/behavior_audit_l3_covar_n10.json | - |
| 19 | Control neither manufactures nor spares signal | historical; secondary | untrained resid 0.498; predictor 0.574 | FINDINGS 10.4 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 20 | Second capacity: the residual after the behavior control replicates | historical; secondary | 0.722 [0.678, 0.763], 8/10; quad 0.704 | FINDINGS 10.5 | artifacts/expB2/behavior_audit_l3_h7_traces.json | - |
| 21 | Second capacity: dissociation NOT met (artifact-conditional) | historical; secondary | survival 0.737 [0.688, 0.780] vs predictor 0.714 [0.687, 0.740]; lead +0.023 < +0.05 | FINDINGS 10.5 | artifacts/expB2/behavior_audit_l3_h7_traces.json | - |
| 22 | Gate 0 re-validated per capacity; hidden=7 frozen | agent-free gate; secondary | oracle 0.922, floor 0.566; hidden=8 regression exact (0.928/0.482); hidden=4 uninformative | FINDINGS 10.5 | PREREGISTRATION_L3.md sec. 12 + scripts/run_expA_l3.py | - |
| 23 | Held-out capacity-variant transfer passes (same recipe/data; robustness within one recipe, 10.6 scope note) | historical; secondary | survival 0.773 [0.722, 0.824], 9/10 vs untrained floor 0.569; rule passes | FINDINGS 10.6 | artifacts/expB2/heldout_l3_h8_summary.json | - |
| 24 | Common-garden control, re-scored: prefix condition remains decodable after restoring authentic dynamics (frozen rule passes both directions; memory and physical footprint not separated) | historical; secondary | survival cg_tail 0.666 forward / 0.684 reverse (>= 0.65 and > untrained + 0.05); late tail 0.586/0.577 decays below bar | FINDINGS 10.6.1 | artifacts/expB2/heldout_l3_h8_cg_rescore.json | - |
| 25 | Reverse transfer direction-dependent: coarse-to-subtle FAILS | historical; secondary | survival 0.638, frozen rule fails | FINDINGS 10.6 | artifacts/expB2/heldout_l3_h7_reverse_summary.json | - |
| 26 | Cross-recipe transfer: the direction reads a random-Fourier-features law fit on the same data (rule passes; interval lower bound 0.654) | historical; secondary | survival 0.684, t 90% CI [0.654, 0.715], 7/10 vs untrained 0.548; rule passes, margin +0.034 | FINDINGS 10.7 | artifacts/l3_crossrecipe/summary.json | - |
| 27 | Exp C: the milestone-3 pilot null was INVALIDATED (13.C); the fixed-code re-run is a validated null under the pooled population readout, which is not a per-individual test (13.E) | no actor-critic; exploratory | contrast -0.002, t 90% CI [-0.013, +0.009]; final treatment AUROC 0.509 | FINDINGS 13.C, 13.D | artifacts/expC/emergence_pilot_summary.json (holds the re-run, `git_commit_at_run` a0cb850) | - |
| 28 | Sensory-echo control passes at hidden 8 | historical; secondary | `resid_obs` 0.731 [0.690, 0.772], 8/10 vs untrained 0.534; joint sensory-plus-behavior 0.670 | FINDINGS 10.4.2 | artifacts/expB2/sensory_echo_l3_h8.json | - |
| 29 | At 300 updates the survival arm without the auxiliary did not meet the bar | historical; primary, replaced by C1 and C2 | survival without it 0.601 [0.549, 0.654], 1/10; predictor 0.589 | FINDINGS 10.8 | artifacts/expB2/arch_baseline_l3_h8_nowm.json | - |
| 30 | Device control: the CPU is not the cause | historical; primary, replaced by C1 and C2 | survival with the auxiliary on the same CPU sandbox 0.730 [0.668, 0.791], 8/10; paired lead over 0.601 +0.128 [+0.089, +0.168] | FINDINGS 10.8 | artifacts/expB2/device_control_l3_h8_wm_cpu.json | - |
| 31 | Second fingerprint instance: survival-specific, below the bar | historical; secondary | survival 0.639 [0.610, 0.669], 5/10 vs predictor 0.534, untrained 0.514 | FINDINGS 10.9 | artifacts/expB2/second_instance_l3_h10_gseed1.json | - |
| 32 | H2 graded seam: the signal collapses to chance with the seam | historical; exploratory | 0.752 at alpha 1 to 0.506 at alpha 0 (ROPE accept), Spearman rho 1.0 | FINDINGS 14 | artifacts/expH2/summary.json | - |
| 33 | H2 texture knockout: the original direction does not transfer to matched-band jitter (no fresh probe, no comparator-trained agents) | historical; exploratory | survival 0.539 vs untrained 0.542 (hidden 8); 0.510 vs 0.520 (hidden 7) | FINDINGS 14.5 | artifacts/expH2/texture_knockout_h8.json, texture_knockout_h7.json | - |
| 34 | Observation-channel localization (hidden 8) | historical; exploratory | vision masked 0.686, interoception masked 0.756, all masked 0.500 | FINDINGS 14.6 | artifacts/expH2/obs_localization_h8.json, obs_localization_h7.json | - |
| 35 | L1 organism negative at the in-band grid | historical | survival 0.533 [0.509, 0.556], 0/10; oracle 0.873 | FINDINGS 14.7 | artifacts/expL1/organism_summary.json | - |
| 36 | Matched-handicap oracle: L2 weaker than L3 at the L3 gate noise | agent-free | L2 ar1 0.646, regime 0.618 vs L3 0.928 at sigma 0.02; L2 1.000 at sigma 2e-4 | FINDINGS 15 | artifacts/expA/l2_inconfig_oracle.json | - |
| 37 | Engagement margin sweep on committed cells | historical gate audit | 80/80 cells pass at margins 0.05 to 0.15; 79/80 from 0.20 to 0.30 | FINDINGS 11, note 2 | artifacts/expB2/bv3_n10_gates.json; artifacts/expB2/engagement_margin_cloud_runs.json | - |
| 38 | No-auxiliary survival arm at 450 updates: above the bar, skill match failed, so no mediation inference | historical; secondary | 0.717 [0.679, 0.755], 8/10; return -0.156 against the window [-0.269, -0.169] | FINDINGS 10.8.1 | artifacts/expB2/skill_matched_l3_h8_nowm_u450.json | - |
| 39 | Agent-accessible detector on the agent's own observations reads the L3 surrogate | agent-free | AUROC 0.991 | FINDINGS 10.1.1 | artifacts/surrogate_diagnostics.json | - |
| 40 | Hand-authored quadratic-drag comparator has no in-band gate-0 value | agent-free | detector 0.601 to 0.768 across eps 0.5 to 16; floor leaves tolerance from eps 4 | texture spec amendment | artifacts/texture/gate0_qd.json | - |
| 41 | Corrected primary run: survival with the auxiliary, 300 updates, n = 10 | corrected; primary | pending | FINDINGS 17 | artifacts/expB2/corrected_l3_h8_wm.json | - |
| 42 | Corrected auxiliary contrast on one device, and the budget curve to 450 updates | corrected; primary (contrast), exploratory (curve) | pending | FINDINGS 17 | artifacts/expB2/corrected_l3_h8_nowm.json; artifacts/budget_curve.json | budget_curve.png |

Status: *agent-free* and *no actor-critic* results are untouched by the 2026-10
trainer correction; *historical* results came from the pre-correction survival trainer
and keep that label until a corrected run replaces them; tiers follow FINDINGS methods
note 9.

## Known gaps

- Rows 38 to 42 added 2026-10-06 (revision): the 450-update arm, the two agent-free
  detector results of revision steps 9 and 10, and the corrected runs, filled when they
  land.
- Rows 28 to 37 added 2026-09-28 for the sections that resolved after July
  (10.4.2, 10.8, 10.9, 11 note 2, 14 to 14.7, 15). Row 12 is narrowed by 10.8, and
  row 27 now records the 13.D re-run. The published GPU L3 cells are not committed,
  so row 37 does not cover them.

- Claims 1-7 resolved (2026-07-16): the recorded 06302026 e2e bundle's step
  metrics (plus the 2026-07-13 k-step rerun and the across-seed stds recovered
  from the bundle log) are promoted to `artifacts/expA/summary.json` and
  `artifacts/expB/summary.json` by `scripts/promote_ab_summaries.py`.
- Claims 9-10 resolved (2026-07-16): per-seed pooled targets promoted to
  `artifacts/expB2/bv3_n10_summary.json` and
  `artifacts/expB2/sysid_ceiling_n10_summary.json` by the same script;
  `scripts/audit_stats_recheck.py` re-verifies every inventory number against
  the committed artifacts.
- Rows 23-24 resolved (spec 2026-07-14, per-seed summary committed as
  `artifacts/expB2/heldout_l3_h8_summary.json`): transfer generalizes, and the
  re-scored common-garden control passes the frozen rule on both directions
  (row 24; forward 0.666, reverse 0.684; prefix condition remains decodable after
  restoring authentic dynamics, FINDINGS §10.6.1);
  the writeup reports both as generality checks, not as a new headline. Rows 20-22 resolved 2026-07-14: the second
  capacity replicates the residual after the behavior control and bounds the
  survival-specific claim to the subtler artifact.
