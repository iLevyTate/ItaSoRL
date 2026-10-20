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
     (L1, L2) that a detector finds and that did not meet the encoding criterion
     registered for L2 (applied to L1 after its run); (c) a learned-dynamics flaw (L3) whose decodability clauses are met for
     survival plus next-observation prediction at the registered budget on the corrected
     primary run, on the one evaluation-world draw all seeds share, while the registered rule is NOT MET as registered (L0 open on the
     registered world pair, 0.559), bounded by that budget,
     where the signal rides on the foraging trajectories and is kept by a trunk shaped
     by prediction, and where a hand-written drag that depends on the agent's own velocity
     is read the same way, with no mechanism claimed (FINDINGS 22.3); (d) a measurement
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
     nearly noiseless inputs; at the L3 gate's σ = 0.02, L2 0.646 and 0.618 on the first noise
     draw (0.604 and 0.503 averaged over eight) against L3 0.928. Privileged detector scores, not agent capabilities.
   - 4.2 Hand-authored flaws (FINDINGS 3, 4, 9, 14.7, 22.1): the prediction-only state stays
     near chance across channels, horizons, and probes at L2; survival coupling 0.610
     (seed bootstrap [0.585, 0.634]; t 90% CI [0.581, 0.639]); supervised reference 0.596 (a reference for what the pooled readout
     can express under direct supervision, not an architectural ceiling); L1 organism 0.551 [0.531, 0.570], 0/10, at the in-band grid on
     the corrected trainer and the stack-independent partition (FINDINGS 22.1, row 35a),
     0.533 as first recorded on the original trainer and the legacy partition (row 35).
     *(Note 2026-10-16: the L2 prediction-only world model received each action one step
     late, pairing each observation with the action that had already produced it, so that arm
     was a weaker predictor than designed. Its prediction-error and open-loop channels are the
     most exposed, and no stored number changes (`docs/CORRECTIONS.md`, 2026-10-16).)*
     *(Note 2026-10-19: retrained on 2026-10-18 under both timings on one machine, three seeds each,
     the drift-0.45 prediction-only state target reads 0.544 ± 0.037 with correct timing and
     0.517 ± 0.056 with the original timing, both inside the recorded row's interval [0.429, 0.591],
     so the label is RESTORE: the negative stands with correct timing. The surprise channel reads
     0.599 ± 0.007 aligned and 0.596 ± 0.004 legacy (`artifacts/expB_aligned_2026-10-18/`,
     `artifacts/expB_legacy_local_2026-10-18/`; rows 4 to 7).)*
     The L2 runs "did not meet the registered encoding criterion"; the L1 organism run did not
     meet that criterion, which was registered for L2 and applied to L1 after its run
     (`docs/CORRECTIONS.md`, 2026-10-16). Experiment B
     predates the first preregistration, so its 0.65 bar is applied after the fact.
   - 4.3 The L3 primary result (FINDINGS 17.2, 17.5): corrected survival with the
     next-observation auxiliary at 300 updates reads 0.733 [0.669, 0.797] (on the one shared
     evaluation-world draw; carried through C1's between-draw spread the bar clause's lower
     bound would be 0.618, FINDINGS 17.5.1 note) against predictor 0.589 and untrained 0.523. The registered rule is NOT MET, because L0 is
     open on the registered world pair (0.559); the decodability clauses are met, and
     across eight independent pairs L0 averages 0.493 (TOST over pairs p 0.022). The
     historical 0.752 [0.698, 0.807] is reported beside it.
     *(Note 2026-10-12: which half of the registered feature [mean h, h_T] carries the 0.733 is
     measured (FINDINGS 20.15, R205). Mean h alone reads 0.692 and h_T alone 0.716, so the
     time-averaged state alone does not carry the reading and the final state alone reads at
     least as high on the seed means; the pre-stated label is BOTH, which means no uniquely dominant block.
     Both halves are individually decodable. The final state's increment from the
     concatenation includes zero.)*
   - 4.4 The auxiliary contrast, bounded by budget (FINDINGS 10.8, 10.8.1, 17): the
     corrected decoder-on minus decoder-off difference, +0.120 [+0.065, +0.174] (0.733
     against 0.613; the registered auxiliary rule is not met because C1 does not meet its
     primary rule, and the lead compares the tested regimes, not general necessity), and the
     budget curve: without the decoder 0.685 at 450 updates;
     at 450 the decoder arm reads 0.821; the return difference, -0.054 [-0.198, +0.090], is
     too wide to show equal foraging. Historical: 0.601 against
     0.730 on the same CPU; 0.717 at 450 updates with a failed skill match.
   - 4.5 Who drives the episodes (FINDINGS 17.6, 17.10): at matched evaluation input a
     prediction-trained trunk reads the world within about 0.01 of the survival trunk, and
     the trained agents (trunk plus its own normalizer) read about 0.10 above the untrained
     one; the signal rides on the foraging trajectories and is kept by a trunk shaped by
     prediction.
     This is the section that withdraws "survival-specific".
     *(Note 2026-10-11: the split of that 0.10 between trunk and normalizer is measured since
     (FINDINGS 20.5, 20.10). On fresh worlds, with every trunk reading identical normalized
     input, the survival trunk's replay margin over the untrained trunk stays above zero: +0.059
     on the survival agent's normalizer and +0.092 on one refit to the replayed streams, against
     +0.076 on each trunk's own. The pre-stated reading is TRUNK on the `C1` agents and SPLIT on
     the `C2` agents; it is TRUNK on all three instance reruns as well (20.11, 20.13, 20.14). The
     figures above stay as recorded.)*
   - 4.5b Controls, named by basis (FINDINGS 10.4, 10.4.1, 10.4.2, 17.8): what remains
     after per-timestep linear residualization on speed, energy, food, and drag
     (historical 0.726; corrected 0.723 with position and heading), after regressing
     out the current and previous observation (historical 0.731; corrected 0.699), and
     after the best-fitting nonlinear control over observations, actions, and behavior
     (corrected 0.620, held-out R² 0.89, under the bar); a decoder of the observation
     stream alone reads 0.724.
   - 4.6 Secondary and exploratory (historical trainer unless stated): second capacity
     (10.5), held-out and cross-recipe transfer (10.6, 10.7), second and third
     fingerprint instances (10.9); the common garden, where prefix condition remains
     decodable after restoring authentic dynamics (0.666 forward, 0.684 reverse;
     FINDINGS 10.6.1), carried mostly by the physical footprint on the corrected agents
     (17.7: reset hidden 0.662, replay 0.561); the graded seam (14); the texture
     comparators (17.9): a hand-authored coherent drag is read through the original
     direction (0.721) and by a fresh probe (0.754) against an untrained floor of 0.612
     that sits outside the 0.1 tolerance (margins +0.109 and +0.142), white jitter is not,
     so nothing is specific to learned dynamics; observation localization (14.6).
   - 4.7 Experiment C (FINDINGS 13.D, 13.E, 13.F): the contrast is flat under the
     pooled population readout (-0.002 [-0.013, +0.009]) and under the per-individual
     readout (no individual at 0.65), and the registered verdict is uninformative
     because the speed positive control (gate 5) fails and the preregistration interprets
     no contrast unless every gate passes; gate 1 fails only against the margin of an
     unratified draft amendment (13.F, second note of 2026-10-10). On known-decodable
     agents the pooled readout
     reads 0.609 where individuals average 0.649 (17.11); regrouped by world seed (2026-10-19) it
     reads 0.617, untrained 0.537, still between the untrained arm and the individuals (row 49).
     Out of the abstract.
5. **Limitations** (FINDINGS 11; `docs/CORRECTIONS.md`)
   - One surrogate per run and one set of evaluation worlds: intervals over agent
     seeds condition on both. L0 depends on the world sample (`artifacts/l0_audit/`).
     One instance of the learned surrogate (hidden 8, G seed 0) serves C1, C2 and every
     other corrected run against it, the matched-world and death-off runs included; the
     three corrected instance reruns (FINDINGS 20.2) are the exception.
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
| 4 | L2 prediction-only state below the 0.65 bar (recurrent state; Experiment B predates the first preregistration, so the bar is applied after the fact; the world model received each action one step late, `docs/CORRECTIONS.md` 2026-10-16; with correct timing it reads the same, RESTORE) | no actor-critic | 0.510 ± 0.039 @ drift 0.45; rerun 2026-10-18 on one machine: aligned 0.544 ± 0.037, legacy 0.517 ± 0.056, both inside the recorded interval [0.429, 0.591] | FINDINGS 3.1 | artifacts/expB/summary.json; artifacts/expB_aligned_2026-10-18/expB_full.json, artifacts/expB_legacy_local_2026-10-18/expB_full.json | expB_incidental.png |
| 5 | L2 surprise channel weak (most exposed to the one-step-late action timing, `docs/CORRECTIONS.md` 2026-10-16; with correct timing it reads the same) | no actor-critic | 0.596 ± 0.007 @ drift 0.45; rerun 2026-10-18: aligned 0.599 ± 0.007, legacy 0.596 ± 0.004 | FINDINGS 3.2 | artifacts/expB/summary.json; artifacts/expB_aligned_2026-10-18/expB_surprise.json, artifacts/expB_legacy_local_2026-10-18/expB_surprise.json | expB_channels.png |
| 6 | No liftoff with horizon (the open-loop rollout shares the one-step-late action timing, `docs/CORRECTIONS.md` 2026-10-16; with correct timing it reads the same) | no actor-critic | 0.48-0.51 across 0/8/16; rerun 2026-10-18: aligned 0.45 to 0.50 across 0/8/16, legacy 0.47 to 0.51 | FINDINGS 3.3 | artifacts/expB/summary.json (recorded rerun; see 3.3 correction); artifacts/expB_aligned_2026-10-18/expB_kstep.json, artifacts/expB_legacy_local_2026-10-18/expB_kstep.json | expB_kstep.png |
| 7 | L2 null holds under a nonlinear probe (same world model and action timing as row 4; with correct timing it reads the same) | no actor-critic | 0.482 ± 0.031; rerun 2026-10-18: aligned 0.459 ± 0.015, legacy 0.486 ± 0.034 | FINDINGS 3.4 | artifacts/expB/summary.json; artifacts/expB_aligned_2026-10-18/expB_nonlinear.json, artifacts/expB_legacy_local_2026-10-18/expB_nonlinear.json | - |
| 8 | Survival coupling does not rescue L2 | historical | 0.523 ± 0.045 (replication) | FINDINGS 9 | artifacts/expB2/expB2_results.json | expB2_survival.png |
| 9 | B-v3 regime negative at scale | historical | 0.610, seed bootstrap 90% [0.585, 0.634] (t 90% CI [0.581, 0.639]), n=10 | FINDINGS 7.1 | artifacts/expB2/bv3_n10_summary.json | - |
| 10 | L2 supervised reference below bar (a reference for what the pooled readout can express under direct supervision, not an architectural ceiling) | historical | 0.596, seed bootstrap 90% [0.577, 0.616] (t 90% CI [0.573, 0.619]), n=10 | FINDINGS 7.1 | artifacts/expB2/sysid_ceiling_n10_summary.json | - |
| 11 | L3 gate frozen in-band | agent-free | oracle 0.928, floor 0.483 (the seed-0 floor; the three-seed mean is 0.482, row 22) | FINDINGS 10.1 | PREREGISTRATION_L3.md sec. 12 | - |
| 12 | L3 decodable from survival-trained agents carrying the next-observation auxiliary at 300 updates (neither objective alone reached the bar at that budget; FINDINGS 10.8) | historical; primary, replaced by C1 | 0.752, t 90% CI [0.698, 0.807], 8/10 seeds | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 13 | L3 predictor baseline near chance | historical; primary, replaced by C1 | 0.573, seed bootstrap [0.546, 0.599] (t 90% CI [0.542, 0.604]) | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 14 | L3 untrained floor at chance | historical; primary, replaced by C1 | 0.488, seed bootstrap [0.461, 0.514] (t 90% CI [0.457, 0.518]) | FINDINGS 10.2 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 15 | Reward leak clean | historical; primary, replaced by C1 | 0.541 (seeds 0.517 to 0.559), clean 10/10 | FINDINGS 10.3 | PREREGISTRATION_L3.md sec. 12 (n=10 audited entry); artifacts/expB2/heldout_l3_h8_summary.json (standard pools of the held-out run, which reproduce this run) | - |
| 16 | No survivorship asymmetry | historical; primary, replaced by C1 | 0 early deaths, 110/110 all pools | FINDINGS 10.3 | PREREGISTRATION_L3.md sec. 12; artifacts/expB2/heldout_l3_h8_summary.json (standard pools) | - |
| 17 | Behavior alone decodes world | historical; secondary | trace 0.803, seed bootstrap [0.763, 0.840] (t 90% CI [0.758, 0.848]) | FINDINGS 10.4 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 18 | Remains after per-timestep linear behavior control (four-channel; position/heading added in 10.4.1) | historical; secondary | 0.726, seed bootstrap [0.685, 0.765] (t 90% CI [0.679, 0.772]), 9/10; quad 0.721; position/heading-controlled 0.723, seed bootstrap [0.682, 0.760] (t 90% CI [0.676, 0.769]), 8/10, quad 0.700 | FINDINGS 10.4, 10.4.1 | artifacts/expB2/behavior_audit_l3_h8_traces.json; artifacts/expB2/behavior_audit_l3_covar_n10.json | - |
| 19 | Control leaves little untrained residual (diagnostic, basis-relative) | historical; secondary | untrained resid 0.498; predictor 0.574 | FINDINGS 10.4 | artifacts/expB2/behavior_audit_l3_h8_traces.json | - |
| 20 | Second capacity: the residual after the behavior control replicates | historical; secondary | 0.722, seed bootstrap [0.678, 0.763] (t 90% CI [0.672, 0.773]), 8/10; quad 0.704 | FINDINGS 10.5 | artifacts/expB2/behavior_audit_l3_h7_traces.json | - |
| 21 | Second capacity: dissociation NOT met (artifact-conditional) | historical; secondary | survival 0.737, seed bootstrap [0.688, 0.780] (t 90% CI [0.682, 0.791]) vs predictor 0.714, seed bootstrap [0.687, 0.740] (t 90% CI [0.683, 0.745]); lead +0.023 < +0.05 | FINDINGS 10.5 | artifacts/expB2/behavior_audit_l3_h7_traces.json | - |
| 22 | Gate 0 re-validated per capacity; hidden=7 frozen | agent-free gate; secondary | oracle 0.922, floor 0.566; hidden=8 regression within its tolerances (oracle 0.928; floor 0.482, the three-seed mean, against the frozen seed-0 0.483); hidden=4 uninformative | FINDINGS 10.5 | PREREGISTRATION_L3.md sec. 12 + scripts/run_expA_l3.py; artifacts/fold_rescore/gate0_h8h7_legacy.json | - |
| 23 | Held-out capacity-variant transfer passes (same recipe/data; robustness within one recipe, 10.6 scope note) | historical; secondary | survival 0.773 [0.722, 0.824], 9/10 vs untrained floor 0.569; rule passes | FINDINGS 10.6 | artifacts/expB2/heldout_l3_h8_summary.json | - |
| 24 | Common-garden control, re-scored: prefix condition remains decodable after restoring authentic dynamics (frozen rule passes both directions; memory and physical footprint not separated) | historical; secondary | survival cg_tail 0.666 forward / 0.684 reverse (>= 0.65 and > untrained + 0.05); late tail 0.586/0.577 decays below bar | FINDINGS 10.6.1 | artifacts/expB2/heldout_l3_h8_cg_rescore.json | - |
| 25 | Reverse transfer direction-dependent: coarse-to-subtle FAILS | historical; secondary | survival 0.638, frozen rule fails | FINDINGS 10.6 | artifacts/expB2/heldout_l3_h7_reverse_summary.json | - |
| 26 | Cross-recipe transfer: the direction reads a random-Fourier-features law fit on the same data (rule passes; interval lower bound 0.654) | historical; secondary | survival 0.684, t 90% CI [0.654, 0.715], 7/10 vs untrained 0.548; rule passes, margin +0.034 | FINDINGS 10.7 | artifacts/l3_crossrecipe/summary.json | - |
| 27 | Exp C: the milestone-3 pilot null was INVALIDATED (13.C); the fixed-code re-run and the per-individual rerun both have a flat contrast, and the registered verdict is uninformative because the speed positive control (gate 5) fails and no contrast is interpreted unless every gate passes; gate 1 fails only against an unratified draft margin (13.F) | no actor-critic; exploratory | re-run (`a0cb850`, legacy partition, no gate battery): contrast -0.002, t 90% CI [-0.013, +0.009], final treatment AUROC 0.509; per-individual rerun (`06ecc10`, explicit partition, full gate battery): contrast -0.002 [-0.013, +0.009], final treatment AUROC 0.510, per-individual final treatment means 0.534 to 0.539, no individual at 0.65 | FINDINGS 13.C, 13.D, 13.F | artifacts/expC/emergence_pilot_summary.json (holds the re-run, `git_commit_at_run` a0cb850); artifacts/expC/emergence_pilot_per_individual_summary.json | - |
| 28 | Sensory-echo control passes at hidden 8 | historical; secondary | `resid_obs` 0.731 [0.690, 0.772], 8/10 vs untrained 0.534; joint sensory-plus-behavior 0.670 | FINDINGS 10.4.2 | artifacts/expB2/sensory_echo_l3_h8.json | - |
| 29 | At 300 updates the survival arm without the auxiliary did not meet the bar | historical; primary, replaced by C1 and C2 | survival without it 0.601 [0.549, 0.654], 1/10; predictor 0.589 | FINDINGS 10.8 | artifacts/expB2/arch_baseline_l3_h8_nowm.json | - |
| 30 | Device control: the CPU is not the cause | historical; primary, replaced by C1 and C2 | survival with the auxiliary on the same CPU sandbox 0.730 [0.668, 0.791], 8/10; paired lead over 0.601 +0.128 [+0.089, +0.168] | FINDINGS 10.8 | artifacts/expB2/device_control_l3_h8_wm_cpu.json | - |
| 31 | Second fingerprint instance: survival above predictor, below the bar | historical; secondary | survival 0.639 [0.610, 0.669], 5/10 vs predictor 0.534, untrained 0.514 | FINDINGS 10.9 | artifacts/expB2/second_instance_l3_h10_gseed1.json | - |
| 32 | H2 graded seam: the signal collapses to chance with the seam | historical; exploratory | 0.752 at alpha 1 to 0.506 at alpha 0 (TOST p 0.005; ROPE accept), Spearman rho 1.0; the specificity reading (flat untrained floor) holds at hidden 8 only | FINDINGS 14 | artifacts/expH2/summary.json | - |
| 33 | H2 texture knockout: the original direction does not transfer to matched-band jitter (no fresh probe, no comparator-trained agents) | historical; exploratory | survival 0.539 vs untrained 0.542 (hidden 8); 0.510 vs 0.520 (hidden 7) | FINDINGS 14.5 | artifacts/expH2/texture_knockout_h8.json, texture_knockout_h7.json | - |
| 34 | Observation-channel localization (hidden 8) | historical; exploratory | vision masked 0.686, interoception masked 0.756, all masked 0.500 (by construction: constant input, positive controls also 0.500); a masked channel enters as -mean/sd | FINDINGS 14.6 | artifacts/expH2/obs_localization_h8.json, obs_localization_h7.json | - |
| 35 | L1 organism negative at the in-band grid | historical; replaced by L1-CORRECTED (row 35a, FINDINGS 22.1) | survival 0.533, seed bootstrap [0.509, 0.556] (t 90% CI [0.506, 0.561]), 0/10; oracle 0.873 | FINDINGS 14.7 | artifacts/expL1/organism_summary.json | - |
| 35a | L1 organism rerun on the corrected trainer: the negative reproduces | corrected; primary | survival 0.551 [0.531, 0.570], 0/10 (stack-independent partition); the historical run scored the same way 0.552; drift-zero 0.548 [0.532, 0.565]; seam minus drift zero, paired by seed, +0.002 [-0.011, +0.016] (descriptive) | FINDINGS 22.1 | artifacts/expB2/corrected_L1-corrected.json | - |
| 35b | L1 state at a supra-band grid: the readout registers the seam, so the negative is specific to the in-band grid | corrected; descriptive | at Δ = 0.06 untrained 0.672 [0.640, 0.704], survival 0.643 [0.619, 0.667], predictor 0.611; survival minus untrained -0.029 [-0.062, +0.005]; (a) SENSITIVE | FINDINGS 22.1 | artifacts/expL1/l1_coarse_grid_corrected.json | - |
| 36 | Matched-handicap oracle: L2 weaker than L3 at the L3 gate noise | agent-free | L2 ar1 0.646, regime 0.618 on noise draw 0 (eight-draw means 0.604, 0.503) vs L3 0.928 at sigma 0.02; L2 1.000 at sigma 2e-4 | FINDINGS 15 | artifacts/expA/l2_inconfig_oracle.json | - |
| 37 | Engagement margin sweep on committed cells | historical gate audit | 80/80 cells pass at margins 0.05 to 0.15; 79/80 from 0.20 to 0.30; the 200 GPU-published and L1 cells (2026-10-10 sweep, inputs recorded) pass 10 of 10 per run at every margin from 0.05 to 0.30 except the two hidden-7 runs (9 of 10 at 0.30, drift 0.45) and the uninformative hidden-4 run (at drift 0.45: 9 of 10 at 0.05, 5 of 10 at 0.10, 3 of 10 at 0.15, 0 of 10 from 0.20) | FINDINGS 11, note 2 | artifacts/expB2/bv3_n10_gates.json; artifacts/expB2/engagement_margin_cloud_runs.json; artifacts/engagement_margin_gpu_runs.json | - |
| 38 | No-auxiliary survival arm at 450 updates: above the bar, skill match failed, so no mediation inference | historical; secondary | 0.717 [0.679, 0.755], 8/10; return -0.156 against the window [-0.269, -0.169] | FINDINGS 10.8.1 | artifacts/expB2/skill_matched_l3_h8_nowm_u450.json | - |
| 39 | Agent-accessible detector on the agent's own observations reads the L3 surrogate | agent-free | AUROC 0.991 | FINDINGS 10.1.1 | artifacts/surrogate_diagnostics.json | - |
| 40 | Hand-authored quadratic-drag comparator has no in-band gate-0 value | agent-free | detector 0.601 to 0.768 across eps 0.5 to 16; floor leaves tolerance from eps 8 (0.560 at eps 4 is inside the 0.1 tolerance) | texture spec amendment | artifacts/texture/gate0_qd.json | - |
| 41 | Corrected primary run: survival with the auxiliary, 300 updates, n = 10; NOT MET as registered (L0 open on the registered pair), decodability clauses met | corrected; primary | 0.733 [0.669, 0.797], 8/10; predictor 0.589, untrained 0.523; margins +0.144 [+0.072, +0.217], +0.210 [+0.135, +0.286]; L0 0.559 (TOST p 0.939) | FINDINGS 17.2 | artifacts/expB2/corrected_l3_h8_wm.json; artifacts/corrected_verdicts.json | - |
| 42 | Corrected auxiliary contrast on one device, and the budget curve to 450 updates | corrected; primary (contrast), exploratory (curve) | without the auxiliary 0.613 [0.552, 0.675], NOT MET; C1 minus C2 +0.120 [+0.065, +0.174] (paired by seed index; unpaired lower bound 0.035); at 450 updates 0.821 with and 0.685 without, return difference -0.054 [-0.198, +0.090] | FINDINGS 17.4, 17.10 | artifacts/expB2/corrected_l3_h8_nowm.json; artifacts/budget_curve.json | budget_curve.png |
| 43 | L0 across independent world-sample pairs; balanced readout | corrected; secondary | eight independent pairs: signed means 0.406 to 0.557, average 0.493 (sd 0.049), TOST over pairs p 0.022 (signed means, a diagnostic not named in the frozen spec; folded about 0.5, p = 0.129), gate accepts at 2 of 8; balanced survival readout 0.767 [0.718, 0.816] | FINDINGS 17.5 | artifacts/l0_audit/corrected_l3_h8_wm.json | - |
| 44 | At matched evaluation input a prediction-trained trunk reads the world within about 0.01 of the survival trunk, and trained agents read about 0.10 above the untrained one | corrected; secondary | replay: survival 0.733, predictor 0.721, exposure-matched predictor 0.738; survival minus predictor +0.012 [-0.002, +0.026], minus predictor_logged -0.005 [-0.038, +0.027], minus untrained +0.097 [+0.066, +0.128]; predictor_logged minus untrained +0.102 [+0.067, +0.137]; each trunk reads the replay through its own normalizer; scripted policy 0.55 to 0.58 for every arm; post hoc TOST at ±0.05 accepts survival against predictor (p 0.0004) and against predictor_logged (p 0.0164). *(Note 2026-10-11: the trunk and normalizer split is measured since, FINDINGS 20.5 and 20.10: TRUNK on the `C1` agents, SPLIT on the `C2` agents; the figures in this row stay as recorded.)* | FINDINGS 17.6 | artifacts/policy_controls/corrected_l3_h8_wm.json | - |
| 45 | Retention under identical input not shown; the common garden reads the physical footprint | corrected; exploratory | replay 0.561 [0.545, 0.577]; reset hidden 0.662 [0.638, 0.687]; factorial physical 0.658, hidden 0.569 | FINDINGS 17.7 | artifacts/persistence/corrected_l3_h8_wm.json | - |
| 46 | The best-fitting control of the battery leaves a signal under the bar at the mean; the input stream alone carries the world | corrected; secondary | MLP joint control 0.620 [0.584, 0.656] (held-out R^2 0.89); sequence GRU on observations and previous actions 0.724; summary features of the observation trace alone 0.715; post hoc paired state minus readout, t 90% CI and TOST p against a borrowed 0.05: sequence GRU +0.009 [-0.011, +0.029] (p 0.002), summary features +0.018 [-0.004, +0.040] (p 0.013), MLP control residual +0.113 [+0.065, +0.161] (p 0.980) | FINDINGS 17.8 | artifacts/control_diagnostics/corrected_l3_h8_wm.json | - |
| 47 | A hand-authored coherent drag is read; white jitter is not | corrected; exploratory | qd transfer 0.721, fresh 0.754 (+0.109 and +0.142 over untrained, whose floor of 0.612 sits outside the 0.1 tolerance); gn transfer 0.527, fresh 0.488 | FINDINGS 17.9 | artifacts/texture/corrected_l3_h8_wm_{gn,qd}.json | - |
| 48 | Decoder effect split between trunk and trajectories (post hoc) | corrected; exploratory | trunk effect +0.056 and +0.080; stream effect +0.039 and +0.064 | FINDINGS 17.10 | artifacts/cross_replay/corrected_c1_c2.json | - |
| 49 | The pooled evolutionary readout is attenuated, not blind | corrected; exploratory | pooled 0.609 (untrained 0.536) against per-individual 0.649 [0.612, 0.686] on known-decodable agents (96-unit trunks; the evolved agents have 8 units); own-world assignment minus the better single policy -0.056 [-0.092, -0.020], best of the two policies per world +0.003 (zero in 9 of 10 seeds). Regrouped by world seed (rerun 2026-10-19): survival 0.617, untrained 0.537, label DIFFERS (the expected EQUAL was not borne out); the claim holds on the world-grouped values, 0.617 above 0.537 and below 0.649. The output records 110 and 108 world groups but not the number of dead pairs | FINDINGS 17.11 | artifacts/population_readout/corrected_l3_h8_wm.json; artifacts/population_readout/corrected_l3_h8_wm_world_grouped.json | - |
| 50 | Drift-0.45 world-sample sensitivity: the level is not a product of the registered draw | historical agents; diagnostic | registered 0.774 (rank 5 of 9); eight independent draws 0.728 to 0.860, sd 0.042, t 90% CI over draws [0.752, 0.808]; 8 of 8 at or above the bar; verdict SECURE under the frozen rule; integrity gate passed; on the C1 agents (post hoc) eight draws 0.685 to 0.829, sd 0.052, registered draw rank 6 of 9 | FINDINGS 17.5.1 | artifacts/l0_audit/d045_world_samples_l3_h8_heldout.json; artifacts/l0_audit/d045_world_samples_corrected_l3_h8_wm.json | - |
| 51 | Reverse direction: an authentic-raised agent reads a surrogate it never lived in | historical agents; secondary | survival 0.760 [0.716, 0.804], 9/10; predictor 0.570, untrained 0.505; margin over untrained +0.255; balanced 0.796; integrity gate passed on 30 cells; on the corrected C1 drift-0 agents the registered readout (rerun 2026-10-18) reads 0.754 [0.721, 0.787] against untrained 0.523 [0.491, 0.555] and predictor 0.572, margin over untrained +0.231 [+0.183, +0.278], R1 CARRIED, descriptive on C1 (integrity: 30 cells, worst deviation 0.0008); the balanced value kept beside it is 0.787 against untrained 0.540 (FINDINGS 20.7) | FINDINGS 17.5.2, 20.7 | artifacts/l0_audit/reverse_direction_l3_h8_heldout.json; artifacts/l0_audit/reverse_direction_corrected_l3_h8_wm.json | - |
| 52 | White-jitter comparator-trained run: UNINFORMATIVE, supports no claim | corrected trainer; set aside | survival 0.522 [0.502, 0.542], 0/10; untrained 0.552; speed positive control 0.686 at its worst against 0.75, 9 of 60 pools short; ran with an 8-unit trunk, not the registered 96 | FINDINGS 14.5.1 | artifacts/texture/T_gn_l3_h8_wm.json | - |
| 53 | L1 stream carries the seam while the state does not | historical agents; exploratory | summary-feature stream 0.795 [0.760, 0.829], supervised GRU 0.590 [0.547, 0.633], state 0.552 [0.530, 0.574] (explicit partition; 0.533 legacy, row 35); untrained stream 0.804 | FINDINGS 14.7.1 | artifacts/control_diagnostics/l1_stream_readout.json | - |
| 54 | The four calibrated fingerprint instances read together: the untrained margin reproduces on all four, the predictor margin on two | historical; secondary, descriptive | survival pooled targets as published 0.752 (hidden 8, G seed 0), 0.737 (hidden 7), 0.676 (hidden 8, G seed 2), 0.639 (hidden 10, G seed 1, CPU); descriptive mean 0.701; margin over the untrained arm +0.265, +0.151, +0.126, +0.126; over the predictor +0.179, +0.023, +0.049, +0.105, so above 0.05 on two | FINDINGS 10.2, 10.5, 10.9 | artifacts/expB2/behavior_audit_l3_h8_traces.json; artifacts/expB2/behavior_audit_l3_h7_traces.json; artifacts/expB2/second_seed_l3_h8_gseed2_gpu.json; artifacts/expB2/second_instance_l3_h10_gseed1.json | - |

Status: *agent-free* and *no actor-critic* results are untouched by the 2026-10
trainer correction; *historical* results came from the pre-correction survival trainer
and keep that label until a corrected run replaces them; tiers follow FINDINGS methods
note 9. Brackets are t-based 90% intervals over seeds unless the row names another estimator;
"seed bootstrap" marks the percentile seed bootstrap an artifact stores, and "±" in rows 4 to 8
is the standard deviation over seeds.

## Known gaps

- 2026-10-19 (full review 1): the manuscript attaches the shared evaluation-world draw to 0.733,
  the tested-regime and general-necessity qualifiers to the auxiliary lead, the drag's
  untrained floor and the jitter-draw qualifier to the texture readings, the later-procedure
  status to the matched-world sensory result, and the use-test limits to the nudge and
  erasure; no row's number or status changes. `docs/CORRECTIONS.md`, 2026-10-19.
- 2026-10-19 (review response 1): the manuscript's section 4.3 gains a registration timeline
  (the tuning run before the first registration, the engagement margin set after the first
  confirmatory run had been read, the executed MLP surrogate first committed after the L3
  registration that describes a recurrent one, the gate-0 handicap sigma_d frozen at 0.05 and
  re-frozen at 0.02 after a retracted organism run had been read, the historical sensory
  control's rule committed with its result); its Appendix D.1 writes the survival reward, the
  normalizations, the nudge score and the erasure score as equations; L2 wording is "below the
  criterion"; the trajectory attribution is dependence plus a trained-trunk advantage with an
  order-dependent split; in-sample and biased estimates are no longer called bounds; the 0.596
  supervised control (section 4.2 item and row 10) is called a reference for what the pooled
  readout can express under direct supervision, not a lower-bound reference (FINDINGS 7.1,
  2026-10-19 note). No row's number or status changes. `docs/CORRECTIONS.md`, 2026-10-19.
- 2026-10-19: rows 4 to 7 gain the Experiment B rerun with correct action timing (RESTORE) beside
  the stored values, and section 4.2's 2026-10-16 note no longer says the effect is unmeasured;
  row 49 and section 4.7 gain the world-grouped pooled readings (DIFFERS; the claim holds); row 51
  gains the registered readout on the corrected C1 drift-0 agents (R1 CARRIED). Rows 9, 10, 13,
  14, 17, 18, 20, 21 and 35 quoted the seed-bootstrap interval the artifact stores without saying
  so; each now names it and gives the t-based interval the manuscript's appendix uses beside it,
  and the Status paragraph says how brackets are read. No stored value or reading changes.
- 2026-10-16: this inventory was last revised row by row on 2026-10-14. Where a row here and
  the manuscript's claims appendix differ, the appendix is current. This pass rewords
  contribution (c) so it leads with the registered outcome (NOT MET as registered, L0 open),
  as section 4.3 and row 41 already did; adds row 35a for the corrected L1 rerun
  (`L1-CORRECTED`, FINDINGS 22.1), which replaces row 35's run, as the appendix numbers it;
  adds the L2 action-timing note to section 4.2 and rows 4 to 7 (`docs/CORRECTIONS.md`,
  2026-10-16); gives the drag's untrained floor beside its readings in section 4.6 and row 47;
  and names the drag by its dependence on the agent's velocity in contribution (c), since
  FINDINGS 22.3 does not show coherence to be the property read.
- Rows 38 to 49 added 2026-10-06 (revision): the 450-update arm, the two agent-free
  detector results of revision steps 9 and 10, the corrected runs, and the readouts on
  the corrected agents (FINDINGS 17).
- Rows 27, 40 and 46 and section 4.7 revised 2026-10-10: row 27 and 4.7 give the 13.F
  verdict (flat contrast, registered verdict uninformative), row 40 says where the
  quadratic-drag floor leaves tolerance as the gate-0 artifact records it (eps 8), and
  row 46 carries the post hoc paired contrasts of FINDINGS 17.8. The audit reads those
  three rows.
- Later on 2026-10-10: row 27 and section 4.7 name gate 5 as the gate that decides the
  Experiment C routing (gate 1 fails only against the margin of an unratified draft) and
  attribute each number to its run; rows 10, 22, 32, 34, 43, 44 and 49 match the FINDINGS
  notes of the same date and the manuscript's wording; row 44 gives the paired contrasts of
  FINDINGS 17.6 and is read by the audit. Rows 50 to 53 add the drift-0.45 world samples, the
  reverse direction, the white-jitter comparator-trained run and the L1 stream readout, with
  the numbers the manuscript's inventory gives the same claims, so rows 1 to 26 and 38 to 53
  of both lists name the same claims.
- Also on 2026-10-10, after the verdict labels were changed to lead with the registered
  outcome: section 4.3 and row 41 say that `C1` does not meet its registered rule because L0
  is open and that the decodability clauses are met (`artifacts/corrected_verdicts.json`).
  Row 4 and section 4.2 say that Experiment B predates the first preregistration, so its
  0.65 bar is applied after the fact. Row 54 records the four calibrated fingerprint
  instances read together (rows 12, 21 and 31 and the hidden 8 new-seed run of FINDINGS 10.9),
  a descriptive summary the manuscript gives.
- Also on 2026-10-10: row 49 names its payoff figure for what it is, the own-world assignment
  minus the better single policy, beside the best-of-two value (FINDINGS 13.E and 17.11 notes);
  rows 15 and 16 cite the held-out run's summary, whose standard pools reproduce the run; row
  22 cites the gate-0 regression artifact, and row 11 says its floor is the seed-0 value.
  Row 46 says the 0.724 sequence decoder reads observations and previous actions, and gives
  the 0.715 of the observation trace alone (FINDINGS 17.8).
- 2026-10-12: section 4.3 carries a dated note on R205, which half of the probe feature carries
  the primary reading (FINDINGS 20.15).
- 2026-10-11: section 4.5 and row 44 carry a dated note that the replay split between trunk
  and normalizer is now measured (FINDINGS 20.5, 20.10). The row's recorded figures are
  unchanged.
- Rows 28 to 37 added 2026-09-28 for the sections that resolved after July
  (10.4.2, 10.8, 10.9, 11 note 2, 14 to 14.7, 15). Row 12 is narrowed by 10.8, and
  row 27 now records the 13.D re-run. The published GPU L3 cells are not committed;
  since 2026-10-19 row 37 covers them through the 2026-10-10 sweep, which records each cell's
  engagement inputs (`artifacts/engagement_margin_gpu_runs.json`).

- Claims 1-7 resolved (2026-07-16): the recorded 06302026 e2e bundle's step
  metrics (plus the 2026-07-13 k-step rerun and the across-seed stds recovered
  from the bundle log) are promoted to `artifacts/expA/summary.json` and
  `artifacts/expB/summary.json` by `scripts/promote_ab_summaries.py`.
- Claims 9-10 resolved (2026-07-16): per-seed pooled targets promoted to
  `artifacts/expB2/bv3_n10_summary.json` and
  `artifacts/expB2/sysid_ceiling_n10_summary.json` by the same script;
  `scripts/audit_stats_recheck.py` recomputes those values from the committed
  artifacts and compares them with its own copies of the quoted numbers. It reads
  numbers from this page only in rows 27, 40, 44 and 46 (notes of 2026-10-10).
- Rows 23-24 resolved (spec 2026-07-14, per-seed summary committed as
  `artifacts/expB2/heldout_l3_h8_summary.json`): transfer generalizes, and the
  re-scored common-garden control passes the frozen rule on both directions
  (row 24; forward 0.666, reverse 0.684; prefix condition remains decodable after
  restoring authentic dynamics, FINDINGS §10.6.1);
  the writeup reports both as generality checks, not as a new headline. Rows 20-22 resolved 2026-07-14: the second
  capacity replicates the residual after the behavior control and bounds the
  survival-over-predictor margin to the subtler artifact (that margin is later
  withdrawn as an objective effect, FINDINGS 17.6).
