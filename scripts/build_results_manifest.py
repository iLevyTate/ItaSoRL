"""Results manifest: which implementation and configuration produced each committed result.

Revision step 1 (docs/REVISION_2026-10.md). Every committed artifact under `artifacts/`
is assigned to exactly one RUN below. A run records the code commit it executed at, its
configuration, the agent seeds, the surrogate (G_motion) seed and capacity, the training
budget, the evaluation seed bases, the CV fold partition its numbers were scored under,
the dependency versions and device where they were recorded, and, the field this
revision turns on, which policy-gradient implementation trained its survival agents.

Recorded provenance (commits, config fingerprints, environment blocks) is read from the
artifacts themselves and placed next to the curated entry, so a disagreement between the
two is visible. Nothing here recomputes a result; `scripts/audit_stats_recheck.py` does
that.

Status vocabulary:
  historical  produced before the 2026-10 corrections; the number stands as a record of
              what that implementation measured and is cited as such.
  corrected   produced by the corrected implementation (revision step 4: the confirmation
              runs C1 and C2 and the readouts on their saved agents).

GAE vocabulary (`survival_trainer` field):
  pre_transition_value   itasorl.experiment_b2.compute_gae as of 679fee6 (2026-06-28) up to
                         and including 4b6e1f3: an episode alive at the 80-step cutoff
                         bootstraps from V(h_T), the critic value at its LAST STORED step,
                         i.e. before its final transition, not from the successor state.
  successor_value        the corrected bootstrap (revision step 2).
  none                   the run trains no actor-critic (oracle, predictor-only, Exp B,
                         neuroevolution).
  inherited: <value>     readout-only on saved agents of the runs in `readout_only_on`.

Usage:
    python scripts/build_results_manifest.py           # write artifacts/results_manifest.json
                                                       # and docs/RESULTS_MANIFEST.md
    python scripts/build_results_manifest.py --check   # exit 1 if stale or incomplete
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts")
OUT_JSON = os.path.join(ART, "results_manifest.json")
OUT_MD = os.path.join(ROOT, "docs", "RESULTS_MANIFEST.md")

# The commit this revision was reviewed at; every entry below is historical relative to it.
REVIEWED_COMMIT = "4b6e1f3"
GAE_PADDING_FIX_COMMIT = "679fee6"  # 2026-06-28: the next-step mask fix; the bootstrap stayed

WORLD_P = "WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)"

# Evaluation seed bases hard-coded in itasorl/experiment_b2.py and scripts/run_expB2.py.
# The same world seeds are used for every agent seed, so across-seed intervals condition on
# this one sample of evaluation worlds (revision step 13).
BV2_EVAL_SEEDS = {
    "pooled_authentic": "800000 + i, i < pool_n",
    "pooled_surrogate": "850000 + i, i < pool_n (a DIFFERENT world sample from the authentic pool)",
    "matched_pair": "700000 + p, p < mp_pairs (both branches share the seed)",
    "engagement": "900000 (+1 random baseline, +2 scripted baseline), 64 episodes",
    "cross_eval_return": "920000, 64 episodes",
    "untrained_norm_warmup": "555000, 8 episodes x 40 steps",
}
BV2_TRAIN_SEEDS = {
    "survival": "100000 + 10000*seed + 16*update + i (16 parallel episodes per update)",
    "predictor": "200000 + 9000*seed + 16*update + i",
}
HELDOUT_EVAL_SEEDS = {
    "transfer_authentic": "860000 + i", "transfer_surrogate": "870000 + i",
    "common_garden": "930000 + p (prefix authentic vs surrogate, tail authentic)",
}

# The standard B-v2 protocol (scripts/run_expB2.py defaults at full scale).
BV2_PROTOCOL = {
    "runner": "scripts/run_expB2.py",
    "world": WORLD_P + " with SURVIVAL_METAB and SURVIVAL_FOOD (itasorl/experiment_b2.py)",
    "drifts": [0.0, 0.45],
    "train_condition": "each (drift, seed) cell trains its agents in ONE condition: authentic "
                       "at drift 0, surrogate at drift 0.45; it is evaluated on an authentic "
                       "pool and a drift pool",
    "updates": 300, "n_eps": 16, "max_steps": 80, "hidden": 96, "embed": 64, "ray_steps": 5,
    "survival_optimizer": "Adam lr 3e-4, gamma 0.99, lambda 0.95, entropy 0.01, value 0.5, "
                          "decoder 1.0, grad clip 1.0",
    "shaping_coef": 1.0,
    "predictor_optimizer": "Adam lr 1e-3 (train_predictor_only default), scripted policy",
    "pool_n": 110, "pool_steps": 24, "mp_pairs": 60, "mp_prefix": 20, "mp_branch": 24,
    "normalizer": "RunningNorm fitted online on the arm's own training observations, frozen "
                  "after training; the untrained arm fits it on 8 x 40 warmup steps",
}

FOLDS_NOTE = {
    "legacy": "installed scikit-learn GroupKFold on the run's own stack (published GPU "
              "numbers: (24,20),(22,22),(21,23),(21,23),(22,22) for 110+110)",
    "explicit": "itasorl.folds explicit partition (five 22/22 folds for 110+110), stack-"
                "independent; equal to current scikit-learn",
    "both": "scored under both; see artifacts/fold_rescore and FINDINGS methods note 8",
    "n/a": "no grouped CV probe",
}


def run(id, title, *, experiment, artifacts, claims=(), survival_trainer="pre_transition_value",
        trains_survival=True, readout_of=None, commit_at_run=None, config=None,
        agent_seeds=None, surrogate=None, budget=None, eval_seeds=None, folds="legacy",
        device=None, deps=None, local_run_dir=None, notes="", status="historical"):
    """One manifest row. Paths in `artifacts` are relative to artifacts/ (file or directory)."""
    return {
        "id": id, "title": title, "experiment": experiment, "claims": list(claims),
        "artifacts": list(artifacts), "status": status,
        "survival_trainer": (survival_trainer if trains_survival else
                             f"inherited: {survival_trainer}" if readout_of else "none"),
        "affected_by_gae_correction": bool(trains_survival or readout_of),
        "readout_only_on": readout_of,
        "commit_at_run": commit_at_run, "config": config or {},
        "agent_seeds": agent_seeds, "surrogate": surrogate, "budget": budget,
        "eval_seeds": eval_seeds, "folds": folds, "device": device, "deps": deps,
        "local_run_dir": local_run_dir, "notes": notes,
    }


S10 = list(range(10))
L3_H8 = {"family": "GMotion (itasorl/surrogate_l3.py)", "hidden": 8, "g_seed": 0,
         "training": "fixed-epoch Adam fit on authentic transitions (train_g_motion defaults)"}
GPU_OWNER = "owner's GPU machine (RTX 4050 Laptop); torch 2.7.0+cu126"
CPU_CLOUD = "cloud CPU sandbox, 4 vCPU, 3 workers; torch 2.14+cpu"
CPU_REVISION = "revision container, 4 vCPU, 4 workers; torch 2.14.1+cpu"

RUNS = [
    # ---------------- agent-free and prediction-only arcs (no actor-critic) ----------------
    run("A-L1L2", "Experiment A oracle ceilings, L1 and L2", experiment="A",
        artifacts=["expA/summary.json", "expA/l1_calib.json"], claims=[1, 2, 3],
        trains_survival=False, survival_trainer="none", commit_at_run="4c16be6",
        config={"runners": ["scripts/run_expA.py", "scripts/run_expA_l2.py",
                            "scripts/run_expA_l1.py"]},
        eval_seeds="matched pairs, seeds per runner", folds="legacy", device="Colab Tesla T4",
        deps="python 3.12.13, torch 2.11.0+cu128 (recorded)", local_run_dir="fullruns/06302026",
        notes="Detector-side measurement noise sigma is part of the detector, not the world."),
    run("A-L2-INCONFIG", "Matched-handicap oracle ceilings across rungs", experiment="A",
        artifacts=["expA/l2_inconfig_oracle.json"], claims=[36], trains_survival=False,
        survival_trainer="none", config={"runner": "scripts/run_expA_l2_inconfig.py",
                                          "sigma_frozen": 0.02},
        folds="legacy", notes="Privileged detector score at a common detector-side noise."),
    run("A-L3-GATE0", "L3 gate 0 oracle and floor, hidden 8 and 7, both fold schemes",
        experiment="A", artifacts=["fold_rescore/gate0_h8h7_legacy.json",
                                   "fold_rescore/gate0_h8h7_explicit.json"],
        claims=[11, 22], trains_survival=False, survival_trainer="none",
        surrogate={**L3_H8, "hidden": [8, 7]}, folds="both",
        config={"runner": "scripts/run_expA_l3.py", "sigma_meas": 0.02}),
    run("B-PRED", "Experiment B prediction-only arc (L2)", experiment="B",
        artifacts=["expB/summary.json"], claims=[4, 5, 6, 7], trains_survival=False,
        survival_trainer="none", commit_at_run="4c16be6",
        config={"agent": "itasorl/agent.py RecurrentWorldModel, next-step prediction"},
        agent_seeds=[0, 1, 2], folds="legacy", device="Colab Tesla T4",
        deps="python 3.12.13, torch 2.11.0+cu128 (recorded)", local_run_dir="fullruns/06302026"),
    # ---------------- B-v2 / B-v3 (L2 surrogates), survival trained ----------------
    run("BV2-L2-AR1", "B-v2 survival coupling, L2 AR(1) drift, canonical replication",
        experiment="B-v2", artifacts=["expB2/expB2_results.json", "expB2/expB2_survival.png"],
        claims=[8], commit_at_run="4c16be6", config={**BV2_PROTOCOL, "drift_mode": "ar1"},
        agent_seeds=[0, 1, 2], budget={"survival_updates": 300, "env_steps_per_update": "<=1280"},
        eval_seeds=BV2_EVAL_SEEDS, folds="legacy", device="Colab Tesla T4",
        deps="torch 2.11.0+cu128", local_run_dir="fullruns/06302026"),
    run("BV2-L2-AR1-LAB", "B-v2 initial lab confirmatory run (archived)", experiment="B-v2",
        artifacts=["expB2/expB2_results_confirmatory_n3.json",
                   "expB2/expB2_survival_confirmatory_n3.png"],
        claims=[], config={**BV2_PROTOCOL, "drift_mode": "ar1"}, agent_seeds=[0, 1, 2],
        eval_seeds=BV2_EVAL_SEEDS, folds="legacy",
        notes="Run commit not recorded. Re-run after the 679fee6 padding fix per "
              "PREREGISTRATION.md (2026-06-28); kept as an archive, not cited as a result."),
    run("BV3-REGIME-N10", "B-v3 per-episode drag regime, n = 10", experiment="B-v3",
        artifacts=["expB2/bv3_n10_summary.json", "expB2/bv3_n10_gates.json"], claims=[9, 37],
        commit_at_run="820849f", config={**BV2_PROTOCOL, "drift_mode": "regime"},
        agent_seeds=S10, budget={"survival_updates": 300}, eval_seeds=BV2_EVAL_SEEDS,
        folds="legacy", device=GPU_OWNER, deps="python 3.13.2, torch 2.7.0+cu126 (recorded)",
        local_run_dir="fullruns/07062026",
        notes="State dumps absent, so not re-scored under the explicit partition."),
    run("BV2-SYSID-N10", "L2 capacity ceiling (sysid auxiliary), n = 10", experiment="B-v2",
        artifacts=["expB2/sysid_ceiling_n10_summary.json"], claims=[10], commit_at_run="f848738",
        config={**BV2_PROTOCOL, "drift_mode": "ar1", "sysid_aux": True}, agent_seeds=S10,
        eval_seeds=BV2_EVAL_SEEDS, folds="legacy", device=GPU_OWNER,
        local_run_dir="fullruns/07092026",
        notes="Supervised ceiling control: breaks readout-not-reward by design."),
    # ---------------- L3 survival-trained runs ----------------
    run("L3-H8-N10", "L3 hidden 8 headline, n = 10 (fullruns/l3_n10, l3_n10_audited, "
        "l3_h8_traces are deterministic replications)", experiment="B-v2 L3",
        artifacts=["expB2/behavior_audit_l3_n10.json", "expB2/behavior_audit_l3_h8_traces.json",
                   "expB2/behavior_audit_l3_covar_n10.json", "fold_rescore/l3_n10.json",
                   "fold_rescore/l3_h8_traces.json"],
        claims=[12, 13, 14, 15, 16, 17, 18, 19], commit_at_run="2888d37 (l3_h8_traces)",
        config={**BV2_PROTOCOL, "drift_mode": "l3"}, agent_seeds=S10, surrogate=L3_H8,
        budget={"survival_updates": 300, "predictor_updates": 300}, eval_seeds=BV2_EVAL_SEEDS,
        folds="both", device=GPU_OWNER + " (local GPU; per-run stack not recorded)",
        local_run_dir="fullruns/l3_h8_traces",
        notes="Primary published numbers on the legacy partition (0.752 / 0.726); explicit "
              "partition 0.774 / 0.750 (FINDINGS methods note 8)."),
    run("L3-H4", "L3 hidden 4 run (gate failure, uninformative)", experiment="B-v2 L3",
        artifacts=["fold_rescore/l3_h4_traces.json"], claims=[22],
        config={**BV2_PROTOCOL, "drift_mode": "l3"}, agent_seeds=S10,
        surrogate={**L3_H8, "hidden": 4}, eval_seeds=BV2_EVAL_SEEDS, folds="both",
        device=GPU_OWNER, local_run_dir="fullruns/l3_h4_traces"),
    run("L3-H7-N10", "L3 hidden 7 second capacity, n = 10", experiment="B-v2 L3",
        artifacts=["expB2/behavior_audit_l3_h7_traces.json", "fold_rescore/l3_h7_traces.json"],
        claims=[20, 21, 22], config={**BV2_PROTOCOL, "drift_mode": "l3"}, agent_seeds=S10,
        surrogate={**L3_H8, "hidden": 7}, eval_seeds=BV2_EVAL_SEEDS, folds="both",
        device=GPU_OWNER, local_run_dir="fullruns/l3_h7_traces",
        notes="Run commit recorded only in the PREREGISTRATION_L3 2026-07-14 entries."),
    run("L3-H8-HELDOUT", "L3 hidden 8 held-out transfer and common garden, n = 10 (saved "
        "agents reused by every readout-only L3 analysis)", experiment="B-v2 L3",
        artifacts=["expB2/heldout_l3_h8_summary.json", "expB2/behavior_audit_l3_h8_heldout.json",
                   "expB2/heldout_l3_h8_cg_rescore.json", "expB2/heldout_l3_h8_mp_rescore.json",
                   "fold_rescore/l3_h8_heldout.json", "fold_rescore/cg_h8_legacy.json",
                   "fold_rescore/cg_h8_explicit.json", "fold_rescore/mp_h8_legacy.json",
                   "fold_rescore/mp_h8_explicit.json"],
        claims=[23, 24], commit_at_run="ed88df0",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "heldout_evals": True, "heldout_hidden": 7,
                "cg_prefix": 20, "cg_steps": 24, "save_agents": True},
        agent_seeds=S10, surrogate={**L3_H8, "heldout": {"hidden": 7, "g_seed": 0}},
        eval_seeds={**BV2_EVAL_SEEDS, **HELDOUT_EVAL_SEEDS}, folds="both", device=GPU_OWNER,
        local_run_dir="fullruns/l3_h8_heldout",
        notes="The held-out G shares recipe, seed and training data with the training G."),
    run("L3-H7-REVERSE", "L3 reverse transfer (train hidden 7, hold out hidden 8), n = 10",
        experiment="B-v2 L3",
        artifacts=["expB2/heldout_l3_h7_reverse_summary.json",
                   "expB2/heldout_l3_h7_reverse_cg_rescore.json",
                   "expB2/heldout_l3_h7_reverse_mp_rescore.json", "fold_rescore/l3_h7_heldout.json",
                   "fold_rescore/cg_h7_legacy.json", "fold_rescore/cg_h7_explicit.json",
                   "fold_rescore/mp_h7_legacy.json", "fold_rescore/mp_h7_explicit.json"],
        claims=[25], commit_at_run="15b1fc7 / 64be5bd (cells span both)",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "heldout_evals": True, "heldout_hidden": 8,
                "save_agents": True},
        agent_seeds=S10, surrogate={**L3_H8, "hidden": 7, "heldout": {"hidden": 8, "g_seed": 0}},
        eval_seeds={**BV2_EVAL_SEEDS, **HELDOUT_EVAL_SEEDS}, folds="both", device=GPU_OWNER,
        local_run_dir="fullruns/l3_h7_heldout"),
    run("L1-ORGANISM", "L1 organism at the in-band grid (delta 0.023), n = 10",
        experiment="B-v2 L1", artifacts=["expL1/organism_summary.json", "fold_rescore/l1_heldout.json"],
        claims=[35], commit_at_run="8a71593",
        config={**BV2_PROTOCOL, "drift_mode": "l1", "drifts": [0.0, 0.023], "l1_delta": 0.023,
                "sensor_sigma": 0.01, "save_agents": True},
        agent_seeds=S10, eval_seeds=BV2_EVAL_SEEDS, folds="both",
        device="cpu, 4 workers (run log header)", local_run_dir="fullruns/l1_heldout"),
    run("L3-H8-NOWM-CPU", "Architecture baseline: no next-observation decoder, n = 10",
        experiment="B-v2 L3", artifacts=["expB2/arch_baseline_l3_h8_nowm.json",
                                         "reviewer_gaps_runs/l3_h8_nowm"],
        claims=[29], commit_at_run="a641ca0",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "world_model": False}, agent_seeds=S10,
        surrogate=L3_H8, budget={"survival_updates": 300}, eval_seeds=BV2_EVAL_SEEDS,
        folds="legacy (= explicit on this stack)", device=CPU_CLOUD, deps="torch 2.14+cpu",
        notes="Untrained arm also built without the decoder; predictor arm unchanged."),
    run("L3-H8-WM-CPU", "Device control: published hidden 8 protocol on the CPU sandbox, n = 10",
        experiment="B-v2 L3", artifacts=["expB2/device_control_l3_h8_wm_cpu.json",
                                         "reviewer_gaps_runs/l3_h8_wm_cpu"],
        claims=[30], commit_at_run="3f36cc2", config={**BV2_PROTOCOL, "drift_mode": "l3"},
        agent_seeds=S10, surrogate=L3_H8, budget={"survival_updates": 300},
        eval_seeds=BV2_EVAL_SEEDS, folds="legacy (= explicit on this stack)", device=CPU_CLOUD,
        deps="torch 2.14+cpu",
        notes="Training code is byte-identical to 4b6e1f3; the like-for-like baseline for "
              "the corrected CPU reruns of step 4."),
    run("L3-H10-GS1-CPU", "Second fingerprint instance (G seed 1, hidden 10), CPU, n = 10",
        experiment="B-v2 L3", artifacts=["expB2/second_instance_l3_h10_gseed1.json",
                                         "reviewer_gaps_runs/l3_h10_gseed1",
                                         "reviewer_gaps_runs/l3_gate0_seed1"],
        claims=[31], commit_at_run="a641ca0", config={**BV2_PROTOCOL, "drift_mode": "l3"},
        agent_seeds=S10, surrogate={**L3_H8, "hidden": 10, "g_seed": 1},
        eval_seeds=BV2_EVAL_SEEDS, folds="legacy (= explicit on this stack)", device=CPU_CLOUD),
    run("L3-H10-GS1-GPU", "Second fingerprint instance re-measured on GPU, n = 10",
        experiment="B-v2 L3", artifacts=["expB2/second_instance_l3_h10_gseed1_gpu.json",
                                         "reviewer_gaps_runs/l3_h10_gseed1_gpu",
                                         "reviewer_gaps_runs/l3_gate0_seed1_gpu"],
        claims=[31], commit_at_run="283f3ab", config={**BV2_PROTOCOL, "drift_mode": "l3"},
        agent_seeds=S10, surrogate={**L3_H8, "hidden": 10, "g_seed": 1},
        eval_seeds=BV2_EVAL_SEEDS, folds="explicit (the itasorl.folds default at 283f3ab)",
        device=GPU_OWNER),
    run("L3-H8-GS2-GPU", "Hidden 8 new-seed instance (G seed 2), n = 10", experiment="B-v2 L3",
        artifacts=["expB2/second_seed_l3_h8_gseed2_gpu.json", "reviewer_gaps_runs/l3_h8_gseed2_gpu",
                   "reviewer_gaps_runs/l3_gate0_seed2_gpu"],
        claims=[31], commit_at_run="d67dd51", config={**BV2_PROTOCOL, "drift_mode": "l3"},
        agent_seeds=S10, surrogate={**L3_H8, "g_seed": 2}, eval_seeds=BV2_EVAL_SEEDS,
        folds="explicit", device=GPU_OWNER + ", ITASORL_FOLDS=explicit"),
    run("L3-H8-NOWM-U450", "Skill-matched no-decoder baseline (survival budget 450), n = 10",
        experiment="B-v2 L3", artifacts=["expB2/skill_matched_l3_h8_nowm_u450.json",
                                         "reviewer_gaps_runs/l3_h8_nowm_skill_u450"],
        claims=[], commit_at_run="9d5d047 / 80948ff (cells span both)",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "world_model": False,
                "survival_updates": 450},
        agent_seeds=S10, surrogate=L3_H8,
        budget={"survival_updates": 450, "predictor_updates": 300}, eval_seeds=BV2_EVAL_SEEDS,
        folds="explicit", device=GPU_OWNER + ", ITASORL_FOLDS=explicit",
        notes="The skill match failed (overshoot 0.0131); no mediation verdict."),
    run("ENGAGE-MARGIN", "Engagement-margin sweep on committed cells", experiment="B-v2",
        artifacts=["expB2/engagement_margin_cloud_runs.json"], claims=[37],
        readout_of=["BV3-REGIME-N10", "L3-H8-NOWM-CPU", "L3-H10-GS1-CPU", "L3-H8-WM-CPU"],
        trains_survival=False, folds="n/a"),
    # ---------------- readout-only analyses on saved survival agents ----------------
    run("L3-CROSSRECIPE", "Cross-recipe transfer probe (RFF ridge family)", experiment="B-v2 L3",
        artifacts=["l3_crossrecipe/summary.json"], claims=[26], trains_survival=False,
        readout_of=["L3-H8-HELDOUT"], commit_at_run="a5c46ff", folds="legacy",
        eval_seeds={"rff": "880000 / 890000", "cd": "940000 / 950000"}, device="local GPU"),
    run("H2-GRADED-SEAM", "H2 graded-seam ablation (A1)", experiment="H2",
        artifacts=["expH2/summary.json"], claims=[32], trains_survival=False,
        readout_of=["L3-H8-HELDOUT"], commit_at_run="c4e4417", folds="legacy", device="cuda",
        notes="Decoding falls with the dynamics difference; this does not identify a "
              "representation of learnedness."),
    run("H2-TEXTURE", "H2 texture knockout and dose response", experiment="H2",
        artifacts=["expH2/texture_knockout_h8.json", "expH2/texture_knockout_h7.json"],
        claims=[33], trains_survival=False, readout_of=["L3-H8-HELDOUT", "L3-H7-REVERSE"],
        folds="legacy", notes="Transfer of the ORIGINAL direction only; no fresh Gaussian "
                              "probe and no Gaussian-trained agents (revision step 9)."),
    run("H2-OBSLOC", "Observation-channel localization (A2)", experiment="H2",
        artifacts=["expH2/obs_localization_h8.json", "expH2/obs_localization_h7.json"],
        claims=[34], trains_survival=False, readout_of=["L3-H8-HELDOUT", "L3-H7-REVERSE"],
        folds="legacy", notes="Vision channels include radial velocity, so masking vision "
                              "removes a motion signal, not only appearance."),
    run("L1-H2", "L1 H2 battery and observation localization", experiment="H2",
        artifacts=["expL1/h2_ablations.json", "expL1/obs_localization.json"], claims=[35],
        trains_survival=False, readout_of=["L1-ORGANISM"], folds="legacy"),
    run("SENSORY-ECHO", "Sensory-echo controls (hidden 8 linear and MLP, hidden 7)",
        experiment="B-v2 L3",
        artifacts=["expB2/sensory_echo_l3_h8.json", "expB2/sensory_echo_l3_h8_mlp.json",
                   "expB2/sensory_echo_l3_h7.json"],
        claims=[28], trains_survival=False, readout_of=["L3-H8-HELDOUT", "L3-H7-REVERSE"],
        folds="legacy", device="cuda",
        notes="Basis is [x_t, x_{t-1}] (and cummean in the secondary variant); the five "
              "previous-action channels fed to the GRU are not in the basis."),
    # ---------------- revision 2026-10 methods artifacts ----------------
    run("FOLDS-EXPLICIT-V1", "Serialized explicit-v1 partitions of the standard designs",
        experiment="methods", artifacts=["folds/explicit_v1.json"], trains_survival=False,
        survival_trainer="none", folds="explicit",
        notes="itasorl.folds.standard_partitions(); tests/test_l0_audit.py regenerates and compares."),
    run("GATE-TABLE", "Gate table, historical and corrected runs, both partitions",
        experiment="methods", artifacts=["gate_table.json"], trains_survival=False,
        readout_of=["BV2-L2-AR1", "BV3-REGIME-N10", "L3-H8-N10", "L3-H7-N10", "L3-H4",
                    "L3-H8-HELDOUT", "L3-H7-REVERSE", "L1-ORGANISM", "L3-H8-NOWM-CPU",
                    "L3-H8-WM-CPU", "L3-H10-GS1-CPU", "L3-H10-GS1-GPU", "L3-H8-GS2-GPU",
                    "L3-H8-NOWM-U450"],
        folds="both", notes="scripts/build_gate_table.py; docs/GATE_TABLE.md."),
    run("CONTRAST-INTERVALS", "Seed-paired intervals for the registered margins, every run",
        experiment="methods", artifacts=["contrast_intervals.json"], trains_survival=False,
        readout_of=["BV3-REGIME-N10", "L3-H8-N10", "L3-H7-N10", "L3-H8-HELDOUT", "L3-H7-REVERSE",
                    "L1-ORGANISM", "L3-H8-NOWM-CPU", "L3-H8-WM-CPU", "L3-H10-GS1-CPU",
                    "L3-H10-GS1-GPU", "L3-H8-GS2-GPU", "L3-H8-NOWM-U450"],
        folds="both", notes="scripts/build_contrast_intervals.py; docs/CONTRAST_INTERVALS.md."),
    run("QD-GATE0", "Gate 0 for the hand-authored quadratic-drag comparator (no eps passes)",
        experiment="methods", artifacts=["texture/gate0_qd.json"], trains_survival=False,
        survival_trainer="none", folds="explicit",
        config={"runner": "scripts/run_expA_l3.py --family qd", "sigma_meas": 0.02,
                "sweep": [0.5, 1, 2, 4, 8, 16]},
        notes="Untrained floor seeds use the corrected trainer's untrained arm (no actor-critic). "
              "docs/specs/2026-10-06-texture-comparator-design.md amendment."),
    run("SURROGATE-DIAG", "Surrogate diagnostics: linear-fit control, held-out and rollout error, "
        "agent-accessible detector", experiment="methods", artifacts=["surrogate_diagnostics.json"],
        trains_survival=False, survival_trainer="none", folds="explicit",
        config={"runner": "scripts/run_surrogate_diagnostics.py"},
        notes="Agent free: G instances retrained deterministically from their recipe."),
    run("SURROGATES", "Frozen L3 fingerprints (G_motion instances), serialized",
        experiment="methods", artifacts=["surrogates"], trains_survival=False,
        survival_trainer="none", folds="n/a",
        config={"runner": "scripts/export_surrogates.py", "instances": "h8 s0, h7 s0, h10 s1, h8 s2, h4 s0"},
        device="cpu", notes="CPU-trained; GPU-published runs trained G on CUDA (see index.json)."),
    run("L0-PRE-INTERVENTION", "Pool membership decoded from the reset observation (agent free)",
        experiment="L0 audit", artifacts=["l0_audit/pre_intervention.json"], trains_survival=False,
        survival_trainer="none", folds="explicit",
        eval_seeds={"standard": "800000 / 850000",
                    "independent": "(1000000 + 100000k) / (1050000 + 100000k), k < 8"},
        notes="itasorl/l0_audit.py; revision step 5."),
    # ---------------- corrected confirmation runs (revision step 4) ----------------
    run("C1", "Corrected trainer, next-observation auxiliary on, n = 10 (replaces L3-H8-WM-CPU)",
        experiment="B-v2 L3", artifacts=["expB2/corrected_l3_h8_wm.json",
                                         "corrected_runs/corrected_l3_h8_wm"],
        claims=[41, 42], survival_trainer="successor_value", status="corrected",
        commit_at_run="f676b95",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "gae_bootstrap": "successor",
                "budget_extend": 450, "budget_snapshots": [100, 200]},
        agent_seeds=S10, surrogate=L3_H8,
        budget={"survival_updates": 300, "budget_curve": [100, 200, 300, 450],
                "predictor_updates": 300},
        eval_seeds=BV2_EVAL_SEEDS, folds="explicit (explicit-v1; equals legacy on this stack)",
        device=CPU_REVISION, deps="torch 2.14.1+cpu, numpy 2.4.6, scikit-learn 1.9.1",
        local_run_dir="fullruns/corrected_l3_h8_wm",
        notes="Frozen protocol: docs/specs/2026-10-06-corrected-trainer-confirmation-design.md. "
              "The headline arm is the 300-update snapshot of a run trained to 450."),
    run("C2", "Corrected trainer, next-observation auxiliary off, n = 10 (replaces L3-H8-NOWM-CPU)",
        experiment="B-v2 L3", artifacts=["expB2/corrected_l3_h8_nowm.json",
                                         "corrected_runs/corrected_l3_h8_nowm"],
        claims=[42], survival_trainer="successor_value", status="corrected",
        commit_at_run="f676b95",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "world_model": False, "gae_bootstrap": "successor",
                "budget_extend": 450, "budget_snapshots": [100, 200]},
        agent_seeds=S10, surrogate=L3_H8,
        budget={"survival_updates": 300, "budget_curve": [100, 200, 300, 450],
                "predictor_updates": 300},
        eval_seeds=BV2_EVAL_SEEDS, folds="explicit (explicit-v1; equals legacy on this stack)",
        device=CPU_REVISION, deps="torch 2.14.1+cpu, numpy 2.4.6, scikit-learn 1.9.1",
        local_run_dir="fullruns/corrected_l3_h8_nowm",
        notes="Untrained arm also built without the decoder; predictor arm unchanged. The 450 point "
              "is a budget point, not a skill match."),
    run("C1-L0-AUDIT", "L0 across independent world-sample pairs and the balanced readout, C1 agents",
        experiment="L0 audit", artifacts=["l0_audit/corrected_l3_h8_wm.json"], trains_survival=False,
        readout_of=["C1"], survival_trainer="successor_value", status="corrected",
        commit_at_run="34e2c0d", folds="explicit",
        eval_seeds={"standard": "800000 / 850000",
                    "independent": "(1000000 + 100000k) / (1050000 + 100000k), k < 8",
                    "balanced": "800000 + i for both pools (same world seed and initial state)"},
        device=CPU_REVISION, notes="scripts/run_l0_audit.py; revision step 5; diagnostic, not a gate."),
    run("C1-POLICY-CONTROLS", "Readouts under own, scripted, and replayed policies, with an "
        "exposure-matched predictor, C1 agents", experiment="B-v2 L3",
        artifacts=["policy_controls/corrected_l3_h8_wm.json"], trains_survival=False,
        readout_of=["C1"], survival_trainer="successor_value", status="corrected",
        commit_at_run="34e2c0d", folds="explicit", eval_seeds=BV2_EVAL_SEEDS, device=CPU_REVISION,
        notes="scripts/run_policy_controlled_readouts.py; revision step 6. Survival agents "
              "retrained with batch logging, bit-identical to the saved agents (10/10); "
              "predictor_logged trains the prediction objective on those batches."),
    run("C1-PERSISTENCE", "Controlled persistence test (replay, common state, factorial, reset), "
        "C1 agents", experiment="B-v2 L3", artifacts=["persistence/corrected_l3_h8_wm.json"],
        trains_survival=False, readout_of=["C1"], survival_trainer="successor_value",
        status="corrected", commit_at_run="34e2c0d", folds="explicit",
        eval_seeds={"persistence": "980000 + p, p < 110"}, device=CPU_REVISION,
        notes="scripts/run_persistence_readout.py; revision step 7; rule frozen in "
              "docs/specs/2026-10-06-controlled-persistence-design.md."),
    run("C1-CONTROL-DIAG", "Behavior and sensory controls with fit diagnostics, and sequence "
        "readouts of the observation stream, C1 agents", experiment="B-v2 L3",
        artifacts=["control_diagnostics/corrected_l3_h8_wm.json"], trains_survival=False,
        readout_of=["C1"], survival_trainer="successor_value", status="corrected",
        commit_at_run="34e2c0d", folds="explicit", eval_seeds=BV2_EVAL_SEEDS, device=CPU_REVISION,
        notes="scripts/run_control_diagnostics.py; revision step 8. Pools regenerated and "
              "bit-matched to the run's state dumps."),
    run("C1-TEXTURE", "Texture comparators on the C1 agents: transfer of the original direction "
        "and a fresh probe", experiment="H2", artifacts=["texture/corrected_l3_h8_wm_gn.json",
                                                "texture/corrected_l3_h8_wm_qd.json"],
        trains_survival=False, readout_of=["C1"], survival_trainer="successor_value",
        status="corrected", commit_at_run="34e2c0d", folds="explicit",
        eval_seeds={"gn": "960000 / 970000", "qd": "1900000 / 1950000"}, device=CPU_REVISION,
        notes="scripts/run_texture_fresh_probe.py; revision step 9; wording rules frozen in "
              "docs/specs/2026-10-06-texture-comparator-design.md."),
    run("C1-POPULATION", "Pooled versus per-individual readout on known-decodable agents, and the "
        "value of world information, C1", experiment="C",
        artifacts=["population_readout/corrected_l3_h8_wm.json"], trains_survival=False,
        readout_of=["C1"], survival_trainer="successor_value", status="corrected",
        commit_at_run="34e2c0d", folds="explicit", eval_seeds=HELDOUT_EVAL_SEEDS, device=CPU_REVISION,
        notes="scripts/validate_population_readout.py; revision step 12."),
    run("BUDGET-CURVE", "Return and decodability against survival updates, decoder on and off",
        experiment="B-v2 L3", artifacts=["budget_curve.json"], trains_survival=False,
        readout_of=["C1", "C2"], survival_trainer="successor_value", status="corrected",
        folds="explicit", notes="scripts/build_budget_curve.py; revision step 11; frozen snapshots "
                                "of one training run per seed; figure docs/figures/budget_curve.png."),
    run("CROSS-REPLAY", "EXPLORATORY: decoder-on and decoder-off survival trunks replayed on each "
        "other's streams", experiment="B-v2 L3", artifacts=["cross_replay/corrected_c1_c2.json"],
        trains_survival=False, readout_of=["C1", "C2"], survival_trainer="successor_value",
        status="corrected", commit_at_run="7870bba", folds="explicit", eval_seeds=BV2_EVAL_SEEDS,
        device=CPU_REVISION, notes="scripts/run_cross_replay.py; post hoc, written after C1 and C2 "
                                   "were read; revision step 11."),
    run("CORRECTED-VERDICTS", "Frozen-rule verdicts, integrity, and correction effect for C1 and C2",
        experiment="methods", artifacts=["corrected_verdicts.json"], trains_survival=False,
        readout_of=["C1", "C2", "L3-H8-WM-CPU", "L3-H8-NOWM-CPU"], survival_trainer="successor_value",
        status="corrected", folds="explicit",
        notes="scripts/build_corrected_verdicts.py; rules frozen in "
              "docs/specs/2026-10-06-corrected-trainer-confirmation-design.md."),
    # ---------------- Experiment C (neuroevolution, no actor-critic) ----------------
    run("C-EMERGENCE", "Experiment C emergence under selection, fixed-code re-run",
        experiment="C", artifacts=["expC/emergence_pilot_summary.json",
                                   "expC/control_layout_derisk.json",
                                   "expC/gate1_steepness_sweep.json"],
        claims=[27], trains_survival=False, survival_trainer="none", commit_at_run="a0cb850",
        agent_seeds=[0, 1, 2], surrogate=L3_H8, folds="legacy",
        notes="Pooled population probe; individual detectors encoded in unaligned "
              "directions are not measured (revision step 12)."),
]


def _rel_files() -> list[str]:
    out = []
    for p in glob.glob(os.path.join(ART, "**", "*"), recursive=True):
        if os.path.isfile(p) and "__pycache__" not in p:
            rel = os.path.relpath(p, ART).replace(os.sep, "/")
            if rel != "results_manifest.json" and os.path.basename(rel) != "README.md":
                out.append(rel)
    return sorted(out)


def _owner(rel: str) -> list[str]:
    return [r["id"] for r in RUNS for a in r["artifacts"]
            if rel == a or rel.startswith(a.rstrip("/") + "/")]


def _recorded(path: str) -> dict:
    """Provenance fields the artifact records about itself (JSON files only)."""
    if not path.endswith(".json"):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {"unreadable": True}
    if isinstance(doc, list):
        return {}
    if "cell" in doc and "fingerprint" in doc:          # a checkpointed run cell
        return {"git_commit": doc.get("git_commit"), "fingerprint": doc.get("fingerprint")}
    keep = ("git_commit", "git_commit_at_run", "git_commit_at_promotion", "fingerprint",
            "environment", "execution", "stack", "spec", "source_run", "run_id", "code",
            "date", "generated_by")
    return {k: doc[k] for k in keep if k in doc}


def _cells_summary(dirpath: str) -> dict:
    commits, fps, n = {}, {}, 0
    for p in sorted(glob.glob(os.path.join(dirpath, "cells", "cell_*.json"))):
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        n += 1
        commits[j.get("git_commit")] = commits.get(j.get("git_commit"), 0) + 1
        fps[j.get("fingerprint")] = fps.get(j.get("fingerprint"), 0) + 1
    return {"n_cells": n, "cell_commits": commits, "cell_fingerprints": fps} if n else {}


def build() -> dict:
    runs = []
    for r in RUNS:
        rec = {}
        for a in r["artifacts"]:
            p = os.path.join(ART, a)
            rec[a] = _cells_summary(p) if os.path.isdir(p) else _recorded(p)
        runs.append({**r, "recorded_provenance": rec})
    return {
        "generated_by": "scripts/build_results_manifest.py",
        "reviewed_commit": REVIEWED_COMMIT,
        "gae_history": {
            "padding_fix": GAE_PADDING_FIX_COMMIT,
            "pre_transition_value": "every survival-trained run from 679fee6 through "
                                    "4b6e1f3 (all runs below with survival_trainer "
                                    "'pre_transition_value')",
            "successor_value": "the corrected bootstrap (revision step 2): the runs with "
                               "status 'corrected'",
        },
        "fold_schemes": FOLDS_NOTE,
        "eval_world_note": "Evaluation worlds are fixed seed bases shared by every agent seed, "
                           "so across-seed intervals condition on one sample of evaluation "
                           "worlds and on the single trained surrogate of each run.",
        "train_seed_schedule": BV2_TRAIN_SEEDS,
        "runs": runs,
    }


def render_md(m: dict) -> str:
    L = ["# Results manifest", "",
         "*Generated by `scripts/build_results_manifest.py` from the registry in that script "
         "and the provenance fields the artifacts record about themselves. Do not edit by hand; "
         "edit the registry and rerun. `--check` fails if this page or "
         "`artifacts/results_manifest.json` is stale, or if any file under `artifacts/` "
         "belongs to no run.*", "",
         f"Reviewed at commit `{m['reviewed_commit']}`. Rows with status **historical** record "
         "what the pre-correction implementation measured. Corrected results are added as new "
         "rows with status **corrected** (trainer `successor_value`); historical rows are never "
         "overwritten.", "",
         "## Which trainer produced each survival agent", "",
         "`pre_transition_value` is `compute_gae` as it stood from "
         f"`{m['gae_history']['padding_fix']}` (2026-06-28) to `{m['reviewed_commit']}`: an "
         "episode still alive at the 80-step cutoff bootstrapped from the critic value at its "
         "last stored step, before its final transition. `none` means the run trains no "
         "actor-critic. A readout-only analysis inherits the trainer of the run whose saved "
         "agents it reads.", "",
         "| Run | Status | Experiment | Claims | Trainer | Readout of | Commit at run | Folds | Device |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in m["runs"]:
        claims = ", ".join(str(c) for c in r["claims"]) or "-"
        ro = ", ".join(r["readout_only_on"]) if r["readout_only_on"] else "-"
        L.append(f"| `{r['id']}` | {r['status']} | {r['experiment']} | {claims} | {r['survival_trainer']} | "
                 f"{ro} | {r['commit_at_run'] or 'not recorded'} | {r['folds']} | "
                 f"{r['device'] or 'not recorded'} |")
    L += ["", "Claims are the row numbers of the claims inventory in `docs/PAPER_OUTLINE.md`.",
          "", "## Evaluation worlds and seeds", "",
          m["eval_world_note"], "",
          "| Readout | Seed base |", "|---|---|"]
    for k, v in BV2_EVAL_SEEDS.items():
        L.append(f"| {k} | {v} |")
    for k, v in HELDOUT_EVAL_SEEDS.items():
        L.append(f"| {k} | {v} |")
    L += ["", "Training seeds: survival "
          f"{m['train_seed_schedule']['survival']}; predictor "
          f"{m['train_seed_schedule']['predictor']}.", "",
          "## Fold partitions", ""]
    for k, v in m["fold_schemes"].items():
        L.append(f"- `{k}`: {v}")
    L += ["", "## Standard B-v2 protocol", "", "| Knob | Value |", "|---|---|"]
    for k, v in BV2_PROTOCOL.items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Runs", ""]
    for r in m["runs"]:
        L += [f"### `{r['id']}`: {r['title']}", ""]
        for key, label in (("status", "Status"), ("survival_trainer", "Trainer"),
                           ("commit_at_run", "Commit at run"), ("agent_seeds", "Agent seeds"),
                           ("surrogate", "Surrogate"), ("budget", "Budget"),
                           ("folds", "Folds"), ("device", "Device"), ("deps", "Dependencies"),
                           ("local_run_dir", "Local run directory (not in git)"),
                           ("notes", "Notes")):
            v = r.get(key)
            if v in (None, "", [], {}):
                continue
            L.append(f"- {label}: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
        diff = {k: v for k, v in r["config"].items() if BV2_PROTOCOL.get(k) != v}
        if diff:
            L.append(f"- Config (beyond the standard protocol): {json.dumps(diff)}")
        L.append("- Artifacts: " + ", ".join(f"`artifacts/{a}`" for a in r["artifacts"]))
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def check(m: dict) -> list[str]:
    errs = []
    for r in RUNS:
        for a in r["artifacts"]:
            if not os.path.exists(os.path.join(ART, a)):
                errs.append(f"{r['id']}: missing artifact artifacts/{a}")
    for rel in _rel_files():
        owners = _owner(rel)
        if not owners:
            errs.append(f"artifacts/{rel} belongs to no run in the manifest")
        elif len(owners) > 1:
            errs.append(f"artifacts/{rel} is claimed by more than one run: {owners}")
    ids = [r["id"] for r in RUNS]
    if len(set(ids)) != len(ids):
        errs.append("duplicate run ids")
    for r in RUNS:
        for ro in r["readout_only_on"] or []:
            if ro not in ids:
                errs.append(f"{r['id']}: readout_only_on names unknown run {ro}")
    for path, want in ((OUT_JSON, json.dumps(m, indent=2, sort_keys=True, default=str) + "\n"),
                       (OUT_MD, render_md(m))):
        try:
            with open(path, encoding="utf-8") as fh:
                have = fh.read()
        except OSError:
            have = None
        if have != want:
            errs.append(f"{os.path.relpath(path, ROOT)} is stale; run "
                        "python scripts/build_results_manifest.py")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    m = build()
    if a.check:
        errs = check(m)
        for e in errs:
            print("MANIFEST:", e)
        print("results manifest: " + ("OK" if not errs else f"{len(errs)} problem(s)"))
        return 1 if errs else 0
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(m, indent=2, sort_keys=True, default=str) + "\n")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_md(m))
    errs = check(m)
    for e in errs:
        print("MANIFEST:", e)
    print(f"wrote {os.path.relpath(OUT_JSON, ROOT)} and {os.path.relpath(OUT_MD, ROOT)}"
          + ("" if not errs else f" ({len(errs)} problem(s))"))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
