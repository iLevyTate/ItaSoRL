"""Results manifest: which implementation and configuration produced each committed result.

Revision step 1 (docs/REVISION_2026-10.md). Every committed artifact under `artifacts/`
is assigned to exactly one RUN below. A run records the code commit it executed at, its
configuration, the agent seeds, the surrogate (G_motion) seed and capacity, the training
budget, the evaluation seed bases, the CV fold partition its numbers were scored under,
the dependency versions and device where they were recorded, and, the field this
revision turns on, which policy-gradient implementation trained its survival agents.

Recorded provenance (commits, config fingerprints, environment blocks) is read from the
artifacts themselves and placed next to the curated entry, so a disagreement between the
two is visible. Each curated `commit_at_run` that names a commit is compared with the run
commits the artifacts record (`git_commit`, `git_commit_at_run`, the cells' commits; never
the promotion commit) and labelled in `commit_at_run_source`: "recorded" when every named
commit is one the artifacts record, otherwise "curated" with the reason, and the page marks
such a commit "(curated)". Nothing here recomputes a result; `scripts/audit_stats_recheck.py`
does that.

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
import subprocess
import json
import os
import re
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
              "numbers: (22,22),(22,22),(22,22),(21,23),(23,21) for 110+110, as measured by "
              "the 2026-09-29 re-score; the split first inferred from fold means, "
              "(24,20),(22,22),(21,23),(21,23),(22,22), is superseded)",
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
        deps="torch 2.11.0+cu128", local_run_dir="fullruns/06302026",
        notes="Figure redrawn 2026-10-10 from the JSON by scripts/plot_record_figures.py "
              "(pooled AUROC per arm, mean \u00b1 std ddof 0, labeled historical)."),
    run("BV2-L2-AR1-LAB", "B-v2 initial lab confirmatory run (archived)", experiment="B-v2",
        artifacts=["expB2/expB2_results_confirmatory_n3.json",
                   "expB2/expB2_survival_confirmatory_n3.png"],
        claims=[], config={**BV2_PROTOCOL, "drift_mode": "ar1"}, agent_seeds=[0, 1, 2],
        eval_seeds=BV2_EVAL_SEEDS, folds="legacy",
        notes="Run commit not recorded. Re-run after the 679fee6 padding fix per "
              "PREREGISTRATION.md (2026-06-28); kept as an archive, not cited as a result. "
              "Figure redrawn 2026-10-10 from the JSON by scripts/plot_record_figures.py "
              "(pooled AUROC per arm, mean \u00b1 std ddof 0, labeled historical)."),
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
    run("ENGAGE-MARGIN-CORRECTED", "Engagement-margin sweep on the corrected cells",
        experiment="B-v2", artifacts=["engagement_margin_corrected.json"], claims=[37],
        readout_of=["C1", "C2"], survival_trainer="successor_value", status="corrected",
        trains_survival=False, folds="n/a",
        notes="scripts/audit_engagement_margin.py on the committed corrected cells; offline, no compute. FINDINGS methods note 2. Extends ENGAGE-MARGIN, which covered historical cells only."),
    run("ENGAGE-MARGIN-GPU", "Engagement-margin sweep on the GPU-published cells, with their "
        "inputs", experiment="B-v2", artifacts=["engagement_margin_gpu_runs.json"], claims=[37],
        readout_of=["L3-H8-N10", "L3-H7-N10", "L3-H4", "L3-H8-HELDOUT", "L3-H7-REVERSE",
                    "L3-H10-GS1-GPU", "L3-H8-GS2-GPU", "L3-H8-NOWM-U450", "L1-ORGANISM"],
        trains_survival=False, folds="n/a",
        notes="scripts/audit_engagement_margin.py --with-inputs (2026-10-10) on the gitignored "
              "cells of fullruns/l3_n10, l3_h8_traces, l3_h7_traces, l3_h4_traces, l3_h8_heldout "
              "and l3_h7_heldout, the committed cells of the three GPU reviewer-gap runs, and "
              "artifacts/expL1/organism_summary.json. Each report carries its cells' engagement "
              "inputs, so every margin recomputes from this file alone (tests/"
              "test_engagement_margin.py). Survival objective only; other objectives are counted "
              "and skipped."),
    run("WALL-CLOCK-GPU", "Wall-clock record of the GPU organism runs", experiment="methods",
        artifacts=["wall_clock_gpu_runs.json"], trains_survival=False, survival_trainer="none",
        folds="n/a",
        notes="scripts/record_wall_clock.py (2026-10-10), read only from the original bundles: "
              "start is the creation time of run.log (or of the run directory), end the last "
              "write of expB2_results.json, and the three chain-launched runs add their chain "
              "logs' UTC stamps, which agree with the file times. No GPU run recorded its own "
              "duration. Covers L3-H8-N10 (l3_n10, l3_h8_traces), L3-H7-N10, L3-H4, "
              "L3-H8-HELDOUT, L3-H7-REVERSE, L3-H10-GS1-GPU, L3-H8-GS2-GPU and L3-H8-NOWM-U450."),
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
    run("H2-GRADED-SEAM-H7", "H2 graded-seam ablation (A1), hidden-7 capacity replication",
        experiment="H2", artifacts=["expH2/summary_h7.json"], claims=[32], trains_survival=False,
        readout_of=["L3-H7-REVERSE"], commit_at_run="27b0dc1", folds="legacy", device="cuda",
        local_run_dir="fullruns/l3_h7_ablation",
        notes="scripts/run_expH2_ablation.py --hidden 7 (the surrogate capacity; the agent trunk "
              "is 96) on the saved fullruns/l3_h7_heldout agents, then "
              "scripts/promote_expH2_summary.py --l3-hidden 7 (2026-10-10). The bundle records no "
              "commit: it was written 2026-07-24 00:18 -0400 by the uncommitted runner that was "
              "committed as 27b0dc1 at 01:12. All three arms are recorded; the untrained floor "
              "rises from 0.474 to 0.586 and the predictor from 0.500 to 0.714 as the seam is "
              "restored, so neither is flat at this capacity."),
    run("H2-TEXTURE", "H2 texture knockout and dose response", experiment="H2",
        artifacts=["expH2/texture_knockout_h8.json", "expH2/texture_knockout_h7.json"],
        claims=[33], trains_survival=False, readout_of=["L3-H8-HELDOUT", "L3-H7-REVERSE"],
        folds="legacy", notes="Transfer of the ORIGINAL direction only; no fresh Gaussian "
                              "probe and no Gaussian-trained agents (revision step 9). A "
                              "qd-trained run (T-qd, 2026-10-06, commit 2606e7f, gitignored "
                              "fullruns/T_qd_l3_h8_wm) was executed and set aside: its family "
                              "failed gate 0, so it is not promoted and bears on no rule "
                              "(FINDINGS 17.9 addendum). A gn-trained run (T-GN-TRAINED) "
                              "was executed 2026-10-08/09 and is uninformative. Both "
                              "comparator-trained runs ran an 8-unit agent trunk (FINDINGS "
                              "14.5.1 correction 2026-10-10): T-qd's 60 saved agents have "
                              "cell.weight_hh (24, 8), its fingerprint 23a620e51d4e26fd "
                              "reproduces only at --hidden 8, and its speed control is short "
                              "in 8 of 60 pools (worst 0.7018)."),
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
    run("EXPC-M3-PER-INDIVIDUAL", "Experiment C milestone 3 rerun with the per-individual "
        "readout (2026-10-07)", experiment="C",
        artifacts=["expC/emergence_pilot_per_individual_summary.json"], claims=[27],
        trains_survival=False, survival_trainer="none", commit_at_run="06ecc10", folds="explicit",
        device="owner's laptop CPU", local_run_dir="fullruns/expC_milestone3",
        notes="scripts/run_expC_milestone3.py then scripts/promote_expC_summary.py; FINDINGS 13.F. "
              "Identical configuration to the re-run pilot, with one probe per individual added on "
              "the same common-garden panel. Routes UNINFORMATIVE: the speed positive control (gate 5) "
              "fails and section 7 interprets no contrast unless every gate passes; gate 1 fails "
              "only against the 0.005 margin of the unratified section-14 draft (FINDINGS 13.F, "
              "second note of 2026-10-10). Evolution is not an actor-critic trainer, so the GAE "
              "correction does not apply to it."),
    run("L1-STREAM-READOUT", "L1 stream and control readout on the saved L1 agents "
        "(frozen 2026-10-09)", experiment="H2",
        artifacts=["control_diagnostics/l1_stream_readout.json"], claims=[35, 53],
        trains_survival=False, readout_of=["L1-ORGANISM"], commit_at_run="5666f59",
        folds="explicit", device="owner's laptop CPU, 4 workers",
        local_run_dir="fullruns/l1_stream_readout",
        notes="scripts/run_control_diagnostics.py --drift-mode l1 --l1-delta 0.023 --l1-sigma "
              "0.01; spec docs/specs/2026-10-09-l1-stream-readout-design.md; FINDINGS 14.7.1. "
              "Asks whether the quantization seam reaches the agent's own observation stream. "
              "Integrity gate passed: all 30 regenerated state pools bit-match the run's dumps "
              "and every per-seed target reproduces the explicit-partition record exactly. "
              "Branch S1: the stream carries it. Pre-correction-trainer agents."),
    run("T-GN-TRAINED", "Comparator-trained run on the white-jitter family (frozen 2026-10-09)",
        experiment="B-v2 L3", artifacts=["texture/T_gn_l3_h8_wm.json"], claims=[52],
        survival_trainer="successor_value", status="corrected", commit_at_run="60eb355",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "l3_family": "gn", "l3_family_param": 0.01,
                "hidden": 8, "hidden_deviation": "the registered trunk is 96; the frozen "
                "spec's own command block (d341091, line 44) carried --hidden 8, conflating the "
                "agent trunk with --l3-hidden, and the launch copied it; the qd-trained run "
                "T-qd had the same 8-unit trunk"},
        agent_seeds=S10, surrogate={"family": "G_gn (iid velocity jitter on the authentic law)",
                                    "sigma_v": 0.01, "gate0": "oracle 0.8653 in band, floor 0.4479"},
        budget={"survival_updates": 300}, eval_seeds=BV2_EVAL_SEEDS, folds="explicit",
        device="owner's laptop CPU, 4 workers",
        local_run_dir="fullruns/T_gn_l3_h8_wm",
        notes="NOT THE REGISTERED CONFIGURATION: the agent trunk was 8 units, not 96, because "
              "the frozen spec's command block carried --hidden 8 beside --l3-hidden 8 and the "
              "launch copied it (PREREGISTRATION_L3 2026-10-10 third correction); the promoter "
              "now reads the width from the saved agents (artifact trunk_check). Uninformative "
              "for that reason "
              "independently of the gate below. scripts/run_expB2.py --l3-family gn then "
              "scripts/promote_texture_trained.py; spec "
              "docs/specs/2026-10-09-gn-comparator-trained-design.md; FINDINGS 14.5.1. "
              "UNINFORMATIVE: the registered positive control (speed probe at least 0.75, "
              "scored in every pool as the project does) is short in 9 of 60 arm-by-cell pools, "
              "worst 0.6862, with 8 of the 9 in the untrained arm the registered margins are "
              "measured against, so the matrix's two verdicts, which both require every gate to "
              "pass, are not reached, and the matrix names no cell for this "
              "failure. Not promoted as a negative; it strengthens no claim. The descriptive "
              "outcome is recorded because the preregistration log and the manuscript quote it."),
    run("H8-HELDOUT-REVERSE-DIRECTION", "Reverse-direction readout: drift-0 agents probed "
        "against a surrogate they never lived in (frozen 2026-10-09)", experiment="L0 audit",
        artifacts=["l0_audit/reverse_direction_l3_h8_heldout.json"], claims=[51], trains_survival=False,
        readout_of=["L3-H8-HELDOUT"], commit_at_run="8aa2b9f", folds="explicit",
        eval_seeds={"authentic pool": "800000 at drift 0",
                    "surrogate pool": "850000 at drift 0.45"},
        device="owner's laptop CPU, one thread per readout",
        local_run_dir="fullruns/reverse_direction",
        notes="scripts/reverse_direction.py; spec docs/specs/2026-10-09-reverse-direction-readout-"
              "design.md; FINDINGS 17.5.2. The agents are the DRIFT-0 ones of the held-out run and "
              "the readout drift is 0.45, the opposite pairing from every other cell. Integrity "
              "gate (the same agents rescored at drift 0 reproduce the recorded drift-0 targets "
              "within 0.01) passed on all 30 cells, worst abs dev 0.0012. Verdict under the "
              "frozen rule: CARRIED. Pre-correction-trainer agents; this measures the "
              "direction-dependence of the estimator, not the corrected headline."),
    run("H8-HELDOUT-D045-WORLD-SAMPLES", "Drift-0.45 world-sample sensitivity on the historical "
        "hidden-8 held-out agents (frozen 2026-10-08)", experiment="L0 audit",
        artifacts=["l0_audit/d045_world_samples_l3_h8_heldout.json"], claims=[50], trains_survival=False,
        readout_of=["L3-H8-HELDOUT"], commit_at_run="865b842", folds="explicit",
        eval_seeds={"registered": "800000 / 850000",
                    "independent": "(1000000 + 100000k) / (1050000 + 100000k), k < 8"},
        device="owner's laptop CPU, one thread per readout", local_run_dir="fullruns/d045_world_samples",
        notes="scripts/run_world_sample_sensitivity.py; spec docs/specs/2026-10-08-drift-045-"
              "world-sample-sensitivity-design.md; FINDINGS 17.5.1. Diagnostic, not a gate: the "
              "nine per-draw survival means at drift 0.45 are signed and never folded. Integrity "
              "gate (registered draw reproduces the recorded explicit-partition targets within "
              "0.01 on every seed) passed, worst abs dev 0.0037. Verdict under the frozen rule: "
              "SECURE. These are pre-correction-trainer agents; this measures the estimator's "
              "sensitivity to the evaluation-world draw, not the corrected headline."),
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
    run("SURROGATE-DIAG", "Surrogate diagnostics: linear-fit control, calibration-sample and "
        "disjoint held-out error, rollout error, agent-accessible detector", experiment="methods",
        artifacts=["surrogate_diagnostics.json"],
        trains_survival=False, survival_trainer="none", folds="explicit",
        config={"runner": "scripts/run_surrogate_diagnostics.py --out <path> --qd-eps 6.0 "
                          "--calibration-sample-77 (the seed-77 per-surrogate diagnostics; "
                          "disjoint block: --qd-eps 6.0 --merge-into "
                          "artifacts/surrogate_diagnostics.json)"},
        notes="Agent free: G instances retrained deterministically from their recipe. The "
              "per-surrogate diagnostics were scored on a seed-77 sample inside every G's "
              "training episodes (the calibration sample of FINDINGS 10.1.1). The "
              "disjoint_heldout block (2026-10-10) rescored them at seeds 100000 to 100059 with "
              "each G's train column on its own seed, on the owner's workstation stack (Python "
              "3.13.2, numpy 1.26.4, torch 2.7.0+cu126); it was written only after the rerun "
              "reproduced every stored detector AUROC exactly and the stored profiles to 6.5e-8. "
              "docs/CORRECTIONS.md 2026-10-10."),
    run("QD-EPS1-CAL", "Deviation calibration of the quadratic-drag comparator at eps 1",
        experiment="methods", artifacts=["texture/qd_eps1_calibration.json"],
        trains_survival=False, survival_trainer="none", folds="explicit",
        config={"runner": "scripts/run_surrogate_diagnostics.py --out <path> --qd-eps 1.0 "
                          "--gn-sigma 0.01 --calibration-sample-77"},
        commit_at_run="82ef5d6", device="cpu",
        notes="Agent free. The matched eps of the qd comparator (6.0) was fixed by scaling the "
              "eps-1 one-step RMS deviation to gn's at sigma 0.01; that calibration had lived "
              "only in the comparator spec. Measured here: qd at eps 1 deviates by 0.002356 and "
              "gn by 0.0142 on the same 60-episode profile, a ratio of 6.02 (FINDINGS 17.9). "
              "The G rows duplicate SURROGATE-DIAG by construction (same recipe, same seeds). "
              "Its diagnostics block is the seed-77 calibration sample; the eps-1 profile value "
              "0.002356 is unaffected."),
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
        notes="scripts/validate_population_readout.py; revision step 12. The artifact records "
              "no commit and no fold scheme (it predates the git_commit field the script gained "
              "in 19b4677), and its run_dir is the revision container's path; the commit and the "
              "explicit-v1 partition above are curated from the revision record, not read from "
              "the artifact."),
    # ---------------- goal and stakes runs (docs/specs/2026-10-07-goal-and-stakes-design.md) ----------------
    run("T-TOUCH", "Pellet-touch objective, death off, registered food, n = 10 (goal-and-stakes)",
        experiment="B-v2 L3", artifacts=["goal_stakes/T-touch"], claims=[],
        survival_trainer="successor_value", status="corrected", commit_at_run="pending",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "gae_bootstrap": "successor",
                "objective": "touch", "mortal": False},
        agent_seeds=S10, surrogate=L3_H8, budget={"survival_updates": 300, "predictor_updates": 300},
        eval_seeds=BV2_EVAL_SEEDS, folds="explicit", device=CPU_REVISION,
        local_run_dir="artifacts/goal_stakes/T-touch",
        notes="Frozen: docs/specs/2026-10-07-goal-and-stakes-design.md. Cells committed from the "
              "cloud session as they land; mechanism.json from scripts/run_mechanism_readouts.py. "
              "Directory registered ahead of the run (.gitkeep placeholder until cells land)."),
    run("S-IMMORTAL", "Survival objective, death off, registered food, n = 10 (goal-and-stakes)",
        experiment="B-v2 L3", artifacts=["goal_stakes/S-immortal"], claims=[],
        survival_trainer="successor_value", status="corrected", commit_at_run="pending",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "gae_bootstrap": "successor", "mortal": False},
        agent_seeds=S10, surrogate=L3_H8, budget={"survival_updates": 300, "predictor_updates": 300},
        eval_seeds=BV2_EVAL_SEEDS, folds="explicit", device=CPU_REVISION,
        local_run_dir="artifacts/goal_stakes/S-immortal",
        notes="Frozen: docs/specs/2026-10-07-goal-and-stakes-design.md. Directory registered "
              "ahead of the run (.gitkeep placeholder until cells land)."),
    run("S-SCARCE", "Survival objective, death on, scarce food (calibrated), n = 10 (goal-and-stakes)",
        experiment="B-v2 L3", artifacts=["goal_stakes/S-scarce", "goal_stakes/calibration.json"],
        claims=[], survival_trainer="successor_value", status="corrected", commit_at_run="pending",
        config={**BV2_PROTOCOL, "drift_mode": "l3", "gae_bootstrap": "successor",
                "n_pellets": "calibrated: 24, the registered density (artifacts/goal_stakes/calibration.json)"},
        agent_seeds=S10, surrogate=L3_H8, budget={"survival_updates": 300, "predictor_updates": 300},
        eval_seeds=BV2_EVAL_SEEDS, folds="explicit", device=CPU_REVISION,
        local_run_dir="artifacts/goal_stakes/S-scarce",
        notes="Frozen: docs/specs/2026-10-07-goal-and-stakes-design.md. NOT RUN: the frozen "
              "calibration (scripts/calibrate_scarcity.py, artifacts/goal_stakes/calibration.json) "
              "chose 24 pellets, the registered density C1 already uses (scripted 80-step death "
              "rate 0.535 at drift 0, 0.515 at drift 0.45), so the run would repeat C1's "
              "configuration and the stakes rule is not testable (PREREGISTRATION_L3, 2026-10-10). "
              "The directory keeps its .gitkeep placeholder; no cell exists."),
    run("C1-MECHANISM", "Mechanism readouts 6 to 9 and the scripted-walk stream decoder, C1 agents",
        experiment="B-v2 L3", artifacts=["goal_stakes/C1"], trains_survival=False,
        readout_of=["C1"], survival_trainer="successor_value", status="corrected",
        commit_at_run="pending", folds="explicit",
        eval_seeds={"mechanism": "1400000 + i (authentic) / 1450000 + i (surrogate), i < 64"},
        device=CPU_REVISION,
        notes="scripts/run_mechanism_readouts.py; goal-and-stakes spec. Output lands as "
              "artifacts/goal_stakes/C1/mechanism.json; the directory is registered ahead of the "
              "readout (.gitkeep placeholder) and owns the file by prefix."),
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


# Commits that artifacts record but that are not ancestors of main (their branches were
# squash-merged). Each group names the branch that still holds them; keep those branches or
# tag their heads, or these provenance hashes stop resolving.
COMMITS_OFF_MAIN = [
    {"branch": "feat/remaining-research", "head": "55afc72",
     "commits": ["4d57253", "80550ca", "80948ff", "9d5d047", "d67dd51"],
     "note": "Recorded by the 2026-09 runs (second instances on GPU, the skill-matched baseline, "
             "the hidden 7 sensory echo). Their content reached main through the squash merge "
             "21ed4df (#111)."},
    {"branch": "claude/affectionate-carson-azhxt2", "head": "fb4dfa5",
     "commits": ["f676b95", "34e2c0d", "7870bba", "2606e7f"],
     "note": "Recorded by the 2026-10 corrected-trainer confirmation runs (every C1 and C2 cell "
             "records f676b95), the verdict builder and readout promotions (34e2c0d), the "
             "exploratory cross-run replay (7870bba), and the set-aside T-qd comparator run "
             "(2606e7f, FINDINGS 17.9 addendum). Their content reached main through the squash "
             "merges 6d85438 (#116) and 552be24 (#117)."},
    {"branch": "claude/colab-guard-and-checkpointing", "head": "06ecc10",
     "commits": ["06ecc10"],
     "note": "Recorded by the Experiment C per-individual rerun checkpoints (FINDINGS 13.F). Its "
             "content reached main through the squash merge 329dfb7 (#119); the tree at 06ecc10 "
             "is identical to main at 329dfb7."},
]


def _git_listed_artifacts() -> list[str] | None:
    """Files under artifacts/ as git sees them: tracked plus untracked-but-not-ignored. A
    gitignored scratch directory (artifacts/clip_audit/) then cannot fail the check on a
    developer machine, while a new artifact that is not yet added still does. None when git
    is unavailable or this is not a checkout (an extracted supplement)."""
    try:
        res = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                              "--", "artifacts"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [n.decode("utf-8", errors="replace") for n in res.stdout.split(b"\0") if n]
    return [n[len("artifacts/"):] for n in names if n.startswith("artifacts/")]


def _rel_files() -> list[str]:
    listed = _git_listed_artifacts()
    if listed is None:
        listed = [os.path.relpath(p, ART).replace(os.sep, "/")
                  for p in glob.glob(os.path.join(ART, "**", "*"), recursive=True)]
    out = []
    for rel in listed:
        if "__pycache__" in rel or not os.path.isfile(os.path.join(ART, rel)):
            continue
        if rel != "results_manifest.json" and os.path.basename(rel) != "README.md":
            out.append(rel)
    return sorted(set(out))


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


_HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
_PURE_HASH = re.compile(r"^[0-9a-f]{7,40}$")


def _run_commits(rec: dict) -> set[str]:
    """Run commits the artifacts record about themselves: git_commit and git_commit_at_run
    (plain hashes only) and the cells' commits. The promotion commit is not a run commit."""
    out: set[str] = set()
    for prov in rec.values():
        if not isinstance(prov, dict):
            continue
        for k in ("git_commit", "git_commit_at_run"):
            v = prov.get(k)
            out |= {x for x in (v if isinstance(v, list) else [v])
                    if isinstance(x, str) and _PURE_HASH.match(x)}
        out |= {c for c in (prov.get("cell_commits") or {})
                if isinstance(c, str) and _PURE_HASH.match(c)}
    return out


def commit_source(curated: str | None, recorded: set[str]) -> str | None:
    """How the curated commit_at_run relates to what the artifacts record (R125). None when
    the curated value names no commit (absent, or "pending")."""
    named = _HASH.findall(curated or "")
    if not named:
        return None
    if not recorded:
        return "curated: no artifact of this run records the commit it ran at"
    hit = [c for c in named if any(r.startswith(c) or c.startswith(r) for r in recorded)]
    if len(hit) == len(named):
        return "recorded"
    have = ", ".join(sorted(r[:7] for r in recorded))
    if hit:
        return f"partly curated: the artifacts record {have}"
    return f"curated: the artifacts record {have} instead"


def build() -> dict:
    runs = []
    for r in RUNS:
        rec = {}
        for a in r["artifacts"]:
            p = os.path.join(ART, a)
            rec[a] = _cells_summary(p) if os.path.isdir(p) else _recorded(p)
        row = {**r, "recorded_provenance": rec}
        src = commit_source(r["commit_at_run"], _run_commits(rec))
        if src is not None:
            row["commit_at_run_source"] = src
        runs.append(row)
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
        "commits_off_main": COMMITS_OFF_MAIN,
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
         "A commit marked (curated) is the registry's value with no matching run commit in the "
         "run's artifacts; the run's section says what the artifacts record instead.", "",
         "| Run | Status | Experiment | Claims | Trainer | Readout of | Commit at run | Folds | Device |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in m["runs"]:
        claims = ", ".join(str(c) for c in r["claims"]) or "-"
        ro = ", ".join(r["readout_only_on"]) if r["readout_only_on"] else "-"
        src = r.get("commit_at_run_source") or ""
        mark = (" (curated)" if src.startswith("curated") else
                " (partly curated)" if src.startswith("partly") else "")
        L.append(f"| `{r['id']}` | {r['status']} | {r['experiment']} | {claims} | {r['survival_trainer']} | "
                 f"{ro} | {r['commit_at_run'] or 'not recorded'}{mark} | {r['folds']} | "
                 f"{r['device'] or 'not recorded'} |")
    L += ["", "Commits recorded by artifacts that are not ancestors of main (squash merges); each "
          "lives only on the branch named, so keep that branch or tag its head:", ""]
    for g in m["commits_off_main"]:
        L.append(f"- `{g['branch']}` (head `{g['head']}`): "
                 f"{', '.join('`' + c + '`' for c in g['commits'])}. {g['note']}")
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
                           ("commit_at_run", "Commit at run"),
                           ("commit_at_run_source", "Commit source"), ("agent_seeds", "Agent seeds"),
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
