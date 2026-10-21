"""L3 sensory-echo control, READOUT-ONLY.

Spec: docs/specs/2026-09-26-l3-sensory-echo-control-design.md.

Reuses the saved L3 agents (default: fullruns/l3_h8_heldout); NO training. For every
(drift, seed, arm) agent the standard pools are regenerated with the SAME seed bases as
`pooled_readout` (800_000 authentic, 850_000 surrogate) so the recurrent states must
bit-match the saved dumps (integrity gate), this time also recording the normalized
observation trace the trunk received. Four readouts per cell:

  target          the headline probe on h_t (must reproduce the published aggregate)
  obs_trace_only  can the observation trace alone decode the world? (ceiling)
  resid_trace     the published per-timestep BEHAVIOR control, seven-channel basis
  resid_obs       the SENSORY control (PRIMARY): regress [x_t, x_{t-1}] out of h_t
  resid_obs_int   integrated variant with cummean(x) added (SECONDARY, over-strict when
                  the inputs carry the label; see the spec amendment)
  resid_obs_beh   instantaneous sensory + full behavior traces jointly

Frozen decision rule (drift 0.45, survival arm, n = 10): resid_obs mean >= 0.65 AND
> untrained resid_obs + 0.05 -> sensory-independent world representation;
[0.60, 0.65) -> attenuated; < 0.60 -> largely sensory-mediated.

Balanced mode (spec docs/specs/2026-10-13-balanced-sensory-echo-design.md): with
`--balanced-seed-base` both pools come from the SAME seed base, surviving pairs are kept,
and each pair shares one fold group, exactly as `l0_audit.paired_pooled_readout`. The
integrity gate then reads each cell's recorded `balanced.target` from `--cells-dir`
instead of bit-comparing state dumps. The same controls run; the aggregate adds the
floor clause (untrained resid_obs within 0.1 of 0.5) ahead of the trained-arm rule.

Fresh-base repeat (closing item G03): with `--integrity-seed-base` (the run's own balanced
base, 910000) next to `--balanced-seed-base`, each cell is rolled twice. The rollout at the
integrity base must reproduce the recorded `balanced.target`, exactly as above; every reading
comes from the rollout at the balanced (read) base. The read base must miss every registered
world-seed base (itasorl.post_review), unless it is the G03-fresh entry registered for this
repeat (2250000); 2200000, the post-review balanced pairs, is refused.

Usage:
    python scripts/audit_sensory_echo.py --agents-dir fullruns/l3_h8_heldout/agents \\
        --states-dir fullruns/l3_h8_heldout/states --out-dir fullruns/l3_h8_sensory_echo \\
        --device cuda
    ... --quick   (seed 0 only, tiny pools, no bit compare)
    python scripts/audit_sensory_echo.py --agents-dir fullruns/balanced_confirmation_wm/agents \\
        --cells-dir fullruns/balanced_confirmation_wm/cells --balanced-seed-base 910000 \\
        --out-dir fullruns/balanced_confirmation_wm/sensory_echo --drifts 0.45 --device cuda
    ... --balanced-seed-base 2250000 --integrity-seed-base 910000   (G03 fresh-base repeat)
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import platform
import re
import sys
import time

import numpy as np
import scipy
import sklearn
import torch

import itasorl.experiment_b2 as b2
from itasorl import folds
from itasorl.behavior_audit import (sensory_residual_probe_auroc,
                                    trace_residual_probe_auroc)
from itasorl.experiment_a import grouped_auroc
from itasorl.experiment_b import episode_features, episode_features_full
from itasorl.experiment_b2 import collect_pool, default_device, load_agent_bundle
from itasorl.stats import mean_ci, t_ci90
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)   # frozen organism world
# drift-0.45 survival reference per capacity: itasorl.folds, keyed by fold scheme
BAR = 0.65
MARGIN = 0.05
FLOOR_DEV = 0.1          # balanced floor clause: untrained resid_obs within 0.1 of 0.5
TARGET_TOL = 5e-4        # balanced integrity: regenerated target reproduces the recorded one
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")
METRICS = ("target", "obs_trace_only", "resid_trace", "resid_obs", "resid_obs_int", "resid_obs_beh",
           "resid_obs_mlp", "resid_obs_beh_mlp", "resid_obs_int_mlp")
# Decision metrics. A partial or non-finite value here is not a smaller sample.
REQUIRED_METRICS = ("target", "resid_obs")
# Frozen balanced-control population: drift 0.45, seeds 10-19, three arms, 110 pairs per
# cell and no dropped pair (spec 2026-10-13). A cell that reports anything else is not a
# cell of this protocol (2026-10-14 review, C02).
BALANCED_PAIRS = 110
BALANCED_SPEC_KEYS = {
    ("0.45", seed, arm)
    for seed in range(10, 20)
    for arm in ("survival", "predictor", "untrained")
}


def balanced_spec_keys(seeds) -> set:
    """The balanced population for `seeds`: drift 0.45, three arms. Seeds 10 to 19 give
    BALANCED_SPEC_KEYS (spec 2026-10-13); seeds 20 to 29 the prospective population of
    spec 2026-10-20."""
    return {("0.45", int(seed), arm) for seed in seeds
            for arm in ("survival", "predictor", "untrained")}


def stack_record(device: str) -> dict:
    """The stack a balanced readout ran on; recorded beside the cells (same block as
    run_expB2.py writes into every training cell)."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "device": device}


def run_provenance(a, device: str) -> dict:
    """The aggregate's provenance block (closing item G46): commit, tracked changes, stack,
    agents dir and argv. A readout of saved agents, so the commit key is
    git_commit_at_promotion."""
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    out = repo_rel(os.path.join(a.out_dir, "aggregate.json"))
    return {**provenance("scripts/audit_sensory_echo.py", commit_key="git_commit_at_promotion",
                         output=out, device=device),
            "agents_dir": repo_rel(a.agents_dir), "argv": rel_flags(list(sys.argv[1:]))}


def cfg():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--states-dir", help="saved state dumps for the pooled integrity gate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--balanced-seed-base", type=int, default=None,
                    help="balanced mode: one seed base for both pools, pairs grouped in CV")
    ap.add_argument("--cells-dir",
                    help="balanced mode: run_expB2 cells carrying the recorded balanced.target")
    ap.add_argument("--integrity-seed-base", type=int, default=None,
                    help="balanced mode: roll each cell at this base too (the run's own pairs, "
                         "910000) for the recorded-target integrity gate, and read every "
                         "metric from --balanced-seed-base (G03 fresh-base repeat)")
    ap.add_argument("--objective", choices=("survival", "touch"), default="survival",
                    help="the objective the agents were trained under (sets the reward the "
                         "pools accumulate and the engagement world)")
    ap.add_argument("--mortal", choices=("on", "off"), default="on",
                    help="death in the evaluation worlds, as in the run being read")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--g-seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["survival", "untrained", "predictor"])
    ap.add_argument("--drifts", type=float, nargs="+", default=[0.45, 0.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)),
                    help="balanced population (default 10 to 19, spec 2026-10-13; 20 to 29 is "
                         "the prospective confirmation of spec 2026-10-20)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--no-abort", action="store_true",
                    help="record an integrity mismatch instead of aborting (diagnostic only)")
    ap.add_argument("--quick", action="store_true", help="seed 0 only, tiny pools, no bit compare")
    ap.add_argument("--reaggregate", action="store_true",
                    help="skip collection; rebuild aggregate.json from the saved cells.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present in --out-dir/cells.json (after an interrupted run)")
    ap.add_argument("--nonlinear", action="store_true",
                    help="also run the MLP residualizers resid_obs_mlp and resid_obs_beh_mlp "
                         "(spec 2026-09-27-local-strengthening-probes, probe B)")
    ap.add_argument("--save-traces", action="store_true",
                    help="save the regenerated H, observation and behavior traces per cell "
                         "as traces_d<drift>_s<seed>_<arm>.npz under --out-dir")
    return ap.parse_args()


def fmt_drift(d: float) -> str:
    return f"{d:.2f}"


def _is_finite(value) -> bool:
    """True only for a real number. NaN fails this; abs(NaN - x) > tol does not."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def cell_key(cell: dict) -> tuple[str, int, str] | None:
    try:
        return (str(cell["drift"]), int(cell["seed"]), str(cell["agent"]))
    except (KeyError, TypeError, ValueError):
        return None


def declared_cell_keys(agents_dir: str, drifts, arms, quick: bool) -> set[tuple[str, int, str]]:
    """The seed, arm and drift set the agent files say this run was supposed to score."""
    keys: set[tuple[str, int, str]] = set()
    if not agents_dir or not os.path.isdir(agents_dir):
        return keys
    drift_ok = {fmt_drift(d) for d in drifts}
    for name in os.listdir(agents_dir):
        m = AGENT_RE.search(name)
        if not m:
            continue
        drift, seed, arm = fmt_drift(float(m.group(1))), int(m.group(2)), m.group(3)
        if arm not in arms or drift not in drift_ok or (quick and seed != 0):
            continue
        keys.add((drift, seed, arm))
    return keys


def adjudication_blockers(cells: list[dict], *, quick: bool, mismatches: list[str],
                          balanced: bool, expected_keys: set | None) -> list[str]:
    """Reasons a positive verdict is not available. Diagnostic scores may still be kept.

    A pass needs a declared cell set, unique keys, finite required metrics, checked
    integrity, and, in balanced mode, the pair bookkeeping. Quick and no-abort
    mismatch paths are diagnostic only.
    """
    problems: list[str] = []
    if quick:
        problems.append("quick or diagnostic run")
    if mismatches:
        problems.append("integrity mismatch")
    keys: list[tuple[str, int, str]] = []
    for cell in cells:
        key = cell_key(cell)
        if key is None:
            problems.append("cell missing drift, seed, or arm")
            continue
        keys.append(key)
        where = f"{key[0]} s{key[1]} {key[2]}"
        if any(not _is_finite(cell.get(met)) for met in REQUIRED_METRICS):
            problems.append(f"non-finite target or resid_obs at {where}")
        match = cell.get("integrity_match")
        if match is None:
            problems.append(f"integrity unchecked at {where}")
        elif not bool(match):
            problems.append(f"integrity_match false at {where}")
        if balanced:
            if cell.get("n_pairs") != BALANCED_PAIRS:
                problems.append(f"n_pairs is not {BALANCED_PAIRS} at {where}")
            dropped = cell.get("dropped_pairs")
            if cell.get("identical_first_state") is not True or dropped != 0:
                problems.append(f"pair bookkeeping failed at {where}")
            # A fresh-base repeat (G03) checks the rollout at the integrity base against the
            # record; its own pair bookkeeping must be clean too.
            itarget = cell["integrity_target"] if "integrity_target" in cell else cell.get("target")
            if "integrity_target" in cell and (
                    cell.get("integrity_n_pairs") != BALANCED_PAIRS
                    or cell.get("integrity_dropped_pairs") != 0
                    or cell.get("integrity_identical_first_state") is not True):
                problems.append(f"integrity pair bookkeeping failed at {where}")
            if not _is_finite(cell.get("recorded_target")) or not _is_finite(itarget):
                problems.append(f"non-finite recorded target at {where}")
            else:
                # The gap is recomputed from the two targets; a cached gap is checked
                # against it, never trusted on its own (2026-10-14 review, C02).
                gap = abs(float(itarget) - float(cell["recorded_target"]))
                cached = cell.get("target_abs_diff")
                if _is_finite(cached) and abs(float(cached) - gap) > 1e-9:
                    problems.append(f"cached target gap disagrees with the targets at {where}")
                if gap > TARGET_TOL:
                    problems.append(f"target gap above tolerance at {where}")
    if len(keys) != len(set(keys)):
        problems.append("duplicate cell key")
    if expected_keys is None:
        problems.append("declared cell set missing")
    elif set(keys) != set(expected_keys):
        problems.append("cell set does not match the declared seed, arm, and drift set")
    seen: list[str] = []
    for problem in problems:
        if problem not in seen:
            seen.append(problem)
    return seen


def _withhold(decision: dict | None, blockers: list[str]) -> dict | None:
    """Keep diagnostic fields, and refuse every pass flag when a blocker fired."""
    if not blockers:
        return decision
    kept = dict(decision or {})
    kept["pass_bar"] = False
    kept["pass_margin"] = False
    kept["t90_lo_above_untrained"] = False
    kept["zone"] = "WITHHELD: " + "; ".join(blockers)
    return kept


def collect_cell(agent, norm, drift: float, *, n_eps: int, steps: int, device: str,
                 balanced_seed_base: int | None):
    """Regenerate one cell's pools. Pooled mode draws authentic and surrogate pools from
    the historical bases (800_000 / 850_000, independent episodes, one group each).
    Balanced mode draws both from `balanced_seed_base`, keeps the pairs that survive in
    both worlds and gives each pair one group id, exactly as
    `l0_audit.paired_pooled_readout`. Returns (H, Ot, Bt, y, groups, meta)."""
    kw = dict(return_anchors=True, return_obs=True, return_index=True)
    base_a, base_s = (800_000, 850_000) if balanced_seed_base is None else (balanced_seed_base,) * 2
    Ha, _, _, _, _, _, bta, Oa, ia = collect_pool(agent, norm, P, 0.0, n_eps, steps, device, base_a, 5, **kw)
    Hs, _, _, _, _, _, bts, Os, is_ = collect_pool(agent, norm, P, drift, n_eps, steps, device, base_s, 5, **kw)
    meta = {"n_auth": int(len(Ha)), "n_surr": int(len(Hs))}
    if balanced_seed_base is None:
        groups = np.arange(len(Ha) + len(Hs))
    else:
        common = np.intersect1d(ia, is_)
        sa, ss = np.searchsorted(ia, common), np.searchsorted(is_, common)
        Ha, bta, Oa = Ha[sa], bta[sa], Oa[sa]
        Hs, bts, Os = Hs[ss], bts[ss], Os[ss]
        groups = np.r_[common, common]
        meta.update(n_pairs=int(len(common)), dropped_pairs=int(n_eps - len(common)),
                    identical_first_state=bool(len(common) and np.array_equal(Ha[:, 0], Hs[:, 0])))
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    return (np.concatenate([Ha, Hs]), np.concatenate([Oa, Os]), np.concatenate([bta, bts]),
            y, groups, meta, (Ha, Hs))


def balanced_integrity(cells_dir: str, drift: float, seed: int, arm: str, cell: dict) -> tuple[bool, str]:
    """Balanced-mode gate: the regenerated target must reproduce the run's recorded
    `balanced.target` to TARGET_TOL, the first states must be identical and no pair
    dropped. Returns (ok, message); the caller records the recorded value and the gap."""
    path = os.path.join(cells_dir, f"cell_d{fmt_drift(drift)}_s{seed}.json")
    with open(path, encoding="utf-8") as fh:
        recorded = json.load(fh)["cell"]["agents"][arm]["balanced"]
    problems = []
    recorded_target = recorded.get("target")
    regenerated = cell.get("target")
    if not _is_finite(recorded_target) or not _is_finite(regenerated):
        problems.append("non-finite target")
        cell["recorded_target"] = recorded_target
        cell["target_abs_diff"] = None
    else:
        cell["recorded_target"] = float(recorded_target)
        cell["target_abs_diff"] = abs(float(regenerated) - float(recorded_target))
        if cell["target_abs_diff"] > TARGET_TOL:
            problems.append(
                f"target {cell['target']:.6f} vs recorded {cell['recorded_target']:.6f}")
    if not cell.get("identical_first_state"):
        problems.append("first states differ")
    if cell.get("dropped_pairs"):
        problems.append(f"{cell['dropped_pairs']} dropped pairs")
    if int(recorded["n_pairs"]) != cell.get("n_pairs"):
        problems.append(f"n_pairs {cell.get('n_pairs')} vs recorded {recorded['n_pairs']}")
    return (not problems), "; ".join(problems)


def read_base_collisions(read_base: int, n_eps: int) -> list[tuple[str, str]]:
    """Registered world-seed bases the G03 read pool would share draws with. Empty when the
    read base is fresh, or when it is the G03-fresh base registered for this repeat."""
    from itasorl import post_review
    _, registered, _ = post_review.registered_later("G03-fresh")
    hits = post_review.base_collisions(new=(("G03 read", int(read_base), int(n_eps)),),
                                       in_use=post_review.all_bases_in_use())
    return [] if int(read_base) == registered else hits


def integrity_rollout(agent, norm, drift: float, *, n_eps: int, steps: int, device: str,
                      integrity_seed_base: int) -> dict:
    """The rollout at the run's own balanced base, which the recorded target was scored on:
    its pair bookkeeping and its registered-probe target, for `balanced_integrity`."""
    H, _, _, y, groups, meta, _ = collect_cell(agent, norm, drift, n_eps=n_eps, steps=steps,
                                               device=device, balanced_seed_base=integrity_seed_base)
    return {**meta, "target": grouped_auroc(episode_features(H), y, groups)}


def integrity_fields(icell: dict) -> dict:
    """The integrity rollout's values a fresh-base cell carries beside its own readings."""
    return {"integrity_target": icell["target"], "recorded_target": icell.get("recorded_target"),
            "target_abs_diff": icell.get("target_abs_diff"),
            "integrity_n_pairs": icell.get("n_pairs"),
            "integrity_dropped_pairs": icell.get("dropped_pairs"),
            "integrity_identical_first_state": icell.get("identical_first_state")}


def balanced_decision(agg: dict, trained_arm: str = "survival") -> dict:
    """Floor clause then trained clause, as frozen in the 2026-10-13 spec. The trained
    zone is read only when the floor is explained."""
    untr = agg.get("d=0.45 untrained", {}).get("resid_obs")
    trained = agg.get(f"d=0.45 {trained_arm}", {}).get("resid_obs")
    out: dict = {"floor_dev": FLOOR_DEV}
    if not untr or "mean" not in untr:
        return out
    dev = abs(untr["mean"] - 0.5)
    out.update(untrained_resid_obs=untr["mean"], untrained_dev=dev,
               floor_explained=bool(dev <= FLOOR_DEV))
    if not trained or "mean" not in trained:
        return out
    s, u = trained["mean"], untr["mean"]
    out.update(trained_resid_obs=s, trained_t90=trained["t90"], pass_bar=bool(s >= BAR),
               pass_margin=bool(s > u + MARGIN), t90_lo_above_untrained=bool(trained["t90"][0] > u))
    if not out["floor_explained"]:
        out["zone"] = "FLOOR NOT EXPLAINED: trained clause not read"
    elif out["pass_bar"] and out["pass_margin"] and out["t90_lo_above_untrained"]:
        out["zone"] = "SENSORY-INDEPENDENT under matched worlds (rule passes)"
    elif s >= 0.60:
        out["zone"] = "ATTENUATED: below-bar trace survives the sensory control"
    else:
        out["zone"] = "LARGELY SENSORY-MEDIATED"
    return out


def _arm_complete(rows: list[dict], expected_keys: set | None, drift: str, arm: str) -> bool:
    if expected_keys is None:
        return False
    wanted = {key for key in expected_keys if key[0] == drift and key[2] == arm}
    found = [cell_key(row) for row in rows]
    return bool(wanted) and set(found) == wanted and len(found) == len(set(found))


def _metric_block(rows: list[dict], met: str, *, complete: bool) -> dict | None:
    """A decision-ready mean exists only when every row has a finite value.

    A missing or non-finite row is not dropped from the sample. The block then
    keeps the diagnostic values and carries no mean.
    """
    if any(met not in row for row in rows):
        return None
    present = [row[met] for row in rows]
    finite = [float(value) for value in present if _is_finite(value)]
    if len(finite) != len(rows) or not complete:
        return {"per_seed": [float(value) if _is_finite(value) else None for value in present],
                "n_seeds": len(finite), "n_rows": len(rows), "complete": False}
    mean, lo, hi = mean_ci(finite)
    tlo, thi = t_ci90(finite) if len(finite) > 1 else (float("nan"), float("nan"))
    return {"per_seed": finite, "mean": float(mean),
            "boot90": [float(lo), float(hi)],
            "t90": [float(tlo), float(thi)], "n_seeds": len(finite),
            "n_ge_065": int(sum(value >= BAR for value in finite))}


def _seal(agg: dict, cells: list[dict], *, quick: bool, mismatches: list[str],
          balanced: bool, expected_keys: set | None) -> dict:
    blockers = adjudication_blockers(cells, quick=quick, mismatches=mismatches,
                                     balanced=balanced, expected_keys=expected_keys)
    agg["integrity"]["adjudication_blockers"] = blockers
    # Global integrity is derived from the per-cell evidence, not asserted beside it.
    evidence = ("integrity", "n_pairs", "pair bookkeeping", "non-finite recorded", "target gap",
                "cached target gap")
    unchecked = any(b.startswith("integrity unchecked") for b in blockers)
    agg["integrity"]["checked"] = bool(not quick and not unchecked)
    agg["integrity"]["all_match"] = (None if quick else bool(
        not mismatches and not any(b.startswith(evidence) for b in blockers)))
    if expected_keys is not None:
        agg["declared_keys"] = [list(key) for key in sorted(expected_keys)]
    key = "decision_balanced" if balanced else "decision"
    sealed = _withhold(agg.get(key), blockers)
    if sealed is not None:
        agg[key] = sealed
    if not balanced and "decision_nonlinear" in agg:
        agg["decision_nonlinear"] = _withhold(agg["decision_nonlinear"], blockers)
    return agg


def aggregate(cells: list[dict], drifts, arms, *, hidden: int, g_seed: int, n_eps: int,
              steps: int, quick: bool, device: str, mismatches: list[str],
              balanced_seed_base: int | None = None, objective: str = "survival",
              mortal: bool = True, expected_keys: set | None = None,
              integrity_seed_base: int | None = None) -> dict:
    """Across-seed aggregation plus the frozen decision rule. Every value is a plain
    Python float/bool/list so the result is JSON-serializable.

    A positive zone requires `expected_keys` and a cell list that matches it, with
    finite required metrics and clean integrity. Otherwise the zone is WITHHELD.
    `integrity_seed_base` (G03) records the base the integrity gate read beside the base
    the metrics were read from; without it the aggregate gains no key.
    """
    balanced = balanced_seed_base is not None
    agg: dict = {"hidden": hidden, "g_seed": g_seed, "n_eps": n_eps, "steps": steps,
                 "quick": quick, "device": device, "bar": BAR, "margin": MARGIN,
                 "mode": "pooled" if not balanced else "balanced",
                 "balanced_seed_base": balanced_seed_base, "objective": objective,
                 "mortal": bool(mortal), "stack": stack_record(device),
                 "integrity": {"checked": not quick, "mismatches": list(mismatches),
                               "all_match": (not mismatches) if not quick else None}}
    if integrity_seed_base is not None:
        agg["integrity_seed_base"] = integrity_seed_base
        agg["read_seed_base"] = balanced_seed_base
    for d in drifts:
        drift = fmt_drift(d)
        for arm in arms:
            rows = [c for c in cells if c["drift"] == drift and c["agent"] == arm]
            if not rows:
                continue
            key = f"d={drift} {arm}"
            agg[key] = {}
            arm_ok = _arm_complete(rows, expected_keys, drift, arm)
            for met in METRICS:
                block = _metric_block(rows, met, complete=arm_ok)
                if block is not None:
                    agg[key][met] = block
    surv = agg.get("d=0.45 survival", {})
    untr = agg.get("d=0.45 untrained", {})
    if balanced_seed_base is not None:
        # The balanced integrity gate is per cell (recorded target); the pooled published
        # reference and the pooled rule do not apply. One decision block, the frozen
        # floor-then-trained rule.
        agg["integrity"]["fold_scheme"] = folds.current_scheme()
        agg["decision_balanced"] = balanced_decision(agg)
        return _seal(agg, cells, quick=quick, mismatches=mismatches, balanced=True,
                     expected_keys=expected_keys)
    ref = folds.reference_survival_target(hidden, strict=False)
    if "target" in surv and "mean" in surv["target"] and not quick:
        agg["integrity"]["survival_target_mean"] = round(surv["target"]["mean"], 3)
        agg["integrity"]["fold_scheme"] = folds.current_scheme()
        agg["integrity"]["published_target"] = ref
        agg["integrity"]["target_reproduced"] = (None if ref is None else bool(
            abs(surv["target"]["mean"] - ref) < 5e-4))
    if ("resid_obs" in surv and "resid_obs" in untr
            and "mean" in surv["resid_obs"] and "mean" in untr["resid_obs"]):
        s, u = surv["resid_obs"]["mean"], untr["resid_obs"]["mean"]
        rule = {"survival_resid_obs": s, "untrained_resid_obs": u,
                "survival_t90": surv["resid_obs"]["t90"],
                "pass_bar": bool(s >= BAR), "pass_margin": bool(s > u + MARGIN),
                "t90_lo_above_untrained": bool(surv["resid_obs"]["t90"][0] > u)}
        if s >= BAR and s > u + MARGIN:
            rule["zone"] = "SENSORY-INDEPENDENT world representation (rule passes)"
        elif s >= 0.60:
            rule["zone"] = "ATTENUATED: below-bar trace survives the sensory control"
        else:
            rule["zone"] = "LARGELY SENSORY-MEDIATED"
        agg["decision"] = rule
    if ("resid_obs_beh_mlp" in surv and "resid_obs_beh_mlp" in untr
            and "mean" in surv["resid_obs_beh_mlp"] and "mean" in untr["resid_obs_beh_mlp"]):
        s, u = surv["resid_obs_beh_mlp"]["mean"], untr["resid_obs_beh_mlp"]["mean"]
        nl = {"survival_resid_obs_beh_mlp": s, "untrained_resid_obs_beh_mlp": u,
              "survival_t90": surv["resid_obs_beh_mlp"]["t90"],
              "pass_bar": bool(s >= BAR), "pass_margin": bool(s > u + MARGIN)}
        if s >= BAR and s > u + MARGIN:
            nl["zone"] = "SURVIVES the nonlinear joint control (rule passes)"
        elif s >= 0.60:
            nl["zone"] = "ATTENUATED under the nonlinear joint control"
        else:
            nl["zone"] = "LARGELY EXPLAINED by nonlinear mirroring of inputs and behavior"
        agg["decision_nonlinear"] = nl
    return _seal(agg, cells, quick=quick, mismatches=mismatches, balanced=False,
                 expected_keys=expected_keys)


def main() -> int:
    a = cfg()
    dev = default_device() if a.device == "auto" else a.device
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)
    balanced = a.balanced_seed_base is not None
    if balanced and not a.quick and not a.cells_dir:
        raise SystemExit("balanced mode needs --cells-dir for the recorded-target integrity gate")
    if not balanced and not a.quick and not a.states_dir:
        raise SystemExit("pooled mode needs --states-dir for the bit-match integrity gate")
    fresh = a.integrity_seed_base is not None
    if fresh and not balanced:
        raise SystemExit("--integrity-seed-base needs --balanced-seed-base (the read base)")
    if fresh:
        hits = read_base_collisions(a.balanced_seed_base, n_eps)
        if hits:
            raise SystemExit(f"read base {a.balanced_seed_base} shares world draws with registered "
                             f"bases {hits}; use the G03-fresh base or a fresh one")
    os.makedirs(a.out_dir, exist_ok=True)
    cells, mismatches = [], []
    t0 = time.time()
    done: set = set()
    if a.resume and os.path.exists(os.path.join(a.out_dir, "cells.json")):
        with open(os.path.join(a.out_dir, "cells.json"), encoding="utf-8") as fh:
            cells = json.load(fh)
        done = {(c["drift"], c["seed"], c["agent"]) for c in cells}
        print(f"resuming: {len(done)} cells already in {a.out_dir}/cells.json")
    if a.reaggregate:
        with open(os.path.join(a.out_dir, "cells.json"), encoding="utf-8") as fh:
            cells = json.load(fh)
        mismatches = [f"{c['drift']}_s{c['seed']}_{c['agent']}" for c in cells
                      if c.get("integrity_match") is False]
        names = []
        print(f"re-aggregating {len(cells)} saved cells from {a.out_dir}/cells.json")
    else:
        b2.DRIFT_MODE = "l3"
        b2.OBJECTIVE = a.objective
        b2.MORTAL = a.mortal == "on"
        b2.setup_l3_surrogate(hidden=a.hidden, device=dev, seed=a.g_seed, params=P)
        print(f"sensory-echo control  device={dev} hidden={a.hidden} g_seed={a.g_seed} "
              f"n_eps={n_eps} steps={steps} quick={a.quick} objective={a.objective} "
              f"mortal={a.mortal} mode={'balanced@' + str(a.balanced_seed_base) if balanced else 'pooled'}")
        names = sorted(n for n in os.listdir(a.agents_dir) if AGENT_RE.search(n))
    for name in names:
        m = AGENT_RE.search(name)
        drift, seed, arm = float(m.group(1)), int(m.group(2)), m.group(3)
        if arm not in a.arms or drift not in a.drifts or (a.quick and seed != 0):
            continue
        if (fmt_drift(drift), seed, arm) in done:
            continue
        agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
        H, Ot, Bt, y, groups, meta, (Ha, Hs) = collect_cell(
            agent, norm, drift, n_eps=n_eps, steps=steps, device=dev,
            balanced_seed_base=a.balanced_seed_base)
        integrity = None
        if not a.quick and not balanced:
            saved = os.path.join(a.states_dir, f"states_d{fmt_drift(drift)}_s{seed}_{arm}.npz")
            if os.path.exists(saved):
                z = np.load(saved)
                integrity = bool(np.array_equal(z["Ha"], Ha) and np.array_equal(z["Hs"], Hs))
                if not integrity:
                    mismatches.append(name)
                    msg = f"INTEGRITY MISMATCH: regenerated pools differ from {saved}"
                    if not a.no_abort:
                        raise SystemExit(msg)
                    print("  " + msg)
        if len(Ha) < 5 or len(Hs) < 5:
            print(f"  {name}: too few survivors ({len(Ha)}/{len(Hs)}), skipped")
            continue
        cell = {"drift": fmt_drift(drift), "seed": seed, "agent": arm, **meta,
                "integrity_match": integrity,
                "target": grouped_auroc(episode_features(H), y, groups),
                "obs_trace_only": grouped_auroc(episode_features_full(Ot), y, groups),
                "resid_trace": trace_residual_probe_auroc(H, Bt, y, groups=groups),
                "resid_obs": sensory_residual_probe_auroc(H, Ot, y, groups=groups),
                "resid_obs_int": sensory_residual_probe_auroc(H, Ot, y, groups=groups, integrated=True),
                "resid_obs_beh": sensory_residual_probe_auroc(H, Ot, y, Bt=Bt, groups=groups)}
        if balanced and not a.quick:
            if fresh:
                icell = integrity_rollout(agent, norm, drift, n_eps=n_eps, steps=steps, device=dev,
                                          integrity_seed_base=a.integrity_seed_base)
                ok, why = balanced_integrity(a.cells_dir, drift, seed, arm, icell)
                cell.update(integrity_fields(icell))
            else:
                ok, why = balanced_integrity(a.cells_dir, drift, seed, arm, cell)
            cell["integrity_match"] = integrity = ok
            if not ok:
                mismatches.append(name)
                msg = f"INTEGRITY MISMATCH ({name}): {why}"
                if not a.no_abort:
                    raise SystemExit(msg)
                print("  " + msg)
        if a.nonlinear:
            cell["resid_obs_mlp"] = sensory_residual_probe_auroc(H, Ot, y, groups=groups, nonlinear=True)
            cell["resid_obs_beh_mlp"] = sensory_residual_probe_auroc(H, Ot, y, Bt=Bt, groups=groups,
                                                                     nonlinear=True)
            # Integrated basis through the MLP regressor (spec 2026-10-20, S2).
            cell["resid_obs_int_mlp"] = sensory_residual_probe_auroc(H, Ot, y, groups=groups,
                                                                     integrated=True, nonlinear=True)
        if a.save_traces:
            k = len(Ha)
            np.savez_compressed(os.path.join(a.out_dir, f"traces_d{fmt_drift(drift)}_s{seed}_{arm}.npz"),
                                Ha=Ha, Hs=Hs, Oa=Ot[:k], Os=Ot[k:], bta=Bt[:k], bts=Bt[k:])
        cells.append(cell)
        print(f"  d={cell['drift']} s={seed} {arm:9s} target={cell['target']:.3f} "
              f"obs_only={cell['obs_trace_only']:.3f} resid_trace={cell['resid_trace']:.3f} "
              f"resid_obs={cell['resid_obs']:.3f} resid_obs_int={cell['resid_obs_int']:.3f} "
              f"resid_obs_beh={cell['resid_obs_beh']:.3f} "
              f"integrity={integrity}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        with open(os.path.join(a.out_dir, "cells.json"), "w", encoding="utf-8") as fh:
            json.dump(cells, fh, indent=1)

    if balanced and not a.quick:
        expected = balanced_spec_keys(a.seeds)
    else:
        expected = declared_cell_keys(a.agents_dir, a.drifts, a.arms, a.quick)
    agg = aggregate(cells, a.drifts, a.arms, hidden=a.hidden, g_seed=a.g_seed, n_eps=n_eps,
                    steps=steps, quick=a.quick, device=dev, mismatches=mismatches,
                    balanced_seed_base=a.balanced_seed_base, objective=a.objective,
                    mortal=a.mortal == "on", expected_keys=expected,
                    integrity_seed_base=a.integrity_seed_base)
    agg["provenance"] = run_provenance(a, dev)
    if "decision_balanced" in agg:
        print("\nDECISION (balanced, frozen 2026-10-13 rule, drift 0.45):",
              json.dumps(agg["decision_balanced"], indent=1))
    elif "decision" in agg:
        print("\nDECISION (frozen rule, drift 0.45):", json.dumps(agg["decision"], indent=1))
    with open(os.path.join(a.out_dir, "aggregate.json"), "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=1)
    print(f"wrote {a.out_dir}/aggregate.json  ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
