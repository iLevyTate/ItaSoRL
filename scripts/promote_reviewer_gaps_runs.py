"""Promote the cloud reviewer-gap organism runs to committed summary artifacts.

The cloud chain (scripts/reviewer_gaps/run_cloud_chain.sh, run_cloud_device_control.sh)
commits its raw outputs under artifacts/reviewer_gaps_runs/<run>/ (expB2_results.json,
behavior_audit.json, cells/). This script computes the decision-relevant aggregates
from those per-seed values, exactly as the FINDINGS text quotes them, and writes one
summary JSON per run under artifacts/expB2/ with provenance:

  * per (drift, arm): per-seed pooled targets, mean, t-based 90% CI, seed-bootstrap
    90% CI, t-based 95% CI, seeds >= 0.65, speed / reward-leak / death means
  * gates: engagement pass count per drift (from cells/), L0 TOST and ROPE on the
    survival drift-0 targets, pooled leakage clean count, untrained floor
  * behavior audit aggregates (resid_trace etc.) with t-based 90% CI recomputed from
    the audit cells
  * gate-0 calibration rows when a calibration.json is given
  * the frozen decision rule for the run (spec named in --spec)
  * with --device-control-against <no-auxiliary run dir>, the device-control rule
    frozen in the 2026-09-27 addendum to the architecture-baseline spec: the
    decoder-carrying survival target on the same device against the no-auxiliary one

Usage:
    python scripts/promote_reviewer_gaps_runs.py --run artifacts/reviewer_gaps_runs/l3_h8_nowm \
        --out artifacts/expB2/arch_baseline_l3_h8_nowm.json --spec <spec path> \
        --label "architecture baseline (no world-model auxiliary)" [--calibration <json>]
    python scripts/promote_reviewer_gaps_runs.py --run artifacts/reviewer_gaps_runs/l3_h8_wm_cpu \
        --out artifacts/expB2/device_control_l3_h8_wm_cpu.json --spec <spec path> \
        --label "device control" --device-control-against artifacts/reviewer_gaps_runs/l3_h8_nowm
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import dataclasses
import glob
import json
import os
import subprocess

import numpy as np

from itasorl.stats import equivalence_test, mean_ci, rope_test, t_ci90

BAR = 0.65
MARGIN = 0.05
ARMS = ("untrained", "predictor", "survival")


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def t_ci95(vals) -> list[float]:
    from scipy import stats
    v = np.asarray(vals, float)
    n = len(v)
    if n < 2:
        return [float("nan"), float("nan")]
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / n ** 0.5
    return [float(v.mean() - h), float(v.mean() + h)]


def summarize_vals(vals) -> dict:
    vals = [float(x) for x in vals]
    mean, lo, hi = mean_ci(vals)
    tlo, thi = t_ci90(vals) if len(vals) > 1 else (float("nan"), float("nan"))
    return {"per_seed": vals, "mean": float(mean), "boot90": [float(lo), float(hi)],
            "t90": [float(tlo), float(thi)], "t95": t_ci95(vals), "n_seeds": len(vals),
            "n_ge_065": int(sum(v >= BAR for v in vals))}


def _nanmean(xs) -> float:
    try:
        return float(np.nanmean(np.asarray(xs, float)))
    except Exception:
        return float("nan")


def device_control(res: dict, compare_res: dict, dmax: str) -> dict:
    """Frozen device-control rule (addendum 2026-09-27, architecture-baseline spec).

    `res` is the decoder-carrying run and `compare_res` the no-auxiliary run, both on
    the same device and seeds. Seeds pair by index, so the paired lead is reported
    alongside the difference of means the rule is written in."""
    wm = [float(x) for x in res[dmax]["survival"]["pool_target"]]
    nowm = [float(x) for x in compare_res[dmax]["survival"]["pool_target"]]
    wm_mean, nowm_mean = float(np.mean(wm)), float(np.mean(nowm))
    wm_t90 = [float(x) for x in t_ci90(wm)]
    lead = wm_mean - nowm_mean
    paired = [a - b for a, b in zip(wm, nowm)] if len(wm) == len(nowm) else []
    paired_t90 = [float(x) for x in t_ci90(paired)] if len(paired) > 1 else [float("nan")] * 2
    if wm_mean < BAR:
        verdict = "DEVICE CONFOUND (verdict withdrawn to not established)"
    elif wm_t90[0] > BAR and lead > MARGIN:
        verdict = "DEVICE NOT THE CAUSE (auxiliary-conditional verdict stands)"
    elif lead <= MARGIN:
        verdict = "INTERMEDIATE (auxiliary contribution not demonstrated on this device)"
    else:
        verdict = "NOT COVERED BY THE FROZEN RULE (mean clears the bar, t-CI does not)"
    pred_same = (res[dmax]["predictor"]["pool_target"] == compare_res[dmax]["predictor"]["pool_target"])
    return {"decoder_survival": wm_mean, "decoder_survival_t90": wm_t90,
            "no_auxiliary_survival": nowm_mean, "lead": lead,
            "paired_lead": float(np.mean(paired)) if paired else float("nan"),
            "paired_lead_t90": paired_t90,
            "pass_bar": wm_mean >= BAR, "t90_excludes_bar": wm_t90[0] > BAR,
            "pass_lead": lead > MARGIN,
            "predictor_arm_identical_to_comparison": bool(pred_same),
            "verdict": verdict}


def skill_match(run_dir: str, res: dict, dmax: str, reference: float, tol: float) -> dict:
    """Frozen skill-match rule (2026-09-29 skill-matched-baseline spec, amended 2026-09-30).

    R is the survival arm's mean eval@dmax return over the stage-2 seeds, read from the
    per-cell `xeval` block. The amendment makes the clause asymmetric: an overshoot is
    not "matched", and what it implies depends on where the primary probe lands. The
    four rows below are the frozen table, reproduced in order."""
    rets = []
    for cp in sorted(glob.glob(os.path.join(run_dir, "cells", "cell_*.json"))):
        with open(cp, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        if f"{float(c['drift']):.2f}" != dmax or dmax not in c.get("xeval", {}):
            continue
        rets.append(float(c["xeval"][dmax]))
    if not rets:
        return {"error": "no xeval returns at dmax in cells/"}
    r = float(np.mean(rets))
    lo, hi = reference - tol, reference + tol
    surv = [float(x) for x in res[dmax]["survival"]["pool_target"]]
    surv_mean = float(np.mean(surv))
    surv_t90 = [float(x) for x in t_ci90(surv)]
    # "Reads below the bar" = mean under the bar, or a t90 that does not exclude it.
    probe_below_bar = surv_mean < BAR or surv_t90[0] <= BAR
    if r < lo:
        verdict = "SKILL NOT MATCHED (no verdict on the confound; the readout is recorded)"
    elif r <= hi:
        verdict = "SKILL MATCHED (the spec's decision table applies unchanged)"
    elif probe_below_bar:
        verdict = "SKILL-ADVANTAGED (decoder-direct reading stands, strengthened)"
    else:
        verdict = "NO SKILL-MEDIATION VERDICT (positive confounded by excess skill)"
    n = len(rets)
    se = float(np.std(rets, ddof=1) / n ** 0.5) if n > 1 else float("nan")
    return {"reference_return": reference, "tol": tol, "window": [lo, hi],
            "match_return": r, "per_seed_returns": rets, "n_seeds": n,
            "return_sd": float(np.std(rets, ddof=1)) if n > 1 else float("nan"),
            "return_se": se,
            "deviation_from_reference": r - reference,
            "overshoot_past_window": max(0.0, r - hi),
            "overshoot_in_se": (r - hi) / se if se and np.isfinite(se) and r > hi else 0.0,
            "in_window": lo <= r <= hi,
            "probe_survival": surv_mean, "probe_survival_t90": surv_t90,
            "probe_below_bar": bool(probe_below_bar),
            "verdict": verdict}


CLOUD_EXECUTION = "cloud CPU sandbox (4 vCPU, 3 workers, torch 2.14+cpu); published runs were GPU"


def promote(run_dir: str, out_path: str, *, spec: str, label: str, calibration: str | None = None,
            head: str | None = None, compare_run: str | None = None,
            execution: str | None = None, skill_match_reference: float | None = None,
            skill_match_tol: float = MARGIN) -> dict:
    with open(os.path.join(run_dir, "expB2_results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    drifts = sorted(res.keys(), key=float)
    dmax = max(drifts, key=float)
    arms_out: dict = {}
    for d in drifts:
        arms_out[d] = {}
        for arm in ARMS:
            c = res[d][arm]
            block = {"pool_target": summarize_vals(c["pool_target"])}
            for k in ("pool_speed", "pool_reward_leak", "pool_deaths_auth", "pool_deaths_surr",
                      "mp_target", "pool_shuffled"):
                if k in c:
                    block[k + "_mean"] = _nanmean(c[k])
            for k in ("pool_leak_clean", "mp_leak_clean"):
                if k in c:
                    block[k + "_count"] = int(sum(bool(x) for x in c[k]))
            arms_out[d][arm] = block
    # engagement from cells
    eng: dict = {}
    for cp in sorted(glob.glob(os.path.join(run_dir, "cells", "cell_*.json"))):
        with open(cp, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        key = f"{float(c['drift']):.2f}"
        eng.setdefault(key, {"pass": 0, "n": 0})
        eng[key]["n"] += 1
        eng[key]["pass"] += int(bool(c["eng"]["engaged"]))
    # L0 equivalence on the survival drift-0 targets
    d0 = min(drifts, key=float)
    l0_vals = [float(x) for x in res[d0]["survival"]["pool_target"]]
    tost = dataclasses.asdict(equivalence_test(l0_vals, 0.5, margin=MARGIN)) if len(l0_vals) > 2 else None
    rope = dataclasses.asdict(rope_test(l0_vals)) if len(l0_vals) > 2 else None
    surv = arms_out[dmax]["survival"]["pool_target"]
    pred = arms_out[dmax]["predictor"]["pool_target"]
    untr = arms_out[dmax]["untrained"]["pool_target"]
    gates = {
        "engagement": eng,
        "l0_survival_mean": float(np.mean(l0_vals)),
        "l0_tost": tost, "l0_rope": rope,
        "speed_min": float(min(_nanmean(res[d][a]["pool_speed"]) for d in drifts for a in ARMS)),
        "pool_leak_clean_all": all(all(bool(x) for x in res[d][a].get("pool_leak_clean", [True]))
                                   for d in drifts for a in ARMS),
        "untrained_floor_at_dmax": untr["mean"],
        "untrained_floor_ok": abs(untr["mean"] - 0.5) < 0.1,
        "deaths_total": float(sum(_nanmean(res[d][a]["pool_deaths_auth"]) + _nanmean(res[d][a]["pool_deaths_surr"])
                                  for d in drifts for a in ARMS if "pool_deaths_auth" in res[d][a])),
    }
    decision = {
        "survival": surv["mean"], "predictor": pred["mean"], "untrained": untr["mean"],
        "survival_t90": surv["t90"],
        "pass_bar": surv["mean"] >= BAR,
        "t90_excludes_bar": surv["t90"][0] > BAR,
        "lead_over_predictor": surv["mean"] - pred["mean"],
        "lead_over_untrained": surv["mean"] - untr["mean"],
        "pass_margin_predictor": surv["mean"] > pred["mean"] + MARGIN,
        "pass_margin_untrained": surv["mean"] > untr["mean"] + MARGIN,
    }
    if decision["pass_bar"] and decision["pass_margin_predictor"] and decision["pass_margin_untrained"]:
        decision["zone"] = "ENCODING INDUCED (all clauses)"
    elif surv["mean"] < 0.55 and pred["mean"] < 0.55 and untr["mean"] < 0.55:
        decision["zone"] = "STRENGTHENED NEGATIVE (all arms near chance)"
    else:
        decision["zone"] = "INTERMEDIATE (above chance, rule not met)"
    audit_out = None
    ba_path = os.path.join(run_dir, "behavior_audit.json")
    if os.path.exists(ba_path):
        with open(ba_path, encoding="utf-8") as fh:
            ba = json.load(fh)
        audit_out = {}
        for arm in ARMS:
            rows = [c for c in ba["cells"] if c["drift"] == dmax and c["agent"] == arm]
            if not rows:
                continue
            audit_out[arm] = {}
            for met in ("target", "behavior_only", "behavior_trace_only", "resid_epmean",
                        "resid_trace", "resid_trace_quad"):
                vals = [r[met] for r in rows if met in r and np.isfinite(r[met])]
                if vals:
                    audit_out[arm][met] = summarize_vals(vals)
    calib_out = None
    if calibration:
        with open(calibration, encoding="utf-8") as fh:
            cal = json.load(fh)
        calib_out = {"g_seed": cal.get("g_seed"), "rows": cal["rows"],
                     "selected_hidden": next((r["hidden"] for r in cal["rows"] if r.get("passes_gate0")), None)}
    dc_out = None
    if compare_run:
        with open(os.path.join(compare_run, "expB2_results.json"), encoding="utf-8") as fh:
            cres = json.load(fh)
        dc_out = {"compare_run": compare_run.replace("\\", "/"), **device_control(res, cres, dmax)}
    out = {
        "source_run": run_dir.replace("\\", "/"),
        "label": label, "spec": spec,
        "world": "WorldParams(k_land=1.5, k_water=1.5, gravity=0.4) [P]",
        "execution": execution or CLOUD_EXECUTION,
        "git_commit_at_promotion": head or git_head(),
        "generated_by": "scripts/promote_reviewer_gaps_runs.py",
        "bars": {"auroc_floor": BAR, "margin": MARGIN},
        "drifts": drifts, "dmax": dmax,
        "arms": arms_out, "gates": gates, "decision": decision,
        "behavior_audit": audit_out, "gate0_calibration": calib_out,
    }
    if dc_out is not None:
        out["device_control"] = dc_out
    if skill_match_reference is not None:
        out["skill_match"] = skill_match(run_dir, res, dmax, skill_match_reference, skill_match_tol)
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--device-control-against", default=None,
                    help="no-auxiliary run dir on the same device; adds the frozen device-control rule")
    ap.add_argument("--execution", default=None,
                    help="where the run executed (default: the cloud CPU sandbox text)")
    ap.add_argument("--skill-match-reference", type=float, default=None,
                    help="reference eval return to match; adds the frozen skill-match rule")
    ap.add_argument("--skill-match-tol", type=float, default=MARGIN,
                    help="half-width of the skill-match window (default: the 0.05 margin)")
    a = ap.parse_args()
    out = promote(a.run, a.out, spec=a.spec, label=a.label, calibration=a.calibration,
                  compare_run=a.device_control_against, execution=a.execution,
                  skill_match_reference=a.skill_match_reference,
                  skill_match_tol=a.skill_match_tol)
    dec = out["decision"]
    print(f"wrote {a.out}: survival {dec['survival']:.3f} t90 [{dec['survival_t90'][0]:.3f}, "
          f"{dec['survival_t90'][1]:.3f}] vs predictor {dec['predictor']:.3f}, untrained "
          f"{dec['untrained']:.3f} -> {dec['zone']}")
    if "device_control" in out:
        dc = out["device_control"]
        print(f"device control: decoder {dc['decoder_survival']:.3f} vs no-auxiliary "
              f"{dc['no_auxiliary_survival']:.3f}, lead {dc['lead']:+.3f} -> {dc['verdict']}")
    if "skill_match" in out:
        sm = out["skill_match"]
        print(f"skill match: R {sm['match_return']:+.4f} against window "
              f"[{sm['window'][0]:+.3f}, {sm['window'][1]:+.3f}] "
              f"(overshoot {sm['overshoot_past_window']:+.4f}, {sm['overshoot_in_se']:.2f} SE) "
              f"-> {sm['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
