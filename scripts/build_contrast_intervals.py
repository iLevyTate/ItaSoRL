"""Seed-paired intervals for the registered contrasts (revision step 13).

The decision rules require survival to exceed the predictor and the untrained arm by 0.05.
Those are claims about differences, so this table gives each difference per run, paired by
agent seed, with its t-based 90% CI, and says whether the MEAN meets the margin (the
registered rule) and whether the CI LOWER BOUND does (the evidence). A mean can meet a
mean-based rule while the interval still straddles the threshold; the two are reported
apart. Every interval is conditional on the run's single trained surrogate and its fixed
evaluation worlds, which all seeds share.

Usage:
    python scripts/build_contrast_intervals.py           # write artifacts/contrast_intervals.json, docs/CONTRAST_INTERVALS.md
    python scripts/build_contrast_intervals.py --check
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import sys

import numpy as np

from itasorl.stats import paired_contrast, t_ci90

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts")
OUT_JSON = os.path.join(ART, "contrast_intervals.json")
OUT_MD = os.path.join(ROOT, "docs", "CONTRAST_INTERVALS.md")
BAR, MARGIN = 0.65, 0.05


def _load(rel):
    with open(os.path.join(ART, rel), encoding="utf-8") as fh:
        return json.load(fh)


def _row(run, partition, metric, arms, source, status="historical"):
    s, p, u = (np.asarray(arms[k], float) for k in ("survival", "predictor", "untrained"))
    lo, hi = t_ci90(s)
    return {"run": run, "partition": partition, "metric": metric, "source": source, "status": status,
            "survival_mean": float(s.mean()), "survival_t90": [float(lo), float(hi)],
            "bar_mean": bool(s.mean() >= BAR), "bar_t90": bool(lo >= BAR),
            "vs_predictor": paired_contrast(s, p, MARGIN), "vs_untrained": paired_contrast(s, u, MARGIN)}


def _from_rescore(run, rel, metric="target"):
    d = _load(rel)
    dmax = max({c["drift"] for c in d["cells"]}, key=float)
    rows = []
    for part in ("legacy", "explicit"):
        arms = {}
        for g in ("survival", "predictor", "untrained"):
            cs = sorted((c for c in d["cells"] if c["drift"] == dmax and c["agent"] == g),
                        key=lambda c: c["seed"])
            arms[g] = [c[part][metric] for c in cs]
        rows.append(_row(run, part, metric, arms, f"artifacts/{rel}"))
    return rows


def _from_summary(run, rel, partition, status="historical"):
    d = _load(rel)
    dm = d["dmax"]
    rows = [_row(run, partition, "target",
                 {g: d["arms"][dm][g]["pool_target"]["per_seed"] for g in ("survival", "predictor", "untrained")},
                 f"artifacts/{rel}", status)]
    ba = d.get("behavior_audit") or {}
    if all(g in ba and "resid_trace" in ba[g] for g in ("survival", "predictor", "untrained")):
        rows.append(_row(run, partition, "resid_trace",
                         {g: ba[g]["resid_trace"]["per_seed"] for g in ("survival", "predictor", "untrained")},
                         f"artifacts/{rel}", status))
    return rows


def build() -> dict:
    rows = []
    bv3 = _load("expB2/bv3_n10_summary.json")["pooled_target_drift045"]
    rows.append(_row("BV3-REGIME-N10", "legacy", "target",
                     {g: bv3[g]["pool_target_per_seed"] for g in ("survival", "predictor", "untrained")},
                     "artifacts/expB2/bv3_n10_summary.json"))
    for run, rel in (("L3-H8-N10", "fold_rescore/l3_h8_traces.json"),
                     ("L3-H7-N10", "fold_rescore/l3_h7_traces.json"),
                     ("L3-H8-HELDOUT", "fold_rescore/l3_h8_heldout.json"),
                     ("L3-H7-REVERSE", "fold_rescore/l3_h7_heldout.json"),
                     ("L1-ORGANISM", "fold_rescore/l1_heldout.json")):
        rows += _from_rescore(run, rel, "target")
        rows += _from_rescore(run, rel, "resid_trace")
    for run, rel, part in (("L3-H8-NOWM-CPU", "expB2/arch_baseline_l3_h8_nowm.json", "explicit (= legacy)"),
                           ("L3-H8-WM-CPU", "expB2/device_control_l3_h8_wm_cpu.json", "explicit (= legacy)"),
                           ("L3-H10-GS1-CPU", "expB2/second_instance_l3_h10_gseed1.json", "explicit (= legacy)"),
                           ("L3-H10-GS1-GPU", "expB2/second_instance_l3_h10_gseed1_gpu.json", "explicit"),
                           ("L3-H8-GS2-GPU", "expB2/second_seed_l3_h8_gseed2_gpu.json", "explicit"),
                           ("L3-H8-NOWM-U450", "expB2/skill_matched_l3_h8_nowm_u450.json", "explicit")):
        rows += _from_summary(run, rel, part)
    for p in sorted(glob.glob(os.path.join(ART, "expB2", "corrected_*.json"))):
        rel = os.path.relpath(p, ART).replace(os.sep, "/")
        rid = "CORRECTED-" + os.path.basename(rel)[len("corrected_"):-len(".json")].upper()
        rows += _from_summary(rid, rel, "explicit", status="corrected")
    # auxiliary contrast on one device: decoder-carrying minus no-decoder survival, by seed
    aux = []
    for name, a_rel, b_rel, status in (
            ("historical CPU (L3-H8-WM-CPU minus L3-H8-NOWM-CPU)", "expB2/device_control_l3_h8_wm_cpu.json",
             "expB2/arch_baseline_l3_h8_nowm.json", "historical"),
            ("corrected CPU (C1 minus C2)", "expB2/corrected_l3_h8_wm.json",
             "expB2/corrected_l3_h8_nowm.json", "corrected")):
        if not (os.path.exists(os.path.join(ART, a_rel)) and os.path.exists(os.path.join(ART, b_rel))):
            continue
        A, B = _load(a_rel), _load(b_rel)
        sa = A["arms"][A["dmax"]]["survival"]["pool_target"]["per_seed"]
        sb = B["arms"][B["dmax"]]["survival"]["pool_target"]["per_seed"]
        aux.append({"comparison": name, "status": status, **paired_contrast(sa, sb, MARGIN)})
    return {"generated_by": "scripts/build_contrast_intervals.py", "bar": BAR, "margin": MARGIN,
            "conditioning": "intervals are across agent seeds and condition on the run's single "
                            "trained surrogate and its fixed evaluation worlds",
            "rows": rows, "auxiliary": aux}


def _f(x):
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


def render_md(t: dict) -> str:
    L = ["# Contrast intervals", "",
         "*Generated by `scripts/build_contrast_intervals.py`. Do not edit by hand.*", "",
         "Each registered margin (survival at least 0.05 above the predictor and above the untrained "
         "arm) is a claim about a difference, so it is shown with a seed-paired t-based 90% CI. "
         "**Rule** says whether the mean meets the registered threshold; **evidence** says whether the "
         "CI lower bound does. The two can disagree: a mean can satisfy a mean-based rule while the "
         "interval still straddles the threshold. Intervals condition on each run's single trained "
         "surrogate and its fixed evaluation worlds, which every seed shares.", "",
         "| Run | Status | Partition | Metric | Survival (t 90%) | Bar rule / evidence | "
         "vs predictor (t 90%) | rule / evidence | vs untrained (t 90%) | rule / evidence |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    yn = {True: "yes", False: "no"}
    for r in t["rows"]:
        vp, vu = r["vs_predictor"], r["vs_untrained"]
        L.append(f"| `{r['run']}` | {r['status']} | {r['partition']} | {r['metric']} | "
                 f"{r['survival_mean']:.3f} [{r['survival_t90'][0]:.3f}, {r['survival_t90'][1]:.3f}] | "
                 f"{yn[r['bar_mean']]} / {yn[r['bar_t90']]} | "
                 f"{_f(vp['mean'])} [{_f(vp['t90'][0])}, {_f(vp['t90'][1])}] | "
                 f"{yn[vp['mean_ge_margin']]} / {yn[vp['t90_lower_ge_margin']]} | "
                 f"{_f(vu['mean'])} [{_f(vu['t90'][0])}, {_f(vu['t90'][1])}] | "
                 f"{yn[vu['mean_ge_margin']]} / {yn[vu['t90_lower_ge_margin']]} |")
    L += ["", "## Auxiliary contrast on one device", "",
          "| Comparison | Status | Paired difference (t 90%) | rule / evidence vs 0.05 |", "|---|---|---|---|"]
    for a in t["auxiliary"]:
        L.append(f"| {a['comparison']} | {a['status']} | {_f(a['mean'])} [{_f(a['t90'][0])}, "
                 f"{_f(a['t90'][1])}] | {yn[a['mean_ge_margin']]} / {yn[a['t90_lower_ge_margin']]} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    t = build()
    want = {OUT_JSON: json.dumps(t, indent=1, default=float) + "\n", OUT_MD: render_md(t)}
    if a.check:
        stale = [p for p, w in want.items()
                 if not os.path.exists(p) or open(p, encoding="utf-8").read() != w]
        for p in stale:
            print("CONTRAST INTERVALS stale:", os.path.relpath(p, ROOT))
        print("contrast intervals: " + ("OK" if not stale else "stale; run python scripts/build_contrast_intervals.py"))
        return 1 if stale else 0
    for p, w in want.items():
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(w)
    print(f"wrote {os.path.relpath(OUT_JSON, ROOT)} and {os.path.relpath(OUT_MD, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
