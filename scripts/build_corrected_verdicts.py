"""Apply the frozen decision rules to the corrected confirmation runs (revision step 4).

Reads the per-seed cells of the corrected runs C1 (auxiliary on) and C2 (auxiliary off) and
of the historical CPU runs they replace like for like, and writes
artifacts/corrected_verdicts.json with, in the order the spec requires
(docs/specs/2026-10-06-corrected-trainer-confirmation-design.md):

  integrity   every cell records gae_bootstrap = successor; the predictor and untrained arms,
              which train no actor-critic, equal the historical cell at the same (drift, seed)
              to the bit; the fold partition of the pooled 110 + 110 design
  gates       engagement in every cell, L0 equivalence (TOST margin 0.05; the bootstrap ROPE
              check beside it), speed control >= 0.75, pooled reward leakage within 0.1 of
              0.5, untrained floor within 0.1 of 0.5, no deaths
  primary     decide_h_b2: MET if every gate passes, survival mean >= 0.65 with the t-based
              90% lower bound >= 0.65, and survival >= 0.05 above both baselines; a mean at
              or above 0.65 with the lower bound below it is INCONCLUSIVE; a failed gate
              leaves the verdict conditional on it
  auxiliary   C1 meets the primary rule and the C1 minus C2 survival difference exceeds 0.05;
              the seed-paired difference and its t-based 90% CI beside it
  correction  C1 minus L3-H8-WM-CPU and C2 minus L3-H8-NOWM-CPU, survival target and
              engagement return, paired by seed (reported, not thresholded)

A run that is absent or incomplete is reported as such and gets no verdict.

Usage:
    python scripts/build_corrected_verdicts.py            # write the artifact
    python scripts/build_corrected_verdicts.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import dataclasses
import glob
import json
import os
import sys

import numpy as np

from itasorl.stats import equivalence_test, paired_contrast, rope_test, t_ci90

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "artifacts", "corrected_verdicts.json")
SPEC = "docs/specs/2026-10-06-corrected-trainer-confirmation-design.md"
RUNS = {
    "C1": {"dir": "artifacts/corrected_runs/corrected_l3_h8_wm", "replaces": "L3-H8-WM-CPU",
           "historical": "artifacts/reviewer_gaps_runs/l3_h8_wm_cpu", "auxiliary": True},
    "C2": {"dir": "artifacts/corrected_runs/corrected_l3_h8_nowm", "replaces": "L3-H8-NOWM-CPU",
           "historical": "artifacts/reviewer_gaps_runs/l3_h8_nowm", "auxiliary": False},
}
ARMS = ("untrained", "predictor", "survival")
DRIFTS = (0.0, 0.45)
SEEDS = tuple(range(10))
BAR, MARGIN = 0.65, 0.05


def load_cells(run_dir: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, run_dir, "cells", "cell_d*_s*.json"))):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)["cell"]
        out[(round(float(c["drift"]), 2), int(c["seed"]))] = c
    return out


def _vals(cells: dict, drift: float, arm: str, key: str = "target") -> list[float]:
    return [float(cells[(drift, s)]["agents"][arm]["pool"][key]) for s in SEEDS]


def _summ(v) -> dict:
    v = np.asarray(v, float)
    return {"per_seed": v.tolist(), "mean": float(v.mean()), "t90": [float(x) for x in t_ci90(v)],
            "seeds_at_bar": int((v >= BAR).sum())}


def fold_record() -> dict:
    from itasorl import folds
    groups, y = folds.STANDARD_DESIGNS["pooled_110x110"][0](), folds.STANDARD_DESIGNS["pooled_110x110"][1]()
    rec = folds.partition_record(groups, y, scheme="explicit")
    with open(os.path.join(ROOT, "artifacts", "folds", "explicit_v1.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    ref_sha = ref["designs"]["pooled_110x110"]["sha256"] if "designs" in ref else None
    rec["matches_serialized"] = ref_sha == rec["sha256"]
    return rec


def integrity(cells: dict, hist: dict) -> dict:
    boot = sorted({c.get("gae_bootstrap", "missing") for c in cells.values()})
    same = {}
    for arm in ("predictor", "untrained"):
        mism = [f"d{d:.2f}_s{s}" for d in DRIFTS for s in SEEDS
                if (d, s) in hist and cells[(d, s)]["agents"][arm]["pool"]["target"]
                != hist[(d, s)]["agents"][arm]["pool"]["target"]]
        same[arm] = {"compared": sum((d, s) in hist for d in DRIFTS for s in SEEDS),
                     "mismatches": mism}
    ok = boot == ["successor"] and all(not v["mismatches"] for v in same.values())
    return {"gae_bootstrap": boot, "bit_identical_to_historical": same, "pass": ok}


def gates(cells: dict) -> dict:
    eng = {f"{d:.2f}": sum(bool(cells[(d, s)]["eng"]["engaged"]) for s in SEEDS) for d in DRIFTS}
    l0 = _vals(cells, 0.0, "survival")
    tost = equivalence_test(l0, 0.5, margin=MARGIN)
    rope = rope_test(l0)
    speed = min(float(cells[(d, s)]["agents"][a]["pool"]["speed"]) for d in DRIFTS for s in SEEDS for a in ARMS)
    leak = max(abs(float(cells[(d, s)]["agents"][a]["pool"]["pool_reward_leak"]) - 0.5)
               for d in DRIFTS for s in SEEDS for a in ARMS)
    floor = float(np.mean(_vals(cells, 0.45, "untrained")))
    deaths = sum(int(cells[(d, s)]["agents"][a]["pool"][k]) for d in DRIFTS for s in SEEDS for a in ARMS
                 for k in ("deaths_auth", "deaths_surr"))
    g = {
        "engagement": {"per_drift": eng, "pass": all(v == len(SEEDS) for v in eng.values())},
        "l0": {"mean": float(np.mean(l0)), "per_seed": l0, "tost_p": float(tost.p_value),
               "pass": bool(tost.equivalent),
               "rope": {"boot_interval": [float(x) for x in rope.boot_interval],
                        "boot_share_in_rope": float(rope.p_in_rope), "accept": bool(rope.accept)}},
        "speed": {"min": speed, "pass": speed >= 0.75},
        "reward_leak": {"max_abs_dev": leak, "pass": leak < 0.1},
        "untrained_floor": {"mean": floor, "pass": abs(floor - 0.5) < 0.1},
        "deaths": {"total": deaths, "pass": deaths == 0},
    }
    g["failed"] = [k for k, v in g.items() if isinstance(v, dict) and not v["pass"]]
    return g


def primary(cells: dict, g: dict) -> dict:
    surv, pred, untr = (_vals(cells, 0.45, a) for a in ("survival", "predictor", "untrained"))
    s = _summ(surv)
    lead_p, lead_u = s["mean"] - float(np.mean(pred)), s["mean"] - float(np.mean(untr))
    clauses = {"mean_at_bar": s["mean"] >= BAR, "lower_bound_at_bar": s["t90"][0] >= BAR,
               "margin_predictor": lead_p >= MARGIN, "margin_untrained": lead_u >= MARGIN}
    if all(clauses.values()):
        dec = "MET"
    elif clauses["mean_at_bar"] and clauses["margin_predictor"] and clauses["margin_untrained"]:
        dec = "INCONCLUSIVE"
    else:
        dec = "NOT MET"
    verdict = dec if not g["failed"] or dec == "NOT MET" else f"{dec} on the decodability clauses, conditional on open gate(s): {', '.join(g['failed'])}"
    return {"survival": s, "predictor": _summ(pred), "untrained": _summ(untr),
            "lead_over_predictor": lead_p, "lead_over_untrained": lead_u,
            "contrast_vs_predictor": paired_contrast(surv, pred),
            "contrast_vs_untrained": paired_contrast(surv, untr),
            "clauses": clauses, "decodability": dec, "verdict": verdict,
            "met": dec == "MET" and not g["failed"]}


def correction(cells: dict, hist: dict) -> dict:
    out = {}
    for d in DRIFTS:
        seeds = [s for s in SEEDS if (d, s) in hist]
        if not seeds:
            continue
        c_t = [float(cells[(d, s)]["agents"]["survival"]["pool"]["target"]) for s in seeds]
        h_t = [float(hist[(d, s)]["agents"]["survival"]["pool"]["target"]) for s in seeds]
        c_r = [float(cells[(d, s)]["eng"]["trained_return"]) for s in seeds]
        h_r = [float(hist[(d, s)]["eng"]["trained_return"]) for s in seeds]
        out[f"{d:.2f}"] = {"seeds": seeds,
                           "survival_target": {"corrected": _summ(c_t), "historical": _summ(h_t),
                                               "difference": paired_contrast(c_t, h_t, 0.0)},
                           "engagement_return": {"corrected": float(np.mean(c_r)),
                                                 "historical": float(np.mean(h_r)),
                                                 "difference": paired_contrast(c_r, h_r, 0.0)}}
    return out


def build() -> dict:
    out = {"generated_by": "scripts/build_corrected_verdicts.py", "spec": SPEC,
           "rules": {"bar": BAR, "margin": MARGIN, "l0_tost_margin": MARGIN}, "runs": {}}
    try:
        out["fold_partition"] = fold_record()
    except Exception as e:  # pragma: no cover
        out["fold_partition"] = {"error": repr(e)}
    loaded = {}
    for name, r in RUNS.items():
        cells = load_cells(r["dir"])
        need = {(d, s) for d in DRIFTS for s in SEEDS}
        missing = sorted(need - set(cells))
        entry = {"dir": r["dir"], "replaces": r["replaces"], "historical": r["historical"],
                 "auxiliary": r["auxiliary"], "cells": len(cells)}
        if missing:
            entry["status"] = f"incomplete: {len(missing)} of {len(need)} cells missing"
            out["runs"][name] = entry
            continue
        hist = load_cells(r["historical"])
        g = gates(cells)
        entry.update({"status": "complete", "integrity": integrity(cells, hist), "gates": g,
                      "primary": primary(cells, g), "correction_effect": correction(cells, hist)})
        out["runs"][name] = entry
        loaded[name] = cells
    if "C1" in loaded and "C2" in loaded:
        a, b = _vals(loaded["C1"], 0.45, "survival"), _vals(loaded["C2"], 0.45, "survival")
        diff = float(np.mean(a) - np.mean(b))
        out["auxiliary_comparison"] = {
            "c1_minus_c2": diff, "paired": paired_contrast(a, b),
            "c1_meets_primary": out["runs"]["C1"]["primary"]["met"],
            "c1_decodability": out["runs"]["C1"]["primary"]["decodability"],
            # "at least" in section 6, so >=, matching the primary clauses above.
            "holds": bool(out["runs"]["C1"]["primary"]["met"] and diff >= MARGIN),
            "holds_on_decodability_clauses": bool(out["runs"]["C1"]["primary"]["decodability"] == "MET"
                                                  and diff >= MARGIN)}
    return out


def _dump(obj) -> str:
    def clean(x):
        if isinstance(x, float):
            return None if not np.isfinite(x) else round(x, 6)
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if dataclasses.is_dataclass(x):
            return clean(dataclasses.asdict(x))
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return clean(x.item())
        return x
    return json.dumps(clean(obj), indent=1) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = _dump(build())
    if a.check:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        ok = cur == text
        print("corrected verdicts: " + ("OK" if ok else "stale; run python scripts/build_corrected_verdicts.py"))
        return 0 if ok else 1
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    d = json.loads(text)
    for name, r in d["runs"].items():
        print(f"{name}: {r['status']}" + (f"; {r['primary']['verdict']}; integrity "
                                          f"{'pass' if r['integrity']['pass'] else 'FAIL'}"
                                          if r["status"] == "complete" else ""))
    if "auxiliary_comparison" in d:
        ac = d["auxiliary_comparison"]
        print(f"auxiliary comparison: C1 - C2 = {ac['c1_minus_c2']:+.3f} {ac['paired']['t90']}; holds={ac['holds']}")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
