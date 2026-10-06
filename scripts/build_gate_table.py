"""Gate table for the historical and corrected B-v2 / L3 / L1 runs (revision step 5).

One row per (run, fold partition). The L0 equivalence test (TOST, margin 0.05, alpha 0.05,
with the ROPE [0.45, 0.55] bootstrap leg beside it) and the untrained floor are recomputed
here from the per-seed values committed under `artifacts/`; the other gates (engagement,
speed control, pooled leakage, deaths) are carried from the run's promoted summary where it
records them and are marked "not recorded" where it does not. Nothing is relaxed: an L0 that
is not shown equivalent reads "inconclusive", whatever the mean.

Every L0 row conditions on one pair of evaluation-world samples (seed bases 800000 and
850000) shared by all agent seeds; `artifacts/l0_audit/` measures how much the statistic
moves across independent world samples.

Usage:
    python scripts/build_gate_table.py           # write artifacts/gate_table.json, docs/GATE_TABLE.md
    python scripts/build_gate_table.py --check   # exit 1 if either is stale
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import sys

import numpy as np

from itasorl.stats import equivalence_test, rope_test

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts")
OUT_JSON = os.path.join(ART, "gate_table.json")
OUT_MD = os.path.join(ROOT, "docs", "GATE_TABLE.md")
FLOOR_TOL = 0.1


def _load(rel):
    with open(os.path.join(ART, rel), encoding="utf-8") as fh:
        return json.load(fh)


def l0_block(vals) -> dict:
    v = [float(x) for x in vals if np.isfinite(x)]
    if len(v) < 2:
        return {"n": len(v), "mean": float(np.mean(v)) if v else float("nan"),
                "tost_p": float("nan"), "status": "not testable (n < 2)"}
    eq = equivalence_test(v)
    rp = rope_test(v)
    status = "pass" if eq.equivalent else "inconclusive (not shown equivalent)"
    return {"n": len(v), "mean": eq.mean, "sd": float(np.std(v, ddof=1)), "tost_p": eq.p_value,
            "equivalent": bool(eq.equivalent), "rope_interval": list(rp.hdi),
            "rope_share_inside": rp.p_in_rope, "rope_accept": bool(rp.accept), "status": status}


def floor_block(vals) -> dict:
    v = [float(x) for x in vals if np.isfinite(x)]
    m = float(np.mean(v))
    return {"mean": m, "ok": bool(abs(m - 0.5) < FLOOR_TOL),
            "status": "pass" if abs(m - 0.5) < FLOOR_TOL else "fail"}


def _rescore_rows(run_id, rel, partitions=("legacy", "explicit"), other=None):
    d = _load(rel)
    rows = []
    for part in partitions:
        def vals(drift, agent):
            cs = sorted((c for c in d["cells"] if c["drift"] == drift and c["agent"] == agent),
                        key=lambda c: c["seed"])
            return [c[part]["target"] for c in cs]
        drifts = sorted({c["drift"] for c in d["cells"]}, key=float)
        dmax = drifts[-1]
        rows.append({"run": run_id, "partition": part, "source": f"artifacts/{rel}",
                     "l0": l0_block(vals(drifts[0], "survival")),
                     "untrained_floor": floor_block(vals(dmax, "untrained")),
                     **(other or {})})
    return rows


def _summary_row(run_id, rel, partition):
    d = _load(rel)
    g = d["gates"]
    l0 = d["arms"]["0.0"]["survival"]["pool_target"]["per_seed"]
    unt = d["arms"][d["dmax"]]["untrained"]["pool_target"]["per_seed"]
    eng = g["engagement"]
    return {"run": run_id, "partition": partition, "source": f"artifacts/{rel}",
            "l0": l0_block(l0), "untrained_floor": floor_block(unt),
            "engagement": f"{sum(v['pass'] for v in eng.values())}/{sum(v['n'] for v in eng.values())}",
            "speed_min": g.get("speed_min"), "pool_leak_clean_all": g.get("pool_leak_clean_all"),
            "deaths_total": g.get("deaths_total")}


def build() -> dict:
    rows = []
    b2 = _load("expB2/expB2_results.json")
    rows.append({"run": "BV2-L2-AR1", "partition": "legacy", "source": "artifacts/expB2/expB2_results.json",
                 "l0": l0_block(b2["0.0"]["survival"]["pool_target"]),
                 "untrained_floor": floor_block(b2["0.45"]["untrained"]["pool_target"])})
    bv3 = _load("expB2/bv3_n10_gates.json")["gates"]
    bv3_unt = [p["target"] for p in _load("expB2/bv3_n10_gates.json")["pools"]
               if p["drift"] == "0.45" and p["agent"] == "untrained"]
    rows.append({"run": "BV3-REGIME-N10", "partition": "legacy",
                 "source": "artifacts/expB2/bv3_n10_gates.json",
                 "l0": l0_block(bv3["l0_control"]["survival_pool_target_per_seed"]),
                 "untrained_floor": floor_block(bv3_unt),
                 "engagement": f"{bv3['engagement']['n_engaged']}/{bv3['engagement']['n_cells']}",
                 "speed_min": min(bv3["speed_positive_control"]["min_speed"].values())})
    rows += _rescore_rows("L3-H8-N10", "fold_rescore/l3_h8_traces.json")
    rows += _rescore_rows("L3-H7-N10", "fold_rescore/l3_h7_traces.json")
    rows += _rescore_rows("L3-H4", "fold_rescore/l3_h4_traces.json")
    rows += _rescore_rows("L3-H8-HELDOUT", "fold_rescore/l3_h8_heldout.json")
    rows += _rescore_rows("L3-H7-REVERSE", "fold_rescore/l3_h7_heldout.json")
    l1g = _load("expL1/organism_summary.json")["gates"]
    rows += _rescore_rows("L1-ORGANISM", "fold_rescore/l1_heldout.json", other={
        "engagement": f"{l1g['engagement']['n_engaged']}/{l1g['engagement']['n_cells']}",
        "speed_min": min(l1g["speed_positive_control"]["min_speed"].values()),
        "pool_leak_clean_all": l1g["leakage"]["pool_leak_clean_all"],
        "deaths_total": l1g["survivorship"]["deaths_total"]})
    for run_id, rel, part in [
        ("L3-H8-NOWM-CPU", "expB2/arch_baseline_l3_h8_nowm.json", "explicit (= legacy on that stack)"),
        ("L3-H8-WM-CPU", "expB2/device_control_l3_h8_wm_cpu.json", "explicit (= legacy on that stack)"),
        ("L3-H10-GS1-CPU", "expB2/second_instance_l3_h10_gseed1.json", "explicit (= legacy on that stack)"),
        ("L3-H10-GS1-GPU", "expB2/second_instance_l3_h10_gseed1_gpu.json", "explicit"),
        ("L3-H8-GS2-GPU", "expB2/second_seed_l3_h8_gseed2_gpu.json", "explicit"),
        ("L3-H8-NOWM-U450", "expB2/skill_matched_l3_h8_nowm_u450.json", "explicit"),
    ]:
        rows.append(_summary_row(run_id, rel, part))
    # corrected runs (revision step 4), added automatically once promoted
    for rel in sorted(glob.glob(os.path.join(ART, "expB2", "corrected_*.json"))):
        rel = os.path.relpath(rel, ART).replace(os.sep, "/")
        rid = "CORRECTED-" + os.path.basename(rel)[len("corrected_"):-len(".json")].upper()
        rows.append({**_summary_row(rid, rel, "explicit"), "status": "corrected"})
    for r in rows:
        r.setdefault("status", "historical")
    return {"generated_by": "scripts/build_gate_table.py", "floor_tol": FLOOR_TOL,
            "l0_rule": "TOST margin 0.05 alpha 0.05 on per-seed drift-0 survival targets",
            "world_sample_note": "every L0 row conditions on one pair of evaluation-world samples "
                                 "shared by all agent seeds (artifacts/l0_audit/)",
            "rows": rows}


def _fmt(x, nd=3):
    if x is None:
        return "not recorded"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if np.isfinite(x) else "n/a"
    return str(x)


def render_md(t: dict) -> str:
    L = ["# Gate table", "",
         "*Generated by `scripts/build_gate_table.py`. Do not edit by hand. L0 TOST and ROPE and "
         "the untrained floor are recomputed from committed per-seed values; the other gates are "
         "carried from each run's promoted summary.*", "",
         "L0 rule: TOST with margin 0.05 and alpha 0.05 on the ten drift-0 survival targets; the "
         "ROPE column is the share of bootstrap means inside [0.45, 0.55] (a bootstrap containment "
         "proportion, not a posterior probability). A test that does not show equivalence is "
         "reported as **inconclusive**, never as a pass, and no margin is changed after the fact. "
         "Every L0 row conditions on one pair of evaluation-world samples shared by all agent "
         "seeds; `artifacts/l0_audit/` measures how far the statistic moves across independent "
         "world samples.", "",
         "| Run | Status | Partition | L0 mean | L0 TOST p | L0 | ROPE share | Untrained floor | "
         "Engagement | Speed min | Pooled leak clean | Deaths |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        l0, fl = r["l0"], r["untrained_floor"]
        L.append(f"| `{r['run']}` | {r['status']} | {r['partition']} | {_fmt(l0['mean'])} | "
                 f"{_fmt(l0.get('tost_p'))} | {l0['status']} | {_fmt(l0.get('rope_share_inside'))} | "
                 f"{_fmt(fl['mean'])} ({fl['status']}) | {_fmt(r.get('engagement'))} | "
                 f"{_fmt(r.get('speed_min'))} | {_fmt(r.get('pool_leak_clean_all'))} | "
                 f"{_fmt(r.get('deaths_total'))} |")
    open_rows = [r for r in t["rows"] if r["l0"]["status"] != "pass"]
    L += ["", f"L0 not shown equivalent: {len(open_rows)} of {len(t['rows'])} rows "
          + ("(" + ", ".join(f"`{r['run']}` {r['partition']}" for r in open_rows) + ")."
             if open_rows else "."), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    t = build()
    want_json = json.dumps(t, indent=1, default=float) + "\n"
    want_md = render_md(t)
    if a.check:
        stale = []
        for path, want in ((OUT_JSON, want_json), (OUT_MD, want_md)):
            try:
                with open(path, encoding="utf-8") as fh:
                    if fh.read() != want:
                        stale.append(path)
            except OSError:
                stale.append(path)
        for p in stale:
            print("GATE TABLE stale:", os.path.relpath(p, ROOT))
        print("gate table: " + ("OK" if not stale else "stale; run python scripts/build_gate_table.py"))
        return 1 if stale else 0
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(want_json)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(want_md)
    print(f"wrote {os.path.relpath(OUT_JSON, ROOT)} and {os.path.relpath(OUT_MD, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
