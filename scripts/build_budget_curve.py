"""Return and decodability against training budget (revision step 11).

Reads the per-cell `budget_curve` blocks of runs made with --budget-extend / --budget-snapshots
(the corrected confirmation runs C1 and C2) and reports, per budget point at the strongest
drift: survival return (the engagement evaluation in the training world), the pooled
world-identity target, and the environment steps consumed, each across seeds with a t-based
90% CI, plus the decoder-on minus decoder-off difference paired by seed. All points of a run
come from ONE training run per seed (frozen snapshots), so the curve separates the
fixed-budget comparison (300 updates) from capability at a larger budget (450).

Two panels share the x axis (updates); environment steps are tabulated, not given a second
y axis.

Usage:
    python scripts/build_budget_curve.py \\
        --run "decoder on=artifacts/corrected_runs/corrected_l3_h8_wm" \\
        --run "decoder off=artifacts/corrected_runs/corrected_l3_h8_nowm" \\
        --out artifacts/budget_curve.json --figure docs/figures/budget_curve.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os

import numpy as np

from itasorl.stats import paired_contrast, t_ci90

SERIES_COLORS = ("#2a78d6", "#eb6834")   # validated categorical slots 1 and 2 (light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_curve(run_dir: str) -> tuple[float, dict]:
    cells = []
    for p in sorted(glob.glob(os.path.join(run_dir, "cells", "cell_d*_s*.json"))):
        with open(p, encoding="utf-8") as fh:
            cells.append(json.load(fh)["cell"])
    dmax = max(c["drift"] for c in cells)
    by_u: dict[int, dict] = {}
    for c in cells:
        if c["drift"] != dmax or "budget_curve" not in c:
            continue
        for u, v in c["budget_curve"].items():
            e = by_u.setdefault(int(u), {"seed": [], "target": [], "return": [], "env_steps": []})
            e["seed"].append(int(c["seed"]))
            e["target"].append(float(v["pool"]["target"]))
            e["return"].append(float(v["eng"]["trained_return"]))
            e["env_steps"].append(float(v["env_steps"]))
    return dmax, dict(sorted(by_u.items()))


def summarize(by_u: dict) -> list[dict]:
    rows = []
    for u, e in by_u.items():
        order = np.argsort(e["seed"])
        row = {"updates": u, "seeds": [e["seed"][i] for i in order]}
        for k in ("target", "return", "env_steps"):
            v = np.asarray(e[k], float)[order]
            row[k] = {"per_seed": v.tolist(), "mean": float(v.mean()),
                      "t90": [float(x) for x in t_ci90(v)] if v.size > 1 else None}
        rows.append(row)
    return rows


def figure(series: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "return", "Survival return (training world)"),
                           (axes[1], "target", "Pooled world-identity target (AUROC)")):
        ax.set_facecolor(SURFACE)
        for (label, rows), color in zip(series.items(), SERIES_COLORS):
            x = [r["updates"] for r in rows]
            m = [r[key]["mean"] for r in rows]
            lo = [r[key]["t90"][0] if r[key]["t90"] else r[key]["mean"] for r in rows]
            hi = [r[key]["t90"][1] if r[key]["t90"] else r[key]["mean"] for r in rows]
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, m, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label)
            ax.annotate(label, (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        if key == "target":
            ax.axhline(0.65, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate("registered bar 0.65", (ax.get_xlim()[0], 0.65), xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color=INK2)
        ax.axvline(300, color=GRID, linewidth=1)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("survival updates (one training run per seed; 300 = registered budget)",
                      fontsize=8, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    n = max(len(r["seeds"]) for rows in series.values() for r in rows)
    fig.text(0.01, 0.01, f"Mean across {n} agent seeds; band = t-based 90% CI, conditional on one "
             "surrogate and one set of evaluation worlds.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure", default=None)
    a = ap.parse_args()
    series, raw, drifts = {}, {}, {}
    for spec in a.run:
        label, d = spec.split("=", 1)
        drifts[label], by_u = load_curve(d)
        raw[label] = by_u
        series[label] = summarize(by_u)
    out = {"generated_by": "scripts/build_budget_curve.py", "runs": dict(a.run and
           [s.split("=", 1) for s in a.run]), "drift": drifts, "series": series, "contrast": []}
    labels = list(series)
    if len(labels) == 2:
        A, B = raw[labels[0]], raw[labels[1]]
        for u in sorted(set(A) & set(B)):
            sa = dict(zip(A[u]["seed"], A[u]["target"]))
            sb = dict(zip(B[u]["seed"], B[u]["target"]))
            ra = dict(zip(A[u]["seed"], A[u]["return"]))
            rb = dict(zip(B[u]["seed"], B[u]["return"]))
            seeds = sorted(set(sa) & set(sb))
            out["contrast"].append({"updates": u, "seeds": seeds,
                                    "target": paired_contrast([sa[s] for s in seeds], [sb[s] for s in seeds]),
                                    "return": paired_contrast([ra[s] for s in seeds], [rb[s] for s in seeds], 0.0)})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for label, rows in series.items():
        for r in rows:
            print(f"{label:12s} u={r['updates']:4d} env_steps={r['env_steps']['mean']:9.0f} "
                  f"return={r['return']['mean']:+.3f} target={r['target']['mean']:.3f} {r['target']['t90']}")
    for c in out["contrast"]:
        print(f"contrast u={c['updates']}: target {c['target']['mean']:+.3f} {c['target']['t90']}  "
              f"return {c['return']['mean']:+.3f} {c['return']['t90']}")
    if a.figure:
        figure(series, a.figure)
        print(f"wrote {a.figure}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
