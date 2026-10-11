"""Promote an H2 substrate-grounding ablation (A1 graded seam) to a committed artifact.

The A1 graded-seam sweep numbers in FINDINGS come from gitignored run bundles: hidden 8 from
`fullruns/expH2_ablation/aggregate.json` and the hidden-7 capacity replication from
`fullruns/l3_h7_ablation/aggregate.json` (scripts/run_expH2_ablation.py --hidden 7). This
script extracts the decision-relevant per-seed values (the pooled targets of every arm the
bundle scored at each alpha, the collapse curve, its monotonicity, and the alpha=0 L0 anchor)
and writes a compact, committed summary JSON with provenance, so every published H2 number is
recomputable in-repo.

Like the Exp C bundle there is no per-cell fingerprint hash; the run's identity is the frozen
world P and the frozen L3 surrogate (hidden = --l3-hidden, G seed 0), recorded verbatim. The
bundles do not record the commit they ran at, so `git_commit_at_run` is whatever
--commit-at-run says (a curated value, labelled as such) or "not recorded in the bundle";
the promote-time HEAD is recorded separately as `git_commit_at_promotion`. (The hidden-8
artifact, promoted before this distinction, carries its promote-time HEAD c4e4417 in both
fields; the runner and that promotion sat on the same commit.)

The L0 anchor's "hdi" is the 95% percentile-bootstrap interval of the seed mean
(itasorl.stats.rope_test), not a posterior interval; the key keeps the bundle's name.

Usage:
    python scripts/promote_expH2_summary.py \\
        --run fullruns/expH2_ablation/aggregate.json \\
        --out artifacts/expH2/summary.json --commit-at-run c4e4417ea69aa21b6594a46e095f53a162f0a4ab
    python scripts/promote_expH2_summary.py \\
        --run fullruns/l3_h7_ablation/aggregate.json --l3-hidden 7 \\
        --out artifacts/expH2/summary_h7.json --commit-at-run "<curated note>"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promote_bv3_gates import repo_rel  # noqa: E402

NOT_RECORDED = "not recorded in the bundle"
INTERVAL_NOTE = ("l0_anchor_alpha0.hdi is the 95% percentile-bootstrap interval of the seed "
                 "mean (itasorl.stats.rope_test, 20000 resamples, seed 0), not a posterior "
                 "interval; p_in_rope is the bootstrap share inside [0.45, 0.55]")


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def summarize(run: dict, *, source: str, l3_hidden: int, commit_at_run: str | None,
              head: str, device: str = "cuda") -> dict:
    alphas = run["alphas"]
    arms = run["arms"]
    cells: dict[str, dict[str, dict]] = {}
    for arm in arms:
        cells[arm] = {}
        for al in alphas:
            key = f"{arm}_a{al:.2f}"
            cells[arm][f"a{al:.2f}"] = {
                "alpha": al,
                "per_seed": run[f"{key}_per_seed"],
                "mean": run[f"{key}_mean"],
                "tci90": run[f"{key}_tci90"],
                "n_ge_065": run[f"{key}_n_ge_065"],
            }
    out = {
        "source_run": repo_rel(source),
        "world": "WorldParams(k_land=1.5, k_water=1.5, gravity=0.4) [P]",
        "surrogate": (f"L3 GMotion hidden={l3_hidden} seed=0 trained on P; A1 seam "
                      "neutralization = GradedGMotion, convex blend "
                      "(1-alpha)*authentic_law + alpha*G"),
        "git_commit_at_run": commit_at_run or run.get("git_commit") or NOT_RECORDED,
        "git_commit_at_promotion": head,
        "generated_by": "scripts/promote_expH2_summary.py",
        "bars": {"auroc_floor": 0.65, "l0_rope": [0.45, 0.55]},
        "config": {"drift": run["drift"], "alphas": alphas, "arms": arms,
                   "n_eps": run["n_eps"], "steps": run["steps"],
                   "n_seeds": len(cells[arms[0]]["a1.00"]["per_seed"]),
                   "device": device, "l3_hidden": l3_hidden},
        "survival_curve": run["survival_curve"],
        "survival_monotonicity_rho": run["survival_monotonicity_rho"],
        "l0_anchor_alpha0": run["l0_anchor_alpha0"],
        "interval_note": INTERVAL_NOTE,
        "integrity": run["integrity"],
        "cells": cells,
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="fullruns/expH2_ablation/aggregate.json")
    ap.add_argument("--out", default="artifacts/expH2/summary.json")
    ap.add_argument("--l3-hidden", type=int, default=8,
                    help="capacity of the frozen L3 surrogate the ablated agents were raised "
                         "against (8 forward, 7 reverse); the agent trunk is 96 either way")
    ap.add_argument("--commit-at-run", default=None,
                    help="the commit the run executed at, when known from outside the bundle "
                         "(the bundles record none); recorded as given")
    ap.add_argument("--device", default="cuda", help="device the bundle's log records")
    a = ap.parse_args(argv)

    with open(a.run, encoding="utf-8") as fh:
        run = json.load(fh)
    out = summarize(run, source=a.run, l3_hidden=a.l3_hidden, commit_at_run=a.commit_at_run,
                    head=git_head(), device=a.device)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, a.out)
    curve = out["survival_curve"]
    print(f"wrote {a.out}  (hidden {a.l3_hidden}; survival curve {curve[0]:.3f}->{curve[-1]:.3f}, "
          f"rho={out['survival_monotonicity_rho']}, L0 accept={out['l0_anchor_alpha0']['accept_equiv']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
