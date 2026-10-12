"""Experiment C selection intensity, per generation (finding Q5).

Milestone 3 fixes each arm's reproduction threshold at the median of that arm's generation-0
fitness (`scripts/run_expC_milestone3.py run_arm`, q = 0.5) and keeps it for all 30
generations. `itasorl.neuroevolution.evolve` counts, before each reproduction, how many
individuals of the current generation reach that fixed threshold (`n_qualifiers`). The
per-individual rerun (fullruns/expC_milestone3/emergence_pilot_per_individual.json, commit
06ecc10) recorded those counts; the main milestone output (emergence_pilot.json, commit
a0cb850) did not. Evolution is deterministic, and the two outputs carry identical fitness
series and thresholds for every seed and arm (checked here), so the counts describe both.
Neither committed summary (artifacts/expC/*_summary.json) carries the series.

Writes, per seed and arm: the frozen threshold, the qualifier count and fraction of the
48 individuals per generation, the first generation with at least 90% qualifiers, the first
generation from which every later one stays at or above 90%, the mean qualifying share over
generations 5 to 29, and the mean-fitness trajectory with its peak and end. Generation g's
record describes generation g's population before its reproduction, so the population that
is panelled at the end (the product of generation 29's reproduction) is not in the series.

Usage:  python scripts/expC_selection_intensity.py [--out artifacts/expC/selection_intensity.json]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

from agent_free_common import provenance, write_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_FULLRUNS = os.environ.get("ITASORL_FULLRUNS",
                                  os.path.join(os.path.dirname(os.path.dirname(ROOT)), "fullruns")
                                  if os.path.basename(os.path.dirname(ROOT)) == ".worktrees"
                                  else os.path.join(ROOT, "fullruns"))
PER_IND = "fullruns/expC_milestone3/emergence_pilot_per_individual.json"
MAIN = "fullruns/expC_milestone3/emergence_pilot.json"
SHARE = 0.90
LATE_FROM = 5
ARMS = (("treat", "treatment"), ("ctrl", "control"))


def _full(rel: str) -> str:
    return os.path.join(PRIMARY_FULLRUNS, rel[len("fullruns/"):])


def arm_block(n: int, threshold: float, quals: list[int], fit: list[float]) -> dict:
    """Everything the artifact reports for one seed and arm, from the two series alone."""
    q = np.asarray(quals, int)
    frac = q / float(n)
    f = np.asarray(fit, float)
    at = np.flatnonzero(frac >= SHARE)
    first = int(at[0]) if at.size else None
    stays = None
    for g in range(len(frac)):
        if np.all(frac[g:] >= SHARE):
            stays = g
            break
    peak = int(np.argmax(f))
    return {"threshold_gen0_median": threshold,
            "n_qualifiers": q.tolist(), "qualifier_fraction": frac.tolist(),
            "removed_per_generation": (n - q).tolist(),
            "first_gen_at_least_90pct": first,
            "first_gen_staying_at_least_90pct": stays,
            "gens_below_90pct_after_first": [int(g) for g in range(first + 1, len(frac))
                                             if frac[g] < SHARE] if first is not None else None,
            f"mean_share_gens_{LATE_FROM}_to_{len(frac) - 1}": float(frac[LATE_FROM:].mean()),
            "min_share_after_gen0": float(frac[1:].min()),
            "mean_fitness": f.tolist(),
            "fitness_gen0": float(f[0]), "fitness_last": float(f[-1]),
            "fitness_last_minus_gen0": float(f[-1] - f[0]),
            "fitness_peak_gen": peak, "fitness_peak": float(f[peak])}


def build(per_ind: dict, main: dict | None) -> dict:
    n = int(per_ind["config"]["n"])
    seeds = []
    same = {}
    for row in per_ind["per_seed"]:
        s = int(row["seed"])
        block = {"seed": s}
        for key, name in ARMS:
            block[name] = arm_block(n, float(row[f"threshold_{key}"]),
                                    row[f"n_qualifiers_{key}"], row[f"fit_series_{key}"])
            block[name]["fit_delta_recorded"] = row[f"fit_delta_{key}"]
            if main is not None:
                mrow = next(r for r in main["per_seed"] if int(r["seed"]) == s)
                same[f"seed {s} {name}"] = bool(
                    mrow[f"fit_series_{key}"] == row[f"fit_series_{key}"]
                    and mrow[f"threshold_{key}"] == row[f"threshold_{key}"])
        seeds.append(block)
    late = f"mean_share_gens_{LATE_FROM}_to_{int(per_ind['config']['generations']) - 1}"
    summary = {name: {"first_gen_at_least_90pct": [b[name]["first_gen_at_least_90pct"] for b in seeds],
                      "first_gen_staying_at_least_90pct":
                          [b[name]["first_gen_staying_at_least_90pct"] for b in seeds],
                      late: [b[name][late] for b in seeds],
                      "fitness_last_minus_gen0": [b[name]["fitness_last_minus_gen0"] for b in seeds]}
               for _, name in ARMS}
    return {"n_individuals": n, "generations": int(per_ind["config"]["generations"]),
            "quantile": per_ind["config"]["q"], "share_bar": SHARE,
            "main_run_series_identical": same if main is not None else None,
            "summary": summary, "per_seed": seeds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/expC/selection_intensity.json")
    a = ap.parse_args()
    with open(_full(PER_IND), encoding="utf-8") as fh:
        per_ind = json.load(fh)
    main_doc = None
    if os.path.exists(_full(MAIN)):
        with open(_full(MAIN), encoding="utf-8") as fh:
            main_doc = json.load(fh)
    body = build(per_ind, main_doc)
    if main_doc is not None and not all(body["main_run_series_identical"].values()):
        raise SystemExit("main milestone output and per-individual rerun disagree on a series")
    doc = {
        **provenance("scripts/expC_selection_intensity.py", "git_commit_at_promotion"),
        "git_commit_at_run": per_ind.get("git_commit_at_run"),
        "finding": "Q5: Experiment C selection intensity under the frozen generation-0 median",
        "status": "DESCRIPTIVE: milestone 3 is routed UNINFORMATIVE (gates 1 and 5 failed)",
        "sources": {"per_individual": {"path": PER_IND,
                                       "git_commit_at_run": per_ind.get("git_commit_at_run")},
                    "main": {"path": MAIN, "git_commit_at_run":
                             main_doc.get("git_commit_at_run") if main_doc else None,
                             "present": main_doc is not None}},
        "definitions": {
            "threshold": "np.quantile(arm's generation-0 fitness, 0.5), fixed for every generation",
            "qualifier": "an individual of generation g whose lifetime fitness is at or above the "
                         "threshold (itasorl.neuroevolution.evolve n_qualifiers), counted before "
                         "generation g reproduces",
            "mean_fitness": "mean lifetime fitness of generation g before its reproduction; the "
                            "final reproduced population is not scored",
            "fit_delta_recorded": "the run's own fit_delta (last minus first of the series, "
                                  "rounded series)"},
        **body,
    }
    out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    write_json(out_path, doc)
    for b in body["per_seed"]:
        for _, name in ARMS:
            e = b[name]
            print(f"seed {b['seed']} {name:9s} first>=90% gen {e['first_gen_at_least_90pct']} "
                  f"(stays from {e['first_gen_staying_at_least_90pct']}) late share "
                  f"{e[f'mean_share_gens_{LATE_FROM}_to_29']:.3f}  fitness {e['fitness_gen0']:+.3f} -> "
                  f"{e['fitness_last']:+.3f} (peak {e['fitness_peak']:+.3f} at gen "
                  f"{e['fitness_peak_gen']})")
    print(f"wrote {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
