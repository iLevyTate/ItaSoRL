"""Budget-curve dynamics: a seed-paired re-read of the stored budget curve (closing item X08, E17).

Agent-free and post hoc. The proposal's numbers were seen in a scratch read before this rule
was written, so every number here is descriptive and changes no stored value or verdict.
Per run it reads the 10 x 4 matrices of pooled target, survival return and trained episode
length at 100, 200, 300 and 450 updates: target and return from artifacts/budget_curve.json
(series[<label>][*].target.per_seed and .return.per_seed), trained_len from the run's 20
cells (cell.budget_curve.<u>.eng.trained_len at the run's strongest drift). Gate: the target
and return arrays the cells hold equal the committed artifact bit for bit, else nothing is
reported and the script exits 1.

Quantities per run:
- paired_t90 (itasorl.novel_readouts) of every adjacent segment (100 to 200, 200 to 300,
  300 to 450) for target, return and trained_len;
- the across-seed Spearman of target against return and against trained_len at each budget,
  with a two-sided permutation p from 10,000 draws;
- the within-seed partial slopes of target on return and on the update count (per 100
  updates), each variable demeaned within seed, with a 90 percent seed-bootstrap interval
  (10,000 resamples of whole seeds).
Labels per run: "tracks return" (return slope lower bound above 0 and the update slope's
interval includes 0), "tracks budget" (update slope lower bound above 0 and the return slope's
interval includes 0), "both" (both lower bounds above 0), otherwise "neither". The 100 to 200
segment reads "decodability moves without return" only if the target interval excludes 0
while the return and trained_len intervals both include 0. The design is thin: four budgets
per seed, collinear regressors, and a survival length that moves about one step across the
ladder.

    python scripts/read_budget_curve_dynamics.py --budget-curve artifacts/budget_curve.json \\
        --run wm=fullruns/corrected_l3_h8_wm --run nowm=fullruns/corrected_l3_h8_nowm \\
        --out fullruns/closing_2026-10/x08/budget_curve_dynamics.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import sys

import numpy as np

GENERATED_BY = "scripts/read_budget_curve_dynamics.py"
BUDGETS = (100, 200, 300, 450)
N_PERM = 10_000
N_BOOT = 10_000
PERM_SEED = 20_261_017
BOOT_SEED = 20_261_018
STATUS = ("post hoc, descriptive (disclosed: the proposal's numbers were seen in a scratch read "
          "before the rule was written); changes no stored value or verdict")


def load_cells(run_dir: str) -> tuple[float, dict]:
    """{updates: {seed: {"target", "return", "trained_len"}}} at the run's strongest drift."""
    cells = []
    for p in sorted(glob.glob(os.path.join(run_dir, "cells", "cell_d*_s*.json"))):
        with open(p, encoding="utf-8") as fh:
            cells.append(json.load(fh)["cell"])
    if not cells:
        raise SystemExit(f"no cells under {run_dir}/cells")
    dmax = max(float(c["drift"]) for c in cells)
    out: dict = {}
    for c in cells:
        if float(c["drift"]) != dmax or "budget_curve" not in c:
            continue
        for u, v in c["budget_curve"].items():
            out.setdefault(int(u), {})[int(c["seed"])] = {
                "target": float(v["pool"]["target"]), "return": float(v["eng"]["trained_return"]),
                "trained_len": float(v["eng"]["trained_len"])}
    return dmax, out


def series_label(curve: dict, run_dir: str) -> str:
    """The budget-curve series whose recorded run directory has the same basename as run_dir."""
    name = os.path.basename(os.path.normpath(run_dir))
    hits = [lab for lab, path in curve["runs"].items()
            if os.path.basename(os.path.normpath(str(path))) == name]
    if len(hits) != 1:
        raise SystemExit(f"{run_dir}: expected one budget-curve series recorded for {name!r}, "
                         f"found {hits}; series runs are {curve['runs']}")
    return hits[0]


def matrices(curve: dict, label: str, by_u: dict) -> dict:
    """Seeds and the (seeds x budgets) target, return and trained_len matrices. Raises
    SystemExit unless the artifact's target and return arrays equal the cells' bit for bit."""
    pts = {int(p["updates"]): p for p in curve["series"][label]}
    if tuple(sorted(pts)) != BUDGETS or tuple(sorted(by_u)) != BUDGETS:
        raise SystemExit(f"{label}: budgets {sorted(pts)} in the artifact and {sorted(by_u)} in the "
                         f"cells; expected {list(BUDGETS)}")
    seeds = [int(s) for s in pts[BUDGETS[0]]["seeds"]]
    mats = {k: np.zeros((len(seeds), len(BUDGETS))) for k in ("target", "return", "trained_len")}
    mismatches = []
    for j, u in enumerate(BUDGETS):
        p = pts[u]
        if [int(s) for s in p["seeds"]] != seeds or sorted(by_u[u]) != sorted(seeds):
            raise SystemExit(f"{label} {u}: seed lists differ between the artifact and the cells")
        for i, s in enumerate(seeds):
            cell = by_u[u][s]
            for k in ("target", "return"):
                stored = float(p[k]["per_seed"][i])
                if stored != cell[k]:
                    mismatches.append(f"{label} u={u} seed {s} {k}: artifact {stored!r} cell {cell[k]!r}")
                mats[k][i, j] = stored
            mats["trained_len"][i, j] = cell["trained_len"]
    if mismatches:
        raise SystemExit("input gate failed (arrays not bit-equal):\n  " + "\n  ".join(mismatches))
    return {"seeds": seeds, **mats}


def segments(M: np.ndarray) -> dict:
    from itasorl.novel_readouts import paired_t90
    return {f"{BUDGETS[j]}->{BUDGETS[j + 1]}": paired_t90(M[:, j + 1], M[:, j])
            for j in range(len(BUDGETS) - 1)}


def spearman_perm(x, y, n_perm: int = N_PERM, seed: int = PERM_SEED) -> dict:
    """Across-seed Spearman rho with a two-sided permutation p ((1 + exceedances) / (1 + draws))."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)

    def rho(a, b):
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt((a @ a) * (b @ b))
        return float(a @ b / den) if den > 0 else float("nan")

    obs = rho(rx, ry)
    if not np.isfinite(obs):
        return {"rho": obs, "p_perm": float("nan"), "n_perm": n_perm}
    rng = np.random.default_rng(seed)
    hits = sum(abs(rho(rx, rng.permutation(ry))) >= abs(obs) - 1e-12 for _ in range(n_perm))
    return {"rho": obs, "p_perm": float((1 + hits) / (1 + n_perm)), "n_perm": n_perm}


def partial_slopes(T: np.ndarray, R: np.ndarray, U: np.ndarray) -> np.ndarray:
    """OLS slopes of target on (return, updates / 100), every variable demeaned within seed."""
    dm = lambda M: (M - M.mean(1, keepdims=True)).ravel()  # noqa: E731
    X = np.column_stack([dm(R), dm(U / 100.0)])
    beta, *_ = np.linalg.lstsq(X, dm(T), rcond=None)
    return beta


def slopes_with_interval(T, R, U, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """The two partial slopes and their 90 percent seed-bootstrap percentile intervals."""
    beta = partial_slopes(T, R, U)
    rng = np.random.default_rng(seed)
    n = T.shape[0]
    draws = np.array([partial_slopes(T[idx], R[idx], U[idx])
                      for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    out = {}
    for j, name in enumerate(("return", "updates_per_100")):
        lo, hi = np.percentile(draws[:, j], [5, 95])
        out[name] = {"slope": float(beta[j]), "boot90": [float(lo), float(hi)], "n_boot": n_boot}
    return out


def _includes_zero(iv) -> bool:
    return bool(iv[0] <= 0.0 <= iv[1])


def label(sl: dict) -> str:
    r, u = sl["return"]["boot90"], sl["updates_per_100"]["boot90"]
    if r[0] > 0 and u[0] > 0:
        return "both"
    if r[0] > 0 and _includes_zero(u):
        return "tracks return"
    if u[0] > 0 and _includes_zero(r):
        return "tracks budget"
    return "neither"


def read_run(curve: dict, name: str, run_dir: str) -> dict:
    lab = series_label(curve, run_dir)
    drift, by_u = load_cells(run_dir)
    if float(curve["drift"][lab]) != drift:
        raise SystemExit(f"{name}: the artifact records drift {curve['drift'][lab]} for {lab!r}, "
                         f"the cells' strongest drift is {drift}")
    m = matrices(curve, lab, by_u)
    T, R, L = m["target"], m["return"], m["trained_len"]
    U = np.tile(np.asarray(BUDGETS, float), (T.shape[0], 1))
    seg = {k: segments(M) for k, M in (("target", T), ("return", R), ("trained_len", L))}
    first = f"{BUDGETS[0]}->{BUDGETS[1]}"
    moves = (not _includes_zero(seg["target"][first]["t90"])
             and _includes_zero(seg["return"][first]["t90"])
             and _includes_zero(seg["trained_len"][first]["t90"]))
    sl = slopes_with_interval(T, R, U)
    return {"series_label": lab, "drift": drift, "seeds": m["seeds"], "budgets": list(BUDGETS),
            "matrices": {"target": T.tolist(), "return": R.tolist(), "trained_len": L.tolist()},
            "segments": seg,
            "spearman": {str(u): {"target_vs_return": spearman_perm(T[:, j], R[:, j]),
                                  "target_vs_trained_len": spearman_perm(T[:, j], L[:, j])}
                         for j, u in enumerate(BUDGETS)},
            "partial_slopes": sl,
            "label": label(sl),
            f"decodability moves without return ({first})": bool(moves)}


def run_provenance(out: str) -> dict:
    from _readout_common import _rel_argv
    from agent_free_common import provenance
    from promote_bv3_gates import repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit", output=repo_rel(out))
    return {**p, "argv": _rel_argv(list(sys.argv[1:]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget-curve", default="artifacts/budget_curve.json")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR",
                    help="a run whose 20 cells supply trained_len (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="check the inputs and print the plan; write nothing")
    a = ap.parse_args(argv)
    runs = {}
    for spec in a.run:
        name, sep, d = spec.partition("=")
        if not sep:
            ap.error(f"--run expects NAME=RUN_DIR, got {spec!r}")
        runs[name] = d
    with open(a.budget_curve, encoding="utf-8") as fh:
        curve = json.load(fh)
    if a.dry_run:
        for name, d in runs.items():
            lab = series_label(curve, d)
            _, by_u = load_cells(d)
            matrices(curve, lab, by_u)
            print(f"{name}: series {lab!r}, {len(by_u)} budgets; input gate passes")
        print(f"would write {a.out}")
        return 0
    from promote_bv3_gates import repo_rel
    body = {"status": STATUS, "budget_curve": repo_rel(a.budget_curve),
            "runs": {name: {"run_dir": repo_rel(d), **read_run(curve, name, d)} for name, d in runs.items()},
            "input_gate": "target and return arrays bit-equal to the committed artifact and the cells"}
    from agent_free_common import write_json
    write_json(a.out, {"provenance": run_provenance(a.out), **body})
    for name, r in body["runs"].items():
        sl = r["partial_slopes"]
        print(f"{name}: {r['label']}; return slope {sl['return']['slope']:+.4f} {sl['return']['boot90']}, "
              f"update slope {sl['updates_per_100']['slope']:+.4f} {sl['updates_per_100']['boot90']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
