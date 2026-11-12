"""Draw-crossed summary of a pooled readout: agents by evaluation-world draws.

Frozen in docs/specs/2026-11-12-draw-crossed-readout-design.md. The registered pooled
readout scores every agent seed on one shared pair of evaluation-world pools, so an
interval over agent seeds is conditional on that draw. This module holds the pure
arithmetic of the crossed design, with no world, agent or probe code in it, so the
verification script can recompute every summary from the per-cell values of the artifact.

A matrix here is seeds by draws: M[s][k] is the reading of agent seed s on draw k. The
readings are signed and are never folded about chance.
"""

from __future__ import annotations

import math

import numpy as np

BAR = 0.65
MARGIN = 0.05
BAND = (0.45, 0.55)
LEVEL = 0.90

NOT_COMPUTABLE = "not computable"
VOID = "VOID (integrity gate failed)"


def crossed_interval(M, level: float = LEVEL) -> dict:
    """Grand mean of a seeds-by-draws matrix with a t-based interval that treats both
    agent seeds and draws as random.

    Two-way crossed layout with one reading per cell. With S seeds and D draws,
    MS_seed = D * sum_s (mean_s - mean)^2 / (S - 1), MS_draw = S * sum_k (mean_k - mean)^2
    / (D - 1) and MS_resid = sum (residual)^2 / ((S - 1)(D - 1)). The variance of the grand
    mean is estimated by (MS_seed + MS_draw - MS_resid) / (S * D), with Satterthwaite
    degrees of freedom for that combination. The interval is approximate. When the variance
    estimate is not positive, the degrees of freedom are below one, or any reading is not
    finite, no interval is returned and `computable` is False: a missing interval is never
    read as a favorable one.
    """
    from scipy import stats

    A = np.asarray(M, dtype=float)
    out = {"n_seeds": int(A.shape[0]) if A.ndim == 2 else 0,
           "n_draws": int(A.shape[1]) if A.ndim == 2 else 0,
           "mean": float("nan"), "lo": None, "hi": None, "se": None, "df": None,
           "level": level, "computable": False,
           "seed_means": [], "draw_means": [], "sd_seed_means": None, "sd_draw_means": None,
           "ms_seed": None, "ms_draw": None, "ms_resid": None}
    if A.ndim != 2 or A.shape[0] < 2 or A.shape[1] < 2 or not np.all(np.isfinite(A)):
        return out
    S, D = A.shape
    g = float(A.mean())
    a = A.mean(axis=1)          # seed means
    b = A.mean(axis=0)          # draw means
    ms_seed = D * float(((a - g) ** 2).sum()) / (S - 1)
    ms_draw = S * float(((b - g) ** 2).sum()) / (D - 1)
    resid = A - a[:, None] - b[None, :] + g
    ms_resid = float((resid ** 2).sum()) / ((S - 1) * (D - 1))
    out.update({"mean": g, "seed_means": [float(x) for x in a],
                "draw_means": [float(x) for x in b],
                "sd_seed_means": float(a.std(ddof=1)), "sd_draw_means": float(b.std(ddof=1)),
                "ms_seed": ms_seed, "ms_draw": ms_draw, "ms_resid": ms_resid})
    comb = ms_seed + ms_draw - ms_resid
    denom = (ms_seed ** 2 / (S - 1) + ms_draw ** 2 / (D - 1)
             + ms_resid ** 2 / ((S - 1) * (D - 1)))
    if not (comb > 0.0) or not (denom > 0.0):
        return out
    df = comb ** 2 / denom
    if not math.isfinite(df) or df < 1.0:
        return out
    se = math.sqrt(comb / (S * D))
    t = float(stats.t.ppf(0.5 + level / 2.0, df))
    out.update({"lo": g - t * se, "hi": g + t * se, "se": se, "df": float(df),
                "computable": True})
    return out


def label_level(ci: dict, bar: float = BAR) -> str:
    """Where a crossed interval lies against the registered bar."""
    if not ci.get("computable"):
        return NOT_COMPUTABLE
    if ci["lo"] >= bar:
        return "above the bar"
    if ci["hi"] < bar:
        return "below the bar"
    return "straddles the bar"


def label_band(ci: dict, band=BAND) -> str:
    """Where a crossed interval lies against the registered equivalence band about chance."""
    if not ci.get("computable"):
        return NOT_COMPUTABLE
    lo_edge, hi_edge = band
    if ci["lo"] >= lo_edge and ci["hi"] <= hi_edge:
        return "inside the band"
    if ci["hi"] < lo_edge or ci["lo"] > hi_edge:
        return "outside the band"
    return "straddles the band edge"


def label_margin(ci: dict, margin: float = MARGIN) -> str:
    """Where a crossed interval of a difference lies against the registered margin."""
    if not ci.get("computable"):
        return NOT_COMPUTABLE
    if ci["lo"] > -margin and ci["hi"] < margin:
        return "within the margin"
    if ci["lo"] >= margin:
        return "above the margin"
    if ci["hi"] < margin:
        return "below the margin"
    return "straddles the margin"


def fmt_ci(ci: dict, signed: bool = False) -> str:
    """A summary as it is quoted in prose: mean and crossed 90% interval to three decimals,
    with a sign on every number of a contrast."""
    def f(x: float) -> str:
        if not signed:
            return f"{x:.3f}"
        s = f"{x:+.3f}"
        return "+0.000" if s == "-0.000" else s   # a value that rounds to zero prints +0.000
    mean = ci.get("mean")
    if mean is None or not math.isfinite(float(mean)):
        return "not computable"
    if not ci.get("computable"):
        return f(mean) + " [not computable]"
    return f"{f(mean)} [{f(ci['lo'])}, {f(ci['hi'])}]"


# The feature sets scored in every (run, drift, seed, draw) cell.
OWN = ("own/survival", "own/predictor", "own/untrained")
REPLAY = ("replay/predictor", "replay/untrained")
REPLAY_SNORM = ("replay_snorm/predictor", "replay_snorm/untrained")
FEATURE_SETS = OWN + REPLAY + REPLAY_SNORM
PROBES = ("nested", "c1")      # the penalty chosen in fold (primary), and the registered C = 1

# Seed-and-draw-paired contrasts inside one run: (name, minuend, subtrahend).
CONTRASTS = (
    ("survival minus predictor, own walks", "own/survival", "own/predictor"),
    ("survival minus untrained, own walks", "own/survival", "own/untrained"),
    ("survival minus predictor core on the survival stream", "own/survival", "replay/predictor"),
    ("survival minus untrained core on the survival stream", "own/survival", "replay/untrained"),
    ("survival minus predictor core on the survival stream, survival normalizer",
     "own/survival", "replay_snorm/predictor"),
    ("survival minus untrained core on the survival stream, survival normalizer",
     "own/survival", "replay_snorm/untrained"),
)


def matrix(cells: list, run: str, drift: float, feature_set: str, probe: str,
           bases: list) -> list:
    """Seeds-by-draws matrix of one reading, rows in seed order and columns in `bases` order.
    A missing cell or draw yields NaN, which makes every summary of it not computable."""
    rows = sorted((c for c in cells if c["run"] == run and float(c["drift"]) == float(drift)),
                  key=lambda c: c["seed"])
    out = []
    for c in rows:
        by_base = {tuple(d["bases"]): d for d in c["draws"]}
        row = []
        for b in bases:
            d = by_base.get(tuple(b))
            v = d["readings"][feature_set][probe] if d is not None else None
            row.append(float(v) if v is not None else float("nan"))
        out.append(row)
    return out


def summarize(cells: list, runs, drifts, fresh_bases, registered_bases,
              integrity: dict) -> dict:
    """Every summary the frozen design names, recomputable from `cells` alone.

    Per run, drift and probe: the crossed interval of each feature set over the fresh
    draws, the crossed interval of each contrast, and the pre-stated labels. The registered
    draw is reported beside them as seed means and enters no crossed interval. A run whose
    integrity gate failed carries VOID in place of every label.
    """
    out = {}
    for run in runs:
        ok = bool(integrity.get(run, {}).get("pass"))
        out[run] = {}
        for drift in drifts:
            dkey = f"{float(drift):.2f}"
            out[run][dkey] = {}
            for probe in PROBES:
                block = {"readings": {}, "contrasts": {}, "registered_draw": {}}
                mats = {fs: matrix(cells, run, drift, fs, probe, fresh_bases)
                        for fs in FEATURE_SETS}
                for fs in FEATURE_SETS:
                    ci = crossed_interval(mats[fs])
                    if fs == "own/survival":
                        lab = label_band(ci) if float(drift) == 0.0 else label_level(ci)
                    elif float(drift) == 0.0:
                        lab = label_band(ci)
                    else:
                        lab = None      # baselines at a nonzero drift carry no label
                    if lab is not None and not ok:
                        lab = VOID
                    block["readings"][fs] = {**ci, "label": lab}
                    reg = matrix(cells, run, drift, fs, probe, [registered_bases])
                    col = [r[0] for r in reg]
                    block["registered_draw"][fs] = {
                        "per_seed": col,
                        "mean": (float(np.mean(col)) if col and np.all(np.isfinite(col))
                                 else float("nan"))}
                for name, plus, minus in CONTRASTS:
                    diff = (np.asarray(mats[plus], float) - np.asarray(mats[minus], float))
                    ci = crossed_interval(diff)
                    lab = label_margin(ci) if float(drift) != 0.0 else None
                    if lab is not None and not ok:
                        lab = VOID
                    block["contrasts"][name] = {**ci, "label": lab}
                out[run][dkey][probe] = block
    return out


def summarize_between_runs(cells: list, run_a: str, run_b: str, drifts, fresh_bases,
                           integrity: dict) -> dict:
    """The survival arm of `run_a` minus that of `run_b` on their own walks, paired by agent
    seed index and by draw. The two runs were trained apart, as the record notes for the
    registered auxiliary comparison."""
    ok = (bool(integrity.get(run_a, {}).get("pass"))
          and bool(integrity.get(run_b, {}).get("pass")))
    out = {}
    for drift in drifts:
        dkey = f"{float(drift):.2f}"
        out[dkey] = {}
        for probe in PROBES:
            a = np.asarray(matrix(cells, run_a, drift, "own/survival", probe, fresh_bases), float)
            b = np.asarray(matrix(cells, run_b, drift, "own/survival", probe, fresh_bases), float)
            ci = crossed_interval(a - b) if a.shape == b.shape else crossed_interval([[]])
            lab = label_margin(ci) if float(drift) != 0.0 else None
            if lab is not None and not ok:
                lab = VOID
            out[dkey][probe] = {**ci, "label": lab}
    return out


def integrity_gate(cells: list, reference: dict, registered_bases, tol: float = 0.01) -> dict:
    """Per run: every agent's registered-draw reading at C = 1 on its own walk against the
    recorded per-seed target. `reference[run][drift_key][arm]` is the recorded list in seed
    order. A run passes only when every one of its agents is checked and within `tol`."""
    out = {}
    for run, ref in reference.items():
        checks = []
        for c in cells:
            if c["run"] != run:
                continue
            dkey = f"{float(c['drift']):.2f}"
            by_base = {tuple(d["bases"]): d for d in c["draws"]}
            d = by_base.get(tuple(registered_bases))
            for arm in ("survival", "predictor", "untrained"):
                want = ref.get(dkey, {}).get(arm)
                if want is None or c["seed"] >= len(want):
                    continue
                got = d["readings"][f"own/{arm}"]["c1"] if d is not None else None
                finite = got is not None and math.isfinite(float(got))
                dev = abs(float(got) - float(want[c["seed"]])) if finite else None
                checks.append({"drift": dkey, "seed": c["seed"], "arm": arm,
                               "recorded": float(want[c["seed"]]),
                               "rescored": (float(got) if finite else None),
                               "abs_dev": dev})
        n_expected = sum(len(v) for dk in ref.values() for v in dk.values())
        devs = [k["abs_dev"] for k in checks]
        all_scored = bool(devs) and all(v is not None for v in devs)
        worst = max(devs) if all_scored else None
        out[run] = {"tolerance": tol, "n_expected": n_expected, "n_checked": len(checks),
                    "worst_abs_dev": worst,
                    "pass": bool(all_scored and len(checks) == n_expected and worst <= tol),
                    "per_agent": checks}
    return out
