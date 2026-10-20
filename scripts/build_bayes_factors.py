"""Bayes factors with an interval null on committed per-seed arrays (closing item X09, E19).

Agent-free, post hoc, and an annotation only: it never relabels a registered reading (the
standing rule forbids re-adjudicating a stored run with a new estimator), and it is kept apart
from the bootstrap ROPE check of itasorl/stats.py, which says it is not a Bayesian analysis.
Every row is conditional on one evaluation-world draw.

Model: a one-sample t-model on each array (a paired array is its per-seed difference), with
the JZS prior on the standardized effect delta = (mu - mu0) / sigma: Cauchy(0, r) on delta and
p(sigma) proportional to 1 / sigma. Integrating sigma out leaves the noncentral-t likelihood
of the t statistic, t ~ nct(df = n - 1, ncp = delta sqrt(n)), so every marginal likelihood
below is a one-dimensional integral over delta (scipy.integrate.quad).
- H0: the effect lies inside the ROPE, |mu - mu0| <= h, with h = 0.05 for every reading here
  (ROPE [0.45, 0.55] for an AUROC read against chance, plus or minus 0.05 for the sweep C
  contrasts and the L1 margins). The raw half-width is put on the delta scale with the
  array's sample standard deviation, c = h / sd (a plug-in, recorded per row).
- H1: the effect lies outside it. BF01 = (posterior odds of H0) / (prior odds of H0) under
  the encompassing JZS model, with prior mass P(|delta| <= c) = (2 / pi) atan(c / r).
- The point-null JZS BF01 (H0: delta = 0) is reported beside it.
Because c uses the observed sd, the interval-null factors are approximate plug-in sensitivity
calculations, not exact Bayes factors for the fixed raw-scale band: that band is
|delta| sigma <= h, and the plug-in does not integrate the uncertainty in sigma.
Scales: r = 0.707 is read; 0.5 and 1.0 are reported beside it. Labels (r = 0.707 only):
BF01 above 3 "evidence for the band", below 1/3 "evidence against the band", otherwise
"indeterminate". Outside the one appendix table that holds them, a row is quoted as
"BF01 = x (annotation, 0.707 scale)" with no band word.

Inputs are fixed here. Gate: every input file must equal its committed version (git HEAD)
byte for byte, and each derived array must reproduce the summary value its artifact stores;
otherwise nothing is reported and the script exits 1.

    python scripts/build_bayes_factors.py --out fullruns/closing_2026-10/x09/interval_null.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

GENERATED_BY = "scripts/build_bayes_factors.py"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCALES = (0.5, 0.707, 1.0)
LABEL_SCALE = 0.707
HALF = 0.05
STATUS = ("annotation only, post hoc: never relabels a registered reading; kept apart from the "
          "bootstrap ROPE check (itasorl/stats.py), which is not a Bayesian analysis; every row is "
          "conditional on one evaluation-world draw")
INPUTS = {
    "c1": "artifacts/corrected_verdicts.json",
    "l0": "artifacts/l0_audit/corrected_l3_h8_wm.json",
    "sweep_c": "artifacts/texture/ar_sweep_c/summary.json",
    "sweep_c_rho0": "artifacts/texture/ar_sweep_c/C_rho0.00.json",
    "sweep_c_followup": "artifacts/texture/ar_sweep_c_followup/summary.json",
    "l1": "artifacts/expB2/corrected_L1-corrected.json",
    "balanced": "artifacts/balanced_pooled/registered_pair.json",
}


# ---------------------------------------------------------------------------
# Bayes factors
# ---------------------------------------------------------------------------

def _t_stat(x: np.ndarray, mu0: float) -> tuple[float, int, float]:
    n = int(x.size)
    sd = float(np.std(x, ddof=1))
    return float((x.mean() - mu0) / (sd / math.sqrt(n))), n, sd


def _likelihood(t: float, n: int):
    from scipy.stats import nct
    df, rt = n - 1, math.sqrt(n)
    return lambda d: float(nct.pdf(t, df, d * rt))


def _integrate(f, lo: float, hi: float, points=()) -> float:
    from scipy.integrate import quad
    if lo >= hi:
        return 0.0
    inner = [p for p in points if lo < p < hi] if math.isfinite(lo) and math.isfinite(hi) else []
    val, _ = quad(f, lo, hi, points=inner or None, limit=400, epsabs=0.0, epsrel=1e-10)
    return float(val)


def _masses(t: float, n: int, r: float, c: float) -> tuple[float, float]:
    """(marginal likelihood inside |delta| <= c, outside it) under the Cauchy(0, r) prior."""
    from scipy.stats import cauchy
    like = _likelihood(t, n)
    f = lambda d: like(d) * cauchy.pdf(d, scale=r)  # noqa: E731
    peak = t / math.sqrt(n)
    A = max(abs(peak) + 40.0, 2.0 * c + 1.0)
    inside = _integrate(f, -c, c, (peak, 0.0))
    # each side computed directly, so a tiny outside mass is not lost to a subtraction
    outside = (_integrate(f, c, A, (peak,)) + _integrate(f, A, math.inf)
               + _integrate(f, -A, -c, (peak,)) + _integrate(f, -math.inf, -A))
    return inside, outside


def point_bf01(x, mu0: float, r: float) -> float:
    """JZS point-null BF01 (delta = 0 against delta ~ Cauchy(0, r))."""
    x = np.asarray(x, float)
    t, n, _ = _t_stat(x, mu0)
    m_in, m_out = _masses(t, n, r, 0.0)
    return _likelihood(t, n)(0.0) / (m_in + m_out)


def interval_bf01(x, mu0: float, half: float, r: float) -> dict:
    """Interval-null BF01 for |mu - mu0| <= half, the half-width put on the delta scale with
    the sample standard deviation."""
    x = np.asarray(x, float)
    t, n, sd = _t_stat(x, mu0)
    c = half / sd if sd > 0 else math.inf
    m_in, m_out = _masses(t, n, r, c)
    p_in = (2.0 / math.pi) * math.atan(c / r)
    if p_in <= 0 or p_in >= 1:
        bf = math.nan
    elif m_out <= 0:
        bf = math.inf
    else:
        bf = (m_in / p_in) / (m_out / (1.0 - p_in))
    return {"bf01": float(bf), "c_delta": float(c), "prior_mass_in": float(p_in),
            "posterior_mass_in": float(m_in / (m_in + m_out)), "t": float(t), "n": n}


def band_label(bf01: float) -> str:
    if not math.isfinite(bf01):
        return "not computable"
    if bf01 > 3.0:
        return "evidence for the band"
    if bf01 < 1.0 / 3.0:
        return "evidence against the band"
    return "indeterminate"


def fmt_bf(bf01: float) -> str:
    """The printed BF01: four significant figures."""
    return f"{bf01:.4g}"


def bf_row(name: str, source: str, values, mu0: float, half: float = HALF) -> dict:
    x = np.asarray(values, float)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: needs at least two finite per-seed values, got {values!r}")
    by_scale = {f"{r:g}": interval_bf01(x, mu0, half, r) for r in SCALES}
    lab = by_scale[f"{LABEL_SCALE:g}"]["bf01"]
    return {"reading": name, "source": source, "per_seed": x.tolist(), "n": int(x.size),
            "mean": float(x.mean()), "mu0": mu0, "rope": [mu0 - half, mu0 + half],
            "bf01_interval": {k: v["bf01"] for k, v in by_scale.items()},
            "bf01_point": {f"{r:g}": point_bf01(x, mu0, r) for r in SCALES},
            "detail_0.707": by_scale[f"{LABEL_SCALE:g}"],
            "label_0.707": band_label(lab),
            "quoted_as": f"BF01 = {fmt_bf(lab)} (annotation, 0.707 scale)"}


# ---------------------------------------------------------------------------
# inputs and the gate
# ---------------------------------------------------------------------------

def git_committed(rel: str) -> bytes | None:
    try:
        return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True,
                              check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _same_bytes(a: bytes, b: bytes) -> bool:
    """Byte equality up to the checkout's line endings (core.autocrlf rewrites LF to CRLF)."""
    return a == b or a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


def load_inputs(root: str = ROOT, committed=git_committed) -> tuple[dict, list]:
    """The parsed input files and the gate failures (empty when every file is the committed one)."""
    docs, fails = {}, []
    for key, rel in INPUTS.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            fails.append(f"{rel}: missing")
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        ref = committed(rel)
        if ref is None:
            fails.append(f"{rel}: no committed version")
        elif not _same_bytes(raw, ref):
            fails.append(f"{rel}: differs from the committed version")
        docs[key] = json.loads(raw.decode("utf-8"))
    return docs, fails


def _check(cond: bool, msg: str, fails: list) -> None:
    if not cond:
        fails.append(msg)


def _margin_by_seed(doc: dict) -> dict:
    by: dict = {}
    for c in doc["cells"]:
        by.setdefault(int(c["seed"]), {})[c["arm"]] = c["fresh"]
    return {s: v["survival"] - v["untrained"] for s, v in by.items() if "survival" in v and "untrained" in v}


def _point(summary: dict, rho: float) -> dict:
    hits = [p for p in summary["points"] if abs(float(p["rho"]) - rho) < 1e-12]
    if len(hits) != 1:
        raise ValueError(f"expected one point at rho {rho}, found {len(hits)}")
    return hits[0]


def readings(docs: dict) -> tuple[list, list]:
    """The five readings' rows and any cross-check failures (a derived array that does not
    reproduce the summary value its artifact stores)."""
    fails: list = []
    rows = []
    c1 = docs["c1"]["runs"]["C1"]["drift_zero"]["survival"]
    rows.append(bf_row("1 C1 L0 registered pair (survival, drift 0)",
                       INPUTS["c1"] + " runs.C1.drift_zero.survival.per_seed", c1["per_seed"], 0.5))
    _check(abs(float(np.mean(c1["per_seed"])) - c1["mean"]) < 5e-6, "c1 mean does not reproduce", fails)

    l0 = docs["l0"]
    pairs = [float(v) for v in l0["l0_summary"]["independent_pair_means"]]
    for p, v in enumerate(pairs, start=1):
        per = [float(r["rows"][p]["target"]) for r in l0["l0"]]
        _check(abs(float(np.mean(per)) - v) < 1e-12, f"l0 pair {p} mean does not reproduce", fails)
    src = INPUTS["l0"] + " l0_summary.independent_pair_means"
    rows.append(bf_row("2 eight independent L0 pairs (signed)", src, pairs, 0.5))
    rows.append(bf_row("2 eight independent L0 pairs (folded, max(a, 1 - a))", src,
                       [max(v, 1.0 - v) for v in pairs], 0.5))

    bal = docs["balanced"]["runs"]
    for rid in ("B1", "B2"):
        blk = bal[rid]["pool_target"]["0.00"]["survival"]
        _check(abs(float(np.mean(blk["per_seed"])) - blk["mean"]) < 1e-12, f"{rid} drift-0 mean", fails)
        rows.append(bf_row(f"3 {rid} drift-0 survival pooled target",
                           INPUTS["balanced"] + f" runs.{rid}.pool_target.0.00.survival.per_seed",
                           blk["per_seed"], 0.5))

    sc, fu = docs["sweep_c"], docs["sweep_c_followup"]
    m0, m99 = _point(sc, 0.0)["M"]["per_seed"], _point(sc, 0.99)["M"]["per_seed"]
    rec0 = _margin_by_seed(docs["sweep_c_rho0"])
    _check([rec0[s] for s in sorted(rec0)] == list(m0), "sweep C M(0) does not equal C_rho0.00.json", fails)
    stored = np.asarray(m99, float) - np.asarray(m0, float)
    key = next(iter(sc["contrast"]))
    _check(abs(float(stored.mean()) - sc["contrast"][key]["mean"]) < 1e-12, "sweep C stored contrast mean", fails)
    rows.append(bf_row("4 sweep C stored contrast M(0.99) - M(0)",
                       INPUTS["sweep_c"] + " points[rho 0.99, 0].M.per_seed", stored, 0.0))
    fm99 = _point(fu, 0.99)["M"]["per_seed"]
    follow = np.asarray(fm99, float) - np.asarray(m0, float)
    fkey = next(iter(fu["contrast"]))
    _check(abs(float(follow.mean()) - fu["contrast"][fkey]["mean"]) < 1e-12, "sweep C follow-up contrast mean", fails)
    rows.append(bf_row("4 sweep C follow-up contrast M(0.99, matched) - M(0, stored)",
                       INPUTS["sweep_c_followup"] + " points[rho 0.99].M.per_seed minus the stored M(0)",
                       follow, 0.0))

    arms = docs["l1"]["arms"]["0.023"]
    surv = np.asarray(arms["survival"]["pool_target"]["per_seed"], float)
    for other in ("untrained", "predictor"):
        rows.append(bf_row(f"5 corrected L1 survival minus {other} (drift 0.023)",
                           INPUTS["l1"] + f' arms["0.023"].survival minus .{other} pool_target.per_seed',
                           surv - np.asarray(arms[other]["pool_target"]["per_seed"], float), 0.0))
    return rows, fails


def run_provenance(out: str) -> dict:
    from agent_free_common import provenance
    from promote_bv3_gates import rel_flags, repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit", output=repo_rel(out))
    return {**p, "argv": rel_flags(list(sys.argv[1:]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="check the inputs and the gate; write nothing")
    a = ap.parse_args(argv)
    docs, fails = load_inputs()
    if not fails:
        rows, fails = readings(docs)
    if fails:
        print("input gate failed; nothing is reported:\n  " + "\n  ".join(fails), file=sys.stderr)
        return 1
    if a.dry_run:
        print(f"input gate passes for {len(INPUTS)} files; {len(rows)} rows; would write {a.out}")
        return 0
    from agent_free_common import write_json
    write_json(a.out, {"provenance": run_provenance(a.out), "status": STATUS,
                       "model": {"prior": "JZS: delta ~ Cauchy(0, r), p(sigma) ~ 1/sigma",
                                 "scales": list(SCALES), "label_scale": LABEL_SCALE, "rope_half_width": HALF,
                                 "rope_on_delta": "half-width / sample sd (plug-in)",
                                 "labels": {"> 3": "evidence for the band", "< 1/3": "evidence against the band",
                                            "otherwise": "indeterminate"}},
                       "inputs": INPUTS, "rows": rows})
    for r in rows:
        print(f"{r['reading']}: mean {r['mean']:.4f}  BF01(0.707) {fmt_bf(r['bf01_interval']['0.707'])}  "
              f"[{r['label_0.707']}]")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
