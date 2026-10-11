"""Pair-clustered intervals for the gate-0 oracle AUROCs (finding R26).

Every gate-0 in-band or out-of-band call in the record is a point estimate: the grouped
5-fold oracle AUROC over 60 matched pairs (`scripts/run_expA_l3.py`), with no interval. This
recomputes the oracle side of gate 0 for each calibrated instance the record cites, keeps the
out-of-fold oracle scores with their pair ids (`run_experiment_a_l3(return_oof=True)`, which
changes no number), and attaches

  * the pair-clustered percentile bootstrap of `itasorl.stats.cluster_auroc_ci` (whole pairs
    resampled, 2000 draws, seed 0) at 95% and 90%. It is an interval for the POOLED
    out-of-fold AUROC, which is reported beside it;
  * the interval aligned with the recorded point estimator, the mean of the five fold
    AUROCs (`itasorl.stats.fold_mean_auroc_ci` with whole pairs resampled inside each test
    fold, 2000 draws, seed 0);
  * the share of pair-bootstrap draws that land inside the band [0.85, 0.95].

Both intervals are conditional on the fitted probe (no refit). Only the oracle side of gate 0
is recomputed; the untrained-floor side needs agents and is not touched.

Before any interval is read, each recomputed point is compared with the recorded one, under
both fold schemes (`explicit`, the current default, and `legacy`, the installed
scikit-learn GroupKFold, which on this stack reproduces the published legacy partition). A
point that does not reproduce is reported with its difference, never replaced.

Fingerprints: the committed CPU exports (`artifacts/surrogates/*.npz`) where one exists.
Several instances were calibrated on CUDA (the published hidden 8 and 7 at seed 0, hidden 4,
the seed-2 hidden 8, the GPU re-measure of hidden 10 seed 1); this uses the CPU export for
them and says so per instance. The seed-1 hidden 7, 8 and 9 maps were never exported; they
are retrained here from their recipe on CPU, one thread, as the cloud CPU calibration fit
them. `G_rff` (D = 80) is a numpy closed-form fit and `G_gn` (sigma_v = 0.01) has no fitted
part, so neither depends on the torch device.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/gate0_pair_intervals.py \\
        [--out artifacts/agent_free/gate0_pair_intervals.json] [--only mlp_h8_s0 ...]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from agent_free_common import pin_one_thread, provenance, write_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIMARY_FULLRUNS = os.environ.get("ITASORL_FULLRUNS",
                                  os.path.join(os.path.dirname(os.path.dirname(ROOT)), "fullruns")
                                  if os.path.basename(os.path.dirname(ROOT)) == ".worktrees"
                                  else os.path.join(ROOT, "fullruns"))
SIGMA_MEAS = 0.02
BAND = (0.85, 0.95)
N_PAIRS, BRANCH, SEED0 = 60, 30, 3000
N_BOOT, BOOT_SEED = 2000, 0
SCHEMES = ("explicit", "legacy")
TOL = 1e-12

_LEGACY = "artifacts/fold_rescore/gate0_h8h7_legacy.json"
_EXPLICIT = "artifacts/fold_rescore/gate0_h8h7_explicit.json"
_SEED1 = "artifacts/reviewer_gaps_runs/l3_gate0_seed1/calibration.json"
_SEED1_GPU = "artifacts/reviewer_gaps_runs/l3_gate0_seed1_gpu/calibration.json"
_SEED2_GPU = "artifacts/reviewer_gaps_runs/l3_gate0_seed2_gpu/calibration.json"
_GN = "artifacts/expH2/texture_knockout_h8.json"
_RFF = "artifacts/l3_crossrecipe/summary.json"


def _row(knob: str, val):
    return lambda d: next(r for r in d["rows"] if r.get(knob) == val)["oracle_auroc"]


# Each instance: how its fingerprint is obtained, where its gate 0 was calibrated, and every
# recorded oracle value with its source (path, getter, fold scheme of that record).
INSTANCES = [
    {"id": "mlp_h8_s0", "family": "mlp", "hidden": 8, "g_seed": 0,
     "fingerprint": "artifacts/surrogates/gmotion_h8_s0.npz",
     "calibration_device": "cuda (owner's GPU machine; the published hidden 8 map)",
     "role": "headline fingerprint, gate 0 PASS (FINDINGS 10.1, 10.5)",
     "recorded": [(_LEGACY, _row("hidden", 8), "legacy", "published value"),
                  (_EXPLICIT, _row("hidden", 8), "explicit", "fold re-score")]},
    {"id": "mlp_h7_s0", "family": "mlp", "hidden": 7, "g_seed": 0,
     "fingerprint": "artifacts/surrogates/gmotion_h7_s0.npz",
     "calibration_device": "cuda (owner's GPU machine)",
     "role": "second capacity, gate 0 PASS under legacy, floor fails under explicit (FINDINGS 10.5, 16)",
     "recorded": [(_LEGACY, _row("hidden", 7), "legacy", "published value"),
                  (_EXPLICIT, _row("hidden", 7), "explicit", "fold re-score")]},
    {"id": "mlp_h4_s0", "family": "mlp", "hidden": 4, "g_seed": 0,
     "fingerprint": "artifacts/surrogates/gmotion_h4_s0.npz",
     "calibration_device": "cuda (owner's GPU machine)",
     "role": "hidden 4 on world P, oracle in band but floor 0.896 fails (FINDINGS 10.5 cites the floor only)",
     "recorded": [("fullruns/l3_gate0_recal/calibration.json", _row("hidden", 4), "legacy",
                   "local calibration JSON (gitignored)")]},
    {"id": "mlp_h8_s1", "family": "mlp", "hidden": 8, "g_seed": 1, "fingerprint": "retrain_cpu",
     "calibration_device": "cpu (cloud sandbox)",
     "role": "seed-1 order 8, 7, 9, 10: in band, floor dirty, FAIL (FINDINGS 10.9)",
     "recorded": [(_SEED1, _row("hidden", 8), "explicit", "cloud CPU calibration")]},
    {"id": "mlp_h7_s1", "family": "mlp", "hidden": 7, "g_seed": 1, "fingerprint": "retrain_cpu",
     "calibration_device": "cpu (cloud sandbox)",
     "role": "seed-1 out-of-band call, FAIL (FINDINGS 10.9)",
     "recorded": [(_SEED1, _row("hidden", 7), "explicit", "cloud CPU calibration")]},
    {"id": "mlp_h9_s1", "family": "mlp", "hidden": 9, "g_seed": 1, "fingerprint": "retrain_cpu",
     "calibration_device": "cpu (cloud sandbox)",
     "role": "seed-1 out-of-band call, FAIL (FINDINGS 10.9)",
     "recorded": [(_SEED1, _row("hidden", 9), "explicit", "cloud CPU calibration")]},
    {"id": "mlp_h10_s1", "family": "mlp", "hidden": 10, "g_seed": 1,
     "fingerprint": "artifacts/surrogates/gmotion_h10_s1.npz",
     "calibration_device": "cpu (cloud sandbox) and cuda (GPU re-measure)",
     "role": "second instance, gate 0 PASS (FINDINGS 10.9)",
     "recorded": [(_SEED1, _row("hidden", 10), "explicit", "cloud CPU calibration"),
                  (_SEED1_GPU, _row("hidden", 10), "explicit", "CUDA re-measure")]},
    {"id": "mlp_h8_s2", "family": "mlp", "hidden": 8, "g_seed": 2,
     "fingerprint": "artifacts/surrogates/gmotion_h8_s2.npz",
     "calibration_device": "cuda (owner's GPU machine)",
     "role": "hidden 8 new seed, gate 0 PASS (FINDINGS 10.9)",
     "recorded": [(_SEED2_GPU, _row("hidden", 8), "explicit", "CUDA calibration")]},
    {"id": "rff_D80", "family": "rff", "D": 80, "fingerprint": "fit_g_rff(D=80, params=P)",
     "calibration_device": "numpy closed-form fit (device-independent); floor side on cuda",
     "role": "cross-recipe family, frozen by bisection (FINDINGS 10.7)",
     "recorded": [("fullruns/l3_crossrecipe/gate0_rff_bisect.json", _row("D", 80), "legacy",
                   "local calibration JSON (gitignored)"),
                  (_RFF, lambda d: d["gate0"]["rff"]["selected_oracle_auroc"], "legacy",
                   "committed summary, rounded to 3 decimals")]},
    {"id": "gn_0.01", "family": "gn", "sigma_v": 0.01, "gn_seed": 0,
     "fingerprint": "make_g_gn(sigma_v=0.01, params=P, seed=0)",
     "calibration_device": "no fitted part (numpy Philox stream); floor side on cuda",
     "role": "white-jitter comparator, in band (FINDINGS 14.5)",
     "recorded": [(_GN, lambda d: d["gate0"]["gn"]["selected_row"]["oracle_auroc"], "legacy",
                   "committed H2 texture-knockout artifact")]},
]


def _resolve(path: str) -> str:
    if path.startswith("fullruns/"):
        return os.path.join(PRIMARY_FULLRUNS, path[len("fullruns/"):])
    return os.path.join(ROOT, path)


def _recorded(inst: dict) -> list[dict]:
    out = []
    for path, get, scheme, note in inst["recorded"]:
        full = _resolve(path)
        rec = {"source": path, "scheme": scheme, "note": note, "value": None}
        if os.path.exists(full):
            with open(full, encoding="utf-8") as fh:
                rec["value"] = float(get(json.load(fh)))
        else:
            rec["missing"] = True
        out.append(rec)
    return out


def _weights_sha(g) -> str:
    h = hashlib.sha256()
    for name in ("_W", "_b"):
        for a in getattr(g, name, []):
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()


def build_g(inst: dict, params):
    """The instance's velocity law and a short provenance record of how it was obtained."""
    fam = inst["family"]
    if fam == "mlp":
        from itasorl.surrogate_l3 import GMotion, train_g_motion
        if inst["fingerprint"] == "retrain_cpu":
            g = train_g_motion(hidden=inst["hidden"], seed=inst["g_seed"], params=params,
                               device="cpu")
            return g, {"how": "retrained here: train_g_motion(hidden=%d, seed=%d, params=P, "
                              "device='cpu') on one thread" % (inst["hidden"], inst["g_seed"]),
                       "weights_sha256": _weights_sha(g)}
        g = GMotion.from_npz(os.path.join(ROOT, inst["fingerprint"]))
        with open(os.path.join(ROOT, inst["fingerprint"]), "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        return g, {"how": "committed CPU export, GMotion.from_npz", "npz_sha256": sha}
    if fam == "rff":
        from itasorl.surrogate_l3_families import fit_g_rff
        g = fit_g_rff(D=inst["D"], params=params)
        return g, {"how": "fit_g_rff(D=80, params=P), frozen defaults lam=1e-3, ell=1.0, "
                          "feature_seed=0, data seed 0"}
    if fam == "gn":
        from itasorl.surrogate_l3_families import make_g_gn
        return (make_g_gn(sigma_v=inst["sigma_v"], params=params, seed=inst["gn_seed"]),
                {"how": "make_g_gn(sigma_v=0.01, params=P, seed=0), fresh stream"})
    raise ValueError(fam)


def cluster_draws(y, s, clusters, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """The draws behind itasorl.stats.cluster_auroc_ci, same RNG path, so the in-band share
    is read from the very bootstrap that gives the interval."""
    from itasorl.stats import auroc
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        out[b] = auroc(y[idx], s[idx])
    return out


def band_call(lo: float, hi: float, band=BAND) -> str:
    if lo >= band[0] and hi <= band[1]:
        return "inside"
    if hi < band[0]:
        return "below"
    if lo > band[1]:
        return "above"
    return "straddles " + " and ".join(
        e for e, cross in (("0.85", lo < band[0] <= hi), ("0.95", lo <= band[1] < hi)) if cross)


def intervals_from_oof(oof: dict) -> dict:
    """Everything the artifact reports, from the stored out-of-fold rows alone (the CI test
    calls this on the committed rows)."""
    from itasorl.stats import auroc, cluster_auroc_ci, fold_mean_auroc_ci
    y = np.asarray(oof["label"], int)
    s = np.asarray(oof["score"], float)
    pair = np.asarray(oof["pair"])
    fold = np.asarray(oof["fold"])
    ks = sorted(set(fold.tolist()))
    fold_auc = [auroc(y[fold == k], s[fold == k]) for k in ks]
    fy = [y[fold == k] for k in ks]
    fp = [s[fold == k] for k in ks]
    fg = [pair[fold == k] for k in ks]
    out = {"point_fold_mean": float(np.mean(fold_auc)), "fold_aurocs": [float(a) for a in fold_auc],
           "pooled_oof_auroc": float(auroc(y, s)), "n_pairs": int(len(np.unique(pair))),
           "n_rows": int(len(y))}
    draws = cluster_draws(y, s, pair)
    for level in (0.95, 0.90):
        tag = f"{int(round(level * 100))}"
        lo, hi = cluster_auroc_ci(y, s, pair, level=level, n_boot=N_BOOT, seed=BOOT_SEED)
        a = (1.0 - level) / 2.0
        chk = (float(np.nanpercentile(draws, 100 * a)), float(np.nanpercentile(draws, 100 * (1 - a))))
        if abs(chk[0] - lo) > TOL or abs(chk[1] - hi) > TOL:
            raise RuntimeError("cluster draws disagree with itasorl.stats.cluster_auroc_ci")
        out[f"pair_cluster_ci{tag}"] = [lo, hi]
        out[f"pair_cluster_ci{tag}_band"] = band_call(lo, hi)
        flo, fhi = fold_mean_auroc_ci(fy, fp, level=level, n_boot=N_BOOT, seed=BOOT_SEED,
                                      fold_groups=fg)
        out[f"fold_mean_ci{tag}"] = [flo, fhi]
        out[f"fold_mean_ci{tag}_band"] = band_call(flo, fhi)
    out["share_draws_in_band"] = float(np.mean((draws >= BAND[0]) & (draws <= BAND[1])))
    out["share_draws_below_band"] = float(np.mean(draws < BAND[0]))
    out["share_draws_above_band"] = float(np.mean(draws > BAND[1]))
    return out


def run_instance(inst: dict, params) -> dict:
    from itasorl import folds
    from itasorl.experiment_a_l3 import generate_l3_pairs, run_experiment_a_l3
    t0 = time.time()
    g, how = build_g(inst, params)
    eps = generate_l3_pairs(g, n_pairs=N_PAIRS, branch=BRANCH, seed0=SEED0, params=params)
    per_scheme = {}
    for scheme in SCHEMES:
        with folds.fold_scheme(scheme):
            plain = run_experiment_a_l3(eps, sigma_meas=SIGMA_MEAS, seed=0)
            res = run_experiment_a_l3(eps, sigma_meas=SIGMA_MEAS, seed=0, return_oof=True)
        if res["oracle_auroc"] != plain["oracle_auroc"]:
            raise RuntimeError("return_oof changed the oracle AUROC")
        iv = intervals_from_oof(res["oracle_oof"])
        if abs(iv["point_fold_mean"] - res["oracle_auroc"]) > TOL:
            raise RuntimeError("fold-mean recomputed from the OOF rows disagrees with the oracle")
        per_scheme[scheme] = {"oracle_auroc": float(res["oracle_auroc"]),
                              "in_band_point": bool(BAND[0] <= res["oracle_auroc"] <= BAND[1]),
                              "mech_leak_pass": bool(res["leakage_pass"]),
                              "leakage": {k: float(v) for k, v in res["leakage"].items()},
                              **iv, "oof": res["oracle_oof"]}
    recorded = _recorded(inst)
    for rec in recorded:
        if rec["value"] is None:
            continue
        mine = per_scheme[rec["scheme"]]["oracle_auroc"]
        rec["recomputed"] = mine
        rec["diff"] = mine - rec["value"]
        rounded = "rounded" in rec["note"]
        rec["reproduces"] = bool(abs(round(mine, 3) - rec["value"]) < 1e-9 if rounded
                                 else abs(rec["diff"]) <= TOL)
        rec["other_scheme_value"] = per_scheme[[s for s in SCHEMES if s != rec["scheme"]][0]]["oracle_auroc"]
    keep = {k: v for k, v in inst.items() if k != "recorded"}
    return {**keep, "fingerprint_provenance": how, "recorded": recorded,
            "schemes": per_scheme, "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/agent_free/gate0_pair_intervals.json")
    ap.add_argument("--only", nargs="+", default=None,
                    help="instance ids to (re)compute; others are kept from --out")
    a = ap.parse_args()
    pin_one_thread()
    from itasorl.world import WorldParams
    params = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    prev = {}
    if a.only and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            prev = {r["id"]: r for r in json.load(fh)["instances"]}
    rows = []
    for inst in INSTANCES:
        if a.only and inst["id"] not in a.only:
            if inst["id"] in prev:
                rows.append(prev[inst["id"]])
            continue
        r = run_instance(inst, params)
        rows.append(r)
        e, lg = r["schemes"]["explicit"], r["schemes"]["legacy"]
        print(f"{r['id']:11s} explicit {e['oracle_auroc']:.4f} pair95 "
              f"[{e['pair_cluster_ci95'][0]:.3f}, {e['pair_cluster_ci95'][1]:.3f}]  legacy "
              f"{lg['oracle_auroc']:.4f} [{lg['pair_cluster_ci95'][0]:.3f}, "
              f"{lg['pair_cluster_ci95'][1]:.3f}]  reproduces "
              f"{[rec.get('reproduces') for rec in r['recorded']]}  ({r['seconds']} s)", flush=True)
    doc = {
        **provenance("scripts/gate0_pair_intervals.py"),
        "finding": "R26: gate-0 in/out-of-band calls carried no interval",
        "protocol": {"world": "P(k_land=1.5, k_water=1.5, gravity=0.4)", "sigma_meas": SIGMA_MEAS,
                     "band": list(BAND), "n_pairs": N_PAIRS, "branch": BRANCH, "seed0": SEED0,
                     "oracle_seed": 0, "pairs": "generate_l3_pairs + run_experiment_a_l3, as "
                     "scripts/run_expA_l3.py", "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                     "cluster": "matched pair (both branches of a pair share a world seed)",
                     "fold_schemes": list(SCHEMES)},
        "reading_notes": [
            "pair_cluster_ci is itasorl.stats.cluster_auroc_ci on the pooled out-of-fold "
            "scores; it brackets pooled_oof_auroc, which differs from the recorded fold-mean "
            "point. fold_mean_ci is aligned with the recorded fold-mean point.",
            "Both intervals hold the fitted oracle probe fixed (no refit) and resample pairs, so "
            "they carry pair-sampling variation only, not measurement-noise or G-fit variation.",
            "Only the oracle side of gate 0 is recomputed; floors are not.",
            "CUDA-calibrated instances are scored here with the committed CPU export or a CPU "
            "retrain; see calibration_device and fingerprint_provenance per instance."],
        "instances": rows,
    }
    write_json(out_path, doc)
    print(f"wrote {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
