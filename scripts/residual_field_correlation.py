"""Correlation of the L3 fingerprints' residual fields on fresh authentic transitions.

FINDINGS 10.5 (scope note of 2026-07-18) quotes Pearson r of about +0.60 between the hidden 8
and hidden 7 residual fields and about +0.20 between hidden 8 and the cross-recipe G_rff, with
no committed artifact. This measures the full correlation matrix.

Residual field of a velocity law G: r_G(x) = G(x) - f(x), with f the authentic world-P law
(1 - k*dt) * vel + a * dt (uniform drag k = 1.5), evaluated on the same inputs x = (vel, a).
Inputs are fresh authentic scripted-policy transitions (`collect_authentic_transitions`,
250 episodes x 40 steps from HELDOUT_SEED = 100000), a range checked disjoint from every
G's training range [s, s + 250) before anything is scored (the overlap guard of
`surrogate_diagnostics`).

Fingerprints: every committed CPU export in artifacts/surrogates (hidden 8, 7, 4 at G seed 0,
hidden 10 at seed 1, hidden 8 at seed 2) plus G_rff at D = 80, rebuilt from its frozen recipe
(`fit_g_rff(D=80, params=P)`, training episodes [0, 250)).

Reported per pair of fields: Pearson r with both velocity components pooled (the headline,
the quantity the scope note quotes), r on each component, and r squared. Intervals: percentile
bootstrap over episodes (2000 draws, seed 0, 95% and 90%), computed from per-episode
sufficient statistics that the artifact stores, so the test recomputes every number without
regenerating a transition.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/residual_field_correlation.py \\
        [--out artifacts/agent_free/residual_field_correlation.json]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from agent_free_common import pin_one_thread, provenance, write_json  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_EPS, STEPS = 250, 40
N_TRAIN_EPS = 250
N_BOOT, BOOT_SEED = 2000, 0
EXPORTS = (("h8_s0", "gmotion_h8_s0.npz", 0), ("h7_s0", "gmotion_h7_s0.npz", 0),
           ("h4_s0", "gmotion_h4_s0.npz", 0), ("h10_s1", "gmotion_h10_s1.npz", 1),
           ("h8_s2", "gmotion_h8_s2.npz", 2))
RFF = ("rff_D80", 80, 0)
QUOTED = {("h8_s0", "h7_s0"): 0.60, ("h8_s0", "rff_D80"): 0.20}


def _overlap(heldout_seed: int, n_ho: int, train_seed: int, n_tr: int) -> bool:
    return heldout_seed < train_seed + n_tr and train_seed < heldout_seed + n_ho


def triu_pairs(m: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(m) for j in range(i, m)]


def episode_stats(R: np.ndarray) -> dict:
    """Per-episode sufficient statistics of residual fields R (M, n_eps, steps) for one
    component: count, sums and the upper-triangular cross-product sums."""
    M = R.shape[0]
    S = R.sum(axis=2).T                                         # (n_eps, M)
    Q = np.stack([(R[i] * R[j]).sum(axis=1) for i, j in triu_pairs(M)], axis=1)
    return {"n": int(R.shape[2]), "S": S.tolist(), "Q": Q.tolist()}


def corr_from_sums(n: float, S: np.ndarray, Q: np.ndarray, M: int) -> np.ndarray:
    mean = S / n
    C = np.empty((M, M))
    for k, (i, j) in enumerate(triu_pairs(M)):
        C[i, j] = C[j, i] = Q[k] / n
    cov = C - np.outer(mean, mean)
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def summarize(views: dict, names: list[str], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Point correlation matrices and episode-bootstrap intervals for the pooled field and
    each component, from the stored per-episode statistics alone."""
    M = len(names)
    st = {c: (np.asarray(views[c]["S"], float), np.asarray(views[c]["Q"], float),
              int(views[c]["n"])) for c in ("x", "y")}
    n_eps = st["x"][0].shape[0]

    def mats(idx):
        out = {}
        for c in ("x", "y"):
            S, Q, n = st[c]
            out[c] = corr_from_sums(n * len(idx), S[idx].sum(0), Q[idx].sum(0), M)
        Sx, Qx, nx = st["x"]
        Sy, Qy, ny = st["y"]
        out["pooled"] = corr_from_sums((nx + ny) * len(idx), Sx[idx].sum(0) + Sy[idx].sum(0),
                                       Qx[idx].sum(0) + Qy[idx].sum(0), M)
        return out

    point = mats(np.arange(n_eps))
    rng = np.random.default_rng(seed)
    draws = {v: np.empty((n_boot, M, M)) for v in ("pooled", "x", "y")}
    for b in range(n_boot):
        m = mats(rng.integers(0, n_eps, n_eps))
        for v in draws:
            draws[v][b] = m[v]
    out = {}
    for v in ("pooled", "x", "y"):
        lo95, hi95 = np.percentile(draws[v], [2.5, 97.5], axis=0)
        lo90, hi90 = np.percentile(draws[v], [5.0, 95.0], axis=0)
        out[v] = {"r": point[v].tolist(), "ci95_lo": lo95.tolist(), "ci95_hi": hi95.tolist(),
                  "ci90_lo": lo90.tolist(), "ci90_hi": hi90.tolist()}
    pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            row = {"a": names[i], "b": names[j]}
            for v in ("pooled", "x", "y"):
                row[f"r_{v}"] = float(point[v][i, j])
                row[f"ci95_{v}"] = [float(out[v]["ci95_lo"][i][j]), float(out[v]["ci95_hi"][i][j])]
            row["ci90_pooled"] = [float(out["pooled"]["ci90_lo"][i][j]),
                                  float(out["pooled"]["ci90_hi"][i][j])]
            row["r2_pooled"] = row["r_pooled"] ** 2
            q = QUOTED.get((names[i], names[j]))
            if q is not None:
                row["quoted_in_findings_10_5"] = q
            pairs.append(row)
    return {"matrices": out, "pairs": pairs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/agent_free/residual_field_correlation.json")
    a = ap.parse_args()
    pin_one_thread()
    from itasorl.surrogate_l3 import HELDOUT_SEED, GMotion, collect_authentic_transitions
    from itasorl.surrogate_l3_families import fit_g_rff
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    t0 = time.time()

    models, meta = [], []
    for name, fn, seed in EXPORTS:
        path = os.path.join(ROOT, "artifacts", "surrogates", fn)
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        models.append((name, GMotion.from_npz(path)))
        meta.append({"name": name, "source": f"artifacts/surrogates/{fn}", "sha256": sha,
                     "train_episodes": [seed, seed + N_TRAIN_EPS]})
    name, D, seed = RFF
    models.append((name, fit_g_rff(D=D, params=P, seed=seed)))
    meta.append({"name": name, "source": "fit_g_rff(D=80, params=P), frozen defaults lam=1e-3, "
                 "ell=1.0, feature_seed=0", "train_episodes": [seed, seed + N_TRAIN_EPS]})
    for m in meta:
        s0, s1 = m["train_episodes"]
        if _overlap(HELDOUT_SEED, N_EPS, s0, s1 - s0):
            raise SystemExit(f"fresh episodes overlap {m['name']}'s training range")

    X, _Y = collect_authentic_transitions(n_eps=N_EPS, steps=STEPS, params=P, seed0=HELDOUT_SEED)
    Xd = X.astype(float)
    f = (1.0 - P.k_land * P.dt) * Xd[:, :2] + Xd[:, 2:] * P.dt
    authentic_check = float(np.abs(f - _Y.astype(float)).max())
    R = np.stack([np.stack([np.asarray(g(x[:2], x[2:], None), float) for x in X]) - f
                  for _, g in models])                        # (M, N, 2)
    names = [n for n, _ in models]
    Rr = R.reshape(len(models), N_EPS, STEPS, 2)
    views = {"x": episode_stats(Rr[..., 0]), "y": episode_stats(Rr[..., 1])}
    summ = summarize(views, names)
    fields = [{"name": n, "rms": float(np.sqrt((R[k] ** 2).sum(1).mean())),
               "mean": R[k].mean(0).tolist(), "sd": R[k].std(0).tolist()}
              for k, n in enumerate(names)]
    doc = {
        **provenance("scripts/residual_field_correlation.py"),
        "finding": "FINDINGS 10.5 scope note (2026-07-18): r about +0.60 (h8 vs h7), about "
                   "+0.20 (h8 vs G_rff), no committed artifact",
        "world": "P(k_land=1.5, k_water=1.5, gravity=0.4)",
        "residual": "G(vel, a) - [(1 - k*dt) vel + a dt], k = 1.5, on the inputs of fresh "
                    "authentic scripted-policy transitions",
        "fresh_sample": {"seed0": HELDOUT_SEED, "n_episodes": N_EPS, "steps": STEPS,
                         "episodes": [HELDOUT_SEED, HELDOUT_SEED + N_EPS],
                         "n_transitions": int(len(X)),
                         "disjoint_from_every_training_range": True,
                         "max_abs_authentic_law_vs_logged_next_velocity": authentic_check},
        "fingerprints": meta, "fields": fields,
        "bootstrap": {"unit": "episode", "n_boot": N_BOOT, "seed": BOOT_SEED,
                      "levels": [0.95, 0.90]},
        "names": names, **summ,
        "episode_stats": views,
        "seconds": round(time.time() - t0, 1),
    }
    out_path = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    write_json(out_path, doc)
    for p in summ["pairs"]:
        print(f"{p['a']:7s} {p['b']:7s} r {p['r_pooled']:+.3f} "
              f"[{p['ci95_pooled'][0]:+.3f}, {p['ci95_pooled'][1]:+.3f}]  x {p['r_x']:+.3f} "
              f"y {p['r_y']:+.3f}" + (f"  (quoted {p['quoted_in_findings_10_5']:+.2f})"
                                      if "quoted_in_findings_10_5" in p else ""))
    print(f"wrote {os.path.relpath(out_path, ROOT)} in {doc['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
