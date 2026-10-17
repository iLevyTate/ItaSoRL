"""Experiment B nonlinear check: a random-forest probe on the same recurrent states.

A bare invocation reproduces the stored run (printed only, no figure).
`--align` and `--out-dir` are documented in expb_common.
"""

import _bootstrap  # noqa: F401

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from expb_common import PARAMS, WORLD_LABEL, parser, seed_stats, subsample, write_out_json
from itasorl import folds
from itasorl.experiment_b import collect_episodes, train_world_model, states_torch, episode_features

T, RAY, POOL, SUB, EPOCHS, HIDDEN = 24, 5, 150, 110, 15, 96
SEEDS = [0, 1, 2]
DRIFTS = [0.0, 0.45]


def rf_auroc(X, y):
    g = np.arange(len(y)); aucs = []
    for tr, te in folds.split(g, 5):
        if len(np.unique(y[te])) < 2: continue
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def main(argv=None):
    a = parser("Experiment B: random-forest probe on the recurrent states.",
               pool=POOL, sub=SUB, epochs=EPOCHS, seeds=SEEDS).parse_args(argv)
    authA = collect_episodes(0, a.pool, T, 0.0, RAY, seed0=10000, params=PARAMS, align=a.align)
    authB = collect_episodes(0, a.pool, T, 0.0, RAY, seed0=20000, params=PARAMS, align=a.align)
    surrD = collect_episodes(1, a.pool, T, 0.45, RAY, seed0=30450, params=PARAMS, align=a.align)

    print("Nonlinear (random-forest) probe on the same recurrent states:")
    res = {}
    for d in DRIFTS:
        other = authB if d == 0 else surrD
        tg, sp, sh = [], [], []
        for seed in a.seeds:
            rng = np.random.default_rng(1000 + seed); np.random.seed(seed)
            eps = ([dict(e, label=0) for e in subsample(authA, a.sub, rng)] +
                   [dict(e, label=1) for e in subsample(other, a.sub, rng)])
            model, norm = train_world_model(eps, epochs=a.epochs, hidden=HIDDEN, seed=seed)
            X = episode_features(states_torch(model, eps, norm)); y = np.array([e["label"] for e in eps])
            spd = np.array([e["speed"] for e in eps]); rr = np.random.default_rng(seed)
            tg.append(rf_auroc(X, y)); sp.append(rf_auroc(X, (spd > np.median(spd)).astype(int))); sh.append(rf_auroc(X, rr.permutation(y)))
        print(f"   drift={d:.2f}:  target={np.mean(tg):.3f}±{np.std(tg):.3f}   "
              f"speed(+ctrl)={np.mean(sp):.3f}±{np.std(sp):.3f}   shuffled={np.mean(sh):.3f}±{np.std(sh):.3f}")
        res[d] = {"target": tg, "speed": sp, "shuffled": sh}

    if a.out_dir:
        config = {"world": WORLD_LABEL, "steps": T, "ray_steps": RAY, "pool": a.pool,
                  "sub": a.sub, "epochs": a.epochs, "hidden": HIDDEN, "seeds": list(a.seeds),
                  "drifts": DRIFTS, "probe": "random forest, 200 trees, random_state 0"}
        sweep = [{"drift": d, **seed_stats(res[d])} for d in DRIFTS]
        path = write_out_json(a.out_dir, "expB_nonlinear", a.align, config,
                              {"drift_sweep": sweep}, argv=argv)
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
