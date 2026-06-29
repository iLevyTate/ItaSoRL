import _bootstrap  # noqa: F401

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from itasorl.world import WorldParams
from itasorl.experiment_b import collect_episodes, train_world_model, states_torch, episode_features

PARAMS = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
T, RAY, POOL, SUB, EPOCHS, HIDDEN = 24, 5, 150, 110, 15, 96
SEEDS = [0, 1, 2]
authA = collect_episodes(0, POOL, T, 0.0, RAY, seed0=10000, params=PARAMS)
authB = collect_episodes(0, POOL, T, 0.0, RAY, seed0=20000, params=PARAMS)
surrD = collect_episodes(1, POOL, T, 0.45, RAY, seed0=30450, params=PARAMS)

def subsample(pool, k, rng):
    return [pool[i] for i in rng.choice(len(pool), k, replace=False)]

def rf_auroc(X, y):
    g = np.arange(len(y)); aucs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[te])) < 2: continue
        clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
        clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))

print("Nonlinear (random-forest) probe on the same recurrent states:")
for d in [0.0, 0.45]:
    other = authB if d == 0 else surrD
    tg, sp, sh = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed); np.random.seed(seed)
        eps = ([dict(e, label=0) for e in subsample(authA, SUB, rng)] +
               [dict(e, label=1) for e in subsample(other, SUB, rng)])
        model, norm = train_world_model(eps, epochs=EPOCHS, hidden=HIDDEN, seed=seed)
        X = episode_features(states_torch(model, eps, norm)); y = np.array([e["label"] for e in eps])
        spd = np.array([e["speed"] for e in eps]); rr = np.random.default_rng(seed)
        tg.append(rf_auroc(X, y)); sp.append(rf_auroc(X, (spd > np.median(spd)).astype(int))); sh.append(rf_auroc(X, rr.permutation(y)))
    print(f"   drift={d:.2f}:  target={np.mean(tg):.3f}±{np.std(tg):.3f}   "
          f"speed(+ctrl)={np.mean(sp):.3f}±{np.std(sp):.3f}   shuffled={np.mean(sh):.3f}±{np.std(sh):.3f}")
