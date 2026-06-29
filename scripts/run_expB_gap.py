import _bootstrap  # noqa: F401

import numpy as np, torch
from itasorl.world import WorldParams
from itasorl.experiment_b import (collect_episodes, train_world_model, states_torch,
                          episode_features, probe_auroc, _stack)

PARAMS = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
T, RAY, POOL, SUB, EPOCHS, HIDDEN = 24, 5, 150, 110, 20, 96
OPEN = 16; CTX = T - OPEN
SEEDS = [0, 1, 2]

authA = collect_episodes(0, POOL, T, 0.0, RAY, seed0=10000, params=PARAMS)
authB = collect_episodes(0, POOL, T, 0.0, RAY, seed0=20000, params=PARAMS)
surrD = collect_episodes(1, POOL, T, 0.45, RAY, seed0=30450, params=PARAMS)

def subsample(pool, k, rng):
    return [pool[i] for i in rng.choice(len(pool), k, replace=False)]

def open_loop_mse(model, norm, eps, context, delta=False):
    mu, sd = norm; obs, act = _stack(eps)
    on = (torch.tensor(obs) - mu) / sd
    with torch.no_grad():
        pred = model.forward_rollout(on, torch.tensor(act), context)
    P = pred[:, context:-1]
    if delta:
        tgt = on[:, context + 1:] - on[:, context:-1]
        return ((P - tgt) ** 2).mean().item(), (tgt ** 2).mean().item(), None
    tgt = on[:, context + 1:]
    pers = ((on[:, context + 1:] - on[:, context - 1:context]) ** 2).mean().item()
    return ((P - tgt) ** 2).mean().item(), (tgt ** 2).mean().item(), pers

# PART 1 - was the original (absolute-obs) open-loop objective even engaged?
rng = np.random.default_rng(0); np.random.seed(0)
eps0 = ([dict(e, label=0) for e in subsample(authA, SUB, rng)] +
        [dict(e, label=1) for e in subsample(surrD, SUB, rng)])
mdl, nrm = train_world_model(eps0, epochs=EPOCHS, hidden=HIDDEN, seed=0, rollout_context=CTX)
m_abs, mean_base, pers_base = open_loop_mse(mdl, nrm, eps0, CTX, delta=False)
print("DIAGNOSIS - absolute-obs open-loop objective (horizon %d):" % OPEN)
print(f"   open-loop MSE={m_abs:.3f}   mean-predictor={mean_base:.3f}   persistence={pers_base:.3f}")
print(f"   verdict: {'ENGAGED' if m_abs < 0.9*min(mean_base, pers_base) else 'did NOT engage (~baseline) -> prior k-step null was inconclusive'}\n")

# PART 2 - delta-rollout objective: engagement check + identity probe
print("DELTA-rollout objective (predict observation change):")
res = {d: {"auc": [], "eng": []} for d in [0.0, 0.45]}
for d in [0.0, 0.45]:
    other = authB if d == 0 else surrD
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed); np.random.seed(seed)
        eps = ([dict(e, label=0) for e in subsample(authA, SUB, rng)] +
               [dict(e, label=1) for e in subsample(other, SUB, rng)])
        model, norm = train_world_model(eps, epochs=EPOCHS, hidden=HIDDEN, seed=seed, rollout_context=CTX, delta=True)
        mse, zbase, _ = open_loop_mse(model, norm, eps, CTX, delta=True)
        res[d]["eng"].append(zbase / max(mse, 1e-9))
        X = episode_features(states_torch(model, eps, norm)); y = np.array([e["label"] for e in eps])
        res[d]["auc"].append(probe_auroc(X, y))
    print(f"   drift={d:.2f}:  target AUROC={np.mean(res[d]['auc']):.3f}±{np.std(res[d]['auc']):.3f}"
          f"   engagement (zero-delta/model MSE)={np.mean(res[d]['eng']):.2f}x  (>1 = beats baseline)")
