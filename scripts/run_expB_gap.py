"""Experiment B open-loop engagement check and the delta-rollout objective.

A bare invocation reproduces the stored run (printed only, no figure).
`--align` and `--out-dir` are documented in expb_common.
"""

import _bootstrap  # noqa: F401

import numpy as np, torch
from expb_common import PARAMS, WORLD_LABEL, parser, seed_stats, subsample, write_out_json
from itasorl.experiment_b import (collect_episodes, train_world_model, states_torch,
                          episode_features, probe_auroc, _stack)

T, RAY, POOL, SUB, EPOCHS, HIDDEN = 24, 5, 150, 110, 20, 96
OPEN = 16; CTX = T - OPEN
SEEDS = [0, 1, 2]
DRIFTS = [0.0, 0.45]


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


def main(argv=None):
    a = parser("Experiment B: open-loop engagement check and delta-rollout probe.",
               pool=POOL, sub=SUB, epochs=EPOCHS, seeds=SEEDS).parse_args(argv)
    authA = collect_episodes(0, a.pool, T, 0.0, RAY, seed0=10000, params=PARAMS, align=a.align)
    authB = collect_episodes(0, a.pool, T, 0.0, RAY, seed0=20000, params=PARAMS, align=a.align)
    surrD = collect_episodes(1, a.pool, T, 0.45, RAY, seed0=30450, params=PARAMS, align=a.align)

    # PART 1 - was the original (absolute-obs) open-loop objective even engaged?
    rng = np.random.default_rng(0); np.random.seed(0)
    eps0 = ([dict(e, label=0) for e in subsample(authA, a.sub, rng)] +
            [dict(e, label=1) for e in subsample(surrD, a.sub, rng)])
    mdl, nrm = train_world_model(eps0, epochs=a.epochs, hidden=HIDDEN, seed=0, rollout_context=CTX)
    m_abs, mean_base, pers_base = open_loop_mse(mdl, nrm, eps0, CTX, delta=False)
    engaged = bool(m_abs < 0.9 * min(mean_base, pers_base))
    print("DIAGNOSIS - absolute-obs open-loop objective (horizon %d):" % OPEN)
    print(f"   open-loop MSE={m_abs:.3f}   mean-predictor={mean_base:.3f}   persistence={pers_base:.3f}")
    print(f"   verdict: {'ENGAGED' if engaged else 'did NOT engage (~baseline) -> prior k-step null was inconclusive'}\n")

    # PART 2 - delta-rollout objective: engagement check + identity probe
    print("DELTA-rollout objective (predict observation change):")
    res = {d: {"auc": [], "eng": []} for d in DRIFTS}
    for d in DRIFTS:
        other = authB if d == 0 else surrD
        for seed in a.seeds:
            rng = np.random.default_rng(1000 + seed); np.random.seed(seed)
            eps = ([dict(e, label=0) for e in subsample(authA, a.sub, rng)] +
                   [dict(e, label=1) for e in subsample(other, a.sub, rng)])
            model, norm = train_world_model(eps, epochs=a.epochs, hidden=HIDDEN, seed=seed, rollout_context=CTX, delta=True)
            mse, zbase, _ = open_loop_mse(model, norm, eps, CTX, delta=True)
            res[d]["eng"].append(zbase / max(mse, 1e-9))
            X = episode_features(states_torch(model, eps, norm)); y = np.array([e["label"] for e in eps])
            res[d]["auc"].append(probe_auroc(X, y))
        print(f"   drift={d:.2f}:  target AUROC={np.mean(res[d]['auc']):.3f}±{np.std(res[d]['auc']):.3f}"
              f"   engagement (zero-delta/model MSE)={np.mean(res[d]['eng']):.2f}x  (>1 = beats baseline)")

    if a.out_dir:
        config = {"world": WORLD_LABEL, "steps": T, "ray_steps": RAY, "pool": a.pool,
                  "sub": a.sub, "epochs": a.epochs, "hidden": HIDDEN, "seeds": list(a.seeds),
                  "open_horizon": OPEN, "rollout_context": CTX, "drifts": DRIFTS}
        diagnosis = {"seed": 0, "drift": 0.45, "open_loop_mse": float(m_abs),
                     "mean_predictor_mse": float(mean_base), "persistence_mse": float(pers_base),
                     "engaged": engaged}
        delta = [{"drift": d, **seed_stats({"target": res[d]["auc"], "engagement": res[d]["eng"]})}
                 for d in DRIFTS]
        path = write_out_json(a.out_dir, "expB_gap", a.align, config,
                              {"absolute_open_loop": diagnosis, "delta_rollout": delta}, argv=argv)
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
