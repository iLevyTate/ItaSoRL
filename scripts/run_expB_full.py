"""Experiment B full run: recurrent-state probe across the L2 drift sweep (FINDINGS 3).

A bare invocation reproduces the stored run and writes docs/figures/expB_incidental.png.
`--align` and `--out-dir` are documented in expb_common.
"""

import _bootstrap  # noqa: F401

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from expb_common import PARAMS, WORLD_LABEL, figure_path, parser, seed_stats, subsample, write_out_json
from itasorl.experiment_b import (collect_episodes, train_world_model, states_torch,
                          episode_features, probe_auroc)

STEPS, RAY, POOL, SUB, EPOCHS, HIDDEN = 28, 5, 150, 110, 15, 96
SEEDS, DRIFTS = [0, 1, 2], [0.0, 0.2, 0.45]


def main(argv=None):
    a = parser("Experiment B: recurrent-state probe across the drift sweep.",
               pool=POOL, sub=SUB, epochs=EPOCHS, seeds=SEEDS).parse_args(argv)
    print("Collecting episode pools ...")
    authA = collect_episodes(0, a.pool, STEPS, 0.0, RAY, seed0=10000, params=PARAMS, align=a.align)
    authB = collect_episodes(0, a.pool, STEPS, 0.0, RAY, seed0=20000, params=PARAMS, align=a.align)  # 'other world' for the drift=0 control
    surr = {d: collect_episodes(1, a.pool, STEPS, d, RAY, seed0=30000 + int(d * 1000), params=PARAMS, align=a.align)
            for d in DRIFTS if d > 0}

    res = {d: {"target": [], "shuffled": [], "speed": []} for d in DRIFTS}
    print("\nTraining + probing (per drift level, across seeds) ...")
    for d in DRIFTS:
        other = authB if d == 0 else surr[d]
        for seed in a.seeds:
            rng = np.random.default_rng(1000 + seed)
            np.random.seed(seed)
            ea = [dict(e, label=0) for e in subsample(authA, a.sub, rng)]
            eb = [dict(e, label=1) for e in subsample(other, a.sub, rng)]
            eps = ea + eb
            model, norm = train_world_model(eps, epochs=a.epochs, hidden=HIDDEN, seed=seed)
            X = episode_features(states_torch(model, eps, norm))
            y = np.array([e["label"] for e in eps]); spd = np.array([e["speed"] for e in eps])
            rr = np.random.default_rng(seed)
            res[d]["target"].append(probe_auroc(X, y))
            res[d]["shuffled"].append(probe_auroc(X, rr.permutation(y)))
            res[d]["speed"].append(probe_auroc(X, (spd > np.median(spd)).astype(int)))
        m = lambda k: (np.mean(res[d][k]), np.std(res[d][k]))
        print(f"  drift={d:.2f}  target={m('target')[0]:.3f}±{m('target')[1]:.3f}   "
              f"shuffled={m('shuffled')[0]:.3f}±{m('shuffled')[1]:.3f}   "
              f"speed(+ctrl)={m('speed')[0]:.3f}±{m('speed')[1]:.3f}")

    tg = [np.mean(res[d]["target"]) for d in DRIFTS]; tge = [np.std(res[d]["target"]) for d in DRIFTS]
    sh = [np.mean(res[d]["shuffled"]) for d in DRIFTS]; she = [np.std(res[d]["shuffled"]) for d in DRIFTS]
    plt.figure(figsize=(7.2, 4.4))
    plt.errorbar(DRIFTS, tg, yerr=tge, fmt="o-", color="#6b46c1", capsize=4, lw=2, label="target: decode world identity")
    plt.errorbar(DRIFTS, sh, yerr=she, fmt="s--", color="#a0aec0", capsize=4, label="negative control (shuffled labels)")
    plt.axhline(0.5, ls=":", color="grey")
    plt.xlabel("surrogate strength  (L2 drift_sigma)")
    plt.ylabel("probe AUROC  (per episode)")
    plt.title("ITASORL Experiment B - incidental detection from the agent's recurrent state")
    plt.ylim(0.3, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig = figure_path(a.out_dir, "expB_incidental.png")
    plt.savefig(fig, dpi=130)
    print(f"\nsaved {fig}")

    if a.out_dir:
        config = {"world": WORLD_LABEL, "steps": STEPS, "ray_steps": RAY, "pool": a.pool,
                  "sub": a.sub, "epochs": a.epochs, "hidden": HIDDEN, "seeds": list(a.seeds),
                  "drifts": DRIFTS}
        sweep = [{"drift": d, **seed_stats(res[d])} for d in DRIFTS]
        path = write_out_json(a.out_dir, "expB_full", a.align, config,
                              {"drift_sweep": sweep, "figure": fig}, argv=argv)
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
