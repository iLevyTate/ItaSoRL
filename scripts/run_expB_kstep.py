"""Experiment B open-loop horizon check: does a longer-horizon objective induce encoding?

A bare invocation reproduces the stored run and writes docs/figures/expB_kstep.png.
`--align` and `--out-dir` are documented in expb_common.
"""

import _bootstrap  # noqa: F401

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from expb_common import PARAMS, WORLD_LABEL, figure_path, parser, seed_stats, subsample, write_out_json
from itasorl.experiment_b import (collect_episodes, train_world_model, states_torch,
                          episode_features, probe_auroc)

T, RAY, POOL, SUB, EPOCHS, HIDDEN = 24, 5, 150, 110, 18, 96
SEEDS = [0, 1, 2]
OPEN = [0, 8, 16]          # open-loop horizon: 0 = pure next-step (original); higher = deeper imagination
DRIFTS = [0.0, 0.45]       # control vs strong drift


def main(argv=None):
    a = parser("Experiment B: recurrent-state probe across open-loop horizons.",
               pool=POOL, sub=SUB, epochs=EPOCHS, seeds=SEEDS).parse_args(argv)
    print("Collecting pools ...")
    authA = collect_episodes(0, a.pool, T, 0.0, RAY, seed0=10000, params=PARAMS, align=a.align)
    authB = collect_episodes(0, a.pool, T, 0.0, RAY, seed0=20000, params=PARAMS, align=a.align)
    surrD = collect_episodes(1, a.pool, T, 0.45, RAY, seed0=30450, params=PARAMS, align=a.align)

    res = {(oh, d): [] for oh in OPEN for d in DRIFTS}
    print("Training across open-loop horizons x drift x seeds ...")
    for oh in OPEN:
        ctx = None if oh == 0 else (T - oh)
        for d in DRIFTS:
            other = authB if d == 0 else surrD
            for seed in a.seeds:
                rng = np.random.default_rng(1000 + seed); np.random.seed(seed)
                ea = [dict(e, label=0) for e in subsample(authA, a.sub, rng)]
                eb = [dict(e, label=1) for e in subsample(other, a.sub, rng)]
                eps = ea + eb
                model, norm = train_world_model(eps, epochs=a.epochs, hidden=HIDDEN, seed=seed, rollout_context=ctx)
                X = episode_features(states_torch(model, eps, norm))
                y = np.array([e["label"] for e in eps])
                res[(oh, d)].append(probe_auroc(X, y))
        print(f"  open_horizon={oh:2d}:  drift0.45 target={np.mean(res[(oh,0.45)]):.3f}±{np.std(res[(oh,0.45)]):.3f}"
              f"   control target={np.mean(res[(oh,0.0)]):.3f}±{np.std(res[(oh,0.0)]):.3f}")

    td = [np.mean(res[(oh, 0.45)]) for oh in OPEN]; tde = [np.std(res[(oh, 0.45)]) for oh in OPEN]
    tc = [np.mean(res[(oh, 0.0)]) for oh in OPEN]; tce = [np.std(res[(oh, 0.0)]) for oh in OPEN]
    plt.figure(figsize=(7.2, 4.4))
    plt.errorbar(OPEN, td, yerr=tde, fmt="o-", color="#6b46c1", capsize=4, lw=2, label="drift=0.45 (surrogate)")
    plt.errorbar(OPEN, tc, yerr=tce, fmt="s--", color="#a0aec0", capsize=4, label="drift=0 (control)")
    plt.axhline(0.5, ls=":", color="grey")
    plt.xlabel("open-loop prediction horizon  (steps imagined from state)")
    plt.ylabel("recurrent-state probe AUROC")
    plt.title("ITASORL Experiment B - does a longer-horizon objective induce encoding?")
    plt.ylim(0.3, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig = figure_path(a.out_dir, "expB_kstep.png")
    plt.savefig(fig, dpi=130)
    print(f"\nsaved {fig}")

    if a.out_dir:
        config = {"world": WORLD_LABEL, "steps": T, "ray_steps": RAY, "pool": a.pool,
                  "sub": a.sub, "epochs": a.epochs, "hidden": HIDDEN, "seeds": list(a.seeds),
                  "open_horizons": OPEN, "drifts": DRIFTS}
        rows = [{"open_horizon": oh, "drift": d, "rollout_context": None if oh == 0 else T - oh,
                 **seed_stats({"target": res[(oh, d)]})}
                for oh in OPEN for d in DRIFTS]
        path = write_out_json(a.out_dir, "expB_kstep", a.align, config,
                              {"rows": rows, "figure": fig}, argv=argv)
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
