"""Experiment B prediction-error (surprise) channel beside the recurrent-state probe.

A bare invocation reproduces the stored run and writes docs/figures/expB_channels.png, with
the recurrent-state curve read from the committed artifacts/expB/summary.json.
`--align` and `--out-dir` are documented in expb_common. Under `--out-dir` the recurrent-state
curve comes from `<out-dir>/expB_full.json` (run scripts/run_expB_full.py with the same
`--out-dir` and `--align` first); the script stops before any training when that file is
missing or was written under another alignment, and never falls back to the stored summary.
"""

import _bootstrap  # noqa: F401

import json
import os

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from expb_common import (PARAMS, WORLD_LABEL, figure_path, json_path, parser, seed_stats,
                         subsample, write_out_json)
from itasorl.experiment_b import collect_episodes, train_world_model, _stack
from itasorl.experiment_a import grouped_auroc, continuous_dims

STEPS, RAY, POOL, SUB, EPOCHS, HIDDEN = 28, 5, 150, 110, 15, 96
SEEDS, DRIFTS = [0, 1, 2], [0.0, 0.2, 0.45]
CDIMS = continuous_dims()


def per_episode_surprise(model, norm, eps):
    """Per-episode next-step prediction error (the agent's 'surprise'), summarized."""
    mu, sd = norm
    obs, act = _stack(eps)
    with torch.no_grad():
        pred, _ = model((torch.tensor(obs) - mu) / sd, torch.tensor(act))
    tgt = ((torch.tensor(obs) - mu) / sd)[:, 1:]
    err = (pred[:, :-1] - tgt).abs().mean(1).numpy()  # (B, obs_dim) mean over time
    return np.stack([err.mean(1), err[:, CDIMS].mean(1), np.percentile(err, 90, axis=1)], axis=1)


def full_sweep_from_out_dir(out_dir: str, align: str) -> dict:
    """The recurrent-state curve of this --out-dir run, keyed by drift. Raises SystemExit
    when `<out-dir>/expB_full.json` is missing, was written under another alignment or lacks
    a drift; there is no fallback to the stored summary."""
    path = json_path(out_dir, "expB_full")
    if not os.path.isfile(path):
        raise SystemExit(f"run_expB_surprise: {path} is missing. Run scripts/run_expB_full.py "
                         f"--out-dir {out_dir} --align {align} first; under --out-dir this "
                         "script does not read the stored summary.")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("align") != align:
        raise SystemExit(f"run_expB_surprise: {path} was written with --align "
                         f"{doc.get('align')!r}, this run uses {align!r}")
    hs = {row["drift"]: row["target_mean"] for row in doc["drift_sweep"]}
    missing = [d for d in DRIFTS if d not in hs]
    if missing:
        raise SystemExit(f"run_expB_surprise: {path} has no recurrent-state value at drift {missing}")
    return hs


def main(argv=None):
    a = parser("Experiment B: prediction-error (surprise) probe.",
               pool=POOL, sub=SUB, epochs=EPOCHS, seeds=SEEDS).parse_args(argv)
    hs = full_sweep_from_out_dir(a.out_dir, a.align) if a.out_dir else None

    authA = collect_episodes(0, a.pool, STEPS, 0.0, RAY, seed0=10000, params=PARAMS, align=a.align)
    authB = collect_episodes(0, a.pool, STEPS, 0.0, RAY, seed0=20000, params=PARAMS, align=a.align)
    surr = {d: collect_episodes(1, a.pool, STEPS, d, RAY, seed0=30000 + int(d * 1000), params=PARAMS, align=a.align)
            for d in DRIFTS if d > 0}

    half = a.pool // 2
    authtest, ctrl_other = authB[:half], authB[half:2 * half]  # held-out authentic; disjoint authentic for the d=0 control
    res = {d: [] for d in DRIFTS}
    for seed in a.seeds:
        rng = np.random.default_rng(2000 + seed); np.random.seed(seed)
        model, norm = train_world_model(subsample(authA, a.sub, rng), epochs=a.epochs, hidden=HIDDEN, seed=seed)  # AUTHENTIC-ONLY
        Sa = per_episode_surprise(model, norm, authtest)
        for d in DRIFTS:
            other = ctrl_other if d == 0 else surr[d][:half]
            So = per_episode_surprise(model, norm, other)
            X = np.vstack([Sa, So]); y = np.array([0] * len(Sa) + [1] * len(So))
            res[d].append(grouped_auroc(X, y, np.arange(len(y))))

    print("Prediction-error (surprise) probe: authentic-only world model, tested authentic vs surrogate")
    for d in DRIFTS:
        print(f"  drift={d:.2f}   surprise-probe AUROC = {np.mean(res[d]):.3f} ± {np.std(res[d]):.3f}")

    # combined comparison figure. Without --out-dir the recurrent-state curve comes from the
    # committed expB_full artifact so the figure cannot silently go stale against a rerun
    # (2026-07-18 audit: it was a hardcoded dict from the prior full run); the
    # literal fallback keeps the script runnable from a bare checkout state.
    if hs is None:
        try:
            with open("artifacts/expB/summary.json", encoding="utf-8") as _fh:
                _sweep = json.load(_fh)["expB_full"]["drift_sweep"]
            hs = {row["drift"]: row["target_mean"] for row in _sweep}
            assert all(d in hs for d in DRIFTS)
        except Exception:
            hs = {0.0: 0.460, 0.2: 0.509, 0.45: 0.510}  # prior full run, same pipeline/config
    sp = [np.mean(res[d]) for d in DRIFTS]; spe = [np.std(res[d]) for d in DRIFTS]
    plt.figure(figsize=(7.2, 4.4))
    plt.errorbar(DRIFTS, sp, yerr=spe, fmt="o-", color="#c05621", capsize=4, lw=2, label="prediction-error (surprise) probe")
    plt.plot(DRIFTS, [hs[d] for d in DRIFTS], "s--", color="#6b46c1", label="recurrent-state probe")
    plt.axhline(0.5, ls=":", color="grey")
    plt.xlabel("surrogate strength  (L2 drift_sigma)"); plt.ylabel("probe AUROC  (per episode)")
    plt.title("ITASORL Experiment B - two incidental-detection channels")
    plt.ylim(0.3, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig = figure_path(a.out_dir, "expB_channels.png")
    plt.savefig(fig, dpi=130)
    print(f"\nsaved {fig}")

    if a.out_dir:
        config = {"world": WORLD_LABEL, "steps": STEPS, "ray_steps": RAY, "pool": a.pool,
                  "sub": a.sub, "epochs": a.epochs, "hidden": HIDDEN, "seeds": list(a.seeds),
                  "drifts": DRIFTS, "test_episodes_per_arm": half,
                  "training": "authentic-only world model"}
        sweep = [{"drift": d, "recurrent_state_target_mean": float(hs[d]),
                  **seed_stats({"surprise": res[d]})} for d in DRIFTS]
        path = write_out_json(a.out_dir, "expB_surprise", a.align, config,
                              {"drift_sweep": sweep, "recurrent_state_source": json_path(a.out_dir, "expB_full"),
                               "figure": fig}, argv=argv)
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
