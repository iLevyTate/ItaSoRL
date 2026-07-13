"""L3-HOLDOUT: does the survival agent's world-signal transfer to UNSEEN fingerprints?

The L3 headline (FINDINGS.md sec. 10) is conditional on ONE frozen fingerprint instance
G_0 (train_g_motion at the frozen capacity and seed). This probe tests generality: train
the world-identity readout on pools collected against G_0 (the fingerprint the agent
LIVED with), then score it, frozen, on pools collected against unseen same-recipe nets
G_k (same capacity, same data recipe, different training seed). A signal that transfers
to a fingerprint the agent never experienced is much harder to dismiss as behavioral
residue, and it resolves the reactive-vs-representational ambiguity at the L3 rung
(FINDINGS.md sec. 10.6).

Design (mirrored as a dated entry in docs/PREREGISTRATION_L3.md sec. 12 BEFORE the full
run; this docstring is the code-side copy):

- Per agent seed s: retrain the survival agent exactly as the headline cells do
  (deterministic; the pipeline reproduces the published 0.752), plus the untrained agent
  as the free floor control (predictor behind --with-predictor, off by default).
- Train pool: authentic + G_0 surrogate episodes (the pooled_readout seed bases).
  Test pools, per held-out net G_k: FRESH authentic episodes (disjoint seed base) +
  G_k surrogate episodes. Probe = episode_features LEVEL pipeline, fit ONCE on the full
  train pool, scored once per test pool (probe_transfer_auroc; no CV across pools).
- Every G_k is pre-gated by the L3 residual oracle (itasorl.experiment_a_l3) at the
  frozen sigma_meas=0.02: oracle AUROC must land in [0.85, 0.95] with mechanical leakage
  clean, else the candidate seed is skipped and the next one substituted (logged).
- Reported per (s, k): transfer AUROC (survival + untrained), the in-pool 5-fold CV
  anchor (must reproduce the headline ~0.75), the within-G_k CV ceiling, death counts.
- Decision rule (fixed in advance): per agent seed take the MEAN transfer AUROC over the
  held-out nets; across seeds report the bootstrap AND Student-t 90% CIs (adjudication
  follows the t interval). STRONG transfer: survival mean >= 0.65 with the t-CI excluding
  0.65 AND >= untrained + 0.05. PARTIAL: mean in [0.55, 0.65). NONE: < 0.55 (the headline
  stands but narrows to G_0-specific encoding). Validity gates: every G_k oracle in-band;
  untrained transfer in [0.45, 0.55] for every net; per-pool deaths reported.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from itasorl.world import WorldParams

# The frozen organism world (identical to run_expB2.py).
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
ORACLE_BAND = (0.85, 0.95)
SIGMA_MEAS = 0.02          # frozen gate-0 sensor-noise floor (PREREGISTRATION_L3 sec.12)
DRIFT = 0.45               # the L3 surrogate arm
BAR, SESOI = 0.65, 0.05    # pre-registered threshold and minimal effect


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="agent seeds (full pre-registered run: 0..9)")
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="candidate G_k training seeds (0 is reserved for G_0)")
    ap.add_argument("--l3-hidden", type=int, default=8, help="frozen gate-0 capacity")
    ap.add_argument("--l3-g0-seed", type=int, default=0, help="the lived-with fingerprint's seed")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--n-eps", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--shaping-coef", type=float, default=1.0)
    ap.add_argument("--pool-n", type=int, default=110)
    ap.add_argument("--pool-steps", type=int, default=24)
    ap.add_argument("--with-predictor", action="store_true",
                    help="also run the predictor arm (doubles training time; exploratory)")
    ap.add_argument("--out-dir", default="fullruns/l3_holdout")
    ap.add_argument("--dump-states", default=None,
                    help="directory for per-(seed, net) state/trace dumps (offline audits)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--workers", type=int, default=1, help="parallel workers over agent seeds")
    ap.add_argument("--resume", action="store_true", help="skip agent seeds already checkpointed")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-scale pass")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_eps, a.max_steps, a.hidden, a.ray_steps = 60, 8, 40, 64, 4
        a.pool_n, a.pool_steps = 40, 16
    return a


def config_fingerprint(cfg: dict) -> str:
    fp = {k: v for k, v in cfg.items() if k not in ("dump_states", "out_dir", "workers", "resume")}
    return hashlib.sha256(json.dumps(fp, sort_keys=True, default=float).encode()).hexdigest()[:16]


def git_commit_short() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def gate_g(seed: int, hidden: int, device: str) -> tuple:
    """Train candidate G_seed and gate it with the L3 residual oracle on world P.
    Returns (GMotion, gate-record dict)."""
    from itasorl.experiment_a_l3 import generate_l3_pairs, run_experiment_a_l3
    from itasorl.surrogate_l3 import train_g_motion
    g = train_g_motion(hidden=hidden, seed=seed, params=P, device=device)
    res = run_experiment_a_l3(generate_l3_pairs(g, params=P), sigma_meas=SIGMA_MEAS)
    in_band = ORACLE_BAND[0] <= res["oracle_auroc"] <= ORACLE_BAND[1]
    rec = {"g_seed": seed, "oracle_auroc": float(res["oracle_auroc"]),
           "mechanical_leakage_pass": bool(res["leakage_pass"]),
           "reward_leak": float(res["reward_leak"]),
           "in_band": bool(in_band and res["leakage_pass"])}
    return g, rec


def select_holdout_nets(a, device: str) -> tuple[list, list[dict]]:
    """Gate candidate seeds in order until every requested slot holds an in-band net.
    Out-of-band candidates are skipped and the next unused seed substituted (logged)."""
    n_needed = len(a.holdout_seeds)
    candidates = list(a.holdout_seeds)
    next_sub = max([a.l3_g0_seed] + candidates) + 1
    nets, records = [], []
    while len(nets) < n_needed:
        if not candidates:
            candidates.append(next_sub)
            next_sub += 1
        seed = candidates.pop(0)
        g, rec = gate_g(seed, a.l3_hidden, device)
        records.append(rec)
        print(f"  G(seed={seed}): oracle={rec['oracle_auroc']:.3f} "
              f"mech_leak_ok={rec['mechanical_leakage_pass']} -> "
              f"{'IN-BAND' if rec['in_band'] else 'SKIPPED (out of band), substituting next seed'}",
              flush=True)
        if rec["in_band"]:
            nets.append((seed, g))
    return nets, records


def run_agent_seed(task: dict) -> dict:
    """Train the agents for one seed against G_0, fit the probe on the G_0 train pool,
    and score frozen transfer on every held-out net's test pool. Self-contained for
    worker processes (mirrors run_expB2.run_cell)."""
    import torch
    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b import episode_features, probe_auroc, probe_transfer_auroc
    from itasorl.surrogate_l3 import GMotion

    torch.set_num_threads(1)
    s, dev, k = task["seed"], task["device"], task
    b2.DRIFT_MODE = "l3"

    def rebuild(payload):  # GMotion crosses process boundaries as plain arrays
        g = GMotion.__new__(GMotion)
        g._W, g._b = [np.asarray(w) for w in payload["W"]], [np.asarray(b) for b in payload["b"]]
        g._xm, g._xs, g._ym, g._ys = (np.asarray(payload[x]) for x in ("xm", "xs", "ym", "ys"))
        return g
    g0 = rebuild(k["g0"])
    holdouts = [(gs, rebuild(pl)) for gs, pl in k["holdouts"]]

    b2.install_l3_surrogate(g0)   # the agent LIVES with G_0: training + train pool
    arms = {"untrained": b2.untrained_agent(P, DRIFT, k["ray_steps"], k["hidden"], 64, True,
                                            dev, seed=s)}
    if k.get("with_predictor"):
        arms["predictor"] = b2.train_predictor_only(DRIFT, P, n_eps=k["n_eps"],
                                                    updates=k["updates"], hidden=k["hidden"],
                                                    max_steps=k["max_steps"],
                                                    ray_steps=k["ray_steps"], seed=s, device=dev)
    sa, sn, _ = b2.train_actor_critic(DRIFT, P, n_eps=k["n_eps"], updates=k["updates"],
                                      hidden=k["hidden"], max_steps=k["max_steps"],
                                      ray_steps=k["ray_steps"], seed=s, device=dev,
                                      shaping_coef=k["shaping_coef"])
    arms["survival"] = (sa, sn)

    out = {"seed": s, "arms": {}}
    for name, (agent, norm) in arms.items():
        rec = {"transfer": {}, "within_ceiling": {}, "deaths": {}}
        b2.install_l3_surrogate(g0)
        Ha, _ = b2.collect_pool(agent, norm, P, 0.0, k["pool_n"], k["pool_steps"], dev,
                                800_000, k["ray_steps"])
        Hs, _ = b2.collect_pool(agent, norm, P, DRIFT, k["pool_n"], k["pool_steps"], dev,
                                850_000, k["ray_steps"])
        Xtr = episode_features(np.concatenate([Ha, Hs]))
        ytr = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
        rec["in_pool_anchor"] = probe_auroc(Xtr, ytr)   # must reproduce the headline
        rec["deaths"]["train"] = {"auth": k["pool_n"] - len(Ha), "surr": k["pool_n"] - len(Hs)}
        for j, (gs, gk) in enumerate(holdouts):
            b2.install_l3_surrogate(gk)
            base = 900_000 + j * 20_000                 # disjoint from the train pool bases
            Hat, _ = b2.collect_pool(agent, norm, P, 0.0, k["pool_n"], k["pool_steps"], dev,
                                     base, k["ray_steps"])
            Hst, _ = b2.collect_pool(agent, norm, P, DRIFT, k["pool_n"], k["pool_steps"], dev,
                                     base + 10_000, k["ray_steps"])
            Xte = episode_features(np.concatenate([Hat, Hst]))
            yte = np.concatenate([np.zeros(len(Hat)), np.ones(len(Hst))]).astype(int)
            rec["transfer"][str(gs)] = probe_transfer_auroc(Xtr, ytr, Xte, yte)
            rec["within_ceiling"][str(gs)] = probe_auroc(Xte, yte)
            rec["deaths"][str(gs)] = {"auth": k["pool_n"] - len(Hat),
                                      "surr": k["pool_n"] - len(Hst)}
            if k.get("dump_states"):
                os.makedirs(k["dump_states"], exist_ok=True)
                np.savez_compressed(os.path.join(k["dump_states"],
                                                 f"holdout_s{s}_g{gs}_{name}.npz"),
                                    Ha=Hat, Hs=Hst)
        out["arms"][name] = rec
    b2.install_l3_surrogate(None)
    return out


def main() -> None:
    a = parse_args()
    dev = ("cuda" if a.workers == 1 else "cpu") if a.device == "auto" else a.device
    try:
        import torch
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
    except Exception:
        dev = "cpu"
    out_dir = Path(a.out_dir)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cfg = {kk: getattr(a, kk) for kk in ("seeds", "holdout_seeds", "l3_hidden", "l3_g0_seed",
                                         "updates", "n_eps", "max_steps", "hidden", "ray_steps",
                                         "shaping_coef", "pool_n", "pool_steps",
                                         "with_predictor", "quick")}
    fingerprint, commit = config_fingerprint(cfg), git_commit_short()
    print(f"L3-HOLDOUT  seeds={a.seeds} holdout_candidates={a.holdout_seeds} "
          f"hidden={a.l3_hidden} device={dev} fingerprint={fingerprint} commit={commit}",
          flush=True)

    print("Gating G_0 and the held-out candidates against the oracle band "
          f"[{ORACLE_BAND[0]}, {ORACLE_BAND[1]}] ...", flush=True)
    g0, g0_rec = gate_g(a.l3_g0_seed, a.l3_hidden, dev)
    if not g0_rec["in_band"]:
        raise SystemExit(f"G_0 (seed={a.l3_g0_seed}) is OUT of the oracle band "
                         f"({g0_rec['oracle_auroc']:.3f}): the frozen headline config no "
                         "longer gates; do not run.")
    nets, gate_records = select_holdout_nets(a, dev)

    def pack(g):  # GMotion -> plain arrays for pickling into workers
        return {"W": [w.tolist() for w in g._W], "b": [b.tolist() for b in g._b],
                "xm": g._xm.tolist(), "xs": g._xs.tolist(),
                "ym": g._ym.tolist(), "ys": g._ys.tolist()}
    base = dict(cfg, device=dev, dump_states=a.dump_states, g0=pack(g0),
                holdouts=[(gs, pack(g)) for gs, g in nets])
    tasks = [dict(base, seed=s) for s in a.seeds]

    done: dict[int, dict] = {}
    if a.resume:
        for f in cells_dir.glob("seed_s*.json"):
            payload = json.loads(f.read_text(encoding="utf-8"))
            if payload["fingerprint"] != fingerprint:
                raise SystemExit(f"Checkpoint {f} has fingerprint {payload['fingerprint']}, "
                                 f"current config is {fingerprint}: different experiment.")
            done[int(payload["cell"]["seed"])] = payload["cell"]
        print(f"  resumed {len(done)} seed(s) from {cells_dir}", flush=True)
    todo = [t for t in tasks if t["seed"] not in done]

    def record(cell: dict) -> None:
        done[cell["seed"]] = cell
        path = cells_dir / f"seed_s{cell['seed']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "git_commit": commit,
                                   "cell": cell}, indent=2, default=float), encoding="utf-8")
        os.replace(tmp, path)
        surv = cell["arms"]["survival"]
        xfer = " ".join(f"g{gs}={v:.3f}" for gs, v in surv["transfer"].items())
        print(f"seed={cell['seed']} done  anchor={surv['in_pool_anchor']:.3f}  "
              f"transfer: {xfer}", flush=True)

    t0 = time.time()
    if a.workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=min(a.workers, len(todo))) as pool:
            for cell in pool.imap_unordered(run_agent_seed, todo):
                record(cell)
    else:
        for t in todo:
            record(run_agent_seed(t))
    print(f"all seeds done in {(time.time() - t0) / 60:.1f} min", flush=True)

    summarize(a, done, gate_records, g0_rec, fingerprint, commit, out_dir)


def summarize(a, done, gate_records, g0_rec, fingerprint, commit, out_dir: Path) -> None:
    from itasorl.stats import mean_ci, mean_ci_t
    cells = [done[s] for s in sorted(done)]
    summary = {"config_fingerprint": fingerprint, "git_commit": commit,
               "g0": g0_rec, "holdout_gates": gate_records, "arms": {}}
    print("\nSUMMARY (mean transfer AUROC over held-out nets, per agent seed):", flush=True)
    for name in cells[0]["arms"]:
        per_seed = [float(np.mean(list(c["arms"][name]["transfer"].values()))) for c in cells]
        anchors = [c["arms"][name]["in_pool_anchor"] for c in cells]
        m, blo, bhi = mean_ci(per_seed)
        _, tlo, thi = mean_ci_t(per_seed)
        summary["arms"][name] = {
            "per_seed_mean_transfer": per_seed, "in_pool_anchor": anchors,
            "mean": m, "boot90": [blo, bhi], "t90": [tlo, thi],
            "per_net": {str(gs): [c["arms"][name]["transfer"][str(gs)] for c in cells]
                        for gs, _v in [(g["g_seed"], None) for g in gate_records if g["in_band"]]},
        }
        print(f"  {name:10s} transfer={m:.3f} boot90=[{blo:.3f},{bhi:.3f}] "
              f"t90=[{tlo:.3f},{thi:.3f}]  anchor={float(np.mean(anchors)):.3f}", flush=True)
    verdict = adjudicate(summary)
    summary["verdict"] = verdict
    print("\n" + verdict, flush=True)
    out = out_dir / "l3_holdout_results.json"
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"saved {out}", flush=True)
    try:
        save_figure(summary, out_dir)
    except Exception as e:  # matplotlib is a reporting nicety, not science
        print(f"figure skipped: {e}", flush=True)


def adjudicate(summary: dict) -> str:
    """The pre-fixed decision rule from the module docstring, applied verbatim."""
    surv = summary["arms"]["survival"]
    untr = summary["arms"].get("untrained")
    gates = []
    if untr:
        bad = [f"{gs}: {float(np.mean(v)):.3f}" for gs, v in untr["per_net"].items()
               if not 0.45 <= float(np.mean(v)) <= 0.55]
        if bad:
            gates.append(f"untrained transfer out of [0.45, 0.55] for net(s) {bad}")
    if gates:
        return "VERDICT: UNINFORMATIVE (validity gate failure: " + "; ".join(gates) + ")"
    m, (tlo, _thi) = surv["mean"], surv["t90"]
    beats_floor = (not untr) or m >= float(np.mean(untr["per_seed_mean_transfer"])) + SESOI
    if m >= BAR and tlo > BAR and beats_floor:
        return (f"VERDICT: STRONG TRANSFER (survival {m:.3f}, t90 lower {tlo:.3f} > {BAR}) - "
                "the world-signal generalizes to unseen fingerprints.")
    if m >= 0.55:
        return (f"VERDICT: PARTIAL TRANSFER (survival {m:.3f} in [0.55, {BAR}) or CI touches "
                "the bar) - some fingerprint-general signal, weaker than in-pool.")
    return (f"VERDICT: NO TRANSFER (survival {m:.3f} < 0.55) - the headline stands but "
            "narrows to G_0-specific encoding.")


def save_figure(summary: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.2, 4.4))
    colors = {"survival": "#6b46c1", "untrained": "#a0aec0", "predictor": "#2b6cb0"}
    for name, arm in summary["arms"].items():
        nets = sorted(arm["per_net"], key=int)
        xs = np.arange(len(nets))
        mean = [float(np.mean(arm["per_net"][g])) for g in nets]
        sd = [float(np.std(arm["per_net"][g])) for g in nets]
        plt.errorbar(xs, mean, yerr=sd, fmt="o-", capsize=4, lw=2,
                     color=colors.get(name, "#444444"), label=f"{name} (transfer)")
        plt.axhline(float(np.mean(arm["in_pool_anchor"])), ls="--", lw=1,
                    color=colors.get(name, "#444444"), alpha=0.5)
        plt.xticks(xs, [f"G_{g}" for g in nets])
    plt.axhline(0.5, ls=":", color="grey")
    plt.axhline(BAR, ls=":", color="#c53030")
    plt.xlabel("held-out fingerprint (G training seed)")
    plt.ylabel("world-identity AUROC")
    plt.title("ITASORL L3 held-out fingerprint probe (dashed = in-pool anchor)")
    plt.ylim(0.3, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = out_dir / "l3_holdout_transfer.png"
    plt.savefig(fig, dpi=130)
    print(f"saved {fig}", flush=True)


if __name__ == "__main__":
    main()
