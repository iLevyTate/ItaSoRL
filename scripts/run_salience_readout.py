"""Whisper or shout: label-free prominence of world identity in the state.

Frozen design: docs/specs/2026-10-15-salience-design.md. Readout-only.

    python scripts/run_salience_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --out artifacts/salience/corrected_l3_h8_wm.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import sys
import time


from _readout_common import AUTH_BASE, DRIFT, SURR_BASE, fingerprint_extra, load, run_tasks, setup_l3, world_params, write_output

SPEC = "docs/specs/2026-10-15-salience-design.md"
ARMS = ("survival", "predictor", "untrained")


def run_seed(t: dict) -> dict:
    from itasorl.eval_protocols import record_trajectories, replay_states
    from itasorl.experiment_b2 import collect_pool
    from itasorl.novel_readouts import salience_measures

    setup_l3()
    P = world_params()
    s, n, T, rs = t["seed"], t["n_eps"], t["steps"], t["ray_steps"]
    t0 = time.time()
    out = {"seed": s, "own": {}, "replay": {}}
    agents = {g: load(t["agents_dir"], DRIFT, s, g) for g in ARMS}
    for g, (ag, nm) in agents.items():
        Ha, _ = collect_pool(ag, nm, P, 0.0, n, T, "cpu", AUTH_BASE, rs)
        Hs, _ = collect_pool(ag, nm, P, DRIFT, n, T, "cpu", SURR_BASE, rs)
        out["own"][g] = salience_measures(Ha, Hs, seed=s)
    sv, sn = agents["survival"]
    Oa, Aa, _ = record_trajectories(sv, sn, P, 0.0, n, T, "cpu", AUTH_BASE, rs)
    Os, As, _ = record_trajectories(sv, sn, P, DRIFT, n, T, "cpu", SURR_BASE, rs)
    for g, (ag, nm) in agents.items():
        out["replay"][g] = salience_measures(replay_states(ag, nm, Oa, Aa), replay_states(ag, nm, Os, As), seed=s)
    d0, n0 = load(t["agents_dir"], 0.0, s, "survival")
    H1, _ = collect_pool(d0, n0, P, 0.0, n, T, "cpu", AUTH_BASE, rs)
    H2, _ = collect_pool(d0, n0, P, 0.0, n, T, "cpu", SURR_BASE, rs)
    out["l0_survival_d0"] = salience_measures(H1, H2, seed=s)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(rows: list) -> dict:
    from itasorl.novel_readouts import paired_t90
    keys = ("share_probe_dir", "share_pc1", "cohen_d_oof", "kmeans_agreement", "kmeans_ari",
            "pc1_sign_agreement")
    agg = {}
    for proto in ("own", "replay"):
        for g in ARMS:
            for k in keys:
                agg[f"{proto} {g} {k}"] = {**paired_t90([r[proto][g][k] for r in rows]),
                                          "per_seed": [r[proto][g][k] for r in rows]}
    for k in keys:
        agg[f"l0 survival_d0 {k}"] = {**paired_t90([r["l0_survival_d0"][k] for r in rows]),
                                     "per_seed": [r["l0_survival_d0"][k] for r in rows]}
    km, sh = agg["own survival kmeans_agreement"], agg["own survival share_probe_dir"]
    shout = (km["mean"] >= 0.65 and km["t90"][0] > 0.60) or sh["mean"] >= 0.25
    whisper = km["t90"][1] < 0.60 and sh["t90"][1] < 0.10
    ukm, ush = agg["replay untrained kmeans_agreement"], agg["replay untrained share_probe_dir"]
    floor = (ukm["mean"] >= 0.65 and ukm["t90"][0] > 0.60) or ush["mean"] >= 0.25
    # The floor above only asks whether the untrained trunk is loud on identical input; it
    # cannot tell two whispers apart (review N03, 2026-10-15). The untrained trunk's own
    # whisper and the paired survival-minus-untrained differences are reported beside it.
    # They are descriptive, not a decomposition of what the input supplies.
    s_rep = [r["replay"]["survival"] for r in rows]
    u_rep = [r["replay"]["untrained"] for r in rows]
    floor_block = {
        "untrained meets reading 1 (shout)": floor,
        "untrained whispers": bool(ukm["t90"][1] < 0.60 and ush["t90"][1] < 0.10),
        "untrained kmeans_agreement": {k: ukm[k] for k in ("mean", "t90", "n")},
        "untrained share_probe_dir": {k: ush[k] for k in ("mean", "t90", "n")},
        "survival minus untrained, share_probe_dir": paired_t90(
            [a["share_probe_dir"] for a in s_rep], [a["share_probe_dir"] for a in u_rep]),
        "survival minus untrained, kmeans_agreement": paired_t90(
            [a["kmeans_agreement"] for a in s_rep], [a["kmeans_agreement"] for a in u_rep]),
        "note": "a floor that only tests whether the untrained trunk is loud cannot tell two "
                "whispers apart; the paired differences are descriptive and are not a causal "
                "decomposition of what the input supplies"}
    l0 = agg["l0 survival_d0 kmeans_agreement"]["mean"]
    l0_at_chance = bool(abs(l0 - 0.5) < 0.1)
    label = "shout" if shout else "whisper" if whisper else "audible"
    if shout and floor:
        label += " (carried by the input)"
    # Spec line 44: when the L0 analogue does not sit within 0.1 of 0.5, every agreement
    # here is reported as carrying the world-sample draw. The boolean below records the
    # gate; the label carries the qualifier so a reader of the label alone sees it.
    if not l0_at_chance:
        label += " (carrying the world-sample draw: L0 analogue not at chance)"
    return {"aggregate": agg, "readings": {
        "1 shout": shout, "2 whisper": whisper, "3 between (audible)": not shout and not whisper,
        "4 floor: untrained meets reading 1 on identical input": floor,
        "4 floor on identical input": floor_block,
        "label": label,
        "l0 kmeans agreement": l0, "l0 within 0.1 of 0.5": l0_at_chance}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    ckpt = a.ckpt or a.out + ".ckpt"
    tasks = [{"agents_dir": a.agents_dir, "seed": s, "n_eps": a.n_eps, "steps": a.steps,
              "ray_steps": a.ray_steps} for s in a.seeds]
    rows = sorted(run_tasks(run_seed, tasks, ckpt, lambda t: f"s{t['seed']}", a.workers,
                            fingerprint_extra(__file__, a.agents_dir, a.seeds, [*ARMS, ("survival", 0.0)])),
                  key=lambda r: r["seed"])
    body = {"spec": SPEC, **summarize(rows), "rows": rows}
    write_output(a.out, "scripts/run_salience_readout.py", sys.argv[1:],
                 {"agents_dir": a.agents_dir, "seeds": a.seeds, "n_eps": a.n_eps, "steps": a.steps,
                  "pools": {"authentic": AUTH_BASE, "surrogate": SURR_BASE}}, SPEC, body)
    for k, v in body["readings"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
