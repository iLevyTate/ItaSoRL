"""Nudge dose-response with a positive-control direction (closing item X01, E01).

Post hoc and descriptive; changes no stored value or verdict. Built from
scripts/run_mechanism_readouts.py (run_seed, common_rows, NUDGE_SD = 1.0, GAP_MIN, WORLD_BASE):
on one run's saved agents at its strongest drift, per seed and arm,
- the unnudged authentic and surrogate rollouts (base 1400000, shared world seeds);
- at each magnitude a (in s units, s the spread of the pooled states along the decoded
  direction u): real rollouts (+a s u authentic, -a s u surrogate) and sham rollouts
  (+/- a s v, v the registered sham direction);
- with --positive-control actor, one authentic rollout nudged by 1 s_p along p, the top right
  singular vector of the actor's rows with the dead emit row dropped (nothing in the world reads
  it, itasorl/agent_ac.py), s_p the spread of the pooled states along p. The push is also read
  through to_env_action on the pooled states, so a push that only moves the emit logit never
  counts as a control.
Survivors are intersected across every condition; one behavior scale comes from the unnudged
pair on those rows, and n_common is written per magnitude beside each magnitude's own
attrition. d_u, the between-world mean shift of the pooled states along u in s units, is
recorded so each magnitude reads against the natural scale.

Gates: (I) --integrity-ref, a run_mechanism_readouts.py output on the same stack: the 1 s real
and sham scores per seed, computed here on run_mechanism_readouts' own six-rollout path,
reproduce it within 0.01, arm by arm; a stored NaN must come back NaN with the same reason and
is excluded; NaN on one side only fails that arm; a failed arm stops the readout and nothing is
written. (i) the control moves the standardized behavior vector by at least GAP_MIN in at least
7 seeds, else the series is "uninformative: no 1 s nudge moves this policy". (ii) a magnitude
with fewer than 7 informative seeds is VOID at that magnitude.

    python scripts/run_nudge_dose_readout.py --run-dir fullruns/corrected_l3_h8_wm \\
        --magnitudes 0.5 1 2 4 --positive-control actor --integrity-ref <B5 mechanism.json> \\
        --workers 2 --ckpt fullruns/closing_2026-10/x01/ckpt --out fullruns/closing_2026-10/x01/out.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from run_mechanism_readouts import GAP_MIN, NUDGE_SD, WORLD_BASE, _cfg_from_cells, common_rows

GENERATED_BY = "scripts/run_nudge_dose_readout.py"
ARMS = ("survival", "untrained")
MAGNITUDES = (0.5, 1.0, 2.0, 4.0)
MIN_SEEDS = 7
RULE = 0.10
INTEGRITY_TOL = 0.01
STATUS = "post hoc, descriptive; changes no stored value or verdict"


# ---------------------------------------------------------------------------
# the actor-sensitive direction
# ---------------------------------------------------------------------------

def kept_actor_rows(agent) -> np.ndarray:
    """The actor's weight rows without the emit logit (the last Bernoulli row)."""
    W = agent.actor.weight.detach().cpu().numpy().astype(np.float64)
    emit = agent.n_cont + agent.n_bin - 1
    return np.delete(W, emit, axis=0)


def actor_direction(agent) -> tuple[np.ndarray, list]:
    """p, the top right singular vector of the kept actor rows (sign: its largest action
    component positive), and the singular values."""
    W = kept_actor_rows(agent)
    _, sv, Vt = np.linalg.svd(W, full_matrices=False)
    p = Vt[0]
    a = W @ p
    if a[np.argmax(np.abs(a))] < 0:
        p = -p
    return p, sv.tolist()


def env_action_shift(agent, H: np.ndarray, p: np.ndarray, size: float) -> dict:
    """Mean absolute change of the env action (thrust, turn, eat, drink; emit left out) when the
    states H (N, hidden) are pushed by size * p, through the actor and to_env_action with the
    Bernoulli probabilities standing in for the thresholded binaries."""
    import torch
    with torch.no_grad():
        h = torch.as_tensor(np.asarray(H, np.float32))
        dp = torch.as_tensor((size * p).astype(np.float32))

        def env(x):
            out = agent.actor(x)
            mu, logits = out[..., :agent.n_cont], out[..., agent.n_cont:]
            return agent.to_env_action(mu, torch.sigmoid(logits))[..., :agent.n_cont + agent.n_bin - 1]

        d = (env(h + dp) - env(h)).abs().mean(0).cpu().numpy()
    return {"columns": ["thrust", "turn", "eat", "drink"], "mean_abs_shift": d.tolist()}


# ---------------------------------------------------------------------------
# one seed
# ---------------------------------------------------------------------------

def _setup(k: dict):
    import torch

    import itasorl.experiment_b2 as b2
    from itasorl.world import WorldParams
    torch.set_num_threads(1)
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    knobs = k["knobs"]
    b2.DRIFT_MODE = "l3"
    b2.OBJECTIVE = k["objective"]
    b2.MORTAL = k["mortal"]
    if knobs.get("n_pellets") is not None:
        b2.SURVIVAL_FOOD["n_pellets"] = knobs["n_pellets"]
    if knobs.get("reach") is not None:
        b2.SURVIVAL_FOOD["reach"] = knobs["reach"]
    if knobs.get("basal_e") is not None:
        b2.SURVIVAL_METAB["basal_E"] = knobs["basal_e"]
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=knobs.get("l3_hidden") or 8, device="cpu",
                              seed=knobs.get("l3_seed") or 0, params=P)
    return P


def _mag_key(a: float) -> str:
    return f"{float(a):g}"


def score_arm(Ba_rows: dict, scale: np.ndarray, mags) -> dict:
    """Forward and reverse real and sham gap-closed scores at each magnitude on common rows."""
    from itasorl import mechanism_readouts as mr
    out = {}
    for a in mags:
        k = _mag_key(a)
        real, gap = mr.gap_closed(Ba_rows["a"], Ba_rows["s"], Ba_rows[f"n{k}"], scale)
        sham, _ = mr.gap_closed(Ba_rows["a"], Ba_rows["s"], Ba_rows[f"sh{k}"], scale)
        real_r, _ = mr.gap_closed(Ba_rows["s"], Ba_rows["a"], Ba_rows[f"nr{k}"], scale)
        sham_r, _ = mr.gap_closed(Ba_rows["s"], Ba_rows["a"], Ba_rows[f"shr{k}"], scale)
        out[k] = {"score_real": real, "score_sham": sham, "score_real_reverse": real_r,
                  "score_sham_reverse": sham_r, "gap": gap,
                  "informative": bool(np.isfinite(gap) and gap >= GAP_MIN)}
    return out


def run_arm(ag, nm, P, k: dict, seed: int, d: float) -> dict:
    from itasorl import mechanism_readouts as mr
    from itasorl.experiment_b2 import collect_pool

    def roll(drift, nudge=None):
        return mr.rollout_behavior(ag, nm, P, drift, n_eps=k["n_eps"], steps=k["steps"],
                                   seed_base=WORLD_BASE, ray_steps=k["ray_steps"], nudge=nudge)

    Ha, _ = collect_pool(ag, nm, P, 0.0, k["pool_n"], k["pool_steps"], "cpu", 800_000, k["ray_steps"])
    Hs, _ = collect_pool(ag, nm, P, d, k["pool_n"], k["pool_steps"], "cpu", 850_000, k["ray_steps"])
    u, s = mr.probe_direction(Ha, Hs)
    hid = Ha.shape[-1]
    ra, rsu = roll(0.0), roll(d)
    blk = {"s": s, "u": u.tolist()}
    if not np.isfinite(s):
        blk.update(informative=False, reason="degenerate probe")
        return blk
    flat = np.concatenate([Ha, Hs]).reshape(-1, hid)
    blk["d_u"] = float((Hs.reshape(-1, hid) @ u).mean() - (Ha.reshape(-1, hid) @ u).mean()) / s
    v = mr.sham_direction(u, seed)
    blk["v"] = v.tolist()
    rolls = {"a": ra, "s": rsu}
    for a in k["magnitudes"]:
        m = _mag_key(a)
        rolls[f"n{m}"] = roll(0.0, nudge=a * NUDGE_SD * s * u)
        rolls[f"sh{m}"] = roll(0.0, nudge=a * NUDGE_SD * s * v)
        rolls[f"nr{m}"] = roll(d, nudge=-a * NUDGE_SD * s * u)
        rolls[f"shr{m}"] = roll(d, nudge=-a * NUDGE_SD * s * v)
    if k["positive_control"] == "actor":
        p, sv = actor_direction(ag)
        s_p = float((flat @ p).std())
        rolls["p"] = roll(0.0, nudge=1.0 * s_p * p)
        blk["control"] = {"direction": "actor rows without the emit row", "p": p.tolist(), "s_p": s_p,
                          "singular_values": sv, "env_action_shift_1sp": env_action_shift(ag, flat, p, s_p)}
    names = list(rolls)
    rows = common_rows(*[rolls[x] for x in names])
    B = dict(zip(names, rows))
    n_common = int(len(B["a"]))
    blk["n_common"] = n_common
    blk["n_survived"] = {x: int(len(rolls[x]["B"])) for x in names}
    blk["n_common_per_magnitude"] = {_mag_key(a): n_common for a in k["magnitudes"]}
    blk["attrition_per_magnitude"] = {
        _mag_key(a): int(len(common_rows(*[rolls[x] for x in ("a", "s", f"n{_mag_key(a)}", f"sh{_mag_key(a)}",
                                                              f"nr{_mag_key(a)}", f"shr{_mag_key(a)}")])[0]))
        for a in k["magnitudes"]}
    if n_common == 0:
        blk.update(informative=False, reason="empty pool")
        return blk
    scale = mr.behavior_scale(B["a"], B["s"])
    blk["magnitudes"] = score_arm(B, scale, k["magnitudes"])
    if "p" in B:
        m5 = mr.N_MEASURES
        shift = (B["p"][:, :m5].mean(0) - B["a"][:, :m5].mean(0)) / scale[:m5]
        blk["control"]["behavior_shift"] = float(np.linalg.norm(shift))
        blk["control"]["moved"] = bool(np.isfinite(blk["control"]["behavior_shift"])
                                       and blk["control"]["behavior_shift"] >= GAP_MIN)
        blk["control"]["score"] = mr.gap_closed(B["a"], B["s"], B["p"], scale)[0]
    if 1.0 in [float(a) for a in k["magnitudes"]]:
        blk["integrity_1s"] = mechanism_path_1s(rolls)
    return blk


def mechanism_path_1s(rolls: dict) -> dict:
    """The 1 s intervention exactly as run_mechanism_readouts.run_seed computes it (its six
    rollouts, their common survivors, its scale), for gate I."""
    from itasorl import mechanism_readouts as mr
    Ba, Bs, Bn, Bsh, Bnr, Bshr = common_rows(*[rolls[x] for x in ("a", "s", "n1", "sh1", "nr1", "shr1")])
    scale = mr.behavior_scale(Ba, Bs)
    real, gap = mr.gap_closed(Ba, Bs, Bn, scale)
    sham, _ = mr.gap_closed(Ba, Bs, Bsh, scale)
    out = {"score_real": real, "score_sham": sham, "gap": gap, "n_common": int(len(Ba)),
           "informative": bool(np.isfinite(gap) and gap >= GAP_MIN)}
    if not out["informative"]:
        out["reason"] = "gap below threshold" if np.isfinite(gap) else "empty pool"
    return out


def run_seed(k: dict) -> dict:
    from itasorl.experiment_b2 import format_drift, load_agent_bundle
    P = _setup(k)
    seed, d = k["seed"], k["drift"]
    t0 = time.time()
    out = {"seed": seed, "drift": d}
    for g in k["arms"]:
        ag, nm = load_agent_bundle(os.path.join(k["run_dir"], "agents", f"agent_d{format_drift(d)}_s{seed}_{g}.pt"))
        out[g] = run_arm(ag, nm, P, k, seed, d)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------
# gates and summary
# ---------------------------------------------------------------------------

def integrity_check(rows: list, ref: dict, arms) -> dict:
    """Gate I, arm by arm, against a run_mechanism_readouts output."""
    by = {int(r["seed"]): r for r in ref["per_seed"]}
    out = {}
    for g in arms:
        fails, excluded, checked = [], [], 0
        for r in rows:
            s = int(r["seed"])
            if s not in by:
                fails.append(f"seed {s}: not in the reference")
                continue
            stored = by[s]["intervention"] if g == "survival" else by[s].get(f"{g}_arm", {}).get("intervention")
            mine = r[g].get("integrity_1s")
            if stored is None or mine is None:
                fails.append(f"seed {s}: no 1 s block on one side")
                continue
            for q in ("score_real", "score_sham"):
                a, b = float(mine[q]), float(stored[q])
                if np.isnan(b):
                    if np.isnan(a) and mine.get("reason") == stored.get("reason"):
                        excluded.append(s)
                    else:
                        fails.append(f"seed {s} {q}: stored NaN ({stored.get('reason')}), got {a!r} ({mine.get('reason')})")
                elif not (np.isfinite(a) and abs(a - b) <= INTEGRITY_TOL):
                    fails.append(f"seed {s} {q}: {a!r} vs stored {b!r}")
                else:
                    checked += 1
        out[g] = {"passed": not fails, "failures": fails, "excluded_nan_seeds": sorted(set(excluded)),
                  "values_checked": checked}
    return out


def summarize(rows: list, arms, mags) -> dict:
    from itasorl.novel_readouts import paired_t90
    out = {}
    for g in arms:
        rr = [r[g] for r in rows]
        ctrl = [b.get("control", {}).get("moved") for b in rr]
        moved = sum(bool(c) for c in ctrl)
        has_ctrl = any(c is not None for c in ctrl)
        blk = {"gate_i": {"seeds_moved": moved, "passed": bool(has_ctrl and moved >= MIN_SEEDS),
                          "behavior_shift": [b.get("control", {}).get("behavior_shift") for b in rr]} if has_ctrl
               else {"passed": None, "note": "no positive control (--positive-control none)"},
               "d_u": paired_t90([b.get("d_u", np.nan) for b in rr]), "magnitudes": {}}
        for a in mags:
            k = _mag_key(a)
            mb = [b["magnitudes"][k] for b in rr if "magnitudes" in b]
            inf = [x for x in mb if x["informative"]]
            m = {"n_informative": len(inf), "VOID": len(inf) < MIN_SEEDS,
                 "real_minus_sham": paired_t90([x["score_real"] for x in inf], [x["score_sham"] for x in inf]),
                 "reverse_real_minus_sham": paired_t90([x["score_real_reverse"] for x in inf],
                                                       [x["score_sham_reverse"] for x in inf]),
                 "mean_abs_real": float(np.mean([abs(x["score_real"]) for x in inf])) if inf else float("nan"),
                 "mean_abs_sham": float(np.mean([abs(x["score_sham"]) for x in inf])) if inf else float("nan")}
            d = m["real_minus_sham"]
            m["rule_0.10_met"] = bool(np.isfinite(d["t90"][0]) and d["mean"] >= RULE and d["t90"][0] > 0)
            m["n_common_per_seed"] = [b.get("n_common_per_magnitude", {}).get(k) for b in rr]
            m["attrition_per_seed"] = [b.get("attrition_per_magnitude", {}).get(k) for b in rr]
            blk["magnitudes"][k] = m
        blk["label"] = label(blk, mags)
        out[g] = blk
    if "untrained" in out:
        fl = out["untrained"]["magnitudes"]
        out["untrained"]["floor_passes_somewhere"] = any(
            np.isfinite(v["real_minus_sham"]["t90"][0]) and v["real_minus_sham"]["t90"][0] > 0 for v in fl.values())
    return out


def label(blk: dict, mags) -> str:
    if blk["gate_i"].get("passed") is False:
        return "uninformative: no 1 s nudge moves this policy"
    M = blk["magnitudes"]
    keys = [_mag_key(a) for a in sorted(mags)]
    use = [k for k in keys if not M[k]["VOID"]]
    top = _mag_key(4.0)
    if top in use and M[top]["mean_abs_real"] > 0.25 and M[top]["mean_abs_sham"] > 0.25:
        M[top]["off_manifold"] = True
        use = [k for k in use if k != top]
    if not use:
        return "VOID at every magnitude"
    means = [M[k]["real_minus_sham"]["mean"] for k in use]
    lb_hi = any(M[k]["real_minus_sham"]["t90"][0] > 0 for k in use if float(k) >= 2.0)
    if lb_hi and all(b >= a for a, b in zip(means, means[1:])):
        return "dose-dependent use"
    inside = all(-RULE <= M[k]["real_minus_sham"]["t90"][0] and M[k]["real_minus_sham"]["t90"][1] <= RULE
                 for k in use)
    if inside and blk["gate_i"].get("passed"):
        return "not used up to 4 s"
    return "not determined"


def run_provenance(a) -> dict:
    from _readout_common import _rel_argv
    from agent_free_common import provenance
    from promote_bv3_gates import repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(a.out), device="cpu")
    return {**p, "agents_dir": repo_rel(os.path.join(a.run_dir, "agents")), "argv": _rel_argv(list(sys.argv[1:]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--n-eps", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--pool-n", type=int, default=110)
    ap.add_argument("--pool-steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--magnitudes", type=float, nargs="+", default=list(MAGNITUDES), help="in s units")
    ap.add_argument("--positive-control", choices=("actor", "none"), default="actor")
    ap.add_argument("--integrity-ref", default=None,
                    help="run_mechanism_readouts.py output whose 1 s scores gate this readout (within 0.01)")
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (default OUT.ckpt)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args(argv)
    if any(m <= 0 for m in a.magnitudes):
        ap.error("--magnitudes must be positive")
    cells = _cfg_from_cells(a.run_dir)
    dmax = max(d for d, _ in cells)
    have = sorted({s for d, s in cells if d == dmax})
    seeds = have if a.seeds is None else list(a.seeds)
    missing = [s for s in seeds if s not in have]
    if missing:
        raise SystemExit(f"no cell at drift {dmax:.2f} for seed(s) {missing}; cells exist for {have}")
    any_cell = next(iter(cells.values()))
    knobs = any_cell.get("knobs") or {}
    if knobs.get("drift_mode") not in (None, "l3") or knobs.get("l3_family") not in (None, "gmotion"):
        raise SystemExit("the nudge readout is defined for the L3 G_motion surrogate only")
    base = {"run_dir": a.run_dir, "drift": dmax, "n_eps": a.n_eps, "steps": a.steps, "pool_n": a.pool_n,
            "pool_steps": a.pool_steps, "ray_steps": a.ray_steps, "arms": list(a.arms),
            "magnitudes": [float(m) for m in a.magnitudes], "positive_control": a.positive_control,
            "objective": any_cell.get("objective", "survival"), "mortal": bool(any_cell.get("mortal", True)),
            "knobs": knobs}
    tasks = [{**base, "seed": s} for s in seeds]
    ckpt = a.ckpt or a.out + ".ckpt"
    if a.dry_run:
        print(f"{len(tasks)} seed(s) {seeds} at drift {dmax}; arms {a.arms}; magnitudes {a.magnitudes}; "
              f"control {a.positive_control}; gate I {'against ' + a.integrity_ref if a.integrity_ref else 'not checked'}; "
              f"checkpoints {ckpt}; would write {a.out}")
        return 0
    from _readout_common import fingerprint_extra, run_tasks
    from itasorl.experiment_b2 import format_drift
    fpx = fingerprint_extra(__file__, os.path.join(a.run_dir, "agents"), seeds, [(g, dmax) for g in a.arms])
    fpx["drift_tag"] = format_drift(dmax)
    rows = sorted(run_tasks(run_seed, tasks, ckpt, lambda t: f"s{t['seed']}", a.workers, fpx),
                  key=lambda r: r["seed"])
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    if a.integrity_ref:
        with open(a.integrity_ref, encoding="utf-8") as fh:
            integ = integrity_check(rows, json.load(fh), a.arms)
        bad = {g: v["failures"] for g, v in integ.items() if not v["passed"]}
        if bad:
            print("gate I failed; nothing written:\n  " + "\n  ".join(f"{g}: {f}" for g, fs in bad.items() for f in fs),
                  file=sys.stderr)
            return 1
        integ = {"reference": a.integrity_ref, "tolerance": INTEGRITY_TOL, "gate_I": "PASS", "arms": integ}
    else:
        integ = {"gate_I": "NOT CHECKED (no --integrity-ref)"}
    from agent_free_common import write_json
    from promote_bv3_gates import repo_rel
    if a.integrity_ref:
        integ["reference"] = repo_rel(a.integrity_ref)
    doc = {"provenance": run_provenance(a), "status": STATUS, "run_dir": repo_rel(a.run_dir), "drift": dmax,
           "seed_bases": {"world": WORLD_BASE, "pools": [800_000, 850_000]},
           "settings": {k: base[k] for k in ("n_eps", "steps", "pool_n", "pool_steps", "magnitudes", "positive_control")}
           | {"nudge_sd": NUDGE_SD, "gap_min": GAP_MIN, "min_seeds": MIN_SEEDS, "rule": RULE},
           "integrity": integ, "summary": summarize(rows, a.arms, a.magnitudes), "per_seed": rows}
    write_json(a.out, doc)
    for g, blk in doc["summary"].items():
        print(f"{g}: {blk['label']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
