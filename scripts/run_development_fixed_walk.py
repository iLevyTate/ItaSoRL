"""Development on a fixed walk: the residual controls on identical input (closing item X02, E02).

Post hoc and descriptive; changes no stored value or verdict. The CLI mirrors
scripts/run_development_readout.py. Per (run, seed): the 300-update survival agent's
registered-pool episodes (bases 800000 and 850000, 110 x 24) are recorded once, as that script's
run_task does, and replayed open loop through the untrained trunk (b = 0) and the 100, 200, 300
and 450-update snapshots. The observation trace Ot and behavior trace Bt are collected once from
the same episodes (collect_pool on the 300-update agent with return_anchors and return_obs), and
the recorded observations, actions and episode indices must equal the collected ones, else the
script stops. At each b, from the battery's own functions (run_development_readout.battery):
replay_target (the pooled probe on the replayed states), resid_obs (residual_probe_with_diagnostics
with history_basis(Ot, lags=1)) and resid_trace (with _trace_phi(Bt)), all on the same walk.

Gate: replay_target per seed and budget equals the stored `summary.aggregate["<b> replay_target"]`
per_seed of --integrity-ref within 1e-6, else nothing is written. Readings per run:
1-fixed: resid_obs(450) - resid_obs(100) lower bound above 0.05 reads "memory beyond the linear
senses grows on identical input"; else replay_target(450) - replay_target(100) lower bound
above 0.05 reads "only the trace sharpens on identical input"; else "no growth on identical
input". 2-fixed: the same rule on resid_trace. Auxiliary-fixed (wm and nowm both given): C1
minus C2 resid_obs at 450 with a lower bound above 0 reads "the auxiliary builds the stored
component on identical input". The 0 and 100-update snapshots are read on a walk their own
policy would not produce, and the residual probes are linear controls, not a memory claim.

    python scripts/run_development_fixed_walk.py --run wm=fullruns/corrected_l3_h8_wm/agents \\
        --run nowm=fullruns/corrected_l3_h8_nowm/agents \\
        --integrity-ref wm=artifacts/development/corrected_l3_h8_wm.json \\
        --integrity-ref nowm=artifacts/development/corrected_l3_h8_nowm.json \\
        --out-dir fullruns/closing_2026-10/x02 --ckpt fullruns/closing_2026-10/x02/ckpt --workers 2
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from _readout_common import AUTH_BASE, DRIFT, SURR_BASE, fingerprint_extra, load, run_tasks, setup_l3, world_params

GENERATED_BY = "scripts/run_development_fixed_walk.py"
BUDGETS = (100, 200, 300, 450)
GATE_TOL = 1e-6
OBS_TOL = 1e-4          # normalized observations: the recorder keeps float32 raw values
STATUS = "post hoc, descriptive; changes no stored value or verdict"


class WalkMismatch(RuntimeError):
    """The recorded episodes are not the collected ones."""


def assert_same_walk(O_raw, A, K, Ot, At, It, norm) -> None:
    """The recorded raw observations (normalized with the 300-update agent's normalizer),
    actions and episode indices equal the collected ones; raises WalkMismatch otherwise."""
    if len(K) != len(It) or not np.array_equal(np.asarray(K), np.asarray(It)):
        raise WalkMismatch(f"episode indices differ: recorded {list(K)[:8]}..., collected {list(It)[:8]}...")
    if np.asarray(A).shape != np.asarray(At).shape or not np.array_equal(np.asarray(A), np.asarray(At)):
        raise WalkMismatch("recorded actions differ from the collected actions")
    On = np.asarray(norm(np.asarray(O_raw, np.float64)), np.float32)
    if On.shape != np.asarray(Ot).shape or not np.allclose(On, Ot, rtol=OBS_TOL, atol=OBS_TOL):
        dev = float(np.max(np.abs(On - Ot))) if On.shape == np.asarray(Ot).shape else float("nan")
        raise WalkMismatch(f"recorded observations differ from the collected ones (max |diff| {dev:.3g})")


def run_task(t: dict) -> dict:
    from itasorl.behavior_audit import _trace_phi
    from itasorl.control_diagnostics import history_basis, residual_probe_with_diagnostics
    from itasorl.eval_protocols import readout_from_states, record_trajectories, replay_states
    from itasorl.experiment_b2 import collect_pool

    setup_l3()
    P = world_params()
    s, n, T, rs = t["seed"], t["n_eps"], t["steps"], t["ray_steps"]
    t0 = time.time()
    a300, n300 = load(t["agents_dir"], DRIFT, s, "survival", 300)
    walk = {}
    for lab, dd, base in (("a", 0.0, AUTH_BASE), ("s", DRIFT, SURR_BASE)):
        O, A, K = record_trajectories(a300, n300, P, dd, n, T, "cpu", base, rs)
        res = collect_pool(a300, n300, P, dd, n, T, "cpu", base, rs, return_anchors=True,
                           return_obs=True, return_actions=True, return_index=True)
        Bt, Ot, At, It = res[6], res[7], res[8], res[9]
        assert_same_walk(O, A, K, Ot, At, It, n300)
        walk[lab] = (O, A, Bt, Ot)
    (Oa, Aa, Bta, Ota), (Os, As, Bts, Ots) = walk["a"], walk["s"]
    Ot, Bt = np.concatenate([Ota, Ots]), np.concatenate([Bta, Bts])
    y = np.r_[np.zeros(len(Oa)), np.ones(len(Os))].astype(int)
    phi_obs, phi_trace = history_basis(Ot, lags=1), _trace_phi(Bt, quad=False)
    # The two pools are different worlds (800000 and 850000), not twins: one CV group per
    # episode, the partition the stored readout used, written out.
    ind = np.arange(len(y))
    au = lambda d: d["auroc"] if isinstance(d, dict) else d  # noqa: E731
    out = {"run": t["run"], "seed": s, "n": [int(len(Oa)), int(len(Os))], "budgets": {}}
    for b in (0,) + BUDGETS:
        ag, nm = load(t["agents_dir"], DRIFT, s, "untrained") if b == 0 else \
            load(t["agents_dir"], DRIFT, s, "survival", b)
        Ha, Hs = replay_states(ag, nm, Oa, Aa), replay_states(ag, nm, Os, As)
        H = np.concatenate([Ha, Hs])
        out["budgets"][str(b)] = {
            "replay_target": readout_from_states(Ha, Hs, seed=s, groups=ind)["target"],
            "resid_obs_fixed": au(residual_probe_with_diagnostics(H, phi_obs, y, groups=ind)),
            "resid_trace_fixed": au(residual_probe_with_diagnostics(H, phi_trace, y, groups=ind))}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def integrity(rows: list, ref: dict) -> list[str]:
    """Gate: replay_target per seed and budget against the stored development readout."""
    seeds = sorted(int(r["seed"]) for r in ref["rows"])
    agg = ref["summary"]["aggregate"]
    fails = []
    for r in rows:
        if int(r["seed"]) not in seeds:
            fails.append(f"seed {r['seed']}: not in the stored readout")
            continue
        i = seeds.index(int(r["seed"]))
        for b in (0,) + BUDGETS:
            want = float(agg[f"{b} replay_target"]["per_seed"][i])
            got = float(r["budgets"][str(b)]["replay_target"])
            if not (np.isfinite(got) and abs(got - want) <= GATE_TOL):
                fails.append(f"seed {r['seed']} b={b}: replay_target {got!r} vs stored {want!r}")
    return fails


def _reading(d_resid: dict, d_target: dict, grows: str) -> str:
    if d_resid["t90"][0] > 0.05:
        return grows
    if d_target["t90"][0] > 0.05:
        return "only the trace sharpens on identical input"
    return "no growth on identical input"


def summarize(rows: list, run: str) -> dict:
    from itasorl.novel_readouts import paired_t90
    rr = sorted([r for r in rows if r["run"] == run], key=lambda r: r["seed"])
    get = lambda b, k: [r["budgets"][str(b)][k] for r in rr]  # noqa: E731
    agg = {f"{b} {k}": {**paired_t90(get(b, k)), "per_seed": get(b, k)}
           for b in (0,) + BUDGETS for k in ("replay_target", "resid_obs_fixed", "resid_trace_fixed")}
    d_obs = paired_t90(get(450, "resid_obs_fixed"), get(100, "resid_obs_fixed"))
    d_tr = paired_t90(get(450, "resid_trace_fixed"), get(100, "resid_trace_fixed"))
    d_t = paired_t90(get(450, "replay_target"), get(100, "replay_target"))
    return {"seeds": [r["seed"] for r in rr], "aggregate": agg, "readings": {
        "replay_target 450 - 100": d_t,
        "1-fixed (resid_obs 450 - 100)": {**d_obs, "label": _reading(
            d_obs, d_t, "memory beyond the linear senses grows on identical input")},
        "2-fixed (resid_trace 450 - 100)": {**d_tr, "label": _reading(
            d_tr, d_t, "the component beyond the trace grows on identical input")}}}


def auxiliary(rows: list) -> dict | None:
    from itasorl.novel_readouts import paired_t90
    by = {}
    for r in rows:
        by.setdefault(r["run"], {})[r["seed"]] = r
    if not {"wm", "nowm"} <= set(by):
        return None
    seeds = sorted(set(by["wm"]) & set(by["nowm"]))
    c = {str(b): paired_t90([by["wm"][s]["budgets"][str(b)]["resid_obs_fixed"] for s in seeds],
                            [by["nowm"][s]["budgets"][str(b)]["resid_obs_fixed"] for s in seeds]) for b in BUDGETS}
    met = bool(np.isfinite(c["450"]["t90"][0]) and c["450"]["t90"][0] > 0)
    return {"C1 minus C2 resid_obs_fixed": c, "seeds": seeds,
            "label": "the auxiliary builds the stored component on identical input" if met else "not met"}


def run_provenance(out: str, agents_dir: str) -> dict:
    from _readout_common import _rel_argv
    from agent_free_common import provenance
    from promote_bv3_gates import repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(out), device="cpu")
    return {**p, "agents_dir": repo_rel(agents_dir), "argv": _rel_argv(list(sys.argv[1:]))}


def _pairs(specs, ap, flag: str) -> dict:
    out = {}
    for spec in specs:
        name, sep, v = spec.partition("=")
        if not sep:
            ap.error(f"{flag} expects name=PATH, got {spec!r}")
        out[name] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="name=agents_dir (wm and nowm)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (default OUT_DIR/ckpt)")
    ap.add_argument("--integrity-ref", action="append", default=[],
                    help="name=PATH of the stored development readout whose replay_target values gate the run")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args(argv)
    runs, refs = _pairs(a.run, ap, "--run"), _pairs(a.integrity_ref, ap, "--integrity-ref")
    unknown = sorted(set(refs) - set(runs))
    if unknown:
        ap.error(f"--integrity-ref names runs not given with --run: {unknown}")
    tasks = [{"run": nm, "agents_dir": d, "seed": s, "n_eps": a.n_eps, "steps": a.steps, "ray_steps": a.ray_steps}
             for nm, d in runs.items() for s in a.seeds]
    ckpt = a.ckpt or os.path.join(a.out_dir, "ckpt")
    if a.dry_run:
        print(f"{len(tasks)} task(s) over runs {sorted(runs)}; gates {sorted(refs) or 'none'}; checkpoints {ckpt}; "
              f"would write {[os.path.join(a.out_dir, f'fixed_walk_{nm}.json') for nm in runs]}")
        return 0
    fpx = {nm: fingerprint_extra(__file__, d, a.seeds, ["survival", "untrained"], (None, *BUDGETS))
           for nm, d in runs.items()}
    rows = run_tasks(run_task, tasks, ckpt, lambda t: f"{t['run']}_s{t['seed']}", a.workers, fpx)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    gates = {}
    for nm, path in refs.items():
        with open(path, encoding="utf-8") as fh:
            fails = integrity([r for r in rows if r["run"] == nm], json.load(fh))
        if fails:
            print(f"{nm}: integrity gate failed; nothing written:\n  " + "\n  ".join(fails), file=sys.stderr)
            return 1
        gates[nm] = path
    from agent_free_common import write_json
    from promote_bv3_gates import repo_rel
    aux = auxiliary(rows)
    for nm, d in runs.items():
        out = os.path.join(a.out_dir, f"fixed_walk_{nm}.json")
        doc = {"provenance": run_provenance(out, d), "status": STATUS, "run": nm,
               "settings": {"pools": [AUTH_BASE, SURR_BASE], "n_eps": a.n_eps, "steps": a.steps,
                            "budgets": [0, *BUDGETS], "walk": "the 300-update survival agent's own episodes"},
               "integrity": ({"reference": repo_rel(gates[nm]), "tolerance": GATE_TOL, "gate": "PASS",
                              "observations": "recorded equal collected (asserted per task)"}
                             if nm in gates else {"gate": "NOT CHECKED (no --integrity-ref)"}),
               "summary": summarize(rows, nm), "auxiliary_fixed": aux,
               "rows": sorted([r for r in rows if r["run"] == nm], key=lambda r: r["seed"])}
        write_json(out, doc)
        print(nm, json.dumps({k: v.get("label") for k, v in doc["summary"]["readings"].items() if "label" in v}))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
