"""Held-out transfer of the 300-update probe across development snapshots (closing item B6).

The development readout (scripts/run_development_readout.py, spec
docs/specs/2026-10-15-what-develops-design.md) scored the frozen 300-update probe on each
snapshot's states in the SAME world seeds it was fit on (AUTH_BASE 800000 and SURR_BASE 850000),
so its stored transfer_from_300 is in-sample: an upper bound, not held-out transfer
(CORRECTIONS 2026-10-16). This readout keeps the fit exactly as stored and moves the scoring to
fresh worlds:

  in-sample (integrity)   fit on the 300-update states of the registered pools, score on the
                          b-update states of the same pools: must reproduce the stored
                          transfer_from_300 of every (run, seed, b) within 1e-9
  held out                the same fitted probe scored on the b-update states of fresh pools at
                          the registered bases B6-auth 2700000 and B6-surr 2750000
                          (itasorl.post_review.REGISTERED_LATER), for b = 100, 200, 300, 450;
                          b = 300 is the self-transfer row (the 300-update probe on its own
                          snapshot in unseen worlds)
  fresh CV target         the registered cross-validated target of each snapshot on the fresh
                          pools, for comparison

Each held-out score goes through experiment_b2.transfer_probe with the opt-in world-overlap
guard (closing hook A7c): one world seed per row on each side (base + episode index), and a
shared world is an error. The script refuses to start when the B6 bases collide with any base
in use (post_review.base_collisions against post_review.all_bases_in_use()) or when --n-eps
exceeds their registered footprint. If any stored transfer_from_300 is not reproduced, the run's
held-out values are withheld: nothing is written and the script exits 1 (OD9 fallback).

Post hoc and descriptive: it changes no stored value or verdict.

    python scripts/run_development_heldout_transfer.py --run wm=fullruns/corrected_l3_h8_wm/agents \\
        --run nowm=fullruns/corrected_l3_h8_nowm/agents \\
        --integrity-ref artifacts/development/corrected_l3_h8_wm.json \\
        --out-dir fullruns/closing_2026-10/b6 --ckpt fullruns/closing_2026-10/b6/ckpt --workers 2 [--dry-run]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import sys
import time

import numpy as np

from _readout_common import AUTH_BASE, DRIFT, SURR_BASE, agent_path, fingerprint_extra, load, run_tasks, setup_l3, world_params

GENERATED_BY = "scripts/run_development_heldout_transfer.py"
SPEC = "docs/specs/2026-10-15-what-develops-design.md (stored estimator); closing item B6"
BUDGETS = (100, 200, 300, 450)
STEPS, RAY_STEPS = 24, 5
TOL = 1e-9
RUN_FILES = {"wm": "corrected_l3_h8_wm", "nowm": "corrected_l3_h8_nowm"}
STATUS = "post hoc, descriptive; changes no stored value or verdict"


def fresh_bases(n_eps: int) -> tuple[int, int]:
    """The registered B6 bases, after the collision and footprint checks."""
    from itasorl import post_review as pr
    own = [pr.registered_later("B6-auth"), pr.registered_later("B6-surr")]
    too_big = [name for name, _, n in own if n_eps > n]
    if too_big:
        raise SystemExit(f"--n-eps {n_eps} exceeds the registered footprint of {too_big}")
    hits = pr.base_collisions(new=tuple((name, base, n_eps) for name, base, _ in own),
                              in_use=pr.all_bases_in_use())
    if hits:
        raise SystemExit(f"B6 bases collide with bases in use: {hits}")
    return own[0][1], own[1][1]


def pools(agent, norm, P, n: int, auth_base: int, surr_base: int):
    """(X, y, worlds) of the pooled readout at two bases, collected as the development battery
    collects them (collect_pool, deterministic policy, full-length episodes only)."""
    from itasorl.experiment_b import episode_features
    from itasorl.experiment_b2 import collect_pool
    Ha, *_, ia = collect_pool(agent, norm, P, 0.0, n, STEPS, "cpu", auth_base, RAY_STEPS, return_index=True)
    Hs, *_, is_ = collect_pool(agent, norm, P, DRIFT, n, STEPS, "cpu", surr_base, RAY_STEPS, return_index=True)
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    worlds = np.r_[auth_base + np.asarray(ia, int), surr_base + np.asarray(is_, int)]
    return X, y, worlds


def run_task(t: dict) -> dict:
    from itasorl.experiment_a import grouped_auroc
    from itasorl.experiment_b2 import transfer_probe

    setup_l3()
    P = world_params()
    s, n = t["seed"], t["n_eps"]
    t0 = time.time()
    reg, fresh = {}, {}
    for b in BUDGETS:
        ag, nm = load(t["agents_dir"], DRIFT, s, "survival", b)
        reg[b] = pools(ag, nm, P, n, AUTH_BASE, SURR_BASE)
        fresh[b] = pools(ag, nm, P, n, t["fresh_auth"], t["fresh_surr"])
    Xtr, ytr, wtr = reg[300]
    out = {"run": t["run"], "seed": s, "budgets": {}}
    for b in BUDGETS:
        Xr, yr, _ = reg[b]
        Xf, yf, wf = fresh[b]
        out["budgets"][str(b)] = {
            "insample_transfer_from_300": float(transfer_probe(Xtr, ytr, Xr, yr)),
            "heldout_transfer_from_300": float(transfer_probe(Xtr, ytr, Xf, yf, train_worlds=wtr, test_worlds=wf)),
            "fresh_cv_target": float(grouped_auroc(Xf, yf, np.arange(len(yf)))),
            "n_registered": [int((yr == 0).sum()), int((yr == 1).sum())],
            "n_fresh": [int((yf == 0).sum()), int((yf == 1).sum())]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def load_references(path: str, runs) -> dict:
    """{run: {seed: {b: stored transfer_from_300}}} from the stored development outputs: the
    file given, and its sibling for the other run, each identified by its own "run" key."""
    if not os.path.exists(path):
        raise SystemExit(f"integrity reference {path} does not exist")
    docs = {}
    for p in {path, *(os.path.join(os.path.dirname(path), f"{RUN_FILES.get(r, r)}.json") for r in runs)}:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
            docs[doc.get("run")] = (p, doc)
    out = {}
    for r in runs:
        if r not in docs:
            raise SystemExit(f"no stored development output for run {r!r} beside {path}")
        p, doc = docs[r]
        out[r] = {"source": p, "values": {int(row["seed"]): {int(b): v.get("transfer_from_300")
                                                             for b, v in row["budgets"].items()}
                                          for row in doc["rows"]}}
    return out


def integrity(rows: list, refs: dict) -> dict:
    out = {}
    for run, ref in refs.items():
        checks, bad = [], []
        for r in sorted([r for r in rows if r["run"] == run], key=lambda r: r["seed"]):
            for b in BUDGETS:
                stored = ref["values"].get(r["seed"], {}).get(b)
                new = r["budgets"][str(b)]["insample_transfer_from_300"]
                diff = abs(new - stored) if stored is not None else float("nan")
                ok = bool(np.isfinite(diff) and diff <= TOL)
                checks.append({"seed": r["seed"], "budget": b, "recomputed": new, "stored": stored,
                               "abs_diff": diff, "ok": ok})
                if not ok:
                    bad.append((r["seed"], b))
        out[run] = {"source": ref["source"], "tolerance": TOL, "n_checked": len(checks), "failed": bad,
                    "worst_abs_diff": float(max((c["abs_diff"] for c in checks), default=float("nan"))),
                    "pass": bool(checks and not bad), "checks": checks}
    return out


def summarize(rows: list) -> dict:
    from itasorl.post_review import paired_summary, seed_summary
    by = {r["seed"]: r["budgets"] for r in rows}
    out = {}
    for b in BUDGETS:
        get = lambda k: {str(s): by[s][str(b)][k] for s in by}  # noqa: E731
        out[str(b)] = {k: {**seed_summary(list(get(k).values())), "per_seed": [by[s][str(b)][k] for s in sorted(by)]}
                       for k in ("insample_transfer_from_300", "heldout_transfer_from_300", "fresh_cv_target")}
        out[str(b)]["heldout_minus_insample"] = paired_summary(get("heldout_transfer_from_300"),
                                                               get("insample_transfer_from_300"))
        out[str(b)]["heldout_minus_fresh_cv_target"] = paired_summary(get("heldout_transfer_from_300"),
                                                                      get("fresh_cv_target"))
    out["seeds"] = sorted(by)
    return out


def run_provenance(out: str, runs: dict, argv: list) -> dict:
    from _readout_common import _rel_argv
    from agent_free_common import provenance
    from promote_bv3_gates import repo_rel
    p = provenance(GENERATED_BY, commit_key="git_commit_at_promotion", output=repo_rel(out), device="cpu")
    return {**p, "agents_dir": {nm: repo_rel(d) for nm, d in runs.items()}, "argv": _rel_argv(list(argv))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="name=agents_dir (wm and nowm)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint directory (default OUT_DIR/ckpt)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--integrity-ref", required=True, metavar="PATH",
                    help="the stored development output (artifacts/development/corrected_l3_h8_wm.json); "
                         "the other run's file is read beside it")
    ap.add_argument("--dry-run", action="store_true", help="check bases, agents and references, then exit")
    a = ap.parse_args(argv)
    argv = list(sys.argv[1:] if argv is None else argv)
    runs = dict(x.split("=", 1) for x in a.run)
    fa, fs = fresh_bases(a.n_eps)
    refs = load_references(a.integrity_ref, list(runs))
    missing = [agent_path(d, DRIFT, s, "survival", b) for d in runs.values() for s in a.seeds for b in BUDGETS
               if not os.path.exists(agent_path(d, DRIFT, s, "survival", b))]
    no_ref = [(r, s, b) for r in runs for s in a.seeds for b in BUDGETS if refs[r]["values"].get(s, {}).get(b) is None]
    print(f"runs {sorted(runs)}, seeds {a.seeds}, budgets {list(BUDGETS)}; fresh bases {fa} / {fs} "
          f"(no collision); missing agents {len(missing)}; cells without a stored value {len(no_ref)}", flush=True)
    if missing or no_ref:
        raise SystemExit(f"missing agents {missing[:5]} or stored values {no_ref[:5]}")
    if a.dry_run:
        print("dry run: nothing collected or written")
        return 0
    tasks = [{"run": nm, "agents_dir": d, "seed": s, "n_eps": a.n_eps, "fresh_auth": fa, "fresh_surr": fs}
             for nm, d in runs.items() for s in a.seeds]
    fpx = {nm: fingerprint_extra(__file__, d, a.seeds, ["survival"], (None, 100, 200, 450)) for nm, d in runs.items()}
    rows = run_tasks(run_task, tasks, a.ckpt or os.path.join(a.out_dir, "ckpt"),
                     lambda t: f"{t['run']}_s{t['seed']}", a.workers, fpx)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    integ = integrity(rows, refs)
    for run, v in integ.items():
        print(f"{run}: in-sample transfer_from_300 reproduced {v['n_checked'] - len(v['failed'])}/{v['n_checked']} "
              f"within {TOL:g} (worst {v['worst_abs_diff']:.3g}); pass {v['pass']}", flush=True)
    if not all(v["pass"] for v in integ.values()):
        print("INTEGRITY FAILED: the held-out values are withheld (OD9 fallback); nothing written",
              file=sys.stderr)
        return 1
    from agent_free_common import write_json
    from promote_bv3_gates import repo_rel
    out = os.path.join(a.out_dir, "heldout_transfer.json")
    doc = {"provenance": run_provenance(out, runs, argv), "status": STATUS, "spec": SPEC,
           "bases": {"registered": [AUTH_BASE, SURR_BASE], "fresh": [fa, fs],
                     "fresh_names": ["B6-auth", "B6-surr"]},
           "settings": {"budgets": list(BUDGETS), "seeds": a.seeds, "n_eps": a.n_eps, "steps": STEPS,
                        "ray_steps": RAY_STEPS, "drift": DRIFT,
                        "probe": "experiment_b2.transfer_probe fit on the 300-update registered pools",
                        "overlap_guard": "transfer_probe(train_worlds=, test_worlds=), world seed = base + episode index"},
           "integrity": {run: {**v, "source": repo_rel(v["source"])} for run, v in integ.items()},
           "runs": {nm: {"summary": summarize([r for r in rows if r["run"] == nm]),
                         "rows": sorted([r for r in rows if r["run"] == nm], key=lambda r: r["seed"])}
                    for nm in runs}}
    write_json(out, doc)
    print("wrote", out, flush=True)
    for nm in runs:
        s = doc["runs"][nm]["summary"]
        print(nm, {b: round(s[str(b)]["heldout_transfer_from_300"]["mean"], 3) for b in BUDGETS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
