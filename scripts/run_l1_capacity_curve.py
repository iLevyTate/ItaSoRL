"""R31, L1 part: the stream-decoder capacity curve on the saved L1 held-out agents.

Spec: docs/specs/2026-10-11-post-review-readouts-design.md, section "Amendment 2026-10-11,
before any L1 readout ran". Readout-only.

For every arm and seed at the run's strongest drift (the thirty drift-0.023 cells of the
historical L1 held-out run, grid spacing 0.023, sensor noise 0.01), regenerate the standard
pools exactly as scripts/run_control_diagnostics.py does at the L1 rung (seed bases 800000 /
850000, its setup_rung, its collect_pool call, its sequence of normalized observation and
previous action), then score the supervised stream decoder of FINDINGS 14.7.1 at 1x, 2x, 4x and
8x the trunk's recurrent width (GRU hidden 96, 192, 384, 768; the per-step encoder stays at
64) and the 1D-conv reader of the L3 part, all with the explicit-v1 folds. One task is one
(arm, seed, width); every finished task is appended to a partial file, so a cut-off session
resumes.

Two checks are reported, never skipped:
  integrity  the regenerated state target of every (arm, seed) against the recorded
             explicit-partition value (artifacts/fold_rescore/l1_heldout.json) within 0.01,
             the frozen L1 spec's gate; and, when the run directory carries states/, a bit
             match of the regenerated pools against the saved dumps. A failed gate voids the
             readout (exit code 1).
  record     the 1x decoder against the recorded L1 stream readout
             (artifacts/control_diagnostics/l1_stream_readout.json): per cell, and whether the
             survival mean reproduces the recorded 0.590 within 0.01.

    ITASORL_FOLDS=explicit python scripts/run_l1_capacity_curve.py \\
        --run-dir fullruns_upload/l1_heldout --out fullruns_upload/l1_capacity_curve.json \\
        --workers 4 [--widths 8x,4x,2x,1x,conv] [--arms survival,predictor,untrained] [--seeds 0 1]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re
import sys
import time
import traceback

import numpy as np

from itasorl import post_review as pr

ROOT = _bootstrap.ROOT
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")
WIDTHS = ("8x", "4x", "2x", "1x", "conv")          # also the scheduling order, longest first
TOL = 0.01                                          # the frozen L1 spec's integrity tolerance
INTEGRITY_REF = os.path.join("artifacts", "fold_rescore", "l1_heldout.json")
RECORD_REF = os.path.join("artifacts", "control_diagnostics", "l1_stream_readout.json")
RECORD_STACK = ("owner's laptop CPU, 4 workers, commit 5666f59 (results manifest, "
                "L1-STREAM-READOUT); the artifact records no package versions")


# ---------------------------------------------------------------------------
# Run discovery and settings
# ---------------------------------------------------------------------------
def discover(run_dir: str) -> dict:
    """The saved agents at the run's strongest drift: {"drift_tag", "drift", "cells": {(arm,
    seed): path}} and the cell fingerprints of that drift."""
    agents_dir = os.path.join(run_dir, "agents")
    found = []
    for name in sorted(os.listdir(agents_dir)):
        m = AGENT_RE.match(name)
        if m:
            found.append((m.group(1), int(m.group(2)), m.group(3), os.path.join(agents_dir, name)))
    if not found:
        raise SystemExit(f"no saved agents under {agents_dir}")
    tag = max({f[0] for f in found}, key=float)
    cells = {(arm, s): p for t, s, arm, p in found if t == tag}
    prints = {}
    for s in sorted({s for _, s in cells}):
        p = os.path.join(run_dir, "cells", f"cell_d{tag}_s{s}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                prints[f"d{tag}_s{s}"] = json.load(fh).get("fingerprint")
    return {"drift_tag": tag, "drift": float(tag), "cells": cells, "fingerprints": prints}


def settings_for(quick: bool, l1_delta: float, l1_sigma: float) -> dict:
    s = {"n_eps": 110, "steps": 24, "ray_steps": 5, "gru_epochs": pr.GRU_EPOCHS,
         "drift_mode": "l1", "l1_delta": float(l1_delta), "l1_sigma": float(l1_sigma),
         "capacity": {r["label"]: {"embed": r["embed"], "hidden": r["hidden"]} for r in pr.capacity_table()},
         "conv": {"channels": list(pr.CONV_CHANNELS), "kernel": pr.CONV_KERNEL}, "quick": False}
    if quick:
        s.update(n_eps=12, steps=8, gru_epochs=2, quick=True)
    return s


def plan_tasks(arms, seeds, widths, done: dict) -> list[tuple[str, str, str]]:
    """(arm, seed, width) still to run; the widest decoders first so workers pack well."""
    return [(a, str(s), w) for w in WIDTHS if w in widths for a in arms for s in seeds
            if (f"{a}/{w}", str(s)) not in done]


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
def regenerate(task: dict) -> dict:
    """The standard pools of one (arm, seed) at the L1 rung, built as run_control_diagnostics
    builds them, with the state target and, when the dump exists, the bit match."""
    from itasorl.experiment_b import episode_features, probe_auroc
    from itasorl.experiment_b2 import collect_pool, load_agent_bundle
    from itasorl.world import WorldParams

    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    st, d = task["settings"], task["drift"]
    agent, norm = load_agent_bundle(task["path"], "cpu")
    pools = {}
    for lab, dd, base in (("a", 0.0, 800_000), ("s", d, 850_000)):
        H, *_, Bt, Ot, At = collect_pool(agent, norm, P, dd, st["n_eps"], st["steps"], "cpu", base,
                                         st["ray_steps"], return_anchors=True, return_obs=True,
                                         return_actions=True)
        pools[lab] = (H, Ot, At)
    out = {"n_auth": int(len(pools["a"][0])), "n_surr": int(len(pools["s"][0])), "dump_bit_match": None}
    dump = task.get("dump")
    if dump and os.path.exists(dump):
        with np.load(dump) as z:
            out["dump_bit_match"] = bool(np.array_equal(z["Ha"], pools["a"][0])
                                         and np.array_equal(z["Hs"], pools["s"][0]))
    if min(out["n_auth"], out["n_surr"]) < 5:
        out["error"] = "too few survivors"
        return out
    H = np.concatenate([pools["a"][0], pools["s"][0]])
    Ot = np.concatenate([pools["a"][1], pools["s"][1]])
    At = np.concatenate([pools["a"][2], pools["s"][2]])
    y = np.r_[np.zeros(out["n_auth"]), np.ones(out["n_surr"])].astype(int)
    out["target"] = float(probe_auroc(episode_features(H), y))
    a_prev = np.concatenate([np.zeros_like(At[:, :1]), At[:, :-1]], axis=1)
    out["seq"], out["y"] = np.concatenate([Ot, a_prev], axis=2), y
    return out


def decode(seq, y, width: str, *, seed: int, epochs: int) -> dict:
    if width == "conv":
        return pr.conv_sequence_auroc(seq, y, epochs=epochs, seed=seed)
    from itasorl.control_diagnostics import sequence_gru_auroc
    row = {r["label"]: r for r in pr.capacity_table()}[width]
    return sequence_gru_auroc(seq, y, embed=row["embed"], hidden=row["hidden"], epochs=epochs, seed=seed)


def run_task(task: dict) -> dict:
    """One (arm, seed, width); never raises, so one failure cannot take down the pool."""
    t0 = time.time()
    key = {"arm": task["arm"], "seed": task["seed"], "width": task["width"]}
    try:
        import torch

        from run_control_diagnostics import setup_rung
        torch.set_num_threads(1)
        setup_rung({"drift_mode": "l1", "l1_delta": task["settings"]["l1_delta"],
                    "l1_sigma": task["settings"]["l1_sigma"]})
        pools = regenerate(task)
        res = {k: v for k, v in pools.items() if k not in ("seq", "y")}
        if "error" not in pools:
            res["decoder"] = decode(pools["seq"], pools["y"], task["width"], seed=int(task["seed"]),
                                    epochs=task["settings"]["gru_epochs"])
        res["seconds"] = round(time.time() - t0, 1)
        return {**key, "result": res}
    except Exception:
        return {**key, "error": traceback.format_exc(), "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Checks and summaries (pure)
# ---------------------------------------------------------------------------
def load_integrity_ref(doc: dict, drift: float) -> dict:
    """{(arm, seed): recorded explicit-partition state target} at `drift`."""
    return {(c["agent"], str(int(c["seed"]))): float(c["explicit"]["target"]) for c in doc["cells"]
            if abs(float(c["drift"]) - drift) < 1e-9 and "explicit" in c}


def load_record(doc: dict, drift: float) -> dict:
    """{(arm, seed): {"target", "seq_gru"}} of the recorded L1 stream readout at `drift`."""
    return {(c["arm"], str(int(c["seed"]))): {"target": float(c["target"]),
                                             "seq_gru": float(c["seq_gru"]["auroc"])}
            for c in doc["cells"] if abs(float(c["drift"]) - drift) < 1e-9 and "seq_gru" in c}


def cell_targets(rows: dict) -> tuple[dict, list]:
    """One regenerated state target per (arm, seed), and every cell whose tasks disagreed
    (the regeneration is deterministic, so any disagreement is reported)."""
    seen: dict = {}
    for (arm, seed, _w), r in rows.items():
        if "target" in r:
            seen.setdefault((arm, seed), set()).add(r["target"])
    return ({k: next(iter(v)) for k, v in seen.items()},
            [{"arm": k[0], "seed": k[1], "targets": sorted(v)} for k, v in seen.items() if len(v) > 1])


def integrity(rows: dict, ref: dict, tol: float = TOL) -> dict:
    targets, inconsistent = cell_targets(rows)
    checks = sorted(({"arm": a, "seed": s, "recorded": ref[(a, s)], "rescored": t,
                      "abs_dev": abs(t - ref[(a, s)]), "bit_equal": t == ref[(a, s)]}
                     for (a, s), t in targets.items() if (a, s) in ref),
                    key=lambda c: (c["arm"], int(c["seed"])))
    dumps = [r["dump_bit_match"] for r in rows.values() if r.get("dump_bit_match") is not None]
    worst = max((c["abs_dev"] for c in checks), default=float("nan"))
    return {"reference": INTEGRITY_REF.replace(os.sep, "/"), "tolerance": tol, "n_checked": len(checks),
            "n_bit_equal": int(sum(c["bit_equal"] for c in checks)), "worst_abs_dev": worst,
            "unreferenced": sorted(f"{a}/{s}" for (a, s) in targets if (a, s) not in ref),
            "pass": bool(checks) and len(checks) == len(targets) and worst <= tol and not inconsistent,
            "dumps": ({"checked": len(dumps), "bit_match": int(sum(dumps))} if dumps
                      else "not checked: the run directory carries no states/"),
            "inconsistent_regenerations": inconsistent, "per_cell": checks}


def record_check(rows: dict, record: dict, tol: float = TOL) -> dict:
    """The 1x decoder against the recorded L1 stream readout, per cell and per arm mean."""
    per_cell, by_arm = [], {}
    for (arm, seed, w), r in sorted(rows.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        if w != "1x" or "decoder" not in r or (arm, seed) not in record:
            continue
        rec = record[(arm, seed)]["seq_gru"]
        got = float(r["decoder"]["auroc"])
        per_cell.append({"arm": arm, "seed": seed, "recorded": rec, "rescored": got,
                         "abs_dev": abs(got - rec), "bit_equal": got == rec})
        by_arm.setdefault(arm, []).append((rec, got))
    arms = {}
    for arm, pairs in by_arm.items():
        rec_m, got_m = float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))
        arms[arm] = {"n": len(pairs), "recorded_mean": rec_m, "rescored_mean": got_m,
                     "abs_dev": abs(got_m - rec_m), "within_tol": abs(got_m - rec_m) <= tol}
    s = arms.get("survival")
    return {"reference": RECORD_REF.replace(os.sep, "/"), "record_stack": RECORD_STACK, "tolerance": tol,
            "note": "a miss on a stack other than the record's is a stack difference and does not void "
                    "the curve; a miss on the record's stack voids it (spec amendment 2026-10-11)",
            "survival_1x_mean_reproduced": None if s is None or s["n"] < 10 else bool(s["within_tol"]),
            "n_bit_equal": int(sum(c["bit_equal"] for c in per_cell)), "arms": arms, "per_cell": per_cell}


def summarize(rows: dict, arms) -> dict:
    """Per arm: the state target and every decoder across seeds, each decoder minus 1x and the
    state minus each decoder, seed-paired, and the pre-stated reading."""
    targets, _ = cell_targets(rows)
    out = {}
    for arm in arms:
        dec = {w: {s: float(r["decoder"]["auroc"]) for (a, s, ww), r in rows.items()
                   if a == arm and ww == w and "decoder" in r} for w in WIDTHS}
        dec = {w: v for w, v in dec.items() if v}
        state = {s: t for (a, s), t in targets.items() if a == arm}
        block = {"state": pr.seed_summary(list(state.values())),
                 "decoders": {w: pr.seed_summary(list(v.values())) for w, v in dec.items()},
                 "minus_1x": {w: pr.paired_summary(v, dec["1x"]) for w, v in dec.items()
                              if w != "1x" and "1x" in dec},
                 "state_minus": {w: pr.paired_summary(state, v) for w, v in dec.items()}}
        block["reading"] = reading(block)
        out[arm] = block
    return out


def reading(block: dict) -> str:
    """SEAM AVAILABLE AT WIDTH: some wider GRU (2x, 4x, 8x) reads the arm's own stream at a mean
    at least 0.05 above the 1x decoder, with the seed-paired interval above zero. READER-RELATIVE:
    every wider GRU has been scored and none does. INCOMPLETE: a width is missing."""
    wider = [block["minus_1x"].get(w) for w in ("2x", "4x", "8x")]
    if any(w is None or not np.isfinite(w.get("mean", np.nan)) for w in wider):
        return "INCOMPLETE"
    if any(pr.interval_label(w) == "ABOVE" and w["mean"] >= 0.05 for w in wider):
        return "SEAM AVAILABLE AT WIDTH"
    return "READER-RELATIVE"


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="holds agents/ and cells/ (states/ optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--arms", default=",".join(pr.ARMS))
    ap.add_argument("--widths", default=",".join(WIDTHS))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--partial", default=None, help="resume file (default <out>.partial.jsonl)")
    ap.add_argument("--l1-delta", type=float, default=0.023, help="the L1 held-out run's grid spacing")
    ap.add_argument("--l1-sigma", type=float, default=0.01, help="the L1 held-out run's sensor noise")
    ap.add_argument("--quick", action="store_true", help="tiny pools and decoders; the checks then fail by design")
    return ap


def _rows_from(done: dict) -> dict:
    """{(arm, seed, width): result} from the partial store's {(arm/width, seed): result}."""
    return {(k.split("/")[0], s, k.split("/")[1]): r for (k, s), r in done.items()}


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if os.environ.get("ITASORL_FOLDS") != "explicit":
        raise SystemExit("set ITASORL_FOLDS=explicit, the partition the recorded targets use")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    widths = [x.strip() for x in a.widths.split(",") if x.strip()]
    if set(arms) - set(pr.ARMS) or set(widths) - set(WIDTHS):
        raise SystemExit(f"arms must come from {pr.ARMS} and widths from {WIDTHS}")
    run = discover(a.run_dir)
    have = sorted({s for _, s in run["cells"]})
    seeds = have if a.seeds is None else list(a.seeds)
    missing = [(arm, s) for arm in arms for s in seeds if (arm, s) not in run["cells"]]
    if missing:
        raise SystemExit(f"no saved agent at drift {run['drift_tag']} for {missing}")
    settings = settings_for(a.quick, a.l1_delta, a.l1_sigma)
    header = {"spec": pr.SPEC, "readout": "r31-l1", "fingerprints": run["fingerprints"],
              "drift_tag": run["drift_tag"], "settings": settings}
    store = pr.PartialStore(a.partial or a.out + ".partial.jsonl", header)
    try:
        done = store.load()
    except ValueError as e:
        raise SystemExit(str(e))
    states = os.path.join(a.run_dir, "states")
    tasks = [{"arm": arm, "seed": s, "width": w, "drift": run["drift"], "settings": settings,
              "path": run["cells"][(arm, int(s))],
              "dump": os.path.join(states, f"states_d{run['drift_tag']}_s{s}_{arm}.npz")}
             for arm, s, w in plan_tasks(arms, seeds, widths, done)]
    print(f"L1 capacity curve on {a.run_dir} (drift {run['drift_tag']}): {len(done)} task(s) done, "
          f"{len(tasks)} to run", flush=True)
    errors = []

    def record(res):
        k = (f"{res['arm']}/{res['width']}", str(res["seed"]))
        if "error" in res:
            store.append(k[0], k[1], error=res["error"])
            errors.append({"arm": res["arm"], "seed": res["seed"], "width": res["width"], "error": res["error"]})
            print(f"  {res['arm']} s{res['seed']} {res['width']}: ERROR ({res['seconds']} s)", flush=True)
        else:
            store.append(k[0], k[1], result=res["result"])
            done[k] = res["result"]
            auc = res["result"].get("decoder", {}).get("auroc", float("nan"))
            print(f"  {res['arm']} s{res['seed']} {res['width']}: {auc:.3f} "
                  f"({res['result']['seconds']} s)", flush=True)

    t0 = time.time()
    if a.workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for res in pool.imap_unordered(run_task, tasks):
                record(res)
    else:
        for t in tasks:
            record(run_task(t))
    rows = _rows_from(done)
    with open(os.path.join(ROOT, INTEGRITY_REF), encoding="utf-8") as fh:
        ref = load_integrity_ref(json.load(fh), run["drift"])
    with open(os.path.join(ROOT, RECORD_REF), encoding="utf-8") as fh:
        rec = load_record(json.load(fh), run["drift"])
    from itasorl import folds
    from itasorl.results_io import git_head
    from run_post_review_readouts import stack_versions, write_json
    integ = integrity(rows, ref)
    want = {(arm, str(s), w) for arm in arms for s in seeds for w in widths}
    doc = {"generated_by": "scripts/run_l1_capacity_curve.py", "spec": pr.SPEC, "readout": "R31, L1 part",
           "git_commit": git_head(), "stack": stack_versions(), "run_dir": a.run_dir,
           "drift": run["drift"], "drift_tag": run["drift_tag"], "cell_fingerprints": run["fingerprints"],
           "fold_scheme": folds.current_scheme(), "fold_version": folds.scheme_version(),
           "seed_bases": {"authentic": 800_000, "surrogate": 850_000,
                          "note": "the registered pools of the L1 run, as run_control_diagnostics.py "
                                  "regenerated them for FINDINGS 14.7.1"},
           "settings": settings, "requested": {"arms": arms, "seeds": seeds, "widths": widths},
           "integrity": integ, "void": integ["pass"] is False,
           "record_check": record_check(rows, rec),
           "cells": [{"arm": arm, "seed": s, "width": w, **r}
                     for (arm, s, w), r in sorted(rows.items(), key=lambda kv: (kv[0][0], int(kv[0][1]),
                                                                                 WIDTHS.index(kv[0][2])))],
           "summary": summarize(rows, pr.ARMS), "errors": errors,
           "complete": bool(want <= set(rows)) and not errors,
           "wall_seconds_this_session": round(time.time() - t0, 1)}
    write_json(a.out, doc)
    rc = doc["record_check"]
    print(f"integrity gate: {'PASS' if integ['pass'] else 'FAIL, readout void'} "
          f"({integ['n_checked']} cells, worst |dev| {integ['worst_abs_dev']:.4f}, dumps {integ['dumps']})", flush=True)
    print(f"1x survival mean reproduces the recorded 0.590 within {TOL}: "
          f"{rc['survival_1x_mean_reproduced']}", flush=True)
    print(f"wrote {a.out} (complete for this request: {doc['complete']})", flush=True)
    return 0 if doc["complete"] and integ["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
