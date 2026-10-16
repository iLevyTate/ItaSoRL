"""Shared plumbing for the 2026-10-15 readout scripts: surrogate setup, agent paths,
resumable per-task checkpoints, and the provenance block."""

from __future__ import annotations

import json
import os

P_KW = {"k_land": 1.5, "k_water": 1.5, "gravity": 0.4}
AUTH_BASE, SURR_BASE = 800_000, 850_000     # the registered pools
DRIFT = 0.45


def world_params():
    from itasorl.world import WorldParams
    return WorldParams(**P_KW)


def setup_l3() -> None:
    """Install the registered L3 surrogate (hidden 8, seed 0) in this process."""
    import torch

    import itasorl.experiment_b2 as b2
    torch.set_num_threads(1)
    b2.DRIFT_MODE = "l3"
    if b2._L3_GMOTION is None:
        b2.setup_l3_surrogate(hidden=8, device="cpu", seed=0, params=world_params())


def agent_path(agents_dir: str, drift: float, seed: int, arm: str, updates: int | None = None) -> str:
    suffix = "" if updates in (None, 300) else f"_u{updates}"
    return os.path.join(agents_dir, f"agent_d{drift:.2f}_s{seed}_{arm}{suffix}.pt")


def load(agents_dir: str, drift: float, seed: int, arm: str, updates: int | None = None):
    from itasorl.experiment_b2 import load_agent_bundle
    return load_agent_bundle(agent_path(agents_dir, drift, seed, arm, updates), "cpu")


def agent_file_hashes(agents_dir: str, seeds, arms, drift: float = DRIFT, updates: int | None = None) -> dict:
    """First 16 hex of the SHA-256 of each agent file a runner loads, keyed by basename, so a
    checkpoint fingerprint changes when an agent file does; an absent file maps to None. arms holds names or (name, drift)."""
    import hashlib
    out = {}
    for arm in arms:
        name, d = arm if isinstance(arm, tuple) else (arm, drift)
        for s in seeds:
            p = agent_path(agents_dir, d, s, name, updates)
            if not os.path.exists(p):
                out[os.path.basename(p)] = None     # absent: recorded, so its arrival changes the fingerprint
                continue
            with open(p, "rb") as fh:
                out[os.path.basename(p)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def fingerprint_extra(script: str, agents_dir: str, seeds, arms, updates=(None,)) -> dict:
    """The invocation-level inputs a checkpoint must match: code commit, agent file hashes,
    fold scheme and the runner's name."""
    from itasorl import folds
    from itasorl.results_io import git_head
    agents: dict = {}
    for u in updates:
        agents.update(agent_file_hashes(agents_dir, seeds, arms, updates=u))
    return {"code": git_head(), "agents": agents, "folds": folds.scheme_version(),
            "script": os.path.basename(script)}


def task_fingerprint(task: dict, extra: dict | None = None) -> str:
    """SHA-256 of the task's scientific inputs plus `extra` (code commit, agent file hashes,
    fold scheme), so a checkpoint is reused only by an invocation with the same inputs."""
    import hashlib
    payload = {"task": {k: v for k, v in task.items() if not k.startswith("_")}, "extra": extra or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_tasks(fn, tasks: list, ckpt_dir: str, key, workers: int = 1, fingerprint_extra: dict | None = None) -> list:
    """Run `fn` on every task whose checkpoint is missing; return every result, loaded
    checkpoints included. key(task) names the checkpoint file. Each checkpoint stores the
    task and its fingerprint; one written for different inputs is refused, never reused."""
    os.makedirs(ckpt_dir, exist_ok=True)
    done, todo = [], []
    for t in tasks:
        p = os.path.join(ckpt_dir, key(t) + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("_fingerprint") != task_fingerprint(t, fingerprint_extra):
                saved = r.get("_task") or {}
                diff = sorted(k for k in set(saved) | set(t) if not k.startswith("_") and saved.get(k) != t.get(k))
                if r.get("_extra") != (fingerprint_extra or {}):
                    diff.append("fingerprint_extra")
                raise RuntimeError(f"checkpoint {p} was written for different inputs; differing fields: "
                                   f"{diff or ['no fingerprint stored']}")
            done.append(r)
        else:
            todo.append(t)
    if done:
        print(f"resumed {len(done)} task(s); {len(todo)} to go", flush=True)

    def save(r, t):
        p = os.path.join(ckpt_dir, key(t) + ".json")
        r = {**r, "_task": t, "_extra": fingerprint_extra or {}, "_fingerprint": task_fingerprint(t, fingerprint_extra)}
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(r, fh, default=float)
        os.replace(p + ".tmp", p)
        done.append(r)
        print(f"  done {key(t)} ({r.get('seconds')} s)", flush=True)

    if workers > 1 and todo:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for t, r in zip(todo, pool.imap(fn, todo)):
                save(r, t)
    else:
        for t in todo:
            save(fn(t), t)
    return done


def check_rows(rows: list, seeds, required_finite=()) -> None:
    """Every row group (by run or arm, when rows carry one) holds each expected seed exactly
    once, and every key in required_finite is finite in every row."""
    import math
    want = sorted(seeds)
    groups: dict = {}
    for r in rows:
        groups.setdefault((r.get("run"), r.get("arm")), []).append(r["seed"])
    if not groups:
        groups[(None, None)] = []
    for g, found in groups.items():
        if sorted(found) != want:
            raise RuntimeError(f"seeds: expected {want}, found {sorted(found)} in group {g}")
    for r in rows:
        for k in required_finite:
            v = r.get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise RuntimeError(f"row seed {r.get('seed')}: {k} is not finite ({v!r})")


def _rel_argv(argv: list) -> list:
    """The recorded argv with every absolute path made repo-relative, including the value of
    a `name=path` pair (the development script's --run arguments)."""
    from promote_bv3_gates import _ABS_PATH, rel_flags, repo_rel
    out = []
    for f in rel_flags(argv):
        name, eq, value = f.partition("=") if isinstance(f, str) else ("", "", "")
        out.append(f"{name}={repo_rel(value)}" if eq and _ABS_PATH.match(value) else f)
    return out


def write_output(path: str, script: str, argv: list, inputs: dict, spec: str, body: dict,
                 required_finite=()) -> None:
    from itasorl.optional_readouts import provenance
    if "rows" in body and "seeds" in inputs:
        check_rows(body["rows"], inputs["seeds"], required_finite)
        body = {**body, "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in body["rows"]]}
    from promote_bv3_gates import repo_rel
    inputs = {k: repo_rel(v) if k.endswith("_dir") else v for k, v in inputs.items()}
    prov = provenance(script, "python " + " ".join([script, *_rel_argv(argv)]), inputs, device="cpu")
    prov["status"] = (f"readout under {spec}; diagnostic, not an adjudication; "
                      "changes no registered rule or verdict")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"provenance": prov, **body}, fh, indent=1, default=float)
    print("wrote", path, flush=True)
