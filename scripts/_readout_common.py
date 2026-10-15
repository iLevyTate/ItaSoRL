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


def run_tasks(fn, tasks: list, ckpt_dir: str, key, workers: int = 1) -> list:
    """Run `fn` on every task whose checkpoint is missing; return every result, loaded
    checkpoints included. key(task) names the checkpoint file."""
    os.makedirs(ckpt_dir, exist_ok=True)
    done, todo = [], []
    for t in tasks:
        p = os.path.join(ckpt_dir, key(t) + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                done.append(json.load(fh))
        else:
            todo.append(t)
    if done:
        print(f"resumed {len(done)} task(s); {len(todo)} to go", flush=True)

    def save(r, t):
        p = os.path.join(ckpt_dir, key(t) + ".json")
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


def _rel_argv(argv: list) -> list:
    """The recorded argv with every absolute path made repo-relative, including the value of
    a `name=path` pair (the development script's --run arguments)."""
    from promote_bv3_gates import _ABS_PATH, rel_flags, repo_rel
    out = []
    for f in rel_flags(argv):
        name, eq, value = f.partition("=") if isinstance(f, str) else ("", "", "")
        out.append(f"{name}={repo_rel(value)}" if eq and _ABS_PATH.match(value) else f)
    return out


def write_output(path: str, script: str, argv: list, inputs: dict, spec: str, body: dict) -> None:
    from itasorl.optional_readouts import provenance
    from promote_bv3_gates import repo_rel
    inputs = {k: repo_rel(v) if k.endswith("_dir") else v for k, v in inputs.items()}
    prov = provenance(script, "python " + " ".join([script, *_rel_argv(argv)]), inputs, device="cpu")
    prov["status"] = (f"readout under {spec}; diagnostic, not an adjudication; "
                      "changes no registered rule or verdict")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"provenance": prov, **body}, fh, indent=1, default=float)
    print("wrote", path, flush=True)
