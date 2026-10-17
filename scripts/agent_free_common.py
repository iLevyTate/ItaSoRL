"""Shared provenance and JSON helpers for the agent-free analyses of 2026-10-11.

Each analysis script (gate-0 pair intervals, residual-field correlation, the historical
random-forest probe, the off-design run promotions, the Experiment C selection intensity)
records the commit, the software stack and the thread count it ran on, and writes its JSON
with numeric lists kept on one line so a committed artifact stays readable.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os
import platform
import sys

from itasorl.results_io import git_head


def stack_record() -> dict:
    """Python, numpy, scipy, scikit-learn and torch versions, the OS and the thread settings."""
    import numpy
    out = {"python": sys.version.split()[0], "numpy": numpy.__version__,
           "platform": platform.platform(), "machine": platform.machine(),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
           "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}
    try:
        import scipy
        out["scipy"] = scipy.__version__
    except ImportError:
        out["scipy"] = None
    try:
        import sklearn
        out["sklearn"] = sklearn.__version__
    except ImportError:
        out["sklearn"] = None
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_threads"] = torch.get_num_threads()
    except ImportError:
        out["torch"] = None
    return out


def pin_one_thread() -> None:
    """One BLAS and torch thread (the machine limit for these analyses, and the way the run
    workers fit the fingerprints)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def tracked_changes(exclude: tuple[str, ...] = ()) -> list[str] | None:
    """Tracked files that differ from HEAD (None outside a git checkout), leaving out the
    paths in `exclude` (the artifact being written, which a multi-call script updates in
    place). Empty means the artifact was generated from exactly the recorded commit."""
    import subprocess
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    skip = {e.replace("\\", "/") for e in exclude}
    return [line[3:] for line in out.splitlines() if line.strip() and line[3:] not in skip]


def provenance(generated_by: str, commit_key: str = "git_commit",
               output: str | None = None, device: str | None = None) -> dict:
    """Generator, commit, tracked changes and stack. An analysis that IS the run (the
    agent-free computations) records `git_commit`; a promotion or readout of an earlier run
    records `git_commit_at_promotion`, because scripts/build_results_manifest.py reads
    `git_commit` as the commit a run executed at. `device`, when given, goes into the stack
    block as the torch device the computation ran on."""
    stack = stack_record()
    if device is not None:
        stack["device"] = str(device)
    return {"generated_by": generated_by, commit_key: git_head(),
            "tracked_changes_at_generation": tracked_changes((output,) if output else ()),
            "stack": stack}


NL = chr(10)


def _scalar(x) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def _enc(o, depth: int) -> str:
    pad, pad1 = " " * depth, " " * (depth + 1)
    if isinstance(o, dict):
        if not o:
            return "{}"
        items = [f"{pad1}{json.dumps(str(k))}: {_enc(v, depth + 1)}" for k, v in o.items()]
        return "{" + NL + ("," + NL).join(items) + NL + pad + "}"
    if isinstance(o, (list, tuple)):
        if all(_scalar(x) for x in o):
            return "[" + ", ".join(json.dumps(x) for x in o) + "]"
        items = [pad1 + _enc(x, depth + 1) for x in o]
        return "[" + NL + ("," + NL).join(items) + NL + pad + "]"
    return json.dumps(o)


def dumps_compact(obj) -> str:
    """Indented JSON with every list of scalars kept on one line."""
    return _enc(obj, 0)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = dumps_compact(obj)
    if json.dumps(json.loads(text)) != json.dumps(json.loads(json.dumps(obj))):
        raise RuntimeError("compact JSON writer changed the content")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
