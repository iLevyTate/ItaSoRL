# Local Run Profiles + Cell-Level Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run any Colab notebook RUN_PROFILE locally on Windows with cell-level checkpoint/resume, so a 12+ hour n=10 B-v3 run survives interruption.

**Architecture:** `run_expB2.py` gains per-(drift, seed) cell checkpoint files with a config fingerprint and an explicit `--resume` flag; `run_e2e.py` forwards `--resume` to the expB2 step; a new `scripts/run_local.py` mirrors the notebook profile table and launches `run_e2e.py` with preflight CUDA/RAM checks.

**Tech Stack:** Python 3.10-3.12, argparse, pytest, ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-03-local-bv3-n10-runner-design.md`

**Branch:** `local-bv3-n10-runner` (stays local; no push, no PR).

---

## File map

- Modify: `scripts/run_expB2.py` - checkpoint helpers + `--resume` + canonical rebuild
- Modify: `scripts/run_e2e.py` - forward `--resume` to expB2 in `build_b2_extra()`
- Create: `scripts/run_local.py` - profile launcher (PROFILES, build_cmd, preflights)
- Create: `tests/test_expb2_resume.py` - checkpoint/resume tests
- Create: `tests/test_run_local.py` - profile mapping tests
- Modify: `notebooks/colab_gpu.ipynb` - keep-in-sync comment on `_PROFILES`
- Modify: `scripts/README.md` - short run_local.py entry

Conventions that matter here:
- Repo runs on Windows under Git Bash; tests must not assume POSIX-only APIs.
- `scripts/` is not a package. Scripts import repo code via `import _bootstrap`
  (in `scripts/`) which puts the repo root on `sys.path`. Tests that import a
  script must insert `ROOT/"scripts"` into `sys.path` first.
- CI runs `ruff check .`; f-strings without placeholders (F541) fail CI.
- `fullruns/` and `results/` must never be committed.

---

### Task 1: Checkpoint helper functions in run_expB2.py

**Files:**
- Modify: `scripts/run_expB2.py` (add imports near top; add helpers after `cfg()`)
- Create: `tests/test_expb2_resume.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_expb2_resume.py`:

```python
"""Cell-level checkpoint/resume for run_expB2.py (no training, no GPU)."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_expB2  # noqa: E402


BASE = dict(updates=300, n_eps=16, max_steps=80, hidden=96, ray_steps=5,
            shaping_coef=1.0, pool_n=110, pool_steps=24, mp_pairs=60,
            mp_prefix=20, mp_branch=24, basal_e=None, n_pellets=None,
            reach=None, dump_states=None, sysid_aux=False, sysid_coef=1.0,
            drift_mode="regime", drifts=[0.0, 0.45], device="cuda")


def make_cell(drift, seed):
    target = 0.55 + 0.03 * seed + 0.01 * drift
    pool = {"target": target, "target_lo": target - 0.07,
            "target_hi": target + 0.07, "target_var": 0.5,
            "target_full": float("nan"), "selectivity": 0.1,
            "selectivity_var": 0.0, "selectivity_full": 0.0, "speed": 0.9,
            "shuffled": 0.5, "anchor_energy": 0.8, "anchor_food": 0.8,
            "ceiling_drag": float("nan")}
    mp = {"target": 0.5, "leakage_clean": True, "leakage_max_dev": 0.0}
    return {"drift": drift, "seed": seed,
            "eng": {"engaged": True, "trained_return": 0.1},
            "xeval": {"0.00": 0.1, "0.45": 0.0},
            "agents": {g: {"pool": dict(pool), "mp": dict(mp)}
                       for g in run_expB2.AG}}


def test_fingerprint_stable_and_ignores_dump_states():
    fp1 = run_expB2.config_fingerprint(dict(BASE))
    fp2 = run_expB2.config_fingerprint(dict(BASE, dump_states="/somewhere/else"))
    assert fp1 == fp2


def test_fingerprint_changes_on_science_knob():
    fp1 = run_expB2.config_fingerprint(dict(BASE))
    for knob, value in [("hidden", 64), ("drift_mode", "ar1"),
                        ("updates", 60), ("device", "cpu"),
                        ("drifts", [0.0])]:
        fp2 = run_expB2.config_fingerprint(dict(BASE, **{knob: value}))
        assert fp1 != fp2, knob


def test_cell_roundtrip_preserves_nan(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    cell = make_cell(0.45, 3)
    run_expB2.write_cell_file(tmp_path, fp, "abc1234", cell)
    done = run_expB2.load_cell_files(tmp_path, fp)
    got = done[(0.45, 3)]
    assert got["seed"] == 3
    assert got["agents"]["survival"]["pool"]["target"] == pytest.approx(
        cell["agents"]["survival"]["pool"]["target"])
    assert math.isnan(got["agents"]["survival"]["pool"]["ceiling_drag"])


def test_load_rejects_fingerprint_mismatch(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    run_expB2.write_cell_file(tmp_path, fp, "abc1234", make_cell(0.0, 0))
    other = run_expB2.config_fingerprint(dict(BASE, hidden=64))
    with pytest.raises(SystemExit):
        run_expB2.load_cell_files(tmp_path, other)


def test_load_rejects_corrupt_file(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    (tmp_path / "cell_d0.00_s0.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_expB2.load_cell_files(tmp_path, fp)


def test_load_missing_dir_returns_empty(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    assert run_expB2.load_cell_files(tmp_path / "nope", fp) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_expb2_resume.py -v`
Expected: FAIL / ERROR with `AttributeError: module 'run_expB2' has no attribute 'config_fingerprint'` (import of `run_expB2` itself succeeds; it pulls in torch and matplotlib, which is expected and slow-ish on first import).

- [ ] **Step 3: Implement the helpers**

In `scripts/run_expB2.py`, extend the stdlib import block near the top (it currently has `argparse`, `json`, `os`):

```python
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
```

Add after the `cfg()` function (before `evaluate_agent`):

```python
def config_fingerprint(base: dict) -> str:
    """Hash of the science-relevant config. Cells from different configs never mix;
    dump_states is a path, not science, so it is excluded."""
    fp = {k: v for k, v in base.items() if k != "dump_states"}
    payload = json.dumps(fp, sort_keys=True, default=float)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cell_file(cells_dir, drift: float, seed: int) -> Path:
    return Path(cells_dir) / f"cell_d{drift:.2f}_s{seed}.json"


def git_commit_short() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_cell_file(cells_dir, fingerprint: str, commit: str, cell: dict) -> Path:
    """Atomic write: a killed process never leaves a half-written checkpoint."""
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    path = cell_file(cells_dir, cell["drift"], cell["seed"])
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"fingerprint": fingerprint, "git_commit": commit,
                   "cell": cell}, f, indent=2, default=float)
    os.replace(tmp, path)
    return path


def load_cell_files(cells_dir, fingerprint: str) -> dict:
    """Load checkpointed cells keyed by (drift, seed). Hard error on corrupt
    files or fingerprint mismatch; warning only on git commit drift."""
    done: dict[tuple[float, int], dict] = {}
    cells_dir = Path(cells_dir)
    if not cells_dir.is_dir():
        return done
    commit = git_commit_short()
    for path in sorted(cells_dir.glob("cell_d*_s*.json")):
        try:
            with open(path) as f:
                payload = json.load(f)
            fp, cell = payload["fingerprint"], payload["cell"]
        except Exception as exc:
            raise SystemExit(
                f"Corrupt checkpoint {path}: {exc}. "
                "Delete this one file and rerun with --resume.")
        if fp != fingerprint:
            raise SystemExit(
                f"Checkpoint {path} has fingerprint {fp}, current config is "
                f"{fingerprint}. It belongs to a different experiment config: "
                "use a fresh --out-dir, or delete the stale cells/ directory.")
        if payload.get("git_commit", "unknown") != commit:
            print(f"  WARNING: {path.name} was produced at commit "
                  f"{payload.get('git_commit')} (now {commit})", flush=True)
        done[(float(cell["drift"]), int(cell["seed"]))] = cell
    return done
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_expb2_resume.py -v`
Expected: 6 passed.

- [ ] **Step 5: Ruff and commit**

```bash
ruff check scripts/run_expB2.py tests/test_expb2_resume.py
git add scripts/run_expB2.py tests/test_expb2_resume.py
git commit -m "feat(expB2): cell checkpoint files with config fingerprint"
```

---

### Task 2: Wire resume into run_expB2.main()

**Files:**
- Modify: `scripts/run_expB2.py` (`cfg()` and `main()`)
- Modify: `tests/test_expb2_resume.py` (append integration tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expb2_resume.py`:

```python
def _run_main(tmp_path, monkeypatch, extra=(), cell_fn=None):
    """Run run_expB2.main() with a stubbed run_cell (no training, seconds not hours).
    --quick gives drifts [0.0, 0.45] x seeds [0, 1] = 4 cells."""
    monkeypatch.setattr(run_expB2, "run_cell",
                        cell_fn or (lambda t: make_cell(t["drift"], t["seed"])))
    argv = ["run_expB2.py", "--quick", "--out-dir", str(tmp_path), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    run_expB2.main()


def test_fresh_run_writes_cell_files_and_results(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    cells = sorted(p.name for p in (tmp_path / "cells").glob("*.json"))
    assert cells == ["cell_d0.00_s0.json", "cell_d0.00_s1.json",
                     "cell_d0.45_s0.json", "cell_d0.45_s1.json"]
    assert (tmp_path / "expB2_results.json").is_file()


def test_fresh_run_refuses_stale_cells(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch)  # no --resume


def test_resume_skips_completed_cells(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)

    def boom(task):
        raise AssertionError("run_cell must not be called on full resume")

    _run_main(tmp_path, monkeypatch, extra=["--resume"], cell_fn=boom)


def test_resume_runs_only_missing_cells(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    (tmp_path / "cells" / "cell_d0.45_s1.json").unlink()
    ran = []

    def spy(task):
        ran.append((task["drift"], task["seed"]))
        return make_cell(task["drift"], task["seed"])

    _run_main(tmp_path, monkeypatch, extra=["--resume"], cell_fn=spy)
    assert ran == [(0.45, 1)]


def test_results_are_seed_ordered_after_resume(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    # remove seed 0 so on resume seed 0 completes AFTER resumed seed 1
    (tmp_path / "cells" / "cell_d0.45_s0.json").unlink()
    _run_main(tmp_path, monkeypatch, extra=["--resume"])
    res = json.loads((tmp_path / "expB2_results.json").read_text())
    got = res["0.45"]["survival"]["pool_target"]
    want = [make_cell(0.45, s)["agents"]["survival"]["pool"]["target"]
            for s in (0, 1)]
    assert got == pytest.approx(want)


def test_resume_rejects_different_config(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch,
                  extra=["--resume", "--shaping_coef", "2.0"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_expb2_resume.py -v`
Expected: the 6 Task-1 tests pass; the new tests FAIL (`unrecognized arguments: --resume` or missing cells dir).

- [ ] **Step 3: Implement main() wiring**

In `scripts/run_expB2.py`:

3a. In `cfg()`, after the `--device` argument, add:

```python
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: load matching cell "
                         "checkpoints from <out-dir>/cells and run only the "
                         "missing (drift, seed) cells")
```

3b. In `main()`, replace the block that builds the initial `res`/`eng_log`
(the big dict comprehension) with a helper so the rebuild reuses it. Add just
above `main()`:

```python
def fresh_results(drifts) -> dict:
    return {d: {g: {"pool_target": [], "pool_target_lo": [], "pool_target_hi": [],
                    "pool_target_var": [], "pool_target_full": [],
                    "pool_selectivity": [], "pool_selectivity_var": [],
                    "pool_selectivity_full": [],
                    "pool_speed": [], "pool_shuffled": [],
                    "pool_anchor_energy": [], "pool_anchor_food": [],
                    "pool_ceiling_drag": [],
                    "mp_target": [], "mp_leak_clean": [], "xeval_return": []}
                for g in AG} for d in drifts}
```

and in `main()` replace the `res = {d: {g: {...}}}` literal with:

```python
    res = fresh_results(a.drifts)
```

3c. Still in `main()`, directly after `tasks = [{**base, "drift": d, "seed": s} ...]`
and before the `done = 0` line, insert:

```python
    cells_dir = Path(a.out_dir) / "cells"
    fingerprint = config_fingerprint(base)
    commit = git_commit_short()
    if a.resume:
        resumed = load_cell_files(cells_dir, fingerprint)
    else:
        resumed = {}
        if cells_dir.is_dir() and any(cells_dir.glob("cell_d*_s*.json")):
            raise SystemExit(
                f"{cells_dir} already contains checkpointed cells. Pass "
                "--resume to continue that run, or use a fresh --out-dir.")
    all_cells = dict(resumed)
    for (d, s) in sorted(resumed):
        print(f"resumed from checkpoint: drift={d:.2f} seed={s}", flush=True)
        record_cell(res, eng_log, resumed[(d, s)])
    if resumed:
        print(f"Resume: {len(resumed)} cell(s) loaded from {cells_dir}, "
              f"{len(tasks) - len(resumed)} to run.", flush=True)
    tasks = [t for t in tasks if (t["drift"], t["seed"]) not in resumed]
```

3d. In both run loops (workers>1 and serial), after `record_cell(res, eng_log, cell)`
and before `checkpoint()`, add the same two lines:

```python
                all_cells[(cell["drift"], cell["seed"])] = cell
                write_cell_file(cells_dir, fingerprint, commit, cell)
```

(indentation: 16 spaces in the pool loop, 12 in the serial loop.)

3e. After both loops complete (just before the `# ---- summary table ----`
comment), rebuild in canonical order so per-seed list positions are
deterministic regardless of `imap_unordered` completion order or resume
interleaving:

```python
    # Canonical rebuild: list positions ordered by (drift, seed), independent of
    # completion order (imap_unordered) and of resume interleaving.
    res = fresh_results(a.drifts)
    eng_log = {d: [] for d in a.drifts}
    for key in sorted(all_cells):
        record_cell(res, eng_log, all_cells[key])
    checkpoint()
```

Note: `checkpoint()` is a closure over the name `res` in `main()`, so it picks
up the rebound dict; no change needed there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_expb2_resume.py -v`
Expected: 12 passed. (The main() tests each take a few seconds: summary stats
and the matplotlib figure still run on the stub data.)

- [ ] **Step 5: Run the existing suite to catch regressions**

Run: `python -m pytest -q`
Expected: all pass (existing tests do not invoke run_expB2.main()).

- [ ] **Step 6: Ruff and commit**

```bash
ruff check scripts/run_expB2.py tests/test_expb2_resume.py
git add scripts/run_expB2.py tests/test_expb2_resume.py
git commit -m "feat(expB2): --resume with fingerprint-gated cell checkpoints and canonical result order"
```

---

### Task 3: Forward --resume through run_e2e.py

**Files:**
- Modify: `scripts/run_e2e.py` (`build_b2_extra()` and its call in `main()`)
- Modify: `tests/test_expb2_resume.py` (append plumbing test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expb2_resume.py`:

```python
import run_e2e  # noqa: E402  (scripts/ already on sys.path from the top of this file)


def _e2e_args(**over):
    defaults = dict(b2_seeds=None, b2_updates=None, b2_hidden=None,
                    b2_dump_states=None, b2_sysid_aux=False, b2_drift_mode=None)
    defaults.update(over)
    import argparse
    return argparse.Namespace(**defaults)


def test_build_b2_extra_emits_resume_only_in_resume_mode():
    assert "--resume" not in run_e2e.build_b2_extra(_e2e_args())
    assert "--resume" in run_e2e.build_b2_extra(_e2e_args(), resume=True)


def test_build_b2_extra_keeps_other_flags_with_resume():
    extra = run_e2e.build_b2_extra(
        _e2e_args(b2_drift_mode="regime", b2_seeds=[0, 1]), resume=True)
    assert extra[-1] == "--resume"
    assert ["--drift-mode", "regime"] == extra[extra.index("--drift-mode"):
                                              extra.index("--drift-mode") + 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_expb2_resume.py -k build_b2_extra -v`
Expected: FAIL with `TypeError: build_b2_extra() got an unexpected keyword argument 'resume'`.

- [ ] **Step 3: Implement**

In `scripts/run_e2e.py`, change the signature and tail of `build_b2_extra`:

```python
def build_b2_extra(args: argparse.Namespace, *, resume: bool = False) -> list[str]:
```

and at the end of the function, before `return extra`:

```python
    if resume:
        extra += ["--resume"]
```

In `main()`, change the call site:

```python
        b2_extra = build_b2_extra(args, resume=resume_dir is not None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_expb2_resume.py -v`
Expected: 14 passed.

- [ ] **Step 5: Ruff and commit**

```bash
ruff check scripts/run_e2e.py tests/test_expb2_resume.py
git add scripts/run_e2e.py tests/test_expb2_resume.py
git commit -m "feat(e2e): forward --resume to the expB2 step"
```

---

### Task 4: run_local.py profile table and command builder

**Files:**
- Create: `scripts/run_local.py`
- Create: `tests/test_run_local.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_local.py`:

```python
"""Profile-to-argv mapping for the local launcher (pure functions, no GPU)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_local  # noqa: E402

RUN_DIR = Path("fullruns") / "01011999"

NOTEBOOK_PROFILES = {"quick", "full", "bv3_regime", "bv3_regime_n10",
                     "bv2_ceiling", "bv3_ceiling", "b2_only", "b2_seed0",
                     "experiments_no_b2"}


def test_profiles_match_notebook_table():
    assert set(run_local.PROFILES) == NOTEBOOK_PROFILES


@pytest.mark.parametrize("name", sorted(NOTEBOOK_PROFILES))
def test_every_profile_builds_a_command(name):
    cmd = run_local.build_cmd(run_local.PROFILES[name], RUN_DIR, resume=False)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("run_e2e.py")
    assert "--results-dir" in cmd and "--resume" not in cmd


def test_bv3_regime_n10_flags():
    cmd = run_local.build_cmd(run_local.PROFILES["bv3_regime_n10"], RUN_DIR,
                              resume=False)
    i = cmd.index("--b2-seeds")
    assert cmd[i + 1:i + 11] == [str(s) for s in range(10)]
    assert cmd[cmd.index("--b2-drift-mode") + 1] == "regime"
    assert cmd[cmd.index("--b2-updates") + 1] == "300"
    assert cmd[cmd.index("--b2-dump-states") + 1] == str(RUN_DIR / "states")
    assert cmd[cmd.index("--only") + 1] == "expb2"


def test_resume_swaps_results_dir_for_resume():
    cmd = run_local.build_cmd(run_local.PROFILES["b2_only"], RUN_DIR, resume=True)
    assert cmd[cmd.index("--resume") + 1] == str(RUN_DIR)
    assert "--results-dir" not in cmd


def test_quick_profile_uses_quick_flag():
    cmd = run_local.build_cmd(run_local.PROFILES["quick"], RUN_DIR, resume=False)
    assert "--quick" in cmd and "--only" not in cmd


def test_experiments_no_b2_skips_and_dumps_nothing():
    cmd = run_local.build_cmd(run_local.PROFILES["experiments_no_b2"], RUN_DIR,
                              resume=False)
    assert cmd[cmd.index("--skip") + 1] == "expB2"
    assert "--b2-dump-states" not in cmd


def test_ceiling_profiles_set_sysid_aux():
    for name in ("bv2_ceiling", "bv3_ceiling"):
        cmd = run_local.build_cmd(run_local.PROFILES[name], RUN_DIR, resume=False)
        assert "--b2-sysid-aux" in cmd, name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_local.py -v`
Expected: collection ERROR with `ModuleNotFoundError: No module named 'run_local'`.

- [ ] **Step 3: Implement scripts/run_local.py (table + builder only)**

Create `scripts/run_local.py`:

```python
"""
ITASORL - run any Colab notebook profile locally, with resume.

Mirrors the RUN_PROFILE presets in notebooks/colab_gpu.ipynb and launches
scripts/run_e2e.py with the mapped flags plus local preflight checks
(CUDA visible, enough free RAM). expB2 cells checkpoint after every
(drift, seed) pair, so an interrupted run continues with --resume losing at
most one cell.

Usage (from repo root, Git Bash or any shell):
    python scripts/run_local.py --list
    python scripts/run_local.py bv3_regime_n10            # fresh start
    python scripts/run_local.py bv3_regime_n10 --resume   # continue latest run
    python scripts/run_local.py quick --allow-cpu         # smoke test w/o GPU
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import subprocess
import sys
from pathlib import Path

from itasorl.results_io import default_run_dir, read_latest_run_dir  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent

# Keep in sync with _PROFILES in notebooks/colab_gpu.ipynb. The notebook cannot
# import repo code before it clones the repo, so the table is duplicated there.
PROFILES = {
    "quick":             dict(run_mode="quick", only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),
    "full":              dict(run_mode="full",  only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),
    "bv3_regime":        dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=False, dump_states=True),
    "bv3_regime_n10":    dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=list(range(10)), b2_updates=300, drift_mode="regime", sysid_aux=False, dump_states=True),
    "bv2_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=True,  dump_states=True),
    "bv3_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=True,  dump_states=True),
    "b2_only":           dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),
    "b2_seed0":          dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=[0],  b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),
    "experiments_no_b2": dict(run_mode="full",  only=None,    skip_steps=["expB2"], b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=False),
}


def build_cmd(profile: dict, run_dir: Path, *, resume: bool) -> list[str]:
    """Map one PROFILES entry onto a run_e2e.py argv (pure, unit-tested)."""
    cmd = [sys.executable, str(SCRIPTS / "run_e2e.py")]
    if resume:
        cmd += ["--resume", str(run_dir)]
    else:
        cmd += ["--results-dir", str(run_dir)]
    if profile["run_mode"] == "quick":
        cmd += ["--quick"]
    if profile["only"]:
        cmd += ["--only", profile["only"]]
    for step in profile["skip_steps"]:
        cmd += ["--skip", step]
    if profile["b2_seeds"] is not None:
        cmd += ["--b2-seeds", *[str(s) for s in profile["b2_seeds"]]]
    if profile["b2_updates"] is not None:
        cmd += ["--b2-updates", str(profile["b2_updates"])]
    if profile["drift_mode"]:
        cmd += ["--b2-drift-mode", profile["drift_mode"]]
    if profile["sysid_aux"]:
        cmd += ["--b2-sysid-aux"]
    if profile["dump_states"]:
        cmd += ["--b2-dump-states", str(Path(run_dir) / "states")]
    return cmd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_local.py -v`
Expected: 15 passed (1 + 9 parametrized + 5).

- [ ] **Step 5: Ruff and commit**

```bash
ruff check scripts/run_local.py tests/test_run_local.py
git add scripts/run_local.py tests/test_run_local.py
git commit -m "feat(local): profile table and run_e2e command builder for local runs"
```

---

### Task 5: run_local.py preflights and entry point

**Files:**
- Modify: `scripts/run_local.py` (append below `build_cmd`)
- Modify: `tests/test_run_local.py` (append CLI tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_local.py`:

```python
def test_list_prints_every_profile(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "--list"])
    run_local.main()
    out = capsys.readouterr().out
    for name in NOTEBOOK_PROFILES:
        assert name in out


def test_profile_required_without_list(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_local.py"])
    with pytest.raises(SystemExit):
        run_local.main()


def test_unknown_profile_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "not_a_profile"])
    with pytest.raises(SystemExit):
        run_local.main()


def test_main_launches_mapped_command(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(run_local, "check_cuda", lambda allow_cpu: None)
    monkeypatch.setattr(run_local, "check_ram", lambda min_free_gb: None)
    monkeypatch.setattr(run_local, "default_run_dir", lambda: tmp_path / "run")

    class Ret:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return Ret()

    monkeypatch.setattr(run_local.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_seed0"])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert exc.value.code == 0
    cmd = calls["cmd"]
    assert cmd[cmd.index("--b2-seeds") + 1] == "0"


def test_main_resume_requires_latest_pointer(monkeypatch):
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_only", "--resume"])
    with pytest.raises(SystemExit):
        run_local.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_local.py -k "list or required or unknown or launches or pointer" -v`
Expected: FAIL with `AttributeError: module 'run_local' has no attribute 'main'`.

- [ ] **Step 3: Implement preflights + main()**

Append to `scripts/run_local.py`:

```python
def check_cuda(allow_cpu: bool) -> None:
    import torch

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return
    if allow_cpu:
        print("WARNING: no CUDA GPU visible to torch; continuing on CPU "
              "(--allow-cpu).", flush=True)
        return
    raise SystemExit("No CUDA GPU visible to torch (the 07032026 Colab run "
                     "silently landed on CPU this way). Fix the driver/torch "
                     "install, or pass --allow-cpu for a deliberate CPU run.")


def free_ram_gb() -> float | None:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "[int]((Get-CimInstance Win32_OperatingSystem)"
                 ".FreePhysicalMemory/1024)"],
                capture_output=True, text=True, check=True).stdout
            return int("".join(ch for ch in out if ch.isdigit())) / 1024.0
        except Exception:
            return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return None


def check_ram(min_free_gb: float) -> None:
    free = free_ram_gb()
    if free is None:
        print("WARNING: could not determine free RAM; continuing.", flush=True)
        return
    if free < min_free_gb:
        raise SystemExit(
            f"Only {free:.1f} GB system RAM free (< {min_free_gb:.0f} GB "
            "needed; episode buffers live in RAM). Free memory first, e.g. "
            "stop the ralph loop.")
    print(f"Free RAM: {free:.1f} GB -- OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a Colab RUN_PROFILE locally with resume support.")
    ap.add_argument("profile", nargs="?", choices=sorted(PROFILES),
                    help="run profile (see --list)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the latest interrupted run")
    ap.add_argument("--list", action="store_true",
                    help="list profiles and exit")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run even without a CUDA GPU")
    ap.add_argument("--min-free-gb", type=float, default=4.0,
                    help="free system RAM required to start (default 4)")
    a = ap.parse_args()

    if a.list:
        for name in PROFILES:
            p = PROFILES[name]
            bits = [p["run_mode"]]
            if p["only"]:
                bits.append(f"only={p['only']}")
            if p["b2_seeds"] is not None:
                bits.append(f"seeds={len(p['b2_seeds'])}")
            if p["drift_mode"]:
                bits.append(f"drift_mode={p['drift_mode']}")
            if p["sysid_aux"]:
                bits.append("sysid_aux")
            if p["skip_steps"]:
                bits.append(f"skip={','.join(p['skip_steps'])}")
            print(f"{name:18s} {' '.join(bits)}")
        return
    if not a.profile:
        ap.error("profile is required (or use --list)")

    if a.resume:
        run_dir = read_latest_run_dir()
        if run_dir is None:
            raise SystemExit("--resume: no latest-run pointer found; "
                             "start a fresh run first.")
    else:
        run_dir = default_run_dir()

    check_cuda(a.allow_cpu)
    check_ram(a.min_free_gb)
    cmd = build_cmd(PROFILES[a.profile], Path(run_dir), resume=a.resume)
    print("Launching: " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
```

Implementation note: before relying on `default_run_dir()` /
`read_latest_run_dir()`, open `itasorl/results_io.py` and read lines 31-60 to
confirm their exact behavior (default dir is `fullruns/<MMDDYYYY>` or
`fullruns/<MMDDYYYY_HHMMSS>` when the date folder is taken; the pointer file is
`results/LATEST_RUN.txt`). Adjust nothing unless the signatures differ from
this plan; if they differ, adapt the two call sites, not results_io.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_local.py -v`
Expected: 20 passed.

- [ ] **Step 5: Manual smoke of the CLI surface**

```bash
python scripts/run_local.py --list
python scripts/run_local.py 2>&1 | head -3
```
Expected: first prints 9 profile lines; second errors with "profile is required (or use --list)" and nonzero exit.

- [ ] **Step 6: Ruff and commit**

```bash
ruff check scripts/run_local.py tests/test_run_local.py
git add scripts/run_local.py tests/test_run_local.py
git commit -m "feat(local): CLI entry point with CUDA/RAM preflights and --resume"
```

---

### Task 6: Keep-in-sync breadcrumbs (notebook + scripts README)

**Files:**
- Modify: `notebooks/colab_gpu.ipynb` (config cell comment)
- Modify: `scripts/README.md` (new entry)

- [ ] **Step 1: Add the sync comment to the notebook config cell**

The notebook JSON must stay valid; edit it with a script, not by hand:

```bash
python - <<'EOF'
import json

path = "notebooks/colab_gpu.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)
target = "# only=\"expb2\" runs B-v2/B-v3 alone"
hit = False
for cell in nb["cells"]:
    for i, line in enumerate(cell.get("source", [])):
        if target in line:
            cell["source"].insert(
                i, "# Keep _PROFILES in sync with PROFILES in scripts/run_local.py.\n")
            hit = True
            break
    if hit:
        break
assert hit, "anchor line not found; inspect the config cell manually"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("ok")
EOF
```

Then verify the notebook still parses and shows the comment:

```bash
python -c "import json; nb=json.load(open('notebooks/colab_gpu.ipynb', encoding='utf-8')); print([l for c in nb['cells'] for l in c.get('source', []) if 'run_local' in l])"
```
Expected: the inserted comment line, once.

- [ ] **Step 2: Add run_local.py to scripts/README.md**

Read `scripts/README.md` first to match its format, then add an entry in the
same style as the neighboring script descriptions, with this content:

```markdown
- `run_local.py` - run any Colab notebook RUN_PROFILE locally with resume:
  `python scripts/run_local.py bv3_regime_n10`, then after any interruption
  `python scripts/run_local.py bv3_regime_n10 --resume`. Preflights: CUDA
  visible (override with `--allow-cpu`) and >= 4 GB free RAM. expB2 progress
  checkpoints per (drift, seed) cell under `<run>/artifacts/cells/`.
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_gpu.ipynb scripts/README.md
git commit -m "docs: keep-in-sync breadcrumbs for local profile table"
```

---

### Task 7: Full verification

- [ ] **Step 1: Full test suite**

Run: `python -m pytest -q`
Expected: all tests pass, including the two new files.

- [ ] **Step 2: Ruff over the whole repo**

Run: `ruff check .`
Expected: no findings (watch for F541).

- [ ] **Step 3: End-to-end interruption drill (cheap, stubbed already covered; this is the real path at quick scale)**

This is the one step that exercises the real `run_e2e.py -> run_expB2.py`
chain. It trains at `--quick` scale (roughly 10-20 min on GPU, longer on CPU;
ask the user before running it on their machine, or hand them the commands):

```bash
python scripts/run_local.py quick --allow-cpu
# Ctrl+C partway through expB2, then:
python scripts/run_local.py quick --allow-cpu --resume
```
Expected: the resumed invocation prints `resumed from checkpoint: drift=...`
lines for completed cells and finishes; `fullruns/<dir>/SUMMARY.md` reports
expB2 ok; `fullruns/<dir>/artifacts/cells/` contains 4 cell files.

- [ ] **Step 4: Confirm nothing from fullruns/ or results/ is staged**

Run: `git status --short`
Expected: clean (fullruns/ is untracked user data; results/ is gitignored).

- [ ] **Step 5: Final commit if anything is outstanding**

```bash
git log --oneline origin/main..HEAD
```
Expected: the five commits from Tasks 1-6. Work stays on
`local-bv3-n10-runner`; do NOT push and do NOT open a PR (user decision
pending).
