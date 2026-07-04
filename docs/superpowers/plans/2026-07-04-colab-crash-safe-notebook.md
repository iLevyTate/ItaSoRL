# Crash-Safe Colab Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `notebooks/colab_gpu.ipynb` a pick-profile-and-Run-all notebook that forces a Drive copy, auto-resumes unfinished runs, and never loses more than ~5 minutes of checkpoints to a Colab disconnect.

**Architecture:** Core durability lives in `itasorl/results_io.py` (fault-tolerant mirror plus timed incremental sync of `artifacts/`). `scripts/run_e2e.py` gains an `auto` sentinel for `--b2-dump-states` so state dumps live inside the run dir and survive resume. The notebook adds a force-copy guard, a Colab form config, auto-resume from Drive, and a watchdog thread with a Drive API fallback.

**Tech Stack:** Python stdlib (shutil, zipfile, threading), pytest, Colab forms (`# @param`), Google Drive FUSE + Drive v3 API (`googleapiclient`, preinstalled on Colab).

**Spec:** `docs/superpowers/specs/2026-07-04-colab-crash-safe-notebook-design.md`

**Conventions:** No en/em dashes in any prose or docs (ASCII only). Run `ruff check .` before every commit; CI runs it on Python 3.10-3.12 and F541 (f-string without placeholders) has failed CI before. The notebook is JSON; after every notebook edit, validate with `python -m json.tool notebooks/colab_gpu.ipynb > /dev/null`.

---

## File Structure

- `itasorl/results_io.py` (modify): fault-tolerant `_sync_mirror`, new `_sync_ckpt_mirror`, delete redundant copy block in `finalize`.
- `scripts/run_e2e.py` (modify): `resolve_dump_states()` sentinel resolution, help text.
- `notebooks/colab_gpu.ipynb` (modify): header, new guard cell, form config cell, mount cell auth, run cell rewrite with auto-resume and watchdog, delete resume-only cells, reanalysis path, tail markdown.
- `tests/test_results_io_mirror.py` (create): mirror fault tolerance and incremental sync.
- `tests/test_run_e2e_flags.py` (create): `auto` sentinel resolution and record/replay.
- `tests/test_run_local.py` (modify): parse the notebook for profile table and dropdown instead of a hardcoded set.
- `results/README.md`, `scripts/README.md` (modify): mirror and resume docs.

---

### Task 1: Fault-tolerant mirror in results_io.py

The Drive FUSE mount can die mid-run ("transport endpoint is not connected"). Today any `shutil.copy2` in `_sync_mirror` would raise inside the recorder and kill the run. Wrap all mirror I/O so it degrades loudly and retries instead.

**Files:**
- Modify: `itasorl/results_io.py` (`_sync_mirror` at ~line 385, `RunRecorder` fields at ~line 258, `finalize` at ~line 491)
- Create: `tests/test_results_io_mirror.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_results_io_mirror.py`:

```python
"""Mirror fault tolerance and incremental checkpoint sync (no GPU, no Colab)."""

import shutil
from pathlib import Path

import pytest

from itasorl import results_io
from itasorl.results_io import RunRecorder


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "LATEST_RUN_PTR", tmp_path / "LATEST_RUN.txt")
    monkeypatch.setenv("ITASORL_DRIVE_SYNC", str(tmp_path / "mirror"))
    return RunRecorder.create(quick=True, out_dir=tmp_path / "run")


def _break_mirror(tmp_path: Path) -> Path:
    """Replace the mirror directory with a plain file so any write raises OSError."""
    mirror = tmp_path / "mirror"
    shutil.rmtree(mirror)
    mirror.write_text("not a directory", encoding="utf-8")
    return mirror


def test_mirror_failure_never_raises(recorder, tmp_path):
    _break_mirror(tmp_path)
    recorder._write_status(current_step="s", step_status="running", force=True)


def test_mirror_degraded_warns_once_then_recovers(recorder, tmp_path, capsys):
    mirror = _break_mirror(tmp_path)
    recorder._write_status(current_step="s", step_status="running", force=True)
    recorder._write_status(current_step="s", step_status="running", force=True)
    out = capsys.readouterr().out
    assert out.count("Drive mirror unreachable") == 1
    mirror.unlink()
    recorder._write_status(current_step="s", step_status="running", force=True)
    assert "Drive mirror recovered" in capsys.readouterr().out
```

Note: `_write_status(force=True)` bypasses the 2 s status throttle, so each call reaches the mirror.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_results_io_mirror.py -v`
Expected: both tests FAIL with `OSError` / `NotADirectoryError` (or `FileExistsError` on Windows) raised out of `_write_status`, because `_sync_mirror` has no error handling yet.

- [ ] **Step 3: Implement the fault-tolerant wrapper**

In `itasorl/results_io.py`, add one field to `RunRecorder` (after `_mirror_dir` at ~line 265):

```python
    _mirror_degraded: bool = field(default=False, repr=False)
```

Replace the existing `_sync_mirror` method (~lines 385-398) with:

```python
    def _sync_mirror(self, *, full: bool = False) -> None:
        if self._mirror_dir is None:
            return
        self._mirror_attempt(lambda: self._sync_mirror_files(full=full))

    def _mirror_attempt(self, fn) -> None:
        """Mirror I/O must never kill the run: degrade loudly, retry next call."""
        try:
            fn()
        except OSError as exc:
            if not self._mirror_degraded:
                print(
                    f"\nWARNING: Drive mirror unreachable ({exc}); the run "
                    "continues on local disk and mirroring keeps retrying.",
                    flush=True,
                )
            self._mirror_degraded = True
            return
        if self._mirror_degraded:
            print("\nDrive mirror recovered; syncing resumed.", flush=True)
        self._mirror_degraded = False

    def _sync_mirror_files(self, *, full: bool) -> None:
        dest_root = self._mirror_dir / self.run_dir.name
        dest_root.mkdir(parents=True, exist_ok=True)
        for name in ("combined.log", "status.json", "manifest.json", "SUMMARY.md", "bundle.zip"):
            src = self.run_dir / name
            if src.is_file():
                shutil.copy2(src, dest_root / name)
        if full:
            for sub in ("steps", "artifacts"):
                src_dir = self.run_dir / sub
                if src_dir.is_dir():
                    shutil.copytree(src_dir, dest_root / sub, dirs_exist_ok=True)
```

In `finalize` (~lines 511-518), the block after `self._sync_mirror(full=True)` that copies `SUMMARY.md` and `bundle.zip` directly is redundant (both files exist before that call and are already in the `_sync_mirror_files` small-file list) and is also unguarded. Delete these lines entirely:

```python
        if self._mirror_dir is not None:
            dest_root = self._mirror_dir / self.run_dir.name
            dest_root.mkdir(parents=True, exist_ok=True)
            for name in ("SUMMARY.md", "bundle.zip"):
                src = self.run_dir / name
                if src.is_file():
                    shutil.copy2(src, dest_root / name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_results_io_mirror.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite and ruff**

Run: `python -m pytest -q && ruff check .`
Expected: all tests pass, no ruff findings.

- [ ] **Step 6: Commit**

```bash
git add itasorl/results_io.py tests/test_results_io_mirror.py
git commit -m "fix(results): Drive mirror failures degrade loudly instead of killing the run"
```

---

### Task 2: Incremental checkpoint sync of artifacts/

expB2 checkpoint cells and dumped states live under `<run_dir>/artifacts/` but are only mirrored when a step finishes. For expb2-only profiles the single step is the whole run, so a disconnect loses everything. Add a timed incremental sync.

**Files:**
- Modify: `itasorl/results_io.py` (`RunRecorder` fields, `_write_status` at ~line 359)
- Modify: `tests/test_results_io_mirror.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_results_io_mirror.py`:

```python
def _add_cell_file(recorder, name: str = "cell_d0.00_s0.json") -> Path:
    cells = recorder.run_dir / "artifacts" / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    path = cells / name
    path.write_text("{}", encoding="utf-8")
    return path


def test_ckpt_sync_copies_new_artifact_files(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "0")
    _add_cell_file(recorder)
    recorder._sync_ckpt_mirror()
    mirrored = (tmp_path / "mirror" / recorder.run_dir.name
                / "artifacts" / "cells" / "cell_d0.00_s0.json")
    assert mirrored.is_file()


def test_ckpt_sync_skips_unchanged_files(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "0")
    _add_cell_file(recorder)
    recorder._sync_ckpt_mirror()
    mirrored = (tmp_path / "mirror" / recorder.run_dir.name
                / "artifacts" / "cells" / "cell_d0.00_s0.json")
    mirrored.write_text("sentinel", encoding="utf-8")  # newer mtime than source
    recorder._sync_ckpt_mirror()
    assert mirrored.read_text(encoding="utf-8") == "sentinel"


def test_ckpt_sync_honors_interval(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "3600")
    _add_cell_file(recorder)
    recorder._sync_ckpt_mirror()  # _ckpt_last_sync was set at create(); 0 s elapsed
    assert not (tmp_path / "mirror" / recorder.run_dir.name / "artifacts").exists()


def test_ckpt_sync_runs_via_write_status(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "0")
    _add_cell_file(recorder, "cell_d0.45_s3.json")
    recorder._write_status(current_step="expB2", step_status="running", force=True)
    mirrored = (tmp_path / "mirror" / recorder.run_dir.name
                / "artifacts" / "cells" / "cell_d0.45_s3.json")
    assert mirrored.is_file()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_results_io_mirror.py -v`
Expected: the four new tests FAIL with `AttributeError: 'RunRecorder' object has no attribute '_sync_ckpt_mirror'` (and the `_write_status` one fails on the missing mirrored file).

- [ ] **Step 3: Implement the incremental sync**

In `itasorl/results_io.py`, add a field to `RunRecorder` after `_mirror_degraded`:

```python
    _ckpt_last_sync: float = field(default_factory=time.perf_counter, repr=False)
```

(`default_factory` stamps instance creation time, so the first sync happens one interval after the run starts, and the interval test above is deterministic.)

Add the method after `_sync_mirror_files`:

```python
    def _sync_ckpt_mirror(self) -> None:
        """Timed incremental mirror of artifacts/ (checkpoint cells, dumped
        states) so a mid-step VM loss costs at most one interval of work."""
        if self._mirror_dir is None:
            return
        interval = float(os.environ.get("ITASORL_CKPT_SYNC_SEC", "300"))
        now = time.perf_counter()
        if (now - self._ckpt_last_sync) < interval:
            return
        self._ckpt_last_sync = now
        self._mirror_attempt(self._sync_ckpt_files)

    def _sync_ckpt_files(self) -> None:
        src_root = self.run_dir / "artifacts"
        if not src_root.is_dir():
            return
        dest_root = self._mirror_dir / self.run_dir.name / "artifacts"
        for src in src_root.rglob("*"):
            if not src.is_file():
                continue
            dest = dest_root / src.relative_to(src_root)
            if dest.is_file() and dest.stat().st_mtime >= src.stat().st_mtime:
                continue  # copy2 preserves mtime, so equal means already synced
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
```

In `_write_status` (~line 383), after the existing `self._sync_mirror()` call, add:

```python
        self._sync_ckpt_mirror()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_results_io_mirror.py -v`
Expected: 6 passed.

- [ ] **Step 5: Update results/README.md**

In `results/README.md`, replace the line (currently ~line 40):

```markdown
- `ITASORL_DRIVE_SYNC` mirrors `combined.log`, `status.json`, `manifest.json`, and per-step outputs to Drive after each step.
```

with:

```markdown
- `ITASORL_DRIVE_SYNC` mirrors `combined.log`, `status.json`, and `manifest.json` continuously, per-step outputs after each step, and `artifacts/` (expB2 checkpoint cells, dumped states) incrementally every `ITASORL_CKPT_SYNC_SEC` seconds (default 300). Mirror failures never stop the run; syncing warns once, retries, and reports recovery.
```

- [ ] **Step 6: Run the full suite and ruff**

Run: `python -m pytest -q && ruff check .`
Expected: all pass, no findings.

- [ ] **Step 7: Commit**

```bash
git add itasorl/results_io.py tests/test_results_io_mirror.py results/README.md
git commit -m "feat(results): incremental Drive sync of artifacts so mid-step checkpoints survive a disconnect"
```

---

### Task 3: `--b2-dump-states auto` sentinel in run_e2e.py

States must live inside the run dir (mirrored, bundled) and the recorded flag must survive resume on a different path. The sentinel `auto` is recorded raw in `b2_flags.json` and resolved against the active run dir at command build time.

**Files:**
- Modify: `scripts/run_e2e.py` (help text ~line 82, new function after `resolve_b2_extra` ~line 133, call site ~line 208)
- Create: `tests/test_run_e2e_flags.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_e2e_flags.py`:

```python
"""The b2 dump-states 'auto' sentinel: recorded raw, resolved per run dir."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_e2e  # noqa: E402


def _ns(**overrides):
    base = dict(b2_seeds=None, b2_updates=None, b2_hidden=None,
                b2_dump_states=None, b2_sysid_aux=False, b2_drift_mode=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_auto_resolves_under_run_dir(tmp_path):
    extra = ["--seeds", "0", "--dump-states", "auto", "--sysid-aux"]
    out = run_e2e.resolve_dump_states(extra, tmp_path)
    assert out[out.index("--dump-states") + 1] == str(tmp_path / "artifacts" / "states")
    assert extra[extra.index("--dump-states") + 1] == "auto"  # input not mutated


def test_explicit_path_untouched(tmp_path):
    extra = ["--dump-states", str(tmp_path / "elsewhere")]
    assert run_e2e.resolve_dump_states(extra, tmp_path) == extra


def test_no_dump_states_flag_untouched(tmp_path):
    extra = ["--seeds", "0", "1"]
    assert run_e2e.resolve_dump_states(extra, tmp_path) == extra


def test_auto_recorded_raw_and_reresolved_on_resume(tmp_path):
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    extra = run_e2e.resolve_b2_extra(_ns(b2_dump_states="auto"),
                                     resume=False, run_dir=fresh)
    recorded = json.loads((fresh / "b2_flags.json").read_text(encoding="utf-8"))
    assert recorded == ["--dump-states", "auto"]

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    shutil.copy2(fresh / "b2_flags.json", resumed / "b2_flags.json")
    replayed = run_e2e.resolve_b2_extra(_ns(), resume=True, run_dir=resumed)
    assert replayed == ["--dump-states", "auto", "--resume"]
    resolved = run_e2e.resolve_dump_states(replayed, resumed)
    assert resolved[1] == str(resumed / "artifacts" / "states")
    assert extra == ["--dump-states", "auto"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_e2e_flags.py -v`
Expected: FAIL with `AttributeError: module 'run_e2e' has no attribute 'resolve_dump_states'`.

- [ ] **Step 3: Implement the sentinel**

In `scripts/run_e2e.py`, update the `--b2-dump-states` help text (~lines 82-85) to:

```python
    ap.add_argument("--b2-dump-states", type=str, default=None,
                    help="Persist B-v2 recurrent states to this dir (forwarded to run_expB2.py "
                         "--dump-states) for offline variance/selectivity re-probing with "
                         "scripts/reanalyze_expB2_states.py. Pass 'auto' to place them under "
                         "<run_dir>/artifacts/states (mirrored, bundled, resume-safe).")
```

Add after `resolve_b2_extra` (~line 133):

```python
DUMP_STATES_AUTO = "auto"


def resolve_dump_states(extra: list[str], run_dir: Path) -> list[str]:
    """Resolve the 'auto' dump-states sentinel against the active run dir.
    b2_flags.json keeps the raw sentinel so a resume on a different path
    (new VM, fullruns/_resume_local copy) re-resolves correctly."""
    out = list(extra)
    for i, tok in enumerate(out[:-1]):
        if tok == "--dump-states" and out[i + 1] == DUMP_STATES_AUTO:
            out[i + 1] = str(run_dir / "artifacts" / "states")
    return out
```

In `main()` (~lines 208-209), change:

```python
        b2_extra = resolve_b2_extra(args, resume=resume_dir is not None,
                                    run_dir=recorder.run_dir)
```

to:

```python
        b2_extra = resolve_b2_extra(args, resume=resume_dir is not None,
                                    run_dir=recorder.run_dir)
        b2_extra = resolve_dump_states(b2_extra, recorder.run_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_e2e_flags.py -v`
Expected: 4 passed.

- [ ] **Step 5: Update scripts/README.md**

In `scripts/README.md`, after the line about resuming (`python scripts/run_e2e.py --resume fullruns/<dir> --only expb2`), add:

```markdown
- `--b2-dump-states auto` places recurrent-state dumps under
  `<run_dir>/artifacts/states` so they are mirrored to Drive, included in
  `bundle.zip`, and survive resume on a different machine. The Colab notebook
  always uses `auto`; `run_local.py` keeps its explicit `<run_dir>/states`.
```

- [ ] **Step 6: Run the full suite and ruff**

Run: `python -m pytest -q && ruff check .`
Expected: all pass, no findings.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_e2e.py tests/test_run_e2e_flags.py scripts/README.md
git commit -m "feat(e2e): --b2-dump-states auto resolves under the active run dir"
```

---

### Task 4: Notebook profile tests parse the real notebook

Replace the hardcoded `NOTEBOOK_PROFILES` set in `tests/test_run_local.py` with parsing of `notebooks/colab_gpu.ipynb`, and assert the (future) form dropdown matches. Written before the notebook rework so the dropdown test FAILS until Task 5 lands; run it, confirm the failure mode, and commit both tasks' work only after Task 5 turns it green. (Tasks 4 and 5 form one TDD cycle; do not commit between them.)

**Files:**
- Modify: `tests/test_run_local.py` (top-of-file constants and `test_profiles_match_notebook_table`)

- [ ] **Step 1: Replace the hardcoded set with notebook parsing**

In `tests/test_run_local.py`, replace:

```python
RUN_DIR = Path("fullruns") / "01011999"

NOTEBOOK_PROFILES = {"quick", "full", "bv3_regime", "bv3_regime_n10",
                     "bv2_ceiling", "bv3_ceiling", "b2_only", "b2_seed0",
                     "experiments_no_b2"}


def test_profiles_match_notebook_table():
    assert set(run_local.PROFILES) == NOTEBOOK_PROFILES
```

with:

```python
RUN_DIR = Path("fullruns") / "01011999"

NB_PATH = ROOT / "notebooks" / "colab_gpu.ipynb"


def _notebook_config_source() -> str:
    import json
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "_PROFILES" in src and "RUN_PROFILE" in src:
            return src
    raise AssertionError("config cell with _PROFILES not found in colab_gpu.ipynb")


NOTEBOOK_PROFILES = {"quick", "full", "bv3_regime", "bv3_regime_n10",
                     "bv2_ceiling", "bv3_ceiling", "b2_only", "b2_seed0",
                     "experiments_no_b2"}


def test_profiles_match_notebook_table():
    import re
    src = _notebook_config_source()
    table_keys = re.findall(r'^\s*"([a-z0-9_]+)":\s*dict\(', src, flags=re.M)
    assert table_keys == list(run_local.PROFILES)
    assert set(run_local.PROFILES) == NOTEBOOK_PROFILES


def test_notebook_dropdown_matches_profiles():
    import re
    src = _notebook_config_source()
    m = re.search(r'RUN_PROFILE\s*=\s*"[^"]*"\s*#\s*@param\s*\[([^\]]*)\]', src)
    assert m, "RUN_PROFILE form dropdown (# @param [...]) missing from config cell"
    dropdown = re.findall(r'"([^"]+)"', m.group(1))
    assert dropdown == list(run_local.PROFILES)
```

- [ ] **Step 2: Run to verify the expected mixed result**

Run: `python -m pytest tests/test_run_local.py -v`
Expected: `test_profiles_match_notebook_table` PASSES (the current notebook has the table); `test_notebook_dropdown_matches_profiles` FAILS with "RUN_PROFILE form dropdown ... missing" (the current cell has no `# @param`). Do NOT commit yet; Task 5 makes it green.

---

### Task 5: Notebook rework, part 1: header, guard cell, form config, mount auth

All notebook edits change the `source` arrays of cells in `notebooks/colab_gpu.ipynb`. Keep existing cell ids where a cell is modified; new cells need a fresh unique id. After each edit run `python -m json.tool notebooks/colab_gpu.ipynb > /dev/null`.

**Files:**
- Modify: `notebooks/colab_gpu.ipynb` (cells `09b5e003`, new guard cell, `nbcell_00`, `4b0534e2`, `2e37dea6`)

- [ ] **Step 1: Rewrite the header markdown (cell `09b5e003`)**

Keep the title, badge, research-question paragraph, and the profiles table exactly as they are. Replace the "How to use" section and the trailing free-tier note with:

```markdown
### How to use
1. **File -> Save a copy in Drive** (the first cell blocks the read-only GitHub copy).
2. **Runtime -> Change runtime type -> GPU** (T4 / L4 / A100).
3. Pick a `RUN_PROFILE` in the config form (dropdown, no code edits).
4. **Runtime -> Run all**. Read the printed `SUMMARY.md` and the variance /
   selectivity re-analysis at the bottom when it finishes.

> **If Colab disconnects** (free tier caps GPU sessions at ~90 min): reopen
> your Drive copy and **Runtime -> Run all** again. The notebook finds the
> unfinished run on Drive and resumes it automatically; checkpoints mirror to
> Drive every ~5 minutes during the run, so at most a few minutes repeat.
```

- [ ] **Step 2: Insert the force-copy guard cell**

Insert a new code cell with id `copyguard01` immediately after the header markdown (before `nbcell_00`), source:

```python
# @title 0. Use your own copy (required on Colab)
# Colab opens GitHub notebooks read-only ("playground"). Config edits and cell
# outputs vanish with the tab unless you work inside your own Drive copy.
try:
    import google.colab  # noqa: F401
    _in_colab = True
except ImportError:
    _in_colab = False

if _in_colab:
    try:
        from google.colab.output import eval_js
        _nb_url = eval_js("window.location.href")
    except Exception as exc:  # no frontend attached; never block a real run
        print(f"Could not determine the notebook URL ({exc}); continuing.")
        _nb_url = ""
    if "/github/" in _nb_url:
        raise SystemExit(
            "\n" + "=" * 72 + "\n"
            "STOP: this is the read-only GitHub copy of the notebook.\n"
            "  1. File -> Save a copy in Drive\n"
            "  2. Colab opens YOUR copy in a new tab; close this one\n"
            "  3. In your copy: Runtime -> Run all\n"
            + "=" * 72
        )
    print("OK: running from your own copy.")
else:
    print("Local mode: copy guard not needed.")
```

- [ ] **Step 3: Rewrite the config markdown (cell `nbcell_00`)**

```markdown
## 1. Configure the run

Pick a `RUN_PROFILE` from the dropdown. `FORCE_FRESH` abandons an unfinished
Drive run and starts over. `RESUME_RUN_DIR` is an advanced override (leave
empty for automatic resume). `BRANCH` other than `main` is for testing
unmerged code only.
```

- [ ] **Step 4: Rewrite the config cell (cell `4b0534e2`)**

Full replacement source (the `_PROFILES` table is byte-identical to today's; keep the keep-in-sync comment):

```python
# @title Configure { display-mode: "both" }
from pathlib import Path

RUN_PROFILE = "bv3_regime"  # @param ["quick", "full", "bv3_regime", "bv3_regime_n10", "bv2_ceiling", "bv3_ceiling", "b2_only", "b2_seed0", "experiments_no_b2"]
BRANCH = "main"  # @param {type:"string"}
FORCE_FRESH = False  # @param {type:"boolean"}
RESUME_RUN_DIR = ""  # @param {type:"string"}

REPO_URL = "https://github.com/iLevyTate/ITASORL.git"

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
    REPO_DIR = "/content/ITASORL"
except ImportError:
    IN_COLAB = False
    REPO_DIR = str(Path.cwd().resolve())
    if not (Path(REPO_DIR) / "scripts" / "run_e2e.py").is_file():
        REPO_DIR = str(Path(REPO_DIR).parent)

# Keep _PROFILES in sync with PROFILES in scripts/run_local.py.
# only="expb2" runs B-v2/B-v3 alone (run_e2e.py skips pytest + Experiment A + B).
_PROFILES = {
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
if RUN_PROFILE not in _PROFILES:
    raise ValueError(f"Unknown RUN_PROFILE={RUN_PROFILE!r}; choose from {list(_PROFILES)}")
_p = _PROFILES[RUN_PROFILE]
RUN_MODE = _p["run_mode"]
E2E_ONLY = _p["only"]                 # None | "pytest" | "experiments" | "expb2"
SKIP_STEPS = list(_p["skip_steps"])
B2_SEEDS = _p["b2_seeds"]
B2_UPDATES = _p["b2_updates"]
DRIFT_MODE = _p["drift_mode"]        # None | "ar1" | "regime" (B-v3)
SYSID_AUX = _p["sysid_aux"]          # capacity-ceiling control (breaks readout-not-reward)
DUMP_STATES = _p["dump_states"]      # persist recurrent states for offline re-probing

MOUNT_DRIVE = IN_COLAB
DRIVE_RESULTS = "/content/drive/MyDrive/ITASORL_results"
SAVE_TO_DRIVE = IN_COLAB
DOWNLOAD_BUNDLE = IN_COLAB
RESUME_RUN_DIR = RESUME_RUN_DIR.strip()

if BRANCH != "main":
    print("!" * 72)
    print(f"WARNING: running branch {BRANCH!r}, not main. For testing unmerged")
    print("code only; do not treat the results as replication runs.")
    print("!" * 72)

print(f"RUN_PROFILE={RUN_PROFILE}  RUN_MODE={RUN_MODE}  only={E2E_ONLY}  skip={SKIP_STEPS or '(none)'}")
print(f"B-v2/3: drift_mode={DRIFT_MODE}  sysid_aux={SYSID_AUX}  dump_states={DUMP_STATES}"
      + (f"  seeds={B2_SEEDS}" if B2_SEEDS is not None else "")
      + (f"  updates={B2_UPDATES}" if B2_UPDATES is not None else ""))
if FORCE_FRESH:
    print("FORCE_FRESH: any unfinished Drive run will be ignored, not resumed.")
```

Removed relative to today: `datetime` import, `FRESH_RUN`, `RESULTS_ON_DRIVE`, `RUN_ID`, `STATES_DIR`, `COPY_DRIVE_RESUME_TO_LOCAL`, and the commented resume instructions (auto-resume replaces them).

- [ ] **Step 5: Add Drive API auth to the mount cell (cell `2e37dea6`)**

Replace the `if MOUNT_DRIVE:` block so it reads:

```python
if MOUNT_DRIVE:
    from google.colab import auth, drive
    auth.authenticate_user()  # Drive v3 API fallback for the watchdog; no popup mid-run
    drive.mount("/content/drive")
    os.makedirs(DRIVE_RESULTS, exist_ok=True)
elif not IN_COLAB:
    print(f"Local mode (repo: {REPO_DIR})")
```

(The imports and `sh` helper above the block stay unchanged.)

- [ ] **Step 6: Validate JSON and run the dropdown test**

Run: `python -m json.tool notebooks/colab_gpu.ipynb > /dev/null && python -m pytest tests/test_run_local.py -v`
Expected: JSON valid; `test_notebook_dropdown_matches_profiles` now PASSES; all of `tests/test_run_local.py` green.

- [ ] **Step 7: Commit (covers Task 4 and Task 5)**

```bash
git add notebooks/colab_gpu.ipynb tests/test_run_local.py
git commit -m "feat(colab): force-copy guard, form config with profile dropdown, Drive API auth"
```

---

### Task 6: Notebook rework, part 2: auto-resume run cell with watchdog

Rewrite the run cell (`23887898`), delete the resume-only section (cells `nbcell_07` and `resume_only_cell`), and renumber the later markdown headings.

**Files:**
- Modify: `notebooks/colab_gpu.ipynb` (cells `nbcell_06`, `23887898`; delete `nbcell_07`, `resume_only_cell`; renumber `nbcell_08` "## 9." to "## 8.", `nbcell_09` "## 10." to "## 9.", `nbcell_11` "## 11." to "## 10.")

- [ ] **Step 1: Update the run markdown (cell `nbcell_06`)**

```markdown
## 7. Run the experiments

Auto-resume: if an unfinished run for this profile exists on Drive, it is
copied local and continued; otherwise a fresh run starts. A watchdog keeps
Drive in sync (remounts if the mount dies, falls back to a Drive API upload
of a resume pack). The printed `$ ...` line shows the exact flags.
```

- [ ] **Step 2: Replace the run cell source (cell `23887898`)**

Full replacement:

```python
import argparse
import json
import threading
import zipfile

if str(Path(REPO_DIR) / "scripts") not in sys.path:
    sys.path.insert(0, str(Path(REPO_DIR) / "scripts"))
import run_e2e as _run_e2e  # for build_b2_extra: single source of truth for flag building


def _expected_b2_flags():
    ns = argparse.Namespace(
        b2_seeds=B2_SEEDS, b2_updates=B2_UPDATES, b2_hidden=None,
        b2_dump_states=("auto" if DUMP_STATES else None),
        b2_sysid_aux=SYSID_AUX, b2_drift_mode=DRIFT_MODE)
    return _run_e2e.build_b2_extra(ns)


def _find_unfinished_drive_run():
    root = Path(DRIVE_RESULTS)
    if not root.is_dir():
        return None
    cands = []
    for mf in sorted(root.glob("*/manifest.json")):
        folder = mf.parent
        if (folder / "bundle.zip").is_file():
            continue  # bundle exists only after a successful finish
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"auto-resume: skipping {folder.name} (unreadable manifest: {exc})", flush=True)
            continue
        cands.append((manifest.get("started_at_utc", ""), folder, manifest))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    _, folder, manifest = cands[-1]
    return folder, manifest


resume_src = None
if RESUME_RUN_DIR:
    resume_src = Path(RESUME_RUN_DIR)
    if not resume_src.is_dir():
        raise FileNotFoundError(f"RESUME_RUN_DIR not found: {resume_src}")
    print("Explicit resume override:", resume_src, flush=True)
elif not FORCE_FRESH and IN_COLAB and MOUNT_DRIVE:
    _found = _find_unfinished_drive_run()
    if _found:
        resume_src, _manifest = _found
        _recorded = []
        _ff = resume_src / "b2_flags.json"
        if _ff.is_file():
            _recorded = json.loads(_ff.read_text(encoding="utf-8"))
        _expected = _expected_b2_flags()
        _quick_ok = bool(_manifest.get("quick", False)) == (RUN_MODE == "quick")
        if _recorded != _expected or not _quick_ok:
            raise SystemExit(
                "\n" + "=" * 72 + "\n"
                f"An unfinished run sits on Drive: {resume_src.name}\n"
                f"  its recorded b2 flags: {_recorded} (quick={_manifest.get('quick')})\n"
                f"  profile {RUN_PROFILE!r} would use: {_expected} (quick={RUN_MODE == 'quick'})\n"
                "Either switch RUN_PROFILE to the one that started that run and\n"
                "re-run, or tick FORCE_FRESH to abandon it and start over.\n"
                + "=" * 72)
        print(f"Auto-resume: unfinished Drive run {resume_src.name!r} matches profile {RUN_PROFILE!r}.",
              flush=True)

run_dir = None
if resume_src is not None:
    if str(resume_src).startswith("/content/drive"):
        run_dir = Path(REPO_DIR) / "fullruns" / "_resume_local" / resume_src.name
        print("Copying Drive run to local disk for resume:", run_dir, flush=True)
        shutil.copytree(resume_src, run_dir, dirs_exist_ok=True)
        _pack = resume_src / "resume_pack.zip"
        _mf = resume_src / "manifest.json"
        if _pack.is_file() and _mf.is_file() and _pack.stat().st_mtime > _mf.stat().st_mtime:
            zipfile.ZipFile(_pack).extractall(run_dir)
            print("Applied fresher resume_pack.zip over the mirror copy.", flush=True)
    else:
        run_dir = resume_src

cmd = [sys.executable, "scripts/run_e2e.py"]
if RUN_MODE == "quick":
    cmd.append("--quick")
elif RUN_MODE != "full":
    raise ValueError("RUN_MODE must be 'quick' or 'full'")
if E2E_ONLY:
    cmd.extend(["--only", E2E_ONLY])
for step in SKIP_STEPS:
    cmd.extend(["--skip", step])
if run_dir is not None:
    cmd.extend(["--resume", str(run_dir)])
    print("Resuming run (recorded --b2-* flags replay automatically):", run_dir, flush=True)
else:
    if B2_SEEDS is not None:
        cmd.extend(["--b2-seeds", *[str(s) for s in B2_SEEDS]])
    if B2_UPDATES is not None:
        cmd.extend(["--b2-updates", str(B2_UPDATES)])
    if DRIFT_MODE is not None:
        cmd.extend(["--b2-drift-mode", DRIFT_MODE])   # B-v3 regime coupling
    if SYSID_AUX:
        cmd.append("--b2-sysid-aux")                  # capacity-ceiling control
    if DUMP_STATES:
        cmd.extend(["--b2-dump-states", "auto"])      # lands in <run_dir>/artifacts/states
    print("New run; results land under fullruns/ and mirror to Drive.", flush=True)

run_env = os.environ.copy()
run_env["PYTHONUNBUFFERED"] = "1"
if MOUNT_DRIVE:
    run_env["ITASORL_DRIVE_SYNC"] = DRIVE_RESULTS
    print("Live mirror -> Drive:", DRIVE_RESULTS, flush=True)


def _current_run_dir():
    if run_dir is not None:
        return Path(run_dir)
    ptr = Path(REPO_DIR) / "results" / "LATEST_RUN.txt"
    return Path(ptr.read_text().strip()) if ptr.is_file() else None


def _make_resume_pack(rd):
    """Everything a resume needs (small files only; states are too big for HTTP)."""
    pack = rd / "resume_pack.zip"
    tmp = rd / "resume_pack.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in ("manifest.json", "b2_flags.json", "status.json"):
            p = rd / rel
            if p.is_file():
                zf.write(p, rel)
        cells = rd / "artifacts" / "cells"
        if cells.is_dir():
            for p in cells.glob("*.json"):
                zf.write(p, f"artifacts/cells/{p.name}")
    os.replace(tmp, pack)
    return pack


def _drive_api_upload(pack, run_name):
    """Upload over HTTP; works even when the FUSE mount is dead."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    svc = build("drive", "v3")

    def folder_id(name, parent):
        q = (f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
             f"and '{parent}' in parents and trashed=false")
        hits = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        if hits:
            return hits[0]["id"]
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent]}
        return svc.files().create(body=meta, fields="id").execute()["id"]

    dest = folder_id(run_name, folder_id("ITASORL_results", "root"))
    q = f"name='resume_pack.zip' and '{dest}' in parents and trashed=false"
    hits = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    media = MediaFileUpload(str(pack), mimetype="application/zip")
    if hits:
        svc.files().update(fileId=hits[0]["id"], media_body=media).execute()
    else:
        svc.files().create(body={"name": "resume_pack.zip", "parents": [dest]},
                           media_body=media, fields="id").execute()


_watchdog_stop = threading.Event()


def _watchdog_loop():
    interval = float(os.environ.get("ITASORL_CKPT_SYNC_SEC", "300"))
    while not _watchdog_stop.wait(interval):
        rd = _current_run_dir()
        if rd is None or not (rd / "status.json").is_file():
            continue
        mirror_status = Path(DRIVE_RESULTS) / rd.name / "status.json"
        try:
            fresh = (mirror_status.is_file()
                     and (rd / "status.json").stat().st_mtime
                     - mirror_status.stat().st_mtime < interval * 2)
        except OSError:
            fresh = False
        if fresh:
            continue
        print("\n" + "!" * 72, flush=True)
        print("WATCHDOG: Drive mirror is stale; attempting force remount...", flush=True)
        print("!" * 72, flush=True)
        try:
            from google.colab import drive as _drive
            _drive.mount("/content/drive", force_remount=True)
            continue  # next tick re-checks freshness
        except Exception as exc:
            print(f"WATCHDOG: remount failed ({exc}); Drive API fallback.", flush=True)
        try:
            _drive_api_upload(_make_resume_pack(rd), rd.name)
            print("WATCHDOG: resume_pack.zip uploaded via Drive API; run stays resumable.",
                  flush=True)
        except Exception as exc:
            print(f"WATCHDOG: Drive API upload failed ({exc}); will retry.", flush=True)


def _print_failure_diagnostics():
    rd = _current_run_dir()
    if rd is None or not rd.is_dir():
        print("No run directory found for diagnostics.", flush=True)
        return
    print("\n" + "=" * 72, flush=True)
    print("RUN FAILED. Diagnostics for:", rd, flush=True)
    mf = rd / "manifest.json"
    if mf.is_file():
        steps = json.loads(mf.read_text()).get("steps", {})
        failed = [n for n, s in steps.items() if s.get("status") == "failed"]
        ok = [n for n, s in steps.items() if s.get("status") == "ok"]
        if ok:
            print("Completed steps:", ", ".join(ok), flush=True)
        if failed:
            print("Failed steps:", ", ".join(failed), flush=True)
            for name in failed:
                log = rd / "steps" / f"{name}.log"
                if log.is_file():
                    tail = log.read_text(errors="replace").splitlines()[-40:]
                    print(f"\n--- tail {name}.log ---", flush=True)
                    print("\n".join(tail), flush=True)
    cl = rd / "combined.log"
    if cl.is_file():
        tail = cl.read_text(errors="replace").splitlines()[-30:]
        print("\n--- tail combined.log ---", flush=True)
        print("\n".join(tail), flush=True)
    print(f"Everything so far is mirrored on Drive: {DRIVE_RESULTS}/{rd.name}", flush=True)
    print("To pick up where this left off: reopen your Drive copy of this", flush=True)
    print("notebook and Runtime -> Run all (auto-resume finds this run).", flush=True)
    print("=" * 72, flush=True)


_watchdog = None
if IN_COLAB and MOUNT_DRIVE:
    _watchdog = threading.Thread(target=_watchdog_loop, daemon=True)
    _watchdog.start()

print("$", " ".join(cmd), flush=True)
t0 = time.perf_counter()
proc = subprocess.run(cmd, cwd=REPO_DIR, env=run_env)
_watchdog_stop.set()
if proc.returncode != 0:
    _print_failure_diagnostics()
    raise subprocess.CalledProcessError(proc.returncode, proc.args)
print(f"Wall time: {(time.perf_counter() - t0) / 60:.1f} min")
```

- [ ] **Step 3: Delete the resume-only section**

Remove cells `nbcell_07` (markdown "## 8. Resume after a timeout") and `resume_only_cell` (the commented-out code) from the `cells` array entirely.

- [ ] **Step 4: Renumber the later markdown headings**

- Cell `nbcell_08`: change `## 9. Read the results` to `## 8. Read the results`.
- Cell `nbcell_09`: change `## 10. Variance & selectivity re-analysis (no GPU)` to `## 9. Variance & selectivity re-analysis (no GPU)`.
- Cell `nbcell_11`: change `## 11. Save & download the bundle` to `## 10. Save & download the bundle`.

- [ ] **Step 5: Validate and test**

Run: `python -m json.tool notebooks/colab_gpu.ipynb > /dev/null && python -m pytest tests/test_run_local.py -q`
Expected: JSON valid, tests pass.

- [ ] **Step 6: Commit**

```bash
git add notebooks/colab_gpu.ipynb
git commit -m "feat(colab): auto-resume from Drive with mount watchdog and Drive API resume pack"
```

---

### Task 7: Notebook rework, part 3: reanalysis path and tail markdown

**Files:**
- Modify: `notebooks/colab_gpu.ipynb` (cells `nbcell_10`, `64d34d5c`)

- [ ] **Step 1: Point the reanalysis cell at the run dir (cell `nbcell_10`)**

Full replacement source (`RUN_DIR` is defined in the "Read the results" cell, which runs first):

```python
# Variance-signature + selectivity re-analysis (NO GPU). Recomputes the world-identity
# probe under level vs dispersion features from the states dumped during the run:
#   target       LEVEL      [mean h, final h]   (the pre-registered headline)
#   target_var   DISPERSION [std h, mean|delta h|]
#   target_full  LEVEL ++ DISPERSION
# target_var/full crossing 0.65 while level target stays ~0.52 => the null was mis-probed.
STATES = RUN_DIR / "artifacts" / "states"
if DUMP_STATES and STATES.is_dir():
    subprocess.run([sys.executable, "scripts/reanalyze_expB2_states.py", str(STATES)],
                   cwd=REPO_DIR, check=False)
else:
    print("No dumped states found at", STATES)
    print("Use a profile with dump_states=True (all B-v2/B-v3 profiles do) to enable this.")
```

- [ ] **Step 2: Update the tail markdown (cell `64d34d5c`)**

Keep the bundle table but add a `artifacts/states/` row, and replace the final three paragraphs. New full table and tail:

```markdown
## What's in the bundle

| File | Purpose |
|------|---------|
| `SUMMARY.md` | Plain English outcome. **Read this first.** |
| `status.json` | Live step + last line (updated during run) |
| `manifest.json` | Step timings, status, artifact index |
| `combined.log` | Full stdout (updated live during run) |
| `steps/*.json` | Parsed metrics (AUROC per drift, etc.) |
| `artifacts/` | Figures + `expB2_results.json` |
| `artifacts/cells/` | Per (drift, seed) checkpoints (resume granularity) |
| `artifacts/states/` | Dumped recurrent states for offline re-probing |

**While running (local):** `python scripts/watch_run.py --follow`

**While running (Colab + Drive):** open `MyDrive/ITASORL_results/<run>/combined.log`

**If Colab disconnected:** reopen your Drive copy of this notebook and
**Runtime -> Run all**. Auto-resume finds the newest unfinished run in
`MyDrive/ITASORL_results`, checks it matches your selected profile, and
continues it. Checkpoints mirror to Drive every ~5 minutes during the run.

Unzip locally and open `SUMMARY.md` to decide whether the organism encoded world identity.
```

- [ ] **Step 3: Validate and test**

Run: `python -m json.tool notebooks/colab_gpu.ipynb > /dev/null && python -m pytest -q && ruff check .`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add notebooks/colab_gpu.ipynb
git commit -m "feat(colab): reanalysis reads states from the run dir; document crash recovery"
```

---

### Task 8: Final verification sweep

**Files:** none new.

- [ ] **Step 1: Full suite, lint, notebook JSON**

Run: `python -m pytest -q && ruff check . && python -m json.tool notebooks/colab_gpu.ipynb > /dev/null`
Expected: everything green.

- [ ] **Step 2: Grep for leftovers**

Run: `grep -rn "STATES_DIR\|FRESH_RUN\|RESULTS_ON_DRIVE\|COPY_DRIVE_RESUME_TO_LOCAL" notebooks/ scripts/ itasorl/ tests/ || echo clean`
Expected: `clean` (no references to the removed config globals anywhere).

- [ ] **Step 3: Dash check on touched docs**

Run: `grep -rlP '[\x{2013}\x{2014}]' results/README.md scripts/README.md || echo clean`
Expected: `clean`.

- [ ] **Step 4: Manual Colab smoke test (user-run, not CI)**

Remind the user to verify in Colab with the `quick` profile: guard blocks the GitHub copy; form renders; fresh run mirrors to Drive; interrupt the runtime mid-expB2 and confirm Run all resumes; watchdog banner appears if Drive is force-unmounted.

---

## Self-Review Notes

- Spec coverage: guard (Task 5), form (Task 5), auto-resume (Task 6), `auto` sentinel (Task 3), incremental mirror + fault tolerance (Tasks 1-2), watchdog + resume pack (Task 6), docs (Tasks 2, 3, 5, 7), tests (Tasks 1-4). Date-named Drive folders need no change (existing `default_run_dir` behavior).
- The `finalize` SUMMARY/bundle copy block is deleted, not wrapped: `_sync_mirror(full=True)` on the line above already copies both files.
- Notebook resume passes only `--resume`; recorded flags replay from `b2_flags.json` (the mismatch guard already proved they equal the profile's flags, and `--quick`/skips are restored from the manifest and re-passed respectively).
