"""Checks for the Colab runner notebook (`notebooks/colab_gpu.ipynb`).

The notebook is hand-edited JSON. These tests guard the invariants that keep
"Runtime -> Run all" safe and that enforce the make-a-copy-first workflow. Colab, a
GPU, and Google Drive are all unavailable in CI, so most tests inspect the source.
Three go further: every flag the notebook passes to a script must be an option that
script's argparse parser defines, the read-only "Read the extras" cell is executed
against small fixtures in the schemas it parses, and the notebook metadata must not
preselect hardware.
"""
import ast
import json
import math
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "colab_gpu.ipynb"


def _load():
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _src(cell):
    # nbformat allows source as a single string or a list of lines; this
    # notebook mixes both, so normalise before matching.
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_nbformat4():
    nb = _load()
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}


def test_every_code_cell_compiles():
    # Colab form magics (`# @title` / `# @param`) are plain comments, so each
    # code cell is valid standalone Python. Catches an edit that corrupts a cell.
    nb = _load()
    for i, cell in enumerate(_code_cells(nb)):
        compile(_src(cell), f"<cell {i}>", "exec")


def test_copy_guard_is_the_first_code_cell():
    # The make-a-copy guard must run before any heavy work so "Run all" stops at
    # it - i.e. before the config form clones the repo or `run_e2e.py` launches.
    nb = _load()
    code = _code_cells(nb)
    guard = _src(code[0])

    assert "I_MADE_A_COPY" in guard
    assert "Save a copy in Drive" in guard
    assert "colab_gpu.ipynb" in guard  # the canonical read-only name it checks
    assert "raise" in guard            # it can actually halt Run all

    rest = "\n".join(_src(c) for c in code[1:])
    assert "run_e2e.py" in rest, "the experiment-run cell moved or vanished"
    assert "run_e2e.py" not in guard, "the guard must precede the run cell"


def test_copy_guard_blocks_when_unverifiable():
    # A GitHub open, the original filename, and an unverifiable result are not
    # a Drive copy. The guard raises in each of those cases. It continues only
    # off-Colab, on positive Drive evidence, or when I_MADE_A_COPY is set in
    # the cell. The skip is not a form checkbox.
    nb = _load()
    guard = _src(_code_cells(nb)[0])
    assert "_in_colab" in guard
    assert "Continuing anyway" not in guard
    assert '"/github/"' in guard
    assert '"/drive/"' in guard
    assert "Copy of " in guard
    assert "file_id" in guard
    assert "_status is False" in guard
    assert "_status is None" in guard
    assert "I_MADE_A_COPY = False  # @param" not in guard
    assert guard.count("raise RuntimeError") >= 2


def test_every_later_code_cell_requires_the_copy():
    nb = _load()
    code = _code_cells(nb)
    for i, cell in enumerate(code[1:], start=1):
        assert "COPY_CHECK_PASSED" in _src(cell), f"code cell {i} can run before the copy check"


def test_copy_guard_sets_a_sentinel_other_cells_can_check():
    nb = _load()
    guard = _src(_code_cells(nb)[0])
    assert "COPY_CHECK_PASSED = True" in guard
    assert "COPY_CHECK_PASSED = False" in guard


def test_heavy_cells_require_the_copy_guard_to_have_passed():
    # Jumping straight to the cell that clones the repo, the main run cell, or
    # the 2026-10 extras run cell must not skip Step 0 - each has to check the
    # guard's sentinel before doing anything real (cloning, installing,
    # launching a training run), so running them out of order does not get
    # around the make-a-copy check.
    nb = _load()
    code = _code_cells(nb)
    markers = {
        "clone/install cell": "git clone",
        "main run cell": 'sys.executable, "scripts/run_e2e.py"',
        "2026-10 extras run cell": "def run_extra(",
    }
    found = set()
    for cell in code[1:]:
        src = _src(cell)
        for label, marker in markers.items():
            if marker in src:
                found.add(label)
                assert "COPY_CHECK_PASSED" in src, (
                    f"the {label} does not check the copy guard's sentinel")
    assert found == set(markers), f"could not locate: {set(markers) - found}"


def test_run_cell_picks_up_the_last_unfinished_drive_folder():
    # A disconnect wipes the VM. The run cell has to find the newest unfinished
    # folder already mirrored under the Drive results root and pass it to
    # --resume, instead of starting a second run beside it.
    nb = _load()
    cells = [_src(c) for c in _code_cells(nb)]
    run = next(s for s in cells if "scripts/run_e2e.py" in s and "--profile" in s)
    assert "finished_at_utc" in run
    assert "manifest.json" in run
    assert "Picking up Drive run" in run
    assert "FORCE_FRESH" in run
    extras = next(s for s in cells if "def run_extra(" in s)
    assert "Restoring" in extras
    assert "_mirror_extra_to_drive" in extras


def test_intro_documents_make_a_copy_first():
    nb = _load()
    markdown = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "Save a copy in Drive" in markdown


def test_world_sample_sensitivity_extra_resolves_and_verifies_the_saved_agents_first():
    # The saved agents are gitignored. The extra must go through the resolver (which checks
    # every sha256 against the committed artifact) before scoring, must checkpoint inside
    # the Drive-mirrored folder, and must be readable against the committed result.
    nb = _load()
    code = "\n".join(_src(c) for c in _code_cells(nb))
    assert "RUN_WORLD_SAMPLE_SENSITIVITY" in code
    assert "AGENTS_ARCHIVE_URL" in code and "AGENTS_DRIVE_DIR" in code
    assert "scripts/fetch_saved_agents.py" in code
    assert "scripts/run_world_sample_sensitivity.py" in code
    assert '"--checkpoint-dir", "fullruns/d045_world_samples/checkpoints"' in code
    run_cell = next(_src(c) for c in _code_cells(nb) if "def run_extra(" in _src(c))
    assert run_cell.index("_saved_agents_dir()") < run_cell.index('"scripts/run_world_sample_sensitivity.py"')
    read_cell = next(_src(c) for c in _code_cells(nb) if "Read the extras" in _src(c))
    assert "artifacts/l0_audit" in read_cell or "d045_world_samples_l3_h8_heldout.json" in read_cell
    assert "verdict" in read_cell


def test_metadata_leaves_the_hardware_choice_to_the_reader():
    # Colab preselects the runtime from metadata.accelerator and metadata.colab.gpuType.
    # The reader picks the hardware (README "Pick your own GPU runtime"), so neither is set.
    meta = _load()["metadata"]
    assert "gpuType" not in meta.get("colab", {})
    assert "accelerator" not in meta


def test_messages_name_cells_that_exist():
    # A guard or hint that says "Run the '<title>' cell" must name a real @title.
    nb = _load()
    code = [_src(c) for c in _code_cells(nb)]
    titles = {m.group(1).strip() for s in code for m in re.finditer(r"# @title (.+?) \{", s)}
    named = {m.group(1) for s in code for m in re.finditer(r"Run the '([^']+)' cell", s)}
    assert named, "no cell-name hints found"
    assert named <= titles, f"messages name missing cells: {sorted(named - titles)}"


def test_notebook_text_matches_the_record():
    # No archive record exists before the review decision; the gn and qd boxes are
    # exploratory; "hidden-8" names the run, not the agents (their trunk is 96 units).
    nb = _load()
    text = "\n".join(_src(c) for c in nb["cells"])
    for stale in ("Zenodo", "paper's archive", "published with the paper", "hidden-8 agents",
                  "inform the paper", "learned-texture claim", "about half a day",
                  "expect noticeably longer", "reproduce the research"):
        assert stale not in text, stale
    assert "learned-texture rule, which remains unmet" in text
    assert "scripts/reproduce.py retrain C1" in text
    assert "22.6 hours" in text


# ---- every flag the notebook passes to a script exists in that script's parser ----

_SCRIPT_RE = re.compile(r"^scripts/(\w+)\.py$")


def _str_elts(lst: ast.List) -> list[str]:
    return [e.value for e in lst.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def _script_in(lst: ast.List):
    for s in _str_elts(lst):
        m = _SCRIPT_RE.match(s)
        if m:
            return m.group(1)
    return None


def _scopes(tree):
    """The module body without its functions, then each function body on its own."""
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return [top] + [f.body for f in funcs]


def _walk(stmts):
    for st in stmts:
        yield from ast.walk(st)


def _argv_flags(source: str) -> list[tuple[str, str]]:
    """(script, flag) for each scripts/<x>.py argv list a cell builds.

    Covers a list literal naming the script, `[script list] + name` where the name is
    bound to list literals in the same scope, and `name += [...]` after
    `name = [..., "scripts/<x>.py", ...]`.
    """
    out = []
    for scope in _scopes(ast.parse(source)):
        bound: dict[str, list[ast.List]] = {}
        var_script: dict[str, str] = {}
        for node in _walk(scope):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        bound.setdefault(t.id, []).append(node.value)
                        if _script_in(node.value):
                            var_script[t.id] = _script_in(node.value)
        for node in _walk(scope):
            if isinstance(node, ast.List) and _script_in(node):
                out += [(_script_in(node), f) for f in _str_elts(node) if f.startswith("--")]
            elif (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                  and isinstance(node.left, ast.List) and _script_in(node.left)
                  and isinstance(node.right, ast.Name)):
                for lst in bound.get(node.right.id, []):
                    out += [(_script_in(node.left), f) for f in _str_elts(lst) if f.startswith("--")]
            elif (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                  and node.target.id in var_script and isinstance(node.value, ast.List)):
                out += [(var_script[node.target.id], f) for f in _str_elts(node.value)
                        if f.startswith("--")]
    return out


def _parser_options(script: str) -> set[str]:
    """Option strings of every add_argument call in scripts/<script>.py, read statically."""
    tree = ast.parse((ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8"))
    opts = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            opts |= {a.value for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-")}
    return opts


def test_every_flag_the_notebook_passes_is_a_real_option():
    nb = _load()
    uses = [u for c in _code_cells(nb) for u in _argv_flags(_src(c))]
    scripts = {s for s, _ in uses}
    # The run cell, the texture boxes, Experiment C and the agent resolver must all be seen.
    assert {"run_e2e", "run_expB2", "run_expC_milestone3", "fetch_saved_agents",
            "run_world_sample_sensitivity"} <= scripts, scripts
    unknown = sorted({(s, f) for s, f in uses if f not in _parser_options(s)})
    assert not unknown, f"flags no parser defines: {unknown}"


def test_argv_extraction_catches_a_bad_flag():
    src = 'run_extra("x", ["scripts/run_expB2.py", "--drift-mode", "l3", "--no-such-flag"], "o")\n'
    uses = _argv_flags(src)
    assert ("run_expB2", "--no-such-flag") in uses
    assert "--no-such-flag" not in _parser_options("run_expB2")


# ---- the read-only "Read the extras" cell runs against fixtures ----

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _raw_arm(targets, speeds):
    return {"pool_target": targets, "pool_speed": speeds, "pool_leak_clean": [True] * len(targets)}


def _run_read_cell(repo: Path, scratch: Path) -> str:
    # The cell is saved as a module and run with runpy, with the two globals the earlier
    # cells would have set. It only reads files under REPO_DIR.
    import contextlib
    import io
    import runpy
    cell = next(_src(c) for c in _code_cells(_load()) if "Read the extras" in _src(c))
    path = scratch / "read_extras_cell.py"
    path.write_text(cell, encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(path), init_globals={"COPY_CHECK_PASSED": True, "REPO_DIR": str(repo)})
    return buf.getvalue()


def test_read_extras_cell_handles_per_seed_lists_partial_runs_and_every_readout(tmp_path):
    # run_expB2.py writes pool_target as one list entry per finished seed; a seed whose pool had
    # too few survivors is None or NaN; a drift with no finished seed is an empty list.
    repo = tmp_path / "repo"
    nan = math.nan
    gn = {
        "0.0": {"survival": _raw_arm([0.54, None, nan, 0.56], [0.88, 0.86, None, 0.80]),
                "predictor": _raw_arm([0.52, 0.51, 0.50, 0.49], [0.85, 0.81, 0.88, 0.80]),
                "untrained": _raw_arm([0.55, 0.53, 0.57, 0.52], [0.74, 0.70, 0.71, 0.76])},
        "0.45": {arm: _raw_arm([], []) for arm in ("survival", "predictor", "untrained")},
    }
    _write_json(repo / "fullruns" / "T_gn_l3_h8_wm" / "expB2_results.json", gn)
    for s in range(4):
        _write_json(repo / "fullruns" / "T_gn_l3_h8_wm" / "cells" / f"cell_d0.00_s{s}.json",
                    {"cell": {"drift": 0.0, "seed": s, "eng": {"engaged": s != 3}}})
    qd = {d: {arm: _raw_arm([0.58, 0.57], [0.90, 0.91]) for arm in ("survival", "predictor", "untrained")}
          for d in ("0.0", "0.45")}
    _write_json(repo / "fullruns" / "T_qd_l3_h8_wm" / "expB2_results.json", qd)
    indiv = {"mean": 0.53, "share_at_or_above_bar": 0.0}
    _write_json(repo / "fullruns" / "expC_milestone3" / "emergence_pilot_per_individual_quick.json",
                {"per_seed": [{"seed": 0, "auroc_gen0": 0.50, "auroc_final_treat": 0.51,
                               "auroc_final_ctrl": 0.50, "indiv_gen0": indiv,
                               "indiv_final_treat": indiv, "indiv_final_ctrl": indiv}],
                 "estimand": {"emergence_claim": False}, "routing": "UNINFORMATIVE (fixture)"})
    ws = ROOT / "artifacts" / "l0_audit" / "d045_world_samples_l3_h8_heldout.json"
    for dest in (repo / "fullruns" / "d045_world_samples" / ws.name,
                 repo / "artifacts" / "l0_audit" / ws.name):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ws, dest)

    out = _run_read_cell(repo, tmp_path)

    assert "drift=0.00  survival=0.550" in out            # mean of the two finite seeds
    assert "seeds 2/4/4" in out
    assert "drift=0.45: no finished seeds yet" in out
    assert "speed control: worst pool 0.700 against 0.75, 3 of 11 pools short" in out
    assert "engagement at drift 0.00: 3 of 4 finished seeds engaged" in out
    assert "learned-texture rule, which remains unmet" in out
    assert "speed control: worst pool 0.900 against 0.75, 0 of 12 pools short" in out
    assert "per-individual mean gen0=0.530" in out
    assert "gate routing: UNINFORMATIVE (fixture)" in out
    committed = json.loads(ws.read_text(encoding="utf-8"))["summary"]["survival"]["verdict"]
    assert f"verdict under the frozen rule: {committed}  (committed: {committed})" in out


def test_read_extras_cell_with_nothing_to_read(tmp_path):
    (tmp_path / "repo").mkdir()
    assert "No revision-extras output yet" in _run_read_cell(tmp_path / "repo", tmp_path)
