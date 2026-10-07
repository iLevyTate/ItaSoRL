"""Structural checks for the Colab runner notebook (`notebooks/colab_gpu.ipynb`).

The notebook is hand-edited JSON. These tests guard the invariants that keep
"Runtime -> Run all" safe and that enforce the make-a-copy-first workflow. They
never execute a cell (Colab, a GPU, and Google Drive are all unavailable in CI),
so they only inspect the notebook's source.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "colab_gpu.ipynb"


def _load():
    return json.loads(NB_PATH.read_text())


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
