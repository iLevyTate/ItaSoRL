"""run_expB2.py refuses an unregistered agent trunk unless the launch says so.

A preregistered comparator run was launched with --hidden 8 (the agent's GRU width) where
--l3-hidden 8 (the surrogate's capacity) was meant, so its trunk was 8 units instead of the
registered 96 and the run is uninformative (FINDINGS 14.5.1). The guard sits at parse time:
it writes nothing into cells and leaves every config fingerprint as it was. The recipes in
use (the corrected C1/C2 script, the cloud step and its runbook flags, the Colab boxes) are
read from their files and must still parse.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")

import run_expB2  # noqa: E402


def test_unregistered_trunk_is_refused_with_a_pointer_to_l3_hidden():
    with pytest.raises(SystemExit, match="l3-hidden"):
        run_expB2.cfg_from_argv(["--drift-mode", "l3", "--hidden", "8"])
    with pytest.raises(SystemExit, match="allow-unregistered-trunk"):
        run_expB2.cfg_from_argv(["--hidden", "64"])          # every drift mode registers 96


def test_the_opt_out_and_the_quick_pass_still_work():
    a = run_expB2.cfg_from_argv(["--drift-mode", "l3", "--hidden", "8", "--allow-unregistered-trunk"])
    assert a.hidden == 8
    assert run_expB2.cfg_from_argv(["--quick", "--hidden", "8"]).hidden == 64   # quick sets its own
    a = run_expB2.cfg_from_argv(["--drift-mode", "l3"])
    assert a.hidden == run_expB2.REGISTERED_TRUNK == 96 and a.allow_unregistered_trunk is False


# ---- the recipes in use parse, and the fingerprints they hash to are unchanged ----

def _recipe_argv(sh_path: Path, subs: dict[str, list[str]]) -> list[str]:
    """The run_expB2.py arguments of the one launch line in a shell script."""
    text = sh_path.read_text(encoding="utf-8").replace("\\\n", " ")
    line = next(ln for ln in text.splitlines() if "scripts/run_expB2.py" in ln)
    tail = line.split("scripts/run_expB2.py", 1)[1]
    tail = re.split(r"\s(?:2>&1|>>|\|)", tail)[0]
    out = []
    for tok in shlex.split(tail):
        if "$" in tok:
            out += subs.get(tok, [])
        else:
            out.append(tok)
    return out


SEEDS = [str(s) for s in range(10)]
CONFIRMATION = ROOT / "scripts" / "revision" / "run_corrected_confirmation.sh"
CLOUD_STEP = ROOT / "scripts" / "cloud_run_step.sh"


def _confirmation_argv():
    return _recipe_argv(CONFIRMATION, {"$SEEDS": SEEDS, "$out/states": ["states"], "$out": ["out"],
                                       "$WORKERS": ["4"]})


def _main_base(a) -> dict:
    """The config dict main() hashes, built with main()'s own key list (read from its source)."""
    tree = ast.parse((ROOT / "scripts" / "run_expB2.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    comp = next(n for n in ast.walk(main) if isinstance(n, ast.DictComp)
                and isinstance(n.generators[0].iter, ast.Tuple))
    keys = [e.value for e in comp.generators[0].iter.elts]
    assert "allow_unregistered_trunk" not in keys
    base = {k: getattr(a, k) for k in keys}
    base.update(drifts=a.drifts, device="cpu", out_dir=a.out_dir, save_agents=a.save_agents)
    base.update(objective=a.objective, mortal=(a.mortal == "on"))
    return base


def _recorded_fingerprint(run: str) -> str:
    cell = ROOT / "artifacts" / "corrected_runs" / run / "cells" / "cell_d0.45_s0.json"
    return json.loads(cell.read_text(encoding="utf-8"))["fingerprint"]


def test_corrected_confirmation_recipe_parses_and_hashes_to_the_committed_cells():
    argv = _confirmation_argv()
    assert "--hidden" not in argv and "--l3-hidden" in argv
    c1 = run_expB2.cfg_from_argv(argv)
    c2 = run_expB2.cfg_from_argv(argv + ["--no-world-model"])
    assert c1.hidden == c2.hidden == 96
    assert run_expB2.config_fingerprint(_main_base(c1)) == _recorded_fingerprint("corrected_l3_h8_wm")
    assert run_expB2.config_fingerprint(_main_base(c2)) == _recorded_fingerprint("corrected_l3_h8_nowm")
    # the opt-out is parse-time only: an 8-unit trunk still hashes apart from both
    c8 = run_expB2.cfg_from_argv(argv + ["--hidden", "8", "--allow-unregistered-trunk"])
    fp8 = run_expB2.config_fingerprint(_main_base(c8))
    assert fp8 not in {_recorded_fingerprint("corrected_l3_h8_wm"), _recorded_fingerprint("corrected_l3_h8_nowm")}


@pytest.mark.parametrize("flags", [[], ["--objective", "touch", "--mortal", "off"], ["--mortal", "off"]])
def test_cloud_step_recipe_with_runbook_flags_does_not_trip_the_guard(flags):
    # docs/specs/2026-10-07-goal-and-stakes-cloud-runbook.md: T-touch and S-immortal flags
    argv = _recipe_argv(CLOUD_STEP, {"${OUT}": ["out"], "$@": flags})
    assert "--hidden" not in argv
    if flags:
        assert argv[-len(flags):] == flags                  # "$@" carried the runbook flags
    a = run_expB2.cfg_from_argv(argv)
    assert a.hidden == 96 and a.drift_mode == "l3" and a.l3_hidden == 8


def test_colab_texture_boxes_do_not_trip_the_guard():
    nb = json.loads((ROOT / "notebooks" / "colab_gpu.ipynb").read_text(encoding="utf-8"))
    src = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                    for c in nb["cells"] if c["cell_type"] == "code")
    boxes = re.findall(r'\["scripts/run_expB2\.py",(.*?)\]', src, flags=re.S)
    assert len(boxes) == 2
    for box in boxes:
        argv = []
        for quoted, star in re.findall(r'"([^"]+)"|(\*_seeds10)', box):
            argv += SEEDS if star else [quoted]
        assert "--hidden" not in argv
        assert run_expB2.cfg_from_argv(argv).hidden == 96
