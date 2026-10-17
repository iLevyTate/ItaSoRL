"""run_expB2.py refuses an unregistered agent trunk unless the launch says so.

A preregistered comparator run was launched with --hidden 8 (the agent's GRU width) where
--l3-hidden 8 (the surrogate's capacity) was meant, so its trunk was 8 units instead of the
registered 96 and the run is uninformative (FINDINGS 14.5.1). The guard sits at parse time:
it writes nothing into cells and leaves every config fingerprint as it was. The recipes in
use (the corrected C1/C2 script, the cloud step and its runbook flags, the Colab boxes) are
read from their files and must still parse. Every run_expB2.py recipe that
`reproduce.py retrain` prints hashes to the fingerprint its committed cells or promoted
artifact record, so a recipe edit that changes the configuration fails here.
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


def _main_base(a, device: str = "cpu") -> dict:
    """The config dict main() hashes, built with main()'s own key list (read from its source)
    and the keys main() adds after it."""
    tree = ast.parse((ROOT / "scripts" / "run_expB2.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    comp = next(n for n in ast.walk(main) if isinstance(n, ast.DictComp)
                and isinstance(n.generators[0].iter, ast.Tuple))
    keys = [e.value for e in comp.generators[0].iter.elts]
    assert "allow_unregistered_trunk" not in keys
    base = {k: getattr(a, k) for k in keys}
    base.update(drifts=a.drifts, device=device, out_dir=a.out_dir, save_agents=a.save_agents)
    base.update(objective=a.objective, mortal=(a.mortal == "on"),
                balanced_seed_base=a.balanced_seed_base)
    if a.heldout_evals:
        base.update(heldout_evals=True, heldout_hidden=a.heldout_hidden,
                    cg_prefix=a.cg_prefix, cg_steps=a.cg_steps)
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


# ---- every run_expB2.py recipe reproduce.py prints hashes to its committed fingerprint ----

import reproduce  # noqa: E402

# The committed fingerprint of each recipe, read from the record: the cells of a run, or the
# promoted artifact (B1 and B2, whose promoter carries the cells' fingerprint since 2026-10-17).
_ART = ROOT / "artifacts"
RECORDED = {
    "L1-CORRECTED": ("cells", _ART / "corrected_runs" / "l1-corrected" / "cells"),
    "B1-BALANCED": ("artifact", _ART / "balanced_confirmation" / "b1.json"),
    "B2-BALANCED": ("artifact", _ART / "balanced_confirmation" / "b2.json"),
    "L3-H8-WM-CPU": ("cells", _ART / "reviewer_gaps_runs" / "l3_h8_wm_cpu" / "cells"),
    "L3-H8-NOWM-CPU": ("cells", _ART / "reviewer_gaps_runs" / "l3_h8_nowm" / "cells"),
    "L3-H10-GS1-CPU": ("cells", _ART / "reviewer_gaps_runs" / "l3_h10_gseed1" / "cells"),
}
# A second, literal pin, so a later re-promotion or cell rewrite cannot move the value silently.
LITERAL = {"C1": "98dfbde61e361984", "L1-CORRECTED": "d8bb95f746f95e4b",
           "B1-BALANCED": "cd7f35c28a343bc2", "B2-BALANCED": "f5e46b22c0c31ecf"}
# The GPU traces run's cells are not committed, so its recipe has no fingerprint to meet.
NO_RECORD = {"L3-H8-N10"}


def _expb2_recipes() -> list[str]:
    return sorted(k for k, cmds in reproduce.RETRAIN.items()
                  if any("scripts/run_expB2.py" in c for c in cmds))


def _retrain_argv(run: str) -> list[str]:
    (line,) = [c for c in reproduce.retrain_lines(run, "dist/retrain_x")[0]
               if "scripts/run_expB2.py" in c]
    return shlex.split(line.split("   #")[0].split("scripts/run_expB2.py", 1)[1])


def _device(a) -> str:
    """The device main() records: cpu for an automatic choice with several workers."""
    if a.device == "auto":
        return "cpu" if a.workers > 1 else run_expB2.default_device()
    return a.device


def _committed_fingerprint(run: str) -> str:
    kind, path = RECORDED[run]
    if kind == "artifact":
        return json.loads(path.read_text(encoding="utf-8"))["fingerprint"]
    fps = {json.loads(p.read_text(encoding="utf-8"))["fingerprint"]
           for p in sorted(path.glob("cell_d*_s*.json"))}
    assert len(fps) == 1, (run, fps)
    return fps.pop()


def test_every_expb2_recipe_has_a_record_or_is_named_as_having_none():
    assert set(_expb2_recipes()) == set(RECORDED) | NO_RECORD


@pytest.mark.parametrize("run", sorted(RECORDED))
def test_retrain_recipe_hashes_to_the_committed_fingerprint(run):
    argv = _retrain_argv(run)
    assert "--hidden" not in argv
    a = run_expB2.cfg_from_argv(argv)
    assert a.hidden == run_expB2.REGISTERED_TRUNK
    fp = run_expB2.config_fingerprint(_main_base(a, _device(a)))
    committed = _committed_fingerprint(run)
    assert fp == committed
    if run in LITERAL:
        assert committed == LITERAL[run]


def test_the_c1_and_c2_recipes_run_the_confirmation_script_and_hash_to_c1():
    assert reproduce.RETRAIN["C1"][0].startswith("bash scripts/revision/run_corrected_confirmation.sh ")
    assert reproduce.RETRAIN["C2"][0].startswith("bash scripts/revision/run_corrected_confirmation.sh ")
    c1 = run_expB2.cfg_from_argv(_confirmation_argv())
    assert run_expB2.config_fingerprint(_main_base(c1)) == LITERAL["C1"] \
        == _recorded_fingerprint("corrected_l3_h8_wm")
