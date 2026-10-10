"""Checks on the rendered site that build_index.py --check cannot see.

build_index.py --check compares bytes, so a page whose logic script fails to parse still
passes it. That happened once: an unescaped apostrophe in the findings list (2026-10-06)
broke the whole logic class, and the live page showed an error badge with the findings
list, the readout chart and the copy button blank. These tests:

  - parse the page's logic script with `node --check` (skipped when node is absent;
    GitHub-hosted runners ship it);
  - check that the script holds no "{{" or "}}" pair, which the page runtime reads as a
    binding;
  - pin every per-seed readout row the section-03 chart draws to the committed artifacts;
  - keep the retired July demo loops off the pages, the phone cut viewable on phones, the
    correction caveats linked to rendered pages, and the page's language and landmark set.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
SCRIPT_RE = re.compile(r'<script type="text/x-dc" data-dc-script="[^"]*"[^>]*>(.*?)</script>', re.S)
ROW_RE = re.compile(
    r"^\s*(\w+): \{ seeds: \[([^\]]*)\], mean: ([0-9.]+), lo: ([0-9.]+), hi: ([0-9.]+) \},$", re.M
)
RETIRED_LOOPS = ("loop-idle-clouds", "loop-survival-clouds", "loop-race", "loop-brain-pair")


def _logic_script() -> str:
    found = SCRIPT_RE.findall(INDEX.read_text(encoding="utf-8"))
    assert len(found) == 1, "index.html should carry exactly one data-dc-script block"
    return found[0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_logic_script_parses(tmp_path):
    src = tmp_path / "page_logic.js"
    src.write_text(_logic_script(), encoding="utf-8")
    res = subprocess.run(["node", "--check", str(src)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr


def test_page_logic_script_has_no_binding_braces():
    script = _logic_script()
    assert "{{" not in script and "}}" not in script


def _artifact_rows() -> dict[str, dict]:
    verdicts = json.loads((ROOT / "artifacts" / "corrected_verdicts.json").read_text(encoding="utf-8"))
    pc = json.loads((ROOT / "artifacts" / "policy_controls" / "corrected_l3_h8_wm.json")
                    .read_text(encoding="utf-8"))["aggregate"]
    nowm = json.loads((ROOT / "artifacts" / "expB2" / "corrected_l3_h8_nowm.json").read_text(encoding="utf-8"))
    c1 = verdicts["runs"]["C1"]
    l0 = {"per_seed": c1["gates"]["l0"]["per_seed"], "mean": c1["gates"]["l0"]["mean"],
          "t90": c1["drift_zero"]["survival"]["t90"]}
    return {
        "own_untrained": pc["untrained own"],
        "own_predictor": pc["predictor own"],
        "own_nopred": nowm["arms"][nowm["dmax"]]["survival"]["pool_target"],
        "own_survival": pc["survival own"],
        "replay_untrained": pc["untrained replay"],
        "replay_predictor": pc["predictor replay"],
        "replay_logged": pc["predictor_logged replay"],
        "replay_survival": pc["survival replay"],
        "l0_survival": l0,
    }


def test_readout_chart_rows_match_the_committed_artifacts():
    rows = {m[0]: m for m in ROW_RE.findall(_logic_script())}
    expected = _artifact_rows()
    assert set(rows) == set(expected)
    for key, art in expected.items():
        _, seeds, mean, lo, hi = rows[key]
        page_seeds = [float(x) for x in seeds.split(",")]
        assert page_seeds == [round(v, 3) for v in art["per_seed"]], key
        assert mean == f"{art['mean']:.3f}", key
        assert (lo, hi) == (f"{art['t90'][0]:.3f}", f"{art['t90'][1]:.3f}"), key
    # The primary rows are the same numbers the corrected verdicts publish.
    c1 = json.loads((ROOT / "artifacts" / "corrected_verdicts.json").read_text(encoding="utf-8"))["runs"]["C1"]
    assert rows["own_survival"][2] == f"{c1['primary']['survival']['mean']:.3f}"
    assert rows["own_predictor"][2] == f"{c1['primary']['predictor']['mean']:.3f}"
    assert rows["own_untrained"][2] == f"{c1['primary']['untrained']['mean']:.3f}"


def test_readout_chart_is_not_a_training_curve():
    page = INDEX.read_text(encoding="utf-8")
    for retired in ("training progress", "survival pressure applied", "schematic; end points"):
        assert retired not in page


@pytest.mark.parametrize("relpath", ["index.html", "index.template.html", "viz/player/index.html",
                                     "assets/film/README.md"])
def test_retired_demo_loops_are_not_served(relpath):
    text = (ROOT / relpath).read_text(encoding="utf-8")
    for name in RETIRED_LOOPS:
        if relpath == "assets/film/README.md":
            # The film README names them once, in the retirement note, never as a served file.
            assert f"!/assets/film/{name}" not in text
            continue
        assert name not in text, f"{relpath} still references {name}"


def test_gitignore_allowlist_drops_the_retired_loops():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in RETIRED_LOOPS:
        assert f"!/assets/film/{name}.mp4" not in text


def test_phone_cut_page_sets_a_viewport():
    head = (ROOT / "viz" / "player" / "index.html").read_text(encoding="utf-8").split("</head>")[0]
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in head


def test_caveat_links_open_rendered_pages_not_raw_markdown():
    # GitHub Pages serves .md as text/markdown, so a relative docs/*.md link opens raw text.
    page = INDEX.read_text(encoding="utf-8")
    assert not re.findall(r'href="(?:\./)?docs/[^"]+\.md(?:#[^"]*)?"', page)


def test_page_language_landmark_and_description():
    page = INDEX.read_text(encoding="utf-8")
    assert page.startswith('<!DOCTYPE html>\n<html lang="en">')
    assert page.count("<main>") == 1 and page.count("</main>") == 1
    assert '<meta name="description" content="' in page
