"""The cross-budget contrast in artifacts/budget_curve.json.

The record quotes the decoder-off arm at 450 updates against the decoder-on arm at the
registered 300 ("out-forages but does not out-read"). scripts/build_budget_curve.py writes it
as `cross_budget`; this recomputes it from the committed cells and pins the quoted values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_budget_curve as bbc  # noqa: E402

ART = ROOT / "artifacts" / "budget_curve.json"
ON = ROOT / "artifacts" / "corrected_runs" / "corrected_l3_h8_wm"
OFF = ROOT / "artifacts" / "corrected_runs" / "corrected_l3_h8_nowm"


def _close(a, b, tol=1e-8):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def test_the_cross_budget_block_recomputes_from_the_committed_cells():
    _, on = bbc.load_curve(str(ON))
    _, off = bbc.load_curve(str(OFF))
    fresh = json.loads(json.dumps(bbc.cross_budget("decoder off", off, "decoder on", on)))
    have = json.loads(ART.read_text(encoding="utf-8"))["cross_budget"]
    assert len(have) == 1 and _close(fresh, have[0])


def test_the_quoted_cross_budget_values():
    c = json.loads(ART.read_text(encoding="utf-8"))["cross_budget"][0]
    assert (c["a"], c["b"]) == ({"label": "decoder off", "updates": 450},
                                {"label": "decoder on", "updates": 300})
    assert c["seeds"] == list(range(10))
    assert [round(x, 3) for x in [c["return"]["mean"], *c["return"]["t90"]]] == [0.208, 0.063, 0.353]
    assert [round(x, 3) for x in [c["target"]["mean"], *c["target"]["t90"]]] == [-0.048, -0.126, 0.030]


def test_cross_budget_is_absent_without_a_point_past_the_registered_budget():
    reg = {300: {"seed": [0, 1], "target": [0.6, 0.7], "return": [-0.5, -0.4], "env_steps": [1, 1]}}
    assert bbc.cross_budget("b", reg, "a", reg) is None
    assert bbc.cross_budget("b", {450: reg[300]}, "a", {}) is None


@pytest.mark.parametrize("key", ["series", "contrast", "drift", "runs"])
def test_the_rest_of_the_artifact_still_regenerates_within_float_noise(key):
    series, raw = {}, {}
    for label, d in (("decoder on", ON), ("decoder off", OFF)):
        _, raw[label] = bbc.load_curve(str(d))
        series[label] = bbc.summarize(raw[label])
    have = json.loads(ART.read_text(encoding="utf-8"))
    if key == "series":
        assert _close(json.loads(json.dumps(series)), have["series"])
    elif key == "runs":
        assert have["runs"] == {"decoder on": "artifacts/corrected_runs/corrected_l3_h8_wm",
                                "decoder off": "artifacts/corrected_runs/corrected_l3_h8_nowm"}
    elif key == "drift":
        assert have["drift"] == {"decoder on": 0.45, "decoder off": 0.45}
    else:
        assert [c["updates"] for c in have["contrast"]] == sorted(raw["decoder on"])
