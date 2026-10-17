"""Closing item X08 (E17): scripts/read_budget_curve_dynamics.py on a tmp budget curve and cells.

The fixture is a two-run budget curve with its cells; the readout must pass its input gate,
record the agent-free commit key, and move every partial slope when one per-seed value flips.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import read_budget_curve_dynamics as bcd  # noqa: E402

SEEDS = list(range(10))


def _values(run: int, flip: tuple | None = None):
    rng = np.random.default_rng(100 + run)
    T = 0.55 + 0.05 * np.arange(4)[None, :] + 0.03 * rng.standard_normal((10, 4))
    R = -0.7 + 0.1 * np.arange(4)[None, :] + 0.05 * rng.standard_normal((10, 4))
    L = 68.0 + rng.standard_normal((10, 4))
    if flip is not None:
        i, j = flip
        T[i, j] = 1.0 - T[i, j]
    return T, R, L


def _fixture(tmp_path: Path, flip: tuple | None = None) -> tuple[Path, dict]:
    curve = {"generated_by": "scripts/build_budget_curve.py", "runs": {}, "drift": {}, "series": {}}
    runs = {}
    for k, (name, label) in enumerate((("run_wm", "decoder on"), ("run_nowm", "decoder off"))):
        T, R, L = _values(k, flip if k == 0 else None)
        run_dir = tmp_path / name
        (run_dir / "cells").mkdir(parents=True)
        for i, s in enumerate(SEEDS):
            bc = {str(u): {"env_steps": 1000 * u, "pool": {"target": float(T[i, j])},
                           "eng": {"trained_return": float(R[i, j]), "trained_len": float(L[i, j])}}
                  for j, u in enumerate(bcd.BUDGETS)}
            cell = {"drift": 0.45, "seed": s, "budget_curve": bc}
            (run_dir / "cells" / f"cell_d0.45_s{s}.json").write_text(json.dumps({"cell": cell}))
        curve["runs"][label] = f"artifacts/corrected_runs/{name}"
        curve["drift"][label] = 0.45
        curve["series"][label] = [{"updates": u, "seeds": SEEDS,
                                   "target": {"per_seed": T[:, j].tolist()},
                                   "return": {"per_seed": R[:, j].tolist()}}
                                  for j, u in enumerate(bcd.BUDGETS)]
        runs[name.split("_")[1]] = str(run_dir)
    path = tmp_path / "budget_curve.json"
    path.write_text(json.dumps(curve))
    return path, runs


def _read(tmp_path: Path, flip=None) -> dict:
    sub = tmp_path / ("flip" if flip else "base")
    sub.mkdir()
    curve, runs = _fixture(sub, flip)
    out = sub / "out.json"
    argv = ["--budget-curve", str(curve), "--out", str(out)]
    for name, d in runs.items():
        argv += ["--run", f"{name}={d}"]
    assert bcd.main(argv) == 0
    return json.loads(out.read_text())


def test_tmp_fixture_reads_and_flipped_value_moves_every_slope(tmp_path):
    base = _read(tmp_path)
    assert "git_commit" in base["provenance"]
    assert "git_commit_at_promotion" not in base["provenance"]
    assert set(base["runs"]) == {"wm", "nowm"}
    wm = base["runs"]["wm"]
    assert wm["series_label"] == "decoder on"
    assert wm["label"] in ("tracks return", "tracks budget", "both", "neither")
    assert set(wm["segments"]["target"]) == {"100->200", "200->300", "300->450"}
    flipped_doc = _read(tmp_path, flip=(3, 2))
    flipped = flipped_doc["runs"]["wm"]
    for key in ("return", "updates_per_100"):
        assert flipped["partial_slopes"][key]["slope"] != wm["partial_slopes"][key]["slope"], key
    # the untouched run reads the same
    assert flipped_doc["runs"]["nowm"]["partial_slopes"] == base["runs"]["nowm"]["partial_slopes"]


def test_input_gate_refuses_an_artifact_that_differs_from_the_cells(tmp_path):
    curve, runs = _fixture(tmp_path)
    doc = json.loads(curve.read_text())
    doc["series"]["decoder on"][1]["target"]["per_seed"][0] += 1e-12
    curve.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="not bit-equal"):
        bcd.main(["--budget-curve", str(curve), "--out", str(tmp_path / "o.json"),
                  "--run", f"wm={runs['wm']}"])
    assert not (tmp_path / "o.json").exists()
