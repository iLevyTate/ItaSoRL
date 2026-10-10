"""Engagement-margin sweep: the objective guard and the committed GPU-run record.

artifacts/engagement_margin_gpu_runs.json is written from the gitignored GPU bundles (and three
committed cell directories plus the L1 organism summary) with --with-inputs, so every number
the manuscript quotes from it is recomputed here from the committed inputs alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_engagement_margin as aem  # noqa: E402

GPU = ROOT / "artifacts" / "engagement_margin_gpu_runs.json"


def _eng(trained, objective=None):
    e = {"trained_return": trained, "random_return": -1.0, "scripted_return": -1.2,
         "trained_len": 70.0, "random_len": 69.0}
    if objective:
        e["objective"] = objective
    return e


def test_readjudicate_refuses_a_non_survival_objective():
    assert aem.readjudicate(_eng(-0.5), 0.15)
    with pytest.raises(ValueError):
        aem.readjudicate(_eng(-0.5, "touch"), 0.15)


def test_touch_cells_are_skipped_and_counted_not_swept(tmp_path):
    cells = tmp_path / "cells"
    cells.mkdir()
    for s, obj in enumerate(("survival", "survival", "touch")):
        (cells / f"cell_d0.45_s{s}.json").write_text(json.dumps(
            {"cell": {"drift": 0.45, "seed": s, "eng": _eng(-0.5, obj)}}), encoding="utf-8")
    rep = aem.summarize(str(cells), with_inputs=True)
    assert rep["n_cells"] == 2 and rep["n_skipped_objective"] == 1
    assert rep["objective"] == "survival"
    assert rep["by_margin"]["0.15"]["0.45"] == {"pass": 2, "n": 2, "min_gap": 0.5}
    assert [r["seed"] for r in rep["inputs"]] == [0, 1]


def test_the_l1_grid_keeps_its_precision_and_the_registered_drifts_keep_two_decimals():
    assert aem.drift_key(0.0) == "0.00" and aem.drift_key(0.45) == "0.45"
    assert aem.drift_key(0.023) == "0.023"


def test_committed_sweeps_reproduce_from_committed_cells():
    for out, srcs in (("engagement_margin_corrected.json",
                       ["corrected_runs/corrected_l3_h8_wm/cells",
                        "corrected_runs/corrected_l3_h8_nowm/cells"]),
                      ("expB2/engagement_margin_cloud_runs.json",
                       ["reviewer_gaps_runs/l3_h8_nowm/cells", "reviewer_gaps_runs/l3_h10_gseed1/cells",
                        "reviewer_gaps_runs/l3_h8_wm_cpu/cells"])):
        have = json.loads((ROOT / "artifacts" / out).read_text(encoding="utf-8"))
        want = [aem.summarize(str(ROOT / "artifacts" / s)) for s in srcs]
        assert have == json.loads(json.dumps(want)), out


def _gpu():
    return {r.get("cells_dir", r.get("source")): r for r in json.loads(GPU.read_text(encoding="utf-8"))}


def test_every_gpu_report_recomputes_from_its_recorded_inputs():
    for src, rep in _gpu().items():
        assert rep["n_cells"] == len(rep["inputs"]) == 20, src
        assert json.loads(json.dumps(aem.sweep(aem.rows_from_report(rep)))) == rep["by_margin"], src


def test_the_gpu_margins_the_manuscript_quotes():
    g = _gpu()
    h8 = g["fullruns/l3_h8_heldout/cells"]["by_margin"]
    h7 = g["fullruns/l3_h7_heldout/cells"]["by_margin"]
    l1 = g["artifacts/expL1/organism_summary.json"]["by_margin"]
    assert round(h8["0.30"]["0.45"]["min_gap"], 3) == 0.417
    assert all(h8[m][d]["pass"] == 10 for m in h8 for d in h8[m])
    assert round(h7["0.30"]["0.45"]["min_gap"], 3) == 0.286
    assert h7["0.25"]["0.45"]["pass"] == 10 and h7["0.30"]["0.45"]["pass"] == 9
    assert round(l1["0.30"]["0.023"]["min_gap"], 3) == 0.520
    assert all(l1[m][d]["pass"] == 10 for m in l1 for d in l1[m])
    # The three GPU runs whose cells are committed pass at every margin.
    for run in ("l3_h10_gseed1_gpu", "l3_h8_gseed2_gpu", "l3_h8_nowm_skill_u450"):
        bm = g[f"artifacts/reviewer_gaps_runs/{run}/cells"]["by_margin"]
        assert all(bm[m][d]["pass"] == bm[m][d]["n"] == 10 for m in bm for d in bm[m]), run
