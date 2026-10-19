"""scripts/compare_run_cells.py (G59): the per-value comparator behind artifacts/repro_check,
moved out of the audit so a reader can run it on a reproduced run. It must reproduce the stored
report's entries, honor --tolerance, and fail on a moved value, a missing leaf, a changed
fingerprint or a file with no committed counterpart."""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import compare_run_cells as crc  # noqa: E402

REP = os.path.join(ROOT, "artifacts", "repro_check")


def _cell(fp="abcd", **vals):
    base = {"drift": "0.45", "seed": 0, "agents": {"survival": {"target": 0.7, "ok": True, "nan": float("nan")}}}
    base["agents"]["survival"].update(vals)
    return {"fingerprint": fp, "git_commit": "2c170f0", "cell": base}


def _write(d, name, doc):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(doc), encoding="utf-8")


def test_the_stored_report_is_what_the_tool_gives():
    if not os.path.isdir(os.path.join(ROOT, "artifacts", "reviewer_gaps_runs", "l3_h8_wm_cpu", "cells")):
        pytest.skip("the committed cells are not in this tree")
    with open(os.path.join(REP, "per_value_4b6e1f3_l3_h8_wm_cpu_s0.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    rep = crc.compare(os.path.join(REP, "rerun_cells"),
                      os.path.join(ROOT, "artifacts", "reviewer_gaps_runs", "l3_h8_wm_cpu", "cells"))
    assert rep["cells"] == stored["cells"]
    assert rep["commit_rerun"] == stored["commit_rerun"] == "4b6e1f3"
    assert rep["all_shared_values_identical"] is stored["all_shared_values_identical"] is True
    assert rep["pass"] is True


def test_identical_dirs_pass_and_nan_equals_nan(tmp_path):
    for side in ("a", "b"):
        _write(tmp_path / side, "cell_d0.45_s0.json", _cell())
    rep = crc.compare(str(tmp_path / "a"), str(tmp_path / "b"))
    e = rep["cells"]["cell_d0.45_s0.json"]
    assert rep["pass"] and e["n_differ"] == 0 and e["shared_kinds"] == {"string": 1, "number": 3, "bool": 1, "nan": 1}
    assert crc.main(["--rerun", str(tmp_path / "a"), "--committed", str(tmp_path / "b")]) == 0


def test_a_moved_value_fails_bit_exact_and_passes_within_tolerance(tmp_path, capsys):
    _write(tmp_path / "a", "c.json", _cell(target=0.7 + 1e-4))
    _write(tmp_path / "b", "c.json", _cell())
    rep = crc.compare(str(tmp_path / "a"), str(tmp_path / "b"))
    d = rep["cells"]["c.json"]["differences"]
    assert not rep["pass"] and len(d) == 1 and d[0]["path"] == "/agents/survival/target"
    assert d[0]["abs_diff"] == pytest.approx(1e-4)
    loose = crc.compare(str(tmp_path / "a"), str(tmp_path / "b"), tol=1e-3)
    assert loose["pass"] and loose["cells"]["c.json"]["n_differ"] == 1
    assert loose["cells"]["c.json"]["max_abs_diff"] == pytest.approx(1e-4)
    assert not loose["all_shared_values_identical"]
    out = tmp_path / "rep.json"
    assert crc.main(["--rerun", str(tmp_path / "a"), "--committed", str(tmp_path / "b"), "--tolerance", "1e-5"]) == 1
    assert crc.main(["--rerun", str(tmp_path / "a"), "--committed", str(tmp_path / "b"), "--tolerance", "1e-3",
                     "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["tolerance"] == 1e-3 and doc["pass"] is True and "stack" in doc
    capsys.readouterr()


def test_a_bool_never_matches_a_number_and_nan_never_matches_a_number():
    assert not crc.same_value(True, 1)
    assert not crc.same_value(float("nan"), 0.5, tol=10)
    assert crc.same_value(float("nan"), float("nan"))
    assert crc.same_value(0.5, 0.5 + 1e-9, tol=1e-6) and not crc.same_value(0.5, 0.5 + 1e-9)


def test_missing_leaves_fingerprints_and_files_fail(tmp_path):
    a, b = _cell(extra=1.0), _cell(fp="ffff")
    _write(tmp_path / "a", "c.json", a)
    _write(tmp_path / "a", "only_rerun.json", {"x": 1})
    _write(tmp_path / "b", "c.json", b)
    rep = crc.compare(str(tmp_path / "a"), str(tmp_path / "b"), tol=1.0)
    e = rep["cells"]["c.json"]
    assert e["only_in_rerun"] == ["/agents/survival/extra"] and not e["fingerprint_match"]
    assert rep["files_only_in_rerun"] == ["only_rerun.json"] and not rep["pass"]


def test_a_readout_json_without_a_cell_block_is_compared_whole(tmp_path):
    _write(tmp_path / "a" / "l0_audit", "x.json", {"summary": {"mean": 0.51, "per_seed": [0.5, 0.52]}})
    _write(tmp_path / "b" / "l0_audit", "x.json", {"summary": {"mean": 0.51, "per_seed": [0.5, 0.53]}})
    rep = crc.compare(str(tmp_path / "a"), str(tmp_path / "b"), tol=0.02)
    e = rep["cells"]["l0_audit/x.json"]
    assert e["fingerprint_match"] and e["n_shared_leaves"] == 3 and e["n_differ"] == 1 and rep["pass"]


def test_the_audit_uses_this_comparator():
    with open(os.path.join(ROOT, "scripts", "audit_stats_recheck.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "from compare_run_cells import compare_docs" in src
    assert "def flat(" not in src
