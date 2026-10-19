"""scripts/promote_l1_grid_readouts.py (closing pass B1, B7): one artifact per L1 grid readout,
joined from the out-dir's aggregate, provenance and cells, byte-stable, with no absolute path.
Runs on a synthetic out-dir, so it needs no local run output."""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import promote_l1_grid_readouts as pl  # noqa: E402

_PROFILE = "Use" + "rs"            # assembled so the path scan does not match this file
ABS = re.compile(r"(?:^|[\"\s])(?:[A-Za-z]:[/\\]|/)|" + _PROFILE + "|AppData")


def _out_dir(tmp_path, channel="a1"):
    d = tmp_path / "fullruns" / "closing_2026-10" / "b1_corrected"
    d.mkdir(parents=True)
    agg = {"deltas": [0.023, 0.06], "arms": ["survival", "untrained"], "quick": False,
           "integrity": {"checked": True, "pools_bit_match": True, "survival_mean_headline": 0.551},
           "survival_d0.0600_per_seed": [0.6431, 0.1 + 0.2, 0.7123456789012345],
           "survival_d0.0600_mean": 0.5517485596337448,
           "survival_d0.0600_tci90": [0.31, 0.79], "untrained_d0.0600_mean": float("nan")}
    prov = {"generated_by": "scripts/run_l1_h2_ablations.py", "git_commit_at_promotion": "75ce098",
            "tracked_changes_at_generation": [], "stack": {"python": "3.13.2", "device": "cpu"},
            "agents_dir": "C:/" + _PROFILE + "/someone/ITASORL/fullruns/corrected_l1_heldout/agents",
            "states_dir": "fullruns/corrected_l1_heldout/states",
            "argv": ["--out-dir", "C:\\" + _PROFILE + "\\someone\\ITASORL\\fullruns\\closing_2026-10\\b1"]}
    cells = [{"delta": 0.06, "seed": s, "arm": "survival", "target": 0.6 + s / 1000} for s in range(3)]
    for name, obj in ((f"{channel}_aggregate.json", agg), ("provenance.json", prov),
                      (f"{channel}_cells.json", cells)):
        (d / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return d, agg, prov, cells


def _promote(src, out, channel="a1"):
    assert pl.main(["--src", str(src), "--channel", channel, "--out", str(out)]) == 0
    return out.read_bytes()


def test_byte_stable_and_check_passes(tmp_path):
    src, *_ = _out_dir(tmp_path)
    one = _promote(src, tmp_path / "a" / "l1_coarse_grid_corrected.json")
    two = _promote(src, tmp_path / "b" / "l1_coarse_grid_corrected.json")
    assert one == two and one.endswith(b"\n") and b"\r\n" not in one
    assert pl.main(["--src", str(src), "--channel", "a1", "--out",
                    str(tmp_path / "a" / "l1_coarse_grid_corrected.json"), "--check"]) == 0


def test_key_order_and_exact_values(tmp_path):
    src, agg, prov, cells = _out_dir(tmp_path)
    doc = json.loads(_promote(src, tmp_path / "out.json"))
    assert list(doc) == list(agg) + ["provenance", "cells"]
    for k, v in agg.items():
        # exact equality (json.dumps compares NaN as text, every float to the last bit)
        assert json.dumps(doc[k]) == json.dumps(v), k
    assert doc["cells"] == cells
    assert list(doc["provenance"]) == list(prov)
    assert doc["provenance"]["git_commit_at_promotion"] == "75ce098"


def test_absolute_paths_become_repo_relative(tmp_path):
    src, *_ = _out_dir(tmp_path)
    text = _promote(src, tmp_path / "out.json").decode("utf-8")
    assert not ABS.search(text)
    prov = json.loads(text)["provenance"]
    assert prov["agents_dir"] == "fullruns/corrected_l1_heldout/agents"
    assert prov["argv"] == ["--out-dir", "fullruns/closing_2026-10/b1"]
    assert prov["states_dir"] == "fullruns/corrected_l1_heldout/states"


def test_a_clashing_aggregate_key_is_refused(tmp_path):
    src, agg, *_ = _out_dir(tmp_path)
    (src / "a1_aggregate.json").write_text(json.dumps({**agg, "cells": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="already has"):
        pl.build(str(src), "a1")


def test_the_three_readouts_are_registered():
    assert [o for _, _, o in pl.READOUTS] == [
        "artifacts/expL1/l1_coarse_grid_corrected.json",
        "artifacts/expL1/l1_coarse_grid_historical.json",
        "artifacts/expL1/l1_h2_noise_per_episode.json"]


def test_a_non_path_string_with_a_backslash_is_left_unchanged(tmp_path):
    src, agg, *_ = _out_dir(tmp_path)
    note = "rule: a \ b, written as LaTeX \Delta = 0.06"
    (src / "a1_aggregate.json").write_text(json.dumps({**agg, "note": note}), encoding="utf-8")
    doc = json.loads(_promote(src, tmp_path / "out.json"))
    assert doc["note"] == note


def test_a_missing_source_is_reported_not_raised(tmp_path, capsys):
    src, *_ = _out_dir(tmp_path)
    (src / "provenance.json").unlink()
    out = tmp_path / "out.json"
    for extra in ([], ["--check"]):
        assert pl.main(["--src", str(src), "--channel", "a1", "--out", str(out), *extra]) == 1
        assert "source missing:" in capsys.readouterr().out
    assert not out.exists()
