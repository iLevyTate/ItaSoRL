"""The PB1/PB2 replication reader (scripts/read_pb_replication.py) on synthetic run folders.

It applies the frozen promotion code to a rerun's own folder, sets the result beside the
committed decision, refuses the recorded run folders and any output under artifacts/, and
leaves the committed record untouched.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import read_pb_replication as rpr  # noqa: E402

# The synthetic run builder of the promotion's own tests.
_spec = importlib.util.spec_from_file_location(
    "_ppb_fixture", ROOT / "tests" / "test_promote_prospective_balanced.py")
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)

RECORD = ROOT / "artifacts" / "prospective_balanced"


def _rerun(tmp_path, run_id="PB1", fingerprint=None, **kw):
    rd = _fx.make_run(tmp_path, run_id, **kw)
    fp = fingerprint or json.loads((RECORD / f"{run_id.lower()}.json").read_text())["fingerprint"]
    for p in (rd / "cells").glob("*.json"):
        doc = json.loads(p.read_text())
        doc["fingerprint"] = fp
        p.write_text(json.dumps(doc))
    return rd


def _digest():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RECORD.glob("*.json"))}


def test_a_complete_rerun_reads_beside_the_record(tmp_path):
    rd = _rerun(tmp_path)
    before = dict(rpr.ppb.RUNS["PB1"])
    r = rpr.read("PB1", str(rd))
    assert rpr.ppb.RUNS["PB1"] == before                  # the promoter's table is restored
    assert r["decision"] == "MET", r["reason"]
    assert r["same_configuration"] is True
    assert r["committed"]["decision"] == "MET"
    assert r["cells_found"] == r["cells_expected"] == 20
    assert "replication" in r["label"] and "unchanged" in r["label"]
    text = "\n".join(r["lines"])
    assert "decision under the frozen rule: MET" in text
    assert "this reading does not change it" in text
    assert "configuration: the recorded one" in text


def test_a_floor_failure_reads_uninformative_and_the_record_still_reads_met(tmp_path):
    r = rpr.read("PB2", str(_rerun(tmp_path, "PB2", untrained_resid=0.62)))
    assert r["decision"] == "UNINFORMATIVE"
    assert r["committed"]["decision"] == "MET"


def test_another_configuration_is_named_as_not_a_rerun(tmp_path):
    r = rpr.read("PB1", str(_rerun(tmp_path, fingerprint="0000000000000000")))
    assert r["same_configuration"] is False
    assert "DIFFERS" in "\n".join(r["lines"])


def test_missing_readout_steps_read_incomplete(tmp_path):
    rd = _rerun(tmp_path)
    os.remove(rd / "matched_gates.json")
    r = rpr.read("PB1", str(rd))
    assert r["decision"] == "INCOMPLETE"
    assert "matched_gates.json is missing" in r["reason"]


def test_a_partial_run_reads_not_run(tmp_path):
    r = rpr.read("PB1", str(tmp_path / "nothing_yet"))
    assert r["decision"] == "NOT RUN" and r["cells_found"] == 0
    assert "not readable yet" in "\n".join(r["lines"])


@pytest.mark.parametrize("argv", [
    ["--run", "PB1", "--dir", str(ROOT / "fullruns" / "prospective_balanced_wm")],
    ["--run", "PB2", "--dir", str(ROOT / "fullruns" / "prospective_balanced_touch" / "cells")],
    ["--run", "PB1", "--dir", "elsewhere", "--out", str(ROOT / "artifacts" / "pb1_replication.json")],
])
def test_recorded_folders_and_artifacts_are_refused(argv):
    with pytest.raises(SystemExit) as exc:
        rpr.main(argv)
    assert exc.value.code == 2


def test_writes_only_where_asked_and_leaves_the_record_alone(tmp_path, capsys):
    before = _digest()
    out = tmp_path / "reading" / "replication_reading.json"
    assert rpr.main(["--run", "PB1", "--dir", str(_rerun(tmp_path)), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["decision"] == "MET" and doc["lines"]
    assert "decision under the frozen rule: MET" in capsys.readouterr().out
    assert _digest() == before
