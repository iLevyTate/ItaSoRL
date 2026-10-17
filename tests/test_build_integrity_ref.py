"""scripts/build_integrity_ref.py (G45, G56, G02) and the committed C1 integrity reference.

The reference is read by reverse_direction._reference and
run_world_sample_sensitivity._reference in the artifacts/fold_rescore schema; the dry checks
of the closing runbook (1.6) are pinned here on the committed file."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_integrity_ref as bir  # noqa: E402
import reverse_direction as rd  # noqa: E402
import run_world_sample_sensitivity as ws  # noqa: E402

REF = os.path.join(ROOT, "artifacts", "fold_rescore", "corrected_l3_h8_wm_integrity_ref.json")
ARMS = ("survival", "predictor", "untrained")


def _target(drift, seed, arm):
    return 0.5 + 0.2 * drift + 0.01 * seed + 0.001 * ARMS.index(arm)


def _cells(tmp_path, drifts=(0.0, 0.45), seeds=(0, 1, 2)):
    d = tmp_path / "fullruns" / "run" / "cells"
    d.mkdir(parents=True)
    for drift in drifts:
        for s in seeds:
            cell = {"drift": drift, "seed": s,
                    "agents": {a: {"pool": {"target": _target(drift, s, a)}} for a in ARMS}}
            (d / f"cell_d{drift:.2f}_s{s}.json").write_text(
                json.dumps({"fingerprint": "f", "git_commit": "c", "cell": cell}), encoding="utf-8")
    return d


def test_the_builder_copies_each_target_in_the_fold_rescore_schema(tmp_path):
    d = _cells(tmp_path)
    out = tmp_path / "ref.json"
    assert bir.main(["--cells-dir", str(d), "--drifts", "0.00", "0.45", "--arms", *ARMS,
                     "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["generated_by"] == "scripts/build_integrity_ref.py" and "git_commit_at_promotion" in doc
    assert doc["cells_dir"] == "fullruns/run/cells" and doc["n_cells"] == 18
    first = doc["cells"][0]
    assert first == {"drift": "0.00", "seed": 0, "agent": "survival",
                     "explicit": {"target": _target(0.0, 0, "survival")},
                     "source": "fullruns/run/cells/cell_d0.00_s0.json"}
    assert {(c["drift"], c["seed"], c["agent"]) for c in doc["cells"]} == {
        (t, s, a) for t in ("0.00", "0.45") for s in (0, 1, 2) for a in ARMS}
    # the two consumers read it as their integrity reference
    assert len(rd._reference(str(out), ARMS)) == 9
    assert ws._reference(str(out), 0.45, "untrained") == {s: _target(0.45, s, "untrained") for s in (0, 1, 2)}


def test_dry_run_writes_nothing(tmp_path):
    d = _cells(tmp_path)
    out = tmp_path / "ref.json"
    assert bir.main(["--cells-dir", str(d), "--drifts", "0.45", "--arms", "survival",
                     "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_a_drift_without_cells_is_a_hard_error(tmp_path):
    d = _cells(tmp_path)
    with pytest.raises(ValueError, match="no cell file for drift 0.0230"):
        bir.build(str(d), ["0.023"], ARMS)
    with pytest.raises(ValueError, match=r"seed\(s\) \[3\]"):
        bir.build(str(d), ["0.45"], ARMS, seeds=[0, 1, 2, 3])


def test_a_non_finite_target_is_refused(tmp_path):
    d = _cells(tmp_path)
    p = d / "cell_d0.45_s1.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["cell"]["agents"]["predictor"]["pool"]["target"] = float("nan")
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="predictor pooled target is not finite"):
        bir.build(str(d), ["0.45"], ARMS)


# --- the committed C1 reference: the runbook 1.6 dry checks ---------------------------------

def test_committed_c1_reference_dry_checks(monkeypatch):
    monkeypatch.delenv("ITASORL_FOLDS", raising=False)
    from itasorl import folds
    assert folds.current_scheme() == "explicit"
    assert len(rd._reference(REF, rd.ARMS)) == 30
    w = ws._reference(REF, 0.45, "survival")
    assert len(w) == 10 and round(float(np.mean(list(w.values()))), 4) == 0.7330
    with open(REF, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["cells_dir"] == "fullruns/corrected_l3_h8_wm/cells" and doc["n_cells"] == 60
    assert all(c["source"].startswith("fullruns/corrected_l3_h8_wm/cells/cell_d") for c in doc["cells"])
