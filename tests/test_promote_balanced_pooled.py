"""scripts/promote_balanced_pooled.py (H01): the B1 and B2 pooled readings on the registered
world pair, copied from the cell records into one artifact, byte-stable on a fixed cell set."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import promote_balanced_pooled as pbp  # noqa: E402

from itasorl.stats import equivalence_test, t_ci90  # noqa: E402

BASE = {"survival": {0.0: 0.56, 0.45: 0.76}, "predictor": {0.0: 0.50, 0.45: 0.58},
        "untrained": {0.0: 0.50, 0.45: 0.52}}


def _target(run_id, arm, drift, seed):
    bump = 0.08 if (run_id == "B2" and arm == "survival" and drift == 0.45) else 0.0
    return BASE[arm][drift] + bump + 0.003 * (seed - 14) + (0.01 if seed % 3 == 0 else -0.004)


def _write_runs(tmp_path, alter=None):
    runs = {}
    for run_id in ("B1", "B2"):
        d = tmp_path / "fullruns" / f"balanced_{run_id.lower()}" / "cells"
        d.mkdir(parents=True)
        for drift in pbp.DRIFTS:
            for seed in pbp.SEEDS:
                cell = {"drift": drift, "seed": seed, "objective": "survival", "mortal": True,
                        "stack": {"device": "cpu"},
                        "agents": {a: {"pool": {"target": _target(run_id, a, drift, seed)}} for a in pbp.ARMS}}
                doc = {"fingerprint": f"fp{run_id}", "git_commit": "2c170f0", "cell": cell}
                if alter:
                    alter(run_id, drift, seed, doc)
                (d / f"cell_d{drift:.2f}_s{seed}.json").write_text(json.dumps(doc), encoding="utf-8")
        runs[run_id] = str(d.parent)
    return runs


def _args(runs, out):
    return ["--run", f"B1={runs['B1']}", "--run", f"B2={runs['B2']}", "--out", str(out)]


def test_the_artifact_is_byte_stable_on_a_fixed_cell_set(tmp_path):
    runs = _write_runs(tmp_path)
    prov = {"generated_by": pbp.GENERATED_BY, "git_commit_at_promotion": "abc1234",
            "tracked_changes_at_generation": [], "stack": {"python": "x"}}
    a = pbp.dumps(pbp.build(runs, prov))
    b = pbp.dumps(pbp.build(runs, prov))
    assert a == b and a.endswith("}\n")
    # two writes through main give the same bytes
    out1, out2 = tmp_path / "o1.json", tmp_path / "o2.json"
    assert pbp.main(_args(runs, out1)) == 0
    assert pbp.main(_args(runs, out2)) == 0
    assert out1.read_bytes() == out2.read_bytes()
    assert json.loads(out1.read_text(encoding="utf-8"))["runs"] == json.loads(a)["runs"]


def test_every_reading_is_the_cell_value_and_its_summary(tmp_path):
    runs = _write_runs(tmp_path)
    doc = pbp.build(runs, {"generated_by": pbp.GENERATED_BY})
    assert doc["scope"].startswith("descriptive") and "decides nothing" in doc["scope"]
    for run_id in ("B1", "B2"):
        b = doc["runs"][run_id]
        assert b["git_commit_at_run"] == "2c170f0" and b["fingerprint"] == f"fp{run_id}"
        assert b["source_run"] == f"fullruns/balanced_{run_id.lower()}"
        for drift in pbp.DRIFTS:
            for arm in pbp.ARMS:
                s = b["pool_target"][f"{drift:.2f}"][arm]
                want = [_target(run_id, arm, drift, seed) for seed in pbp.SEEDS]
                assert s["per_seed"] == want
                assert s["mean"] == float(np.mean(want)) and s["n"] == 10
                assert s["t90"] == [float(x) for x in t_ci90(want)]
                assert s["n_ge_065"] == sum(v >= 0.65 for v in want)
        l0 = [_target(run_id, "survival", 0.0, seed) for seed in pbp.SEEDS]
        assert b["l0_tost"]["p_value"] == float(equivalence_test(l0, 0.5, margin=0.05).p_value)
        m = b["margins_0.45"]["over_untrained"]
        assert m["mean"] == pytest.approx(np.mean([_target(run_id, "survival", 0.45, s)
                                                   - _target(run_id, "untrained", 0.45, s)
                                                   for s in pbp.SEEDS]))
        name = "cell_d0.45_s13.json"
        raw = open(os.path.join(runs[run_id], "cells", name), "rb").read()
        assert b["cells_sha256"][name] == hashlib.sha256(raw).hexdigest()
        assert len(b["cells_sha256"]) == 20
    assert doc["baselines_identical_across_runs"] == {"predictor": True, "untrained": True}


def test_dry_run_writes_nothing_and_check_catches_a_changed_cell(tmp_path, capsys):
    runs = _write_runs(tmp_path)
    out = tmp_path / "artifacts" / "balanced_pooled" / "registered_pair.json"
    assert pbp.main(_args(runs, out) + ["--dry-run"]) == 0
    assert not out.exists() and "B1 d0.45 survival" in capsys.readouterr().out
    assert pbp.main(_args(runs, out) + ["--check"]) == 1
    assert pbp.main(_args(runs, out)) == 0
    assert pbp.main(_args(runs, out) + ["--check"]) == 0
    p = os.path.join(runs["B2"], "cells", "cell_d0.00_s12.json")
    doc = json.load(open(p, encoding="utf-8"))
    doc["cell"]["agents"]["survival"]["pool"]["target"] += 0.001
    open(p, "w", encoding="utf-8").write(json.dumps(doc))
    assert pbp.main(_args(runs, out) + ["--check"]) == 1


@pytest.mark.parametrize("what,alter,match", [
    ("fingerprint", lambda r, d, s, doc: doc.update(fingerprint="other") if (r, d, s) == ("B1", 0.45, 15) else None,
     "more than one fingerprint"),
    ("git_commit", lambda r, d, s, doc: doc.pop("git_commit") if (r, d, s) == ("B2", 0.0, 11) else None,
     r"cell_d0\.00_s11\.json: cell file has no top-level git_commit"),
    ("nan", lambda r, d, s, doc: doc["cell"]["agents"]["predictor"]["pool"].update(target=float("nan"))
     if (r, d, s) == ("B1", 0.0, 19) else None, "not finite"),
    ("seed", lambda r, d, s, doc: doc["cell"].update(seed=20) if (r, d, s) == ("B2", 0.45, 19) else None,
     "differs from the registered seeds"),
])
def test_a_bad_cell_set_is_refused(tmp_path, what, alter, match):
    runs = _write_runs(tmp_path, alter)
    with pytest.raises(ValueError, match=match):
        pbp.build(runs, {"generated_by": pbp.GENERATED_BY})


def test_a_missing_cell_is_refused(tmp_path):
    runs = _write_runs(tmp_path)
    os.remove(os.path.join(runs["B1"], "cells", "cell_d0.00_s10.json"))
    with pytest.raises(ValueError, match=r"missing \[\(0\.0, 10\)\]"):
        pbp.build(runs, {"generated_by": pbp.GENERATED_BY})
