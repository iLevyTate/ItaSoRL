"""R31, L1 part (docs/specs/2026-10-11-post-review-readouts-design.md, amendment of
2026-10-11): scripts/run_l1_capacity_curve.py. No rollout and no torch training: the pools,
the decoders and the agent loader are monkeypatched, and the checks read synthetic rows or
the committed record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_l1_capacity_curve as l1c  # noqa: E402
from itasorl import post_review as pr  # noqa: E402


def _ref_and_record():
    ref = l1c.load_integrity_ref(json.loads((ROOT / l1c.INTEGRITY_REF).read_text(encoding="utf-8")), 0.023)
    rec = l1c.load_record(json.loads((ROOT / l1c.RECORD_REF).read_text(encoding="utf-8")), 0.023)
    return ref, rec


def test_reference_loaders_read_the_committed_record():
    ref, rec = _ref_and_record()
    assert len(ref) == len(rec) == 30
    assert all(rec[k]["target"] == ref[k] for k in rec)           # 14.7.1: the gate held exactly
    surv = [rec[("survival", str(s))]["seq_gru"] for s in range(10)]
    assert np.mean(surv) == pytest.approx(0.5900413223140496)


def _rows(arms=("survival",), seeds=range(10), widths=("1x",), auc=None, target=None):
    ref, rec = _ref_and_record()
    rows = {}
    for a in arms:
        for s in map(str, seeds):
            for w in widths:
                rows[(a, s, w)] = {"target": ref[(a, s)] if target is None else target,
                                   "dump_bit_match": None,
                                   "decoder": {"auroc": rec[(a, s)]["seq_gru"] if auc is None else auc(a, s, w)}}
    return rows, ref, rec


def test_integrity_gate_and_record_check_on_the_recorded_values():
    rows, ref, rec = _rows()
    g = l1c.integrity(rows, ref)
    assert g["pass"] and g["n_checked"] == 10 and g["n_bit_equal"] == 10 and g["worst_abs_dev"] == 0.0
    assert g["dumps"].startswith("not checked")
    rc = l1c.record_check(rows, rec)
    assert rc["survival_1x_mean_reproduced"] is True and rc["n_bit_equal"] == 10
    off, _, _ = _rows(auc=lambda a, s, w: rec[(a, s)]["seq_gru"] + 0.02)
    assert l1c.record_check(off, rec)["survival_1x_mean_reproduced"] is False
    few = {k: v for k, v in rows.items() if int(k[1]) < 5}
    assert l1c.record_check(few, rec)["survival_1x_mean_reproduced"] is None


def test_integrity_gate_fails_on_drift_inconsistency_or_an_unreferenced_cell():
    rows, ref, _ = _rows()
    bad = dict(rows)
    bad[("survival", "0", "1x")] = {**bad[("survival", "0", "1x")], "target": ref[("survival", "0")] + 0.02}
    assert l1c.integrity(bad, ref)["pass"] is False
    twice = dict(rows)
    twice[("survival", "0", "2x")] = {**rows[("survival", "0", "1x")], "target": ref[("survival", "0")] + 1e-12}
    g = l1c.integrity(twice, ref)
    assert g["pass"] is False and g["inconsistent_regenerations"][0]["seed"] == "0"
    extra = dict(rows)
    extra[("survival", "42", "1x")] = {"target": 0.5, "dump_bit_match": True}
    g = l1c.integrity(extra, ref)
    assert g["pass"] is False and g["unreferenced"] == ["survival/42"] and g["dumps"] == {"checked": 1, "bit_match": 1}


def test_reading_labels():
    lift = {"1x": 0.0, "2x": 0.01, "4x": 0.08, "8x": 0.12, "conv": 0.0}
    rng = np.random.default_rng(0)
    noise = {(s, w): rng.normal(0, 0.005) for s in map(str, range(10)) for w in lift}
    rows, _, _ = _rows(widths=tuple(lift), auc=lambda a, s, w: 0.59 + lift[w] + noise[(s, w)])
    s = l1c.summarize(rows, ["survival"])["survival"]
    assert s["reading"] == "SEAM AVAILABLE AT WIDTH"
    assert s["decoders"]["8x"]["mean"] == pytest.approx(0.71, abs=0.01)
    assert s["state_minus"]["1x"]["n"] == 10
    flat, _, _ = _rows(widths=tuple(lift), auc=lambda a, s, w: 0.59 + noise[(s, w)])
    assert l1c.summarize(flat, ["survival"])["survival"]["reading"] == "READER-RELATIVE"
    part, _, _ = _rows(widths=("1x", "2x"), auc=lambda a, s, w: 0.59)
    assert l1c.summarize(part, ["survival"])["survival"]["reading"] == "INCOMPLETE"


def test_plan_tasks_runs_the_widest_first_and_skips_finished_tasks():
    todo = l1c.plan_tasks(["survival", "untrained"], [0, 1], ["1x", "8x", "conv"], {("survival/8x", "0"): {}})
    assert todo[:3] == [("survival", "1", "8x"), ("untrained", "0", "8x"), ("untrained", "1", "8x")]
    assert [t[2] for t in todo] == ["8x"] * 3 + ["1x"] * 4 + ["conv"] * 4


def _fake_run(tmp_path, seeds=(0, 1)):
    run = tmp_path / "l1_heldout"
    (run / "agents").mkdir(parents=True)
    (run / "cells").mkdir()
    for tag in ("0.00", "0.0230"):
        for s in seeds:
            for a in pr.ARMS:
                (run / "agents" / f"agent_d{tag}_s{s}_{a}.pt").write_bytes(b"")
            (run / "cells" / f"cell_d{tag}_s{s}.json").write_text(
                json.dumps({"fingerprint": f"fp{tag}", "cell": {"drift": float(tag), "seed": s}}), encoding="utf-8")
    return run


def test_discover_takes_the_strongest_drift(tmp_path):
    run = l1c.discover(str(_fake_run(tmp_path)))
    assert run["drift_tag"] == "0.0230" and run["drift"] == pytest.approx(0.023)
    assert sorted(run["cells"]) == sorted((a, s) for a in pr.ARMS for s in (0, 1))
    assert run["fingerprints"] == {"d0.0230_s0": "fp0.0230", "d0.0230_s1": "fp0.0230"}


def test_run_task_regenerates_as_the_battery_and_scores_the_named_width(tmp_path, monkeypatch):
    import itasorl.control_diagnostics as cd
    import itasorl.experiment_b2 as b2
    for name in ("DRIFT_MODE", "L1_DELTA", "SENSOR_SIGMA"):
        monkeypatch.setattr(b2, name, getattr(b2, name))
    monkeypatch.setattr(b2, "load_agent_bundle", lambda path, device="cpu": ("A", "N"))
    rng = np.random.default_rng(1)
    pools, calls, seen = {}, [], {}

    def fake_pool(agent, norm, P, drift, n, steps, dev, base, rs, return_anchors, return_obs, return_actions):
        seen["rung"] = (b2.DRIFT_MODE, b2.L1_DELTA, b2.SENSOR_SIGMA)
        H = rng.normal(size=(n, steps, 4)) + drift * 50
        At = rng.random((n, steps, 5))
        pools[base] = H
        return (H, None, None, None, None, None, rng.random((n, steps, 7)), rng.normal(size=(n, steps, 3)), At)

    def fake_gru(seq, y, *, embed, hidden, epochs, seed):
        calls.append(("gru", seq.shape, int(y.sum()), embed, hidden, epochs, seed))
        return {"auroc": 0.6, "final_train_loss": [0.0]}

    monkeypatch.setattr(b2, "collect_pool", fake_pool)
    monkeypatch.setattr(cd, "sequence_gru_auroc", fake_gru)
    monkeypatch.setattr(pr, "conv_sequence_auroc", lambda seq, y, *, epochs, seed:
                        calls.append(("conv", seq.shape, epochs, seed)) or {"auroc": 0.55})
    settings = l1c.settings_for(True, 0.023, 0.01)
    dump = tmp_path / "states.npz"
    base = {"arm": "survival", "seed": "3", "drift": 0.023, "settings": settings, "path": "x.pt", "dump": str(dump)}
    r1 = l1c.run_task({**base, "width": "1x"})
    assert "error" not in r1, r1.get("error")
    assert seen["rung"] == ("l1", 0.023, 0.01)
    assert calls[0] == ("gru", (24, 8, 8), 12, 64, 96, 2, 3)    # 12 + 12 episodes, obs 3 + action 5
    assert r1["result"]["dump_bit_match"] is None and 0.5 <= r1["result"]["target"] <= 1.0
    np.savez(dump, Ha=pools[800_000], Hs=pools[850_000] + 1.0)
    r8 = l1c.run_task({**base, "width": "8x"})
    assert calls[1][3:5] == (64, 768) and r8["result"]["dump_bit_match"] is False
    rc = l1c.run_task({**base, "width": "conv"})
    assert calls[2] == ("conv", (24, 8, 8), 2, 3) and rc["result"]["decoder"]["auroc"] == 0.55


def test_main_resumes_and_reports_both_checks(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    out = tmp_path / "curve.json"
    monkeypatch.setenv("ITASORL_FOLDS", "explicit")
    ref, rec = _ref_and_record()
    ran = []

    def fake(fail):
        def run_task(task):
            ran.append((task["arm"], task["seed"], task["width"]))
            assert task["path"].endswith(f"agent_d0.0230_s{task['seed']}_{task['arm']}.pt")
            if (task["seed"], task["width"]) == fail:
                return {"arm": task["arm"], "seed": task["seed"], "width": task["width"], "error": "boom", "seconds": 0}
            auc = rec[(task["arm"], task["seed"])]["seq_gru"] if task["width"] == "1x" else 0.6
            return {"arm": task["arm"], "seed": task["seed"], "width": task["width"],
                    "result": {"target": ref[(task["arm"], task["seed"])], "dump_bit_match": None,
                               "decoder": {"auroc": auc}, "seconds": 0}}
        return run_task

    argv = ["--run-dir", str(run), "--out", str(out), "--arms", "survival", "--widths", "1x,conv"]
    monkeypatch.setattr(l1c, "run_task", fake(("1", "conv")))
    assert l1c.main(argv) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["complete"] is False
    ran.clear()
    monkeypatch.setattr(l1c, "run_task", fake(None))
    assert l1c.main(argv) == 0
    assert ran == [("survival", "1", "conv")]
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["integrity"]["pass"] and doc["integrity"]["n_checked"] == 2 and not doc["void"]
    assert doc["record_check"]["n_bit_equal"] == 2 and doc["record_check"]["survival_1x_mean_reproduced"] is None
    assert doc["summary"]["survival"]["decoders"]["conv"]["mean"] == pytest.approx(0.6)
    assert doc["cell_fingerprints"] == {"d0.0230_s0": "fp0.0230", "d0.0230_s1": "fp0.0230"}
    src = (ROOT / "scripts" / "run_l1_capacity_curve.py").read_text(encoding="utf-8")
    assert '"git_commit": git_head()' in src
