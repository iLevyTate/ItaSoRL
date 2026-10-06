"""Seed-level checkpoint/resume for run_expC_milestone3.py (no training, no GPU)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_expC_milestone3 as m3  # noqa: E402


BASE = dict(n=4, generations=2, seeds=[0, 1], sigma=0.03, q=0.5, drift_sigma=1.0,
            n_eps=2, max_steps=80, embed=8, hidden=8, l3_hidden=8, l3_seed=0,
            ctrl_n_pellets=80, ctrl_reach=0.25, panel_pairs=10, panel_prefix=4,
            panel_tail=4, panel_seed_base=930_000, base_seed_base=320_000,
            json="x.json", device="cpu", panel_every=1,
            checkpoint_dir=None, resume=False)


def _args(**over):
    import argparse
    d = dict(BASE)
    d.update(over)
    return argparse.Namespace(**d)


def _panel(tail_target=0.55):
    return {"cg_tail_target": tail_target, "cg_tail_lo": tail_target - 0.1,
            "cg_tail_hi": tail_target + 0.1, "cg_n_pairs": 10,
            "cg_latetail_target": tail_target, "l0_auroc": 0.5, "l0_latetail": 0.5,
            "l0_n_pairs": 10, "survival": {"deaths": 0}, "leakage": {"reward_sum": 0.5},
            "leak_clean": True, "speed_control": 0.9, "speed_control_pass": True}


def _indiv(mean=0.5):
    return {"per_individual": [mean] * 4, "n_individuals": 4, "n_scored": 4,
            "mean": mean, "median": mean, "q10_q90": [mean, mean], "max": mean,
            "share_at_or_above_bar": 0.0, "pooled_probe_same_tails": mean}


def _fake_gate1(**kw):
    return {"treatment_gap_mean": 0.1, "treatment_ci90": [0.05, 0.15],
            "control_gap_mean": 0.0, "passes_treatment": True,
            "passes_control": True, "passes_gate1": True}


def _fake_run_arm(pop0, food_override, *, rng_seed, **kw):
    return list(pop0), 0.0, [0.0, 0.5], [2, 2], []


def _fake_common_garden_panel(pop, gen=0, **kw):
    return _panel()


def _fake_individual_probe_panel(pop, **kw):
    return _indiv()


def _run_main(tmp_path, monkeypatch, extra=(), run_arm_fn=None):
    # main() mutates two kinds of process-global state that outlive this test
    # unless undone, both caught for real by running this file immediately
    # before tests/test_experiment_b2.py: (1) torch.use_deterministic_algorithms
    # is a process-wide flag, not something these tests exercise, so it's
    # stubbed to a no-op outright; (2) `b2.DRIFT_MODE = "l3"` is a plain
    # attribute assignment on the shared itasorl.experiment_b2 module (not a
    # local/return value), which main() never resets - it broke
    # test_experiment_b2.py's authentic-vs-surrogate divergence check two
    # files later, because that test's own rollout silently picked up "l3"
    # mode with no L3 surrogate actually installed (setup_l3_surrogate is
    # stubbed below) instead of its expected default. monkeypatch.setattr on
    # its OWN current value registers it for guaranteed restoration at
    # teardown, regardless of what main() later assigns to it directly.
    monkeypatch.setattr(m3.torch, "use_deterministic_algorithms", lambda *a, **kw: None)
    monkeypatch.setattr(m3.b2, "DRIFT_MODE", m3.b2.DRIFT_MODE)
    monkeypatch.setattr(m3.b2, "setup_l3_surrogate", lambda **kw: None)
    monkeypatch.setattr(m3, "gate1_exploitability", lambda **kw: _fake_gate1())
    monkeypatch.setattr(m3, "run_arm", run_arm_fn or _fake_run_arm)
    monkeypatch.setattr(m3, "common_garden_panel", _fake_common_garden_panel)
    monkeypatch.setattr(m3, "individual_probe_panel", _fake_individual_probe_panel)
    out_json = tmp_path / "out.json"
    argv = ["run_expC_milestone3.py", "--n", "4", "--generations", "2",
            "--seeds", "0", "1", "--panel-pairs", "10", "--panel-prefix", "4",
            "--panel-tail", "4", "--json", str(out_json), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    m3.main()
    return out_json


# ---------------------------------------------------------------------------
# config_fingerprint / checkpoint file helpers (pure functions, no main())
# ---------------------------------------------------------------------------


def test_fingerprint_stable_and_ignores_paths_and_seeds():
    fp1 = m3.config_fingerprint(_args())
    fp2 = m3.config_fingerprint(_args(json="elsewhere.json",
                                      checkpoint_dir="/somewhere/else",
                                      seeds=[0, 1, 2, 9], resume=True))
    assert fp1 == fp2


def test_fingerprint_changes_on_science_knob():
    fp1 = m3.config_fingerprint(_args())
    for knob, value in [("n", 8), ("generations", 5), ("sigma", 0.1),
                        ("hidden", 16), ("device", "cuda"), ("ctrl_reach", 0.5)]:
        fp2 = m3.config_fingerprint(_args(**{knob: value}))
        assert fp1 != fp2, knob


def test_seed_checkpoint_roundtrip(tmp_path):
    fp = m3.config_fingerprint(_args())
    row = {"seed": 2, "auroc_gen0": 0.5}
    m3.write_seed_checkpoint(str(tmp_path), fp, "abc1234", row)
    done = m3.load_seed_checkpoints(str(tmp_path), [2], fp, "abc1234")
    assert done[2]["seed"] == 2


def test_load_rejects_fingerprint_mismatch(tmp_path):
    fp = m3.config_fingerprint(_args())
    m3.write_seed_checkpoint(str(tmp_path), fp, "abc1234", {"seed": 0})
    other = m3.config_fingerprint(_args(hidden=16))
    with pytest.raises(SystemExit):
        m3.load_seed_checkpoints(str(tmp_path), [0], other, "abc1234")


def test_load_rejects_corrupt_file(tmp_path):
    fp = m3.config_fingerprint(_args())
    (tmp_path / "seed_0.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        m3.load_seed_checkpoints(str(tmp_path), [0], fp, "abc1234")


def test_load_missing_dir_returns_empty():
    fp = m3.config_fingerprint(_args())
    assert m3.load_seed_checkpoints(str(Path("no") / "such" / "dir"), [0, 1], fp, "abc1234") == {}


# ---------------------------------------------------------------------------
# main() integration: fresh run, refusal, resume (run_arm/panels stubbed out,
# so these run in milliseconds, not hours)
# ---------------------------------------------------------------------------


def test_fresh_run_writes_seed_checkpoints_and_result(tmp_path, monkeypatch):
    out_json = _run_main(tmp_path, monkeypatch)
    ckpt_dir = tmp_path / "out_checkpoints"
    assert sorted(p.name for p in ckpt_dir.glob("seed_*.json")) == [
        "seed_0.json", "seed_1.json"]
    assert out_json.is_file()


def test_fresh_run_refuses_stale_checkpoints(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch)  # no --resume


def test_resume_skips_completed_seeds(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    calls = []

    def spy(pop0, food_override, *, rng_seed, **kw):
        calls.append((rng_seed, food_override))
        return _fake_run_arm(pop0, food_override, rng_seed=rng_seed, **kw)

    _run_main(tmp_path, monkeypatch, extra=["--resume"], run_arm_fn=spy)
    # Both seeds are already checkpointed, so the main per-seed loop calls
    # run_arm zero times; only the end-of-run determinism recheck (seed 0,
    # treatment arm only) still runs, unconditionally, regardless of resume.
    assert calls == [(0, None)]


def test_resume_runs_only_missing_seeds(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    (tmp_path / "out_checkpoints" / "seed_1.json").unlink()
    calls = []

    def spy(pop0, food_override, *, rng_seed, **kw):
        calls.append((rng_seed, food_override))
        return _fake_run_arm(pop0, food_override, rng_seed=rng_seed, **kw)

    _run_main(tmp_path, monkeypatch, extra=["--resume"], run_arm_fn=spy)
    # seed 1's treat + control arms recompute; seed 0 is not touched by the
    # main loop (resumed), only by the unconditional determinism recheck.
    assert [c[0] for c in calls] == [1, 1, 0]
    assert calls.count((0, None)) == 1


def test_resume_rejects_different_config(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, extra=["--resume", "--hidden", "16"])


def test_resumed_seeds_recorded_in_output(tmp_path, monkeypatch):
    import json as _json
    _run_main(tmp_path, monkeypatch)
    (tmp_path / "out_checkpoints" / "seed_1.json").unlink()
    out_json = _run_main(tmp_path, monkeypatch, extra=["--resume"])
    out = _json.loads(out_json.read_text())
    assert out["resumed_seeds"] == [0]
    assert [row["seed"] for row in out["per_seed"]] == [0, 1]
