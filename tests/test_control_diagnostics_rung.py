"""The control battery must be able to run at the L1 rung as well as L3
(spec docs/specs/2026-10-09-l1-stream-readout-design.md).

The battery itself does not change. What changes is that the rung is selectable, so the same
readouts can be scored on the saved L1 agents. These tests pin two things: the L3 default is
byte-identical to what it was, and L1 mode actually installs the grid spacing and sensor noise
the L1 run used, rather than silently scoring an L3 world.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

run_control_diagnostics = pytest.importorskip("run_control_diagnostics")


def test_the_default_rung_is_still_l3_and_needs_no_new_arguments():
    a = run_control_diagnostics.build_parser().parse_args(["--run-dir", "x", "--out", "y"])
    assert a.drift_mode == "l3"
    assert a.l3_hidden == 8 and a.l3_seed == 0


def test_l1_mode_takes_the_grid_spacing_and_sensor_noise_the_l1_run_used():
    a = run_control_diagnostics.build_parser().parse_args(
        ["--run-dir", "x", "--out", "y", "--drift-mode", "l1",
         "--l1-delta", "0.023", "--l1-sigma", "0.01"])
    assert a.drift_mode == "l1"
    assert a.l1_delta == pytest.approx(0.023)
    assert a.l1_sigma == pytest.approx(0.01)


def test_setting_up_the_rung_installs_the_l1_globals_and_does_not_build_a_surrogate():
    import itasorl.experiment_b2 as b2

    before_mode, before_delta, before_sigma = b2.DRIFT_MODE, b2.L1_DELTA, b2.SENSOR_SIGMA
    before_g = b2._L3_GMOTION
    try:
        run_control_diagnostics.setup_rung({"drift_mode": "l1", "l1_delta": 0.023,
                                            "l1_sigma": 0.01, "l3_hidden": 8, "l3_seed": 0})
        assert b2.DRIFT_MODE == "l1"
        assert b2.L1_DELTA == pytest.approx(0.023)
        assert b2.SENSOR_SIGMA == pytest.approx(0.01)
        assert b2._L3_GMOTION is before_g, "the L1 rung must not train a learned surrogate"
    finally:
        b2.DRIFT_MODE, b2.L1_DELTA, b2.SENSOR_SIGMA = before_mode, before_delta, before_sigma


def test_the_l3_path_still_installs_the_surrogate_and_leaves_the_l1_globals_alone():
    import itasorl.experiment_b2 as b2

    before_delta, before_sigma = b2.L1_DELTA, b2.SENSOR_SIGMA
    before_mode = b2.DRIFT_MODE
    calls = []
    real = b2.setup_l3_surrogate
    b2.setup_l3_surrogate = lambda **kw: calls.append(kw)
    try:
        b2._L3_GMOTION = None
        run_control_diagnostics.setup_rung({"drift_mode": "l3", "l1_delta": 0.5,
                                            "l1_sigma": 0.9, "l3_hidden": 8, "l3_seed": 0})
        assert b2.DRIFT_MODE == "l3"
        assert len(calls) == 1 and calls[0]["hidden"] == 8
        assert b2.L1_DELTA == before_delta and b2.SENSOR_SIGMA == before_sigma, (
            "L1 arguments must not leak into an L3 run")
    finally:
        b2.setup_l3_surrogate = real
        b2.DRIFT_MODE, b2.L1_DELTA, b2.SENSOR_SIGMA = before_mode, before_delta, before_sigma


def test_the_rung_reaches_every_worker_through_the_task_payload():
    """Workers are separate processes, so the rung must travel in the task dict, not in a
    module global of the parent."""
    import inspect

    src = inspect.getsource(run_control_diagnostics.run_one)
    assert "setup_rung(task)" in src, "the worker must install the rung from its own task"
    assert 'b2.DRIFT_MODE = "l3"' not in src, "the rung must not be hard-coded in the worker"
    build = inspect.getsource(run_control_diagnostics.main)
    for key in ("drift_mode", "l1_delta", "l1_sigma"):
        assert key in build, f"{key} must be put into the task payload"


def test_the_battery_checkpoints_each_cell_so_a_crash_costs_one_cell():
    """Thirty cells at about twenty minutes each is too long to restart from zero. A Windows
    console-close event killed all four workers six cells in on 2026-10-09; with per-cell
    checkpoints that would have cost one cell, not six."""
    import inspect

    src = inspect.getsource(run_control_diagnostics)
    assert "--checkpoint-dir" in src, "the battery must be able to checkpoint"
    main = inspect.getsource(run_control_diagnostics.main)
    # The resume must narrow the task list before any worker is dispatched, and both dispatch
    # paths must save each finished cell.
    assert main.index("split_resumable") < main.index("imap_unordered"), "resume before dispatch"
    assert main.count("save_checkpoint(a.checkpoint_dir, r)") == 2, "both paths must save"


def test_a_checkpoint_round_trips_a_cell(tmp_path):
    cell = {"drift": 0.45, "seed": 3, "arm": "survival", "target": 0.52}
    p = run_control_diagnostics.checkpoint_path(str(tmp_path), cell)
    run_control_diagnostics.save_checkpoint(str(tmp_path), cell)
    assert p.endswith(".json")
    got = run_control_diagnostics.load_checkpoints(str(tmp_path))
    assert got == [cell]
    assert run_control_diagnostics.load_checkpoints(str(tmp_path / "missing")) == []


def test_resuming_skips_only_the_cells_already_done(tmp_path):
    done = [{"drift": 0.45, "seed": 0, "arm": "survival"},
            {"drift": 0.45, "seed": 1, "arm": "untrained"}]
    for c in done:
        run_control_diagnostics.save_checkpoint(str(tmp_path), c)
    tasks = [{"drift": 0.45, "seed": s, "arm": a}
             for s in (0, 1) for a in ("survival", "untrained")]
    todo, resumed = run_control_diagnostics.split_resumable(tasks, str(tmp_path))
    assert len(resumed) == 2 and len(todo) == 2
    assert {(t["seed"], t["arm"]) for t in todo} == {(0, "untrained"), (1, "survival")}
