"""Tests for viz/collect.py pure helpers (no agent bundle, no surrogate training).

The anonymized supplement ships viz/collect.py but not viz/player/beats.json or viz/data/scene.json,
so outside a git checkout the checks on the committed beats and scene skip; in a checkout a missing
file still fails.
"""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("viz_collect", ROOT / "viz" / "collect.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def _in_git_checkout() -> bool:
    """True when ROOT is the top of a git checkout (an extracted archive placed inside some
    other repository is still an extracted archive)."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        return False
    return bool(top) and Path(top).resolve() == ROOT


def _beats(surv="73%", untr="52%", orac="99%", replay="72%", l0="56%"):
    return {
        "numbers": {
            "oracle_l2": {"display": orac},
            "probe_chance": {"display": untr},
            "probe_survival": {"display": surv},
            "probe_replay": {"display": replay},
            "l0_gate": {"display": l0},
        },
        "beats": [
            {"id": "observer", "gauge": {"display": orac}},
            {"id": "nocare", "gauge": {"display": untr}},
            {"id": "survival", "gauge": {"display": surv}},
            {"id": "close", "endcard": {"recap": [{"pct": orac}, {"pct": untr}, {"pct": surv}, {"pct": replay}],
                                        "foot": f"... reads {replay} ... read {l0} instead of 50% ..."}},
        ],
    }


def _record(surv=0.733, untr=0.5226, lo=0.4914, hi=0.5538, orac=0.9909, replay=0.721, l0=0.5594):
    return {"survival": surv, "untrained": untr, "untrained_t90": [lo, hi],
            "oracle": orac, "replay": replay, "l0": l0}


def test_verify_numbers_passes_on_published_values():
    got = vc.verify_numbers(_beats(), _record())
    assert got["survival"] == 0.733 and got["untrained"] == 0.5226


def test_verify_numbers_fails_on_drifted_survival():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(), _record(surv=0.61))


def test_verify_numbers_fails_when_untrained_leaves_chance():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(untr="60%"), _record(untr=0.60, lo=0.57, hi=0.63))


def test_verify_numbers_fails_when_the_end_card_drops_the_replay_reading():
    b = _beats()
    b["beats"][-1]["endcard"]["recap"] = b["beats"][-1]["endcard"]["recap"][:3]
    with pytest.raises(SystemExit):
        vc.verify_numbers(b, _record())


def test_verify_numbers_fails_when_the_foot_drops_the_l0_reading():
    b = _beats()
    b["beats"][-1]["endcard"]["foot"] = "reads 72%"
    with pytest.raises(SystemExit):
        vc.verify_numbers(b, _record())


def test_committed_beats_match_the_corrected_record():
    """The film's numbers, read from the committed beats.json, agree with the committed
    corrected-trainer artifacts. Runs in CI: no agents, no surrogate training."""
    path = ROOT / "viz" / "player" / "beats.json"
    if not path.exists() and not _in_git_checkout():
        pytest.skip("viz/player/beats.json is not shipped in the anonymized supplement")
    beats = json.loads(path.read_text(encoding="utf-8"))
    vc.verify_numbers(beats, vc.load_record(ROOT))


def test_sample_fields_normalized_grid():
    from itasorl.experiment_b2 import _seeds, make_world
    w = make_world(vc.P, 0.0, vc.RAY_STEPS)
    w.reset(_seeds(1234))
    height, wet = vc.sample_fields(w, 8)
    assert height.shape == (64,) and wet.shape == (64,)
    assert height.min() == 0.0 and height.max() == 1.0
    assert set(np.unique(wet)) <= {0.0, 1.0}


def test_in_split_window():
    inside = [[0.5, 0.5, 0.0, 1.0]] * 300
    assert vc.in_split_window(inside)
    outside = [list(p) for p in inside]
    outside[150][0] = 0.9  # wanders out of the split-panel strip mid-beat
    assert not vc.in_split_window(outside)


def test_build_scene_schema_and_coverage():
    r = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 900, "pellets_t": [[[0.1, 0.2]]] * 900}
    scene = vc.build_scene({"source": "collect.py"}, 8, np.zeros(64), np.zeros(64), r, r)
    for key in ("meta", "grid_n", "height", "wet", "pellets", "trajs", "pellets_t", "step_ms"):
        assert key in scene
    json.dumps(scene)  # fully JSON-serializable
    assert len(scene["trajs"]["auth"]) * scene["step_ms"] >= vc.LAST_WORLD_MS


def test_build_scene_rejects_short_traj():
    r = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 100, "pellets_t": [[[0.1, 0.2]]] * 100}
    with pytest.raises(AssertionError):
        vc.build_scene({"source": "collect.py"}, 8, np.zeros(64), np.zeros(64), r, r)


def test_build_scene_adds_the_untrained_walk():
    r = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 900, "pellets_t": [[[0.1, 0.2]]] * 900}
    ru = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 47, "pellets_t": [[[0.1, 0.2]]] * 47, "n": 47}
    scene = vc.build_scene({"source": "collect.py"}, 8, np.zeros(64), np.zeros(64), r, r, ru=ru)
    assert len(scene["trajs"]["untr"]) == 47 and scene["meta"]["n_untr"] == 47
    assert len(scene["pellets_t"]["untr"]) == 47


def test_committed_scene_carries_the_untrained_walk_the_52_beat_draws():
    scene_path = ROOT / "viz" / "data" / "scene.json"
    beats_path = ROOT / "viz" / "player" / "beats.json"
    if not (scene_path.exists() and beats_path.exists()) and not _in_git_checkout():
        pytest.skip("viz/data/scene.json and viz/player/beats.json are not shipped in the anonymized supplement")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    beats = json.loads(beats_path.read_text(encoding="utf-8"))
    nocare = next(b for b in beats["beats"] if b["id"] == "nocare")
    w = nocare["world"]
    assert w["traj"] == "untr" and "untr" in scene["trajs"]
    # the slowed window must end before the untrained walk does (no death on screen)
    span = w["worldT0"] + (nocare["t1"] - nocare["t0"]) * w["clock"]
    assert span < len(scene["trajs"]["untr"]) * scene["step_ms"]
