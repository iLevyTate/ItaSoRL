"""Goal-and-stakes runs (docs/specs/2026-10-07-goal-and-stakes-design.md): the world's
mortality switch, the touch objective, the engagement rule per objective, and the two
runner flags' fingerprint behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def _world(drift=0.0):
    import itasorl.experiment_b2 as b2
    w = b2.make_world(P, drift, 5)
    w.reset(b2._seeds(123))
    return w


def test_world_is_mortal_by_default_and_immortal_when_switched():
    w = _world()
    assert w.mortal is True
    w.E = 0.0
    assert w._focal_dead() is True
    w.mortal = False
    assert w._focal_dead() is False
    w.Hyd = 0.0
    assert w._focal_dead() is False


def test_immortal_world_step_never_terminates_and_never_pays_the_death_penalty():
    w = _world()
    w.mortal = False
    w.E = 0.0
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    for _ in range(5):
        r = w.step(a)
        assert r.terminated is False
        assert r.reward > -0.5          # the -1 death transition penalty is never charged
    assert w.alive is True
