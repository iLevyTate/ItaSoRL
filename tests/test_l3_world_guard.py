"""make_world must not build an authentic world when an l3 surrogate was asked for.

Before the guard, l3 mode at drift > 0 with no learned law installed fell through to the
authentic velocity law, so a forgotten setup_l3_surrogate call produced a surrogate pool
that was an authentic pool under another seed base, and the target read near 0.5 with
no error.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import itasorl.experiment_b2 as b2  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_l3_world_at_drift_without_a_surrogate_raises(monkeypatch):
    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", None)
    with pytest.raises(RuntimeError, match="setup_l3_surrogate"):
        b2.make_world(P, 0.45, 5)
    # The authentic side never takes a learned law, so it needs none.
    w = b2.make_world(P, 0.0, 5)
    assert w._g_motion is None


def test_the_guard_leaves_an_installed_surrogate_and_other_modes_alone(monkeypatch):
    def sentinel(vel, a, drag=None):
        return np.zeros(2)

    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", sentinel)
    assert b2.make_world(P, 0.45, 5)._g_motion is sentinel
    monkeypatch.setattr(b2, "_L3_GMOTION", None)
    for mode in ("ar1", "regime", "l1"):
        monkeypatch.setattr(b2, "DRIFT_MODE", mode)
        assert b2.make_world(P, 0.45, 5)._g_motion is None
