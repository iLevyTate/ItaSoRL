"""Pin the scripted-action / terrain stream coupling disclosed in docs/CORRECTIONS.md
(2026-10-16, "scripted action streams share a PCG64 stream with the terrain").

SeedBundle.streams() seeds the world stream with the raw world integer, and the scripted
collectors (experiment_b2._collect_scripted, experiment_b.collect_episodes, and others listed in
that entry) seed their action generator with the same integer. The first scripted thrust is
therefore a function of the first terrain frequency. The seeding was left unchanged on purpose,
because changing it would move the stored bits of every run the entry lists. These tests fail
if the seeding changes, so that any change is deliberate (an opt-in decoupling flag must
default off) and comes with a new correction entry.
"""

from __future__ import annotations

import numpy as np
import pytest

from itasorl.experiment_b import scripted_policy
from itasorl.experiment_b2 import _seeds
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.world import SeedBundle, WorldParams

SEEDS = [0, 1, 7, 200_000, 1_400_000]


@pytest.mark.parametrize("s", SEEDS)
def test_world_stream_is_the_raw_world_integer(s):
    world = SeedBundle(world=s, weather=0, ecology=0).streams()["world"]
    assert world.bit_generator.state == np.random.default_rng(s).bit_generator.state


@pytest.mark.parametrize("s", SEEDS)
def test_collector_seed_bundle_world_is_the_action_seed(s):
    """experiment_b2._seeds keeps the world integer equal to the base that
    _collect_scripted also hands np.random.default_rng for the scripted actions."""
    assert _seeds(s).world == s


@pytest.mark.parametrize("s", SEEDS)
def test_first_scripted_thrust_equals_the_first_terrain_uniform(s):
    a = scripted_policy(np.random.default_rng(s))      # the collectors' action generator
    w = PatchOfEarthV0(WorldParams())
    w.reset(_seeds(s))
    u_terrain = (w._tf[0, 0] / (2 * np.pi) - 1.5) / 2.0  # _tf = uniform(1.5, 3.5) * 2 pi
    assert float(a[0]) / 0.6 == pytest.approx(u_terrain, abs=1e-6)   # float32 rounding
    assert float(a[0]) / 0.6 == pytest.approx(np.random.default_rng(s).random(), abs=1e-6)
