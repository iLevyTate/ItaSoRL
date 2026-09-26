"""The two new run_expB2 knobs are science-relevant and must change the config
fingerprint (cells from different configs never mix); the gate-0 script exposes
the fingerprint seed; and train_g_motion's seed is live.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_expB2  # noqa: E402

BASE = {"updates": 1, "n_eps": 1, "max_steps": 8, "hidden": 8, "ray_steps": 5,
        "shaping_coef": 1.0, "pool_n": 4, "pool_steps": 4, "mp_pairs": 2, "mp_prefix": 2,
        "mp_branch": 2, "basal_e": None, "n_pellets": None, "reach": None,
        "dump_states": None, "sysid_aux": False, "sysid_coef": 1.0, "drift_mode": "l3",
        "l3_hidden": 8, "l1_delta": 1 / 64, "sensor_sigma": 0.01, "l3_seed": 0,
        "world_model": True, "drifts": [0.0, 0.45], "device": "cpu", "out_dir": ".",
        "save_agents": False}


def test_fingerprint_changes_with_world_model_flag():
    a = run_expB2.config_fingerprint(BASE)
    b = run_expB2.config_fingerprint({**BASE, "world_model": False})
    assert a != b


def test_fingerprint_changes_with_l3_seed():
    a = run_expB2.config_fingerprint(BASE)
    b = run_expB2.config_fingerprint({**BASE, "l3_seed": 1})
    assert a != b


def test_fingerprint_ignores_paths_only():
    a = run_expB2.config_fingerprint(BASE)
    b = run_expB2.config_fingerprint({**BASE, "dump_states": "elsewhere", "out_dir": "x",
                                      "save_agents": True})
    assert a == b


def test_g_motion_seed_is_live():
    pytest.importorskip("torch")
    from itasorl.surrogate_l3 import train_g_motion
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    g0 = train_g_motion(hidden=4, n_eps=4, steps=10, epochs=3, seed=0, params=P)
    g1 = train_g_motion(hidden=4, n_eps=4, steps=10, epochs=3, seed=1, params=P)
    g0b = train_g_motion(hidden=4, n_eps=4, steps=10, epochs=3, seed=0, params=P)
    v, a = np.array([0.02, -0.01]), np.array([0.3, 0.1])
    assert np.allclose(g0(v, a), g0b(v, a))            # deterministic at a fixed seed
    assert not np.allclose(g0(v, a), g1(v, a))         # the seed changes the fingerprint


def test_untrained_agent_honours_world_model_flag():
    pytest.importorskip("torch")
    from itasorl.experiment_b2 import untrained_agent
    from itasorl.world import WorldParams
    P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
    with_wm, _ = untrained_agent(P, 0.0, 5, 8, 8, True, "cpu", seed=0)
    without, _ = untrained_agent(P, 0.0, 5, 8, 8, False, "cpu", seed=0)
    assert with_wm.world_model is True and without.world_model is False
    assert hasattr(with_wm, "decoder") and not hasattr(without, "decoder")
