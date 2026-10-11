"""Revision step 10: the linear-fit control, held-out and rollout errors, and the
agent-accessible (observation-only) detector. 2026-10-10: the held-out sample must not
overlap the training episodes, and the committed artifact gains the disjoint-seed block
without losing a stored value."""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")

from itasorl.experiment_a_l3 import observation_law_detector  # noqa: E402
from itasorl.surrogate_l3 import HELDOUT_SEED, surrogate_diagnostics, train_g_motion  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def test_linear_fit_recovers_the_uniform_drag_law_and_g_does_not():
    g = train_g_motion(hidden=4, n_eps=20, steps=10, epochs=30, params=P, ray_steps=3)
    d = surrogate_diagnostics(g, params=P, n_train_eps=20, train_steps=10, heldout_eps=6,
                              heldout_steps=12, horizons=(1, 10), ray_steps=3)
    assert d["uniform_drag"]
    assert d["linear_fit"]["rms_heldout"] < 1e-6
    assert d["linear_fit"]["coef_vel_x"][0] == pytest.approx(1 - 1.5 * P.dt, abs=1e-5)
    assert d["g_one_step"]["rms_heldout"] > 100 * d["linear_fit"]["rms_heldout"]
    assert d["rollout_rms_velocity_gap"]["10"] >= d["rollout_rms_velocity_gap"]["1"]


def test_observation_law_detector_is_chance_on_one_world_and_reads_g():
    import run_surrogate_diagnostics as rsd
    a = rsd.scripted_obs_pool(None, 800_000, 20, 8, ray_steps=3)
    b = rsd.scripted_obs_pool(None, 850_000, 20, 8, ray_steps=3)
    same = observation_law_detector(a, b, dt=P.dt)["auroc"]
    assert 0.2 < same < 0.8                                   # two authentic samples
    g = train_g_motion(hidden=4, n_eps=20, steps=10, epochs=30, params=P, ray_steps=3)
    s = rsd.scripted_obs_pool(g, 850_000, 20, 8, ray_steps=3)
    assert observation_law_detector(a, s, dt=P.dt)["auroc"] > 0.9


def test_held_out_episodes_may_not_overlap_the_training_episodes():
    # Episode i of a sample is seeded by seed0 + i, so seed 77 with 60 episodes is a slice of
    # the 250 training episodes that start at seed 0. The guard refuses before any rollout.
    with pytest.raises(ValueError, match="overlap"):
        surrogate_diagnostics(None, params=P, train_seed=0, n_train_eps=250,
                              heldout_seed=77, heldout_eps=60)
    with pytest.raises(ValueError, match="overlap"):
        surrogate_diagnostics(None, params=P, train_seed=2, n_train_eps=250,
                              heldout_seed=0, heldout_eps=5)


def test_default_held_out_range_is_disjoint_from_every_trained_g():
    # The project trained G at seeds 0, 1 and 2, each on 250 episodes; the runner scores
    # 60 held-out episodes.
    for gseed in (0, 1, 2):
        assert HELDOUT_SEED >= gseed + 250 or HELDOUT_SEED + 60 <= gseed


def _stored(auroc=0.99, rms=0.0349, lag1=0.835):
    prof = {"rms_one_step": rms, "linear_explained_share": 0.59, "lag1_autocorr": lag1,
            "mean_dev": [-0.001, -0.001], "n_transitions": 2400}
    return {"generated_by": "scripts/run_surrogate_diagnostics.py",
            "surrogates": {"g": {"diagnostics": {"g_one_step": {"rms_heldout": rms},
                                                  "train_seed": 0},
                                 "profile": prof,
                                 "agent_accessible_detector": {"auroc": auroc}}}}


def _fresh(**kw):
    doc = _stored(**kw)
    doc["surrogates"]["g"]["diagnostics"] = {"g_one_step": {"rms_heldout": 0.0354},
                                             "train_seed": 0}
    doc["surrogates"]["g"]["profile_heldout"] = {"lag1_autocorr": 0.833}
    doc.update(heldout_seed=HELDOUT_SEED, environment={"numpy": "x"}, wall_seconds=1.0)
    return doc


def test_merge_keeps_every_stored_value_and_adds_the_disjoint_block():
    import run_surrogate_diagnostics as rsd
    old = _stored()
    merged = rsd.merge_disjoint_block(old, _fresh(lag1=0.835 + 1e-8), date="2026-10-10")
    assert merged["surrogates"] == old["surrogates"]
    block = merged[rsd.MERGE_KEY]
    assert block["surrogates"]["g"]["diagnostics"]["g_one_step"]["rms_heldout"] == 0.0354
    assert block["surrogates"]["g"]["profile_heldout"]["lag1_autocorr"] == 0.833
    assert block["heldout_seed"] == HELDOUT_SEED
    assert block["determinism"]["profile_max_abs_diff"] < rsd.PROFILE_TOL


def test_merge_refuses_a_rerun_that_does_not_reproduce_the_stored_detector_or_profile():
    import run_surrogate_diagnostics as rsd
    with pytest.raises(ValueError, match="detector"):
        rsd.merge_disjoint_block(_stored(), _fresh(auroc=0.98), date="2026-10-10")
    with pytest.raises(ValueError, match="profile"):
        rsd.merge_disjoint_block(_stored(), _fresh(rms=0.0360), date="2026-10-10")


def test_the_calibration_sample_can_be_regenerated_only_when_labeled_in_sample():
    """The committed top-level diagnostics were scored on a held-out sample inside G's training
    episodes. allow_overlap reproduces them, and the output then says the sample overlaps."""
    g = train_g_motion(hidden=4, n_eps=20, steps=10, epochs=5, params=P, ray_steps=3)
    kw = dict(params=P, n_train_eps=20, train_steps=10, heldout_eps=6, heldout_steps=12,
              horizons=(1,), ray_steps=3, train_seed=0)
    with pytest.raises(ValueError, match="overlap"):
        surrogate_diagnostics(g, heldout_seed=5, **kw)
    d = surrogate_diagnostics(g, heldout_seed=5, allow_overlap=True, **kw)
    assert d["heldout_overlaps_training"] is True and d["heldout_seed"] == 5
    assert "heldout_overlaps_training" not in surrogate_diagnostics(g, heldout_seed=500, **kw)
    import run_surrogate_diagnostics as rsd
    assert rsd.CALIBRATION_SEED == 77
