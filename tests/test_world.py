"""Regression tests for the ITASORL world contract and surrogate ladder.

The keystone guarantees (spec sec. 9/11/12): a byte-identical obs format between
authentic and surrogate, exact snapshot/restore, an L0 branch that is identical
to the authentic branch (the chance control), and an L1 branch that differs from
authentic *only* by the documented observation quantization.
"""

import numpy as np

from itasorl.patch_of_earth import PatchOfEarthV0, first_config_obs_spec
from itasorl.world import (
    DEFAULT_OBS_SPEC,
    L0Identity,
    L1Discretize,
    Level,
    SeedBundle,
    matched_pair_rollout,
)

CONST_ACTION = np.array([0.5, 0.1, 1.0, 0.0, 0.0], dtype=np.float32)


def _const_policy(obs):
    # Obs-independent on purpose: L1 then cannot cause behavioral divergence, so
    # the only authentic/surrogate difference is the observation quantization.
    return CONST_ACTION


def _reactive_policy(obs):
    # Obs-dependent: exposes any state field dropped by get_state/set_state,
    # because the branches would diverge once the policy reacts to it.
    a = np.zeros(5, dtype=np.float32)
    a[0] = 0.5 + 0.4 * np.tanh(obs[0])  # thrust <- first ray distance
    a[1] = float(np.tanh(obs[132]))     # turn   <- velocity x (intero)
    a[2] = float(obs[4] > 0.0)          # eat    <- radial velocity
    return a


def _make_world():
    w = PatchOfEarthV0()
    w.ray_steps = 4  # light raymarch keeps the tests fast
    return w


def _seeds():
    return SeedBundle(world=7, weather=8, ecology=9)


# --- obs format contract (spec sec. 9) --------------------------------------

def test_obs_spec_size():
    assert DEFAULT_OBS_SPEC.size == 24 * 5 + 4 * 3 + 14


def test_mask_preserves_length_and_format_hash():
    masked = first_config_obs_spec()  # smell channel disabled
    assert masked.size == DEFAULT_OBS_SPEC.size  # length preserved
    # masking changes the mask, not the *format* identity (names/sizes/order/dtype)
    assert masked.identity_hash() == DEFAULT_OBS_SPEC.identity_hash()


def test_masked_channel_is_zero_filled_not_removed():
    masked = first_config_obs_spec()
    smell = masked.slices()["smell"]
    v = masked.assemble({"smell": np.ones(4 * 3, dtype=np.float32)})
    assert np.all(v[smell] == 0.0)


# --- snapshot / restore determinism (spec sec. 11/12) -----------------------

def test_get_set_state_roundtrip_is_exact():
    live = _make_world()
    live.reset(_seeds())
    for _ in range(5):
        live.step(CONST_ACTION)
    snap = live.get_state()

    restored = _make_world()
    restored.reset(_seeds())
    restored.set_state(snap)

    a = live.step(CONST_ACTION)
    b = restored.step(CONST_ACTION)
    np.testing.assert_array_equal(a.obs, b.obs)
    assert a.reward == b.reward
    assert a.terminated == b.terminated


def _forager_world():
    """Dense, easy-to-reach food so a random-action rollout genuinely EATS: pellets
    deplete, energy climbs, and the ecology RNG advances - the state a lazy snapshot
    would return stale."""
    w = PatchOfEarthV0()
    w.ray_steps = 4
    w.n_pellets = 40
    w.reach = 0.5
    return w


def _random_actions(n: int, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.array([rng.uniform(0.2, 1.0), rng.uniform(-1.0, 1.0), 1.0,
                      float(rng.random() < 0.5), 0.0], np.float32) for _ in range(n)]


def test_get_set_state_roundtrip_after_state_exercising_rollout():
    """Stronger than the 5-step constant-action roundtrip: run a seeded random-action
    rollout that actually eats (pellet_amt partially depleted, energy shifted) before
    snapshotting, then continue vs restore-and-continue for many steps. Any field
    get_state returns fresh/stale (e.g. pellet_amt reset to ones) breaks bit-identity
    the moment a pellet depletes and respawns on one branch only."""
    acts = _random_actions(12 + 25)

    live = _forager_world()
    live.reset(_seeds())
    for a in acts[:12]:
        live.step(a)
    snap = live.get_state()
    # Guard that the scenario really exercised the state: eating has begun, so the
    # snapshot's pellet ledger and energy must differ from their reset values. This
    # also catches a get_state that returns a FRESH pellet_amt outright.
    assert not np.all(snap["pellet_amt"] == 1.0), "no pellet was touched - vacuous roundtrip"
    assert snap["E"] != live.E0
    assert snap["alive"]

    restored = _forager_world()
    restored.reset(_seeds())
    restored.set_state(snap)

    for i, a in enumerate(acts[12:]):
        r_live = live.step(a)
        r_rest = restored.step(a)
        np.testing.assert_array_equal(r_live.obs, r_rest.obs,
                                      err_msg=f"obs diverged at continuation step {i}")
        assert r_live.reward == r_rest.reward, f"reward diverged at continuation step {i}"
        assert r_live.terminated == r_rest.terminated


# --- ecology invariants (intake cap + one-shot death penalty) ----------------

EAT_ONLY = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def test_eating_at_energy_cap_gains_nothing_and_depletes_nothing():
    """Intake is capped by remaining energy capacity: at E == Emax an eat action on a
    reachable pellet consumes NOTHING - zero intake, no pellet depletion, and the
    reward stays purely homeostatic (the metabolic cost, no positive food term). A
    below-cap control on the identical world proves the setup genuinely eats, so the
    cap case is not passing vacuously."""
    w = _forager_world()
    w.reset(_seeds())
    w.pos = w.pellets[0].copy()          # stand on a pellet
    w.E = w.Emax                          # sated to the cap
    before = w.pellet_amt.copy()
    r = w.step(EAT_ONLY)
    assert r.info["intake"] == 0.0, "intake at the cap must be zero"
    assert np.array_equal(w.pellet_amt, before), "pellet depleted for free at the cap"
    assert r.reward < 0.0, "reward must stay homeostatic (cost only), no free food reward"
    assert w.E <= w.Emax

    ctrl = _forager_world()
    ctrl.reset(_seeds())
    ctrl.pos = ctrl.pellets[0].copy()
    ctrl.E = ctrl.Emax - 1.0              # room to eat: same setup must gain for real
    r2 = ctrl.step(EAT_ONLY)
    assert r2.info["intake"] > 0.0
    assert ctrl.pellet_amt[0] < 1.0       # pellet genuinely depleted below the cap
    assert r2.reward > 0.0                # intake dominates the metabolic cost


def test_death_penalty_applied_exactly_once_on_the_transition():
    """The -1.0 death penalty lands on the death TRANSITION only. A misbehaving
    caller that keeps stepping the dead world must see plain homeostatic rewards
    (here ~ -cost*dt ~ -0.2), never the penalty again."""
    w = _make_world()
    w.reset(_seeds())
    w.E = 0.05                            # nearly starved ...
    w.basal_E = 4.0                       # ... with a drain that kills this step
    r1 = w.step(EAT_ONLY * 0.0)
    assert r1.terminated is True
    assert w.alive is False
    assert r1.reward < -0.9, "death transition must carry the -1.0 penalty"

    for _ in range(3):                    # step PAST termination (misbehaving caller)
        r = w.step(EAT_ONLY * 0.0)
        assert r.terminated is True       # stays dead
        assert r.reward > -0.9, "the -1.0 penalty reappeared after the death transition"


# --- matched-pair confound control (spec sec. 11) ---------------------------

def test_matched_pair_L0_is_bit_identical():
    pe = matched_pair_rollout(
        make_world=_make_world,
        make_surrogate=L0Identity,
        seeds=_seeds(),
        policy=_const_policy,
        prefix_steps=5,
        branch_steps=8,
        pair_id=0,
        rng=np.random.default_rng(0),
    )
    assert pe.level == Level.L0
    assert len(pe.authentic) == len(pe.surrogate) == 8
    for a, s in zip(pe.authentic, pe.surrogate):
        np.testing.assert_array_equal(a.obs, s.obs)
        assert a.reward == s.reward


def test_matched_pair_L0_identical_under_reactive_policy():
    # Stronger than the constant-policy case: a reactive policy would surface an
    # incomplete snapshot the moment it reacts to the affected observation.
    pe = matched_pair_rollout(
        make_world=_make_world,
        make_surrogate=L0Identity,
        seeds=_seeds(),
        policy=_reactive_policy,
        prefix_steps=20,
        branch_steps=40,
        pair_id=0,
        rng=np.random.default_rng(0),
    )
    for a, s in zip(pe.authentic, pe.surrogate):
        np.testing.assert_array_equal(a.obs, s.obs)
        assert a.reward == s.reward


def test_L1_surrogate_preserves_obs_format():
    # spec sec. 9: the surrogate obs must match the authentic format (dtype,
    # shape, format hash) so detection rides the discretization signal, not an
    # implementation artifact like float32-vs-float64.
    auth = _make_world()
    auth.reset(_seeds())
    authentic_obs = auth.observe()
    base = _make_world()
    base.reset(_seeds())
    surr = L1Discretize(base, delta=1.0 / 32)
    surrogate_obs = surr.observe()
    assert surrogate_obs.dtype == authentic_obs.dtype
    assert surrogate_obs.shape == authentic_obs.shape
    assert surr.obs_spec.identity_hash() == auth.obs_spec.identity_hash()


def test_matched_pair_L1_only_quantizes_observation():
    delta = 1.0 / 32
    pe = matched_pair_rollout(
        make_world=_make_world,
        make_surrogate=lambda base: L1Discretize(base, delta=delta),
        seeds=_seeds(),
        policy=_const_policy,
        prefix_steps=5,
        branch_steps=6,
        pair_id=1,
        rng=np.random.default_rng(1),
    )
    assert pe.level == Level.L1
    for a, s in zip(pe.authentic, pe.surrogate):
        np.testing.assert_allclose(s.obs, np.round(a.obs / delta) * delta, atol=1e-6)


def _l3_rollout(drift_mode, g_motion=None, log=False, steps=10):
    """Run a fixed authentic-policy rollout; return (obs trajectory, world)."""
    from itasorl.world import WorldParams
    w = PatchOfEarthV0(WorldParams(), drift_mode=drift_mode)
    w.ray_steps = 4
    w._g_motion = g_motion
    if log:
        w._log_motion = []
    w.reset(SeedBundle(world=1, weather=2, ecology=3))
    rng = np.random.default_rng(0)
    traj = []
    for _ in range(steps):
        a = np.array([rng.uniform(0, 0.6), rng.uniform(-1, 1), 0.0, 0.0, 0.0], np.float32)
        traj.append(w.step(a).obs.copy())
    return np.asarray(traj), w


def test_l3_hook_preserves_authentic_when_unset():
    """The L3 motion hook must be a NO-OP when no G_motion is attached: `l3` mode with
    `_g_motion=None` is bit-identical to the authentic world (keystone L0 invariant - the
    edit to _integrate_motion must not perturb authentic dynamics)."""
    base, _ = _l3_rollout("ar1")               # authentic
    l3_noG, _ = _l3_rollout("l3", g_motion=None)
    assert np.array_equal(base, l3_noG), "l3 mode without G_motion diverged from authentic"


def test_l3_hook_logs_and_overrides():
    """`_log_motion` records one (vel, a, drag, vel_next) transition per step; a set
    `_g_motion` replaces the velocity law and changes the dynamics."""
    base, _ = _l3_rollout("ar1")
    _, wlog = _l3_rollout("l3", g_motion=None, log=True)
    assert len(wlog._log_motion) == 10
    vel, a, drag, vnext = wlog._log_motion[0]
    assert vel.shape == (2,) and a.shape == (2,) and vnext.shape == (2,) and isinstance(drag, float)

    called = {}
    def frozen_g(vel, a, drag):
        called["hit"] = True
        return np.zeros(2)                     # freeze velocity -> different trajectory
    frozen, _ = _l3_rollout("l3", g_motion=frozen_g)
    assert called.get("hit"), "G_motion was not called in l3 mode"
    assert not np.array_equal(base, frozen), "G_motion did not change the dynamics"
