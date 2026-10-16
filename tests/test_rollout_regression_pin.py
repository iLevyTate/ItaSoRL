"""Numeric regression pin for the agent rollout to pooled-target path (CPU).

Every other CI test of collect_pool / pooled_readout checks shapes, guards or same-process
determinism; the saved agents that would let CI rescore a recorded cell are gitignored. This
pin runs the production path end to end on a fixed seeded untrained agent: untrained_agent
(init plus the stochastic normalizer warmup), RunningNorm, collect_pool under the registered
L3 surrogate (artifacts/surrogates/gmotion_h8_s0.npz) at drift 0.45, episode_features and
the grouped probe. A change to any link moves the pinned states or scalars.

The values were measured 2026-10-16 on the workstation (Windows, Python 3.13, CPU, one
torch thread) and reproduce bit for bit across separate processes there. They are pinned to
a tolerance, not a byte hash, because torch is unpinned and CI runs other Python, torch and
BLAS builds. Regenerate with `python tests/test_rollout_regression_pin.py` only together
with a dated docs/CORRECTIONS.md entry that says why the rollout changed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIFT, RS, N_EPS, STEPS = 0.45, 4, 12, 6

PIN = {
    "n_auth": 12,
    "n_surr": 12,
    "target": 0.5333333333333333,
    "shuffled": 0.5375000000000001,
    "pool_reward_leak": 0.4,
    "Ha_sum": -40.62130748620257,
    "Ha_sq": 100.96155934149974,
    "Hs_sum": -42.97785356687382,
    "Hs_sq": 103.58997134271766,
    "norm_mean_sum": 25.496115330798546,
    "Ha_final_first3": [
        [0.501478, -0.6603888, 0.4781358, -0.823126, 0.1374572, 0.2674893, -0.3464201, -0.716853],
        [0.6002504, -0.6397032, 0.5046569, -0.8462605, 0.1829207, 0.2131836, -0.392958, -0.6947504],
        [0.518783, -0.465728, 0.4360017, -0.6168952, 0.0659299, 0.465798, -0.1138212, -0.6523887],
    ],
    "Hs_final_first3": [
        [0.6449564, -0.5684864, 0.417287, -0.7823339, 0.1561559, 0.3146883, -0.2743047, -0.6610588],
        [0.6437095, -0.6658568, 0.6486537, -0.8433972, 0.0639334, 0.2619856, -0.3768423, -0.5987744],
        [0.5414366, -0.5087029, 0.452409, -0.7355893, 0.0788772, 0.3732787, -0.2091946, -0.6533876],
    ],
}


def _measure() -> dict:
    """Run the pinned configuration and return the measured values. Restores every global
    it touches (torch threads, deterministic mode, the experiment_b2 drift mode and law)."""
    import itasorl.experiment_b2 as b2
    from itasorl.surrogate_l3 import GMotion
    from itasorl.world import WorldParams

    saved = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
             b2.DRIFT_MODE, b2._L3_GMOTION)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        b2.DRIFT_MODE = "l3"
        b2._L3_GMOTION = GMotion.from_npz(os.path.join(ROOT, "artifacts", "surrogates", "gmotion_h8_s0.npz"))
        P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
        agent, norm = b2.untrained_agent(P, DRIFT, RS, 8, 16, True, "cpu", seed=0)
        out, (Ha, Hs) = b2.pooled_readout(agent, norm, P, DRIFT, n_eps=N_EPS, steps=STEPS, ray_steps=RS,
                                          device="cpu", seed=0, return_pools=True)
    finally:
        torch.set_num_threads(saved[0])
        torch.use_deterministic_algorithms(saved[1])
        b2.DRIFT_MODE, b2._L3_GMOTION = saved[2], saved[3]
    Ha64, Hs64 = Ha.astype(np.float64), Hs.astype(np.float64)
    return {"n_auth": int(out["n_auth"]), "n_surr": int(out["n_surr"]),
            "target": float(out["target"]), "shuffled": float(out["shuffled"]),
            "pool_reward_leak": float(out["pool_reward_leak"]),
            "Ha_sum": float(Ha64.sum()), "Ha_sq": float((Ha64 ** 2).sum()),
            "Hs_sum": float(Hs64.sum()), "Hs_sq": float((Hs64 ** 2).sum()),
            "norm_mean_sum": float(np.sum(norm.mean)),
            "Ha_final_first3": Ha64[:3, -1].round(7).tolist(),
            "Hs_final_first3": Hs64[:3, -1].round(7).tolist()}


def test_pooled_readout_rollout_pin():
    m = _measure()
    assert (m["n_auth"], m["n_surr"]) == (PIN["n_auth"], PIN["n_surr"]), "survivor filtering changed"
    for k in ("target", "shuffled", "pool_reward_leak"):
        assert m[k] == pytest.approx(PIN[k], abs=1e-6), f"{k} moved: {m[k]} vs pinned {PIN[k]}"
    # Sums over 576 float32 states: a 1e-6 per-value build difference stays far below 1e-3.
    for k in ("Ha_sum", "Ha_sq", "Hs_sum", "Hs_sq", "norm_mean_sum"):
        assert m[k] == pytest.approx(PIN[k], abs=1e-3), f"{k} moved: {m[k]} vs pinned {PIN[k]}"
    for k in ("Ha_final_first3", "Hs_final_first3"):
        np.testing.assert_allclose(m[k], PIN[k], rtol=0, atol=1e-5, err_msg=f"{k} moved")


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, ROOT)
    print(json.dumps(_measure(), indent=4))
