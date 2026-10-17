"""H56: every committed surrogate export is compared with a fresh retrain.

tests/test_reproducibility_package.py retrains only the headline file (gmotion_h8_s0.npz)
and holds it to allclose(atol=1e-6) on five probe inputs; the other four files under
artifacts/surrogates/ were pinned by sha256 alone, although scripts/gate0_pair_intervals.py
and scripts/residual_field_correlation.py read them from disk. This test retrains each
entry of index.json with train_g_motion defaults on its recorded hidden size and seed,
single-threaded as the export was, and compares the normalization statistics, the weights
and 200 probe outputs with the committed file.

The bound is a cross-stack one, atol 1e-4 and rtol 0, not the exporting stack's 1e-6: on
the workstation stack (torch 2.7.0, NumPy 1.26.4, Python 3.13.2, one thread) the measured
worst case is 4.3e-5 on the weights (h8_s0) and 2.9e-5 on the outputs (h8_s2), while
h7_s0, h10_s1 and h4_s0 agree under 1.1e-6. The measured maxima go into the assertion
message. Each retrain takes 15 to 20 s here, under the 60 s mark that would call for a
slow marker. The normalization statistics come from a deterministic float64 rollout, so
they are held much tighter.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from itasorl.surrogate_l3 import GMotion, train_g_motion  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SURR = os.path.join(ROOT, "artifacts", "surrogates")
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
ATOL_WEIGHTS = 1e-4
ATOL_OUTPUTS = 1e-4
ATOL_NORM = 1e-9
N_PROBES = 200


def _index() -> dict:
    with open(os.path.join(SURR, "index.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _retrain_single_thread(hidden: int, seed: int) -> GMotion:
    import torch
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return train_g_motion(hidden=hidden, seed=seed, params=P)
    finally:
        torch.set_num_threads(n_threads)


def test_index_lists_the_five_committed_exports():
    files = _index()["files"]
    assert sorted(files) == ["gmotion_h10_s1.npz", "gmotion_h4_s0.npz", "gmotion_h7_s0.npz",
                             "gmotion_h8_s0.npz", "gmotion_h8_s2.npz"]
    for name, meta in files.items():
        assert os.path.exists(os.path.join(SURR, name)), name
        assert name == f"gmotion_h{meta['hidden']}_s{meta['g_seed']}.npz"


@pytest.mark.parametrize("name", sorted(_index()["files"]))
def test_each_committed_export_matches_a_fresh_single_thread_retrain(name):
    meta = _index()["files"][name]
    g = GMotion.from_npz(os.path.join(SURR, name))
    fresh = _retrain_single_thread(meta["hidden"], meta["g_seed"])

    assert len(g._W) == len(fresh._W) == 3
    for a, b in zip(g._W + g._b, fresh._W + fresh._b):
        assert a.shape == b.shape

    norm_diff = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                    for a, b in zip((g._xm, g._xs, g._ym, g._ys),
                                    (fresh._xm, fresh._xs, fresh._ym, fresh._ys)))
    weight_diff = max(float(np.max(np.abs(a - b)))
                      for a, b in zip(g._W + g._b, fresh._W + fresh._b))
    rng = np.random.default_rng(0)
    out_diff = 0.0
    for _ in range(N_PROBES):
        v, a = rng.normal(size=2), rng.normal(size=2)
        out_diff = max(out_diff, float(np.max(np.abs(g(v, a) - fresh(v, a)))))

    msg = (f"{name}: max |diff| normalization {norm_diff:.3e}, weights {weight_diff:.3e}, "
           f"outputs over {N_PROBES} rng(0) probes {out_diff:.3e}")
    assert norm_diff <= ATOL_NORM, msg
    assert weight_diff <= ATOL_WEIGHTS, msg
    assert out_diff <= ATOL_OUTPUTS, msg
