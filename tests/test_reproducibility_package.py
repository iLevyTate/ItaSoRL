"""Revision step 15: frozen surrogates, the reproduce entry point, and supplement scrubbing."""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from itasorl.surrogate_l3 import GMotion, train_g_motion  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_gmotion_roundtrips_through_npz(tmp_path):
    g = train_g_motion(hidden=4, n_eps=10, steps=8, epochs=20, params=P, ray_steps=3)
    p = tmp_path / "g.npz"
    g.to_npz(str(p))
    h = GMotion.from_npz(str(p))
    for v, a in ((np.array([0.1, -0.2]), np.array([0.3, 0.0])), (np.zeros(2), np.ones(2))):
        assert np.array_equal(g(v, a), h(v, a))


def test_exported_surrogates_match_their_index_and_a_fresh_retrain():
    d = os.path.join(ROOT, "artifacts", "surrogates")
    with open(os.path.join(d, "index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    for name, meta in index["files"].items():
        with open(os.path.join(d, name), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == meta["sha256"], name
    g = GMotion.from_npz(os.path.join(d, "gmotion_h8_s0.npz"))
    fresh = train_g_motion(hidden=8, seed=0, params=P)          # the recipe, on CPU
    rng = np.random.default_rng(0)
    for _ in range(5):
        v, a = rng.normal(size=2), rng.normal(size=2)
        assert np.allclose(g(v, a), fresh(v, a), atol=1e-6)


def test_scrub_removes_identity_words_without_touching_ordinary_words():
    import reproduce
    rules = [(p, r) for p, r in reproduce.GENERIC_SCRUB]
    import re
    for t in ("Tate", "Levy", "iLevyTate"):
        rules.append((re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![a-z0-9])"), "<author>"))
    text = "state estimate Tate Levy iLevyTate C:/Users/someone/x a@b.co /home/user/x"
    out = reproduce.scrub(text, rules)
    assert out.startswith("state estimate <author> <author> <author>")
    assert "someone" not in out and "a@b.co" not in out and "/home/user" not in out
