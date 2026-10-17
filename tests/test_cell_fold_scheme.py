"""A new pooled-readout record says which fold scheme and partition produced it (H39).

ITASORL_FOLDS sits outside the config fingerprint, so before this a cell file alone could
not say whether its target came from the explicit or the legacy partition. pooled_readout
now writes `folds` (the key the results manifest uses), `fold_version` and
`fold_partition`. Stored cells are not rewritten, and every key the record carried before
keeps its value.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl import folds  # noqa: E402
from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, pooled_readout  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4
NEW_KEYS = {"folds", "fold_version", "fold_partition"}


def _agent_norm():
    from itasorl.patch_of_earth import PatchOfEarthV0
    w = PatchOfEarthV0(P)
    od, ad = w.obs_spec.size, w.action_spec.size
    torch.manual_seed(0)
    agent = RecurrentActorCritic(od, ad, embed=16, hidden=8).train(False)
    return agent, RunningNorm(od).freeze()


def _read(monkeypatch, scheme):
    if scheme is None:
        monkeypatch.delenv("ITASORL_FOLDS", raising=False)
    else:
        monkeypatch.setenv("ITASORL_FOLDS", scheme)
    agent, norm = _agent_norm()
    return pooled_readout(agent, norm, P, 0.45, n_eps=10, steps=6, ray_steps=RS, device="cpu")


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_default_cell_records_explicit(monkeypatch):
    out = _read(monkeypatch, None)
    assert out["folds"] == "explicit"
    assert out["fold_version"] == folds.scheme_version("explicit")
    rec = out["fold_partition"]
    assert rec["scheme"] == "explicit" and rec["n_samples"] == out["n"]
    assert rec["groups_intact"] is True


def test_legacy_env_is_recorded(monkeypatch):
    explicit = _read(monkeypatch, "explicit")
    legacy = _read(monkeypatch, "legacy")
    assert legacy["folds"] == "legacy"
    assert legacy["fold_partition"]["scheme"] == "legacy"
    assert legacy["n"] == explicit["n"]


def test_partition_record_is_the_probe_partition_and_deterministic(monkeypatch):
    """The partition described is the one the probe used (one group per row, as
    _auroc_with_ci defaults), and a repeat read gives the same value for every key. The
    pre-existing keys are untouched by construction: the three keys are assigned after the
    record is built and read nothing back from it."""
    out = _read(monkeypatch, None)
    assert NEW_KEYS <= set(out)
    y = np.concatenate([np.zeros(out["n_auth"]), np.ones(out["n_surr"])]).astype(int)
    assert out["fold_partition"] == folds.partition_record(np.arange(out["n"]), y, scheme="explicit")
    again = _read(monkeypatch, None)
    for k in set(out) - NEW_KEYS:
        assert _same(out[k], again[k]), k
