"""Round-trip of scripts/rescore_fold_split.py on tiny synthetic pooled-state dumps."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import rescore_fold_split as rfs  # noqa: E402


def _dump(path: Path, shift: float, seed: int, n: int = 40, T: int = 6, hid: int = 8, traces: bool = True):
    rng = np.random.default_rng(seed)
    Ha = rng.normal(size=(n, T, hid))
    Hs = rng.normal(size=(n, T, hid)) + shift
    kw = {"Ha": Ha, "Hs": Hs}
    if traces:
        kw["bta"] = rng.normal(size=(n, T, 7))
        kw["bts"] = rng.normal(size=(n, T, 7))
    np.savez_compressed(path, **kw)


def test_rescore_round_trip(tmp_path):
    d = tmp_path / "run" / "states"
    d.mkdir(parents=True)
    for s in range(3):
        _dump(d / f"states_d0.45_s{s}_survival.npz", 0.4, s)
        _dump(d / f"states_d0.45_s{s}_untrained.npz", 0.0, 100 + s, traces=False)
    np.savez_compressed(d / "states_d0.45_s0_survival_h7transfer.npz", other=np.zeros(3))  # skipped
    out = rfs.rescore(str(d), "synthetic")
    assert out["n_cells"] == 6
    surv = out["aggregate"]["d=0.45 survival"]
    assert surv["seeds"] == [0, 1, 2]
    for scheme in ("legacy", "explicit"):
        assert surv["target"][scheme]["n_seeds"] == 3
        assert surv["target"][scheme]["mean"] > 0.6          # the planted shift is decodable
    assert "resid_trace" in surv
    assert "resid_trace" not in out["aggregate"]["d=0.45 untrained"]   # no traces dumped
    assert out["partition_110_110"]["explicit"] == [[22, 22]] * 5 or \
        out["partition_110_110"]["explicit"] == [(22, 22)] * 5
    from sklearn.model_selection import GroupKFold
    if 'kind="stable"' in inspect.getsource(GroupKFold._iter_test_indices):
        assert surv["target"]["mean_shift"] == 0.0             # the schemes coincide on this stack
