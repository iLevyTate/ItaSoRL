"""Twin pools need pair groups in readout_from_states (correction 2026-10-16).

When the two pools are same-world twins (row j of each pool starts from one world seed and
one initial state, as in the erasure readout's refit probe), the default one-group-per-episode
split puts twin j and twin k+j in different folds whenever the pair count k is not a multiple
of the fold count. The probe then reads the train twin's label off the near-identical test
twin and the AUROC falls far below 0.5 (experiment_b2.probe_world_identity states the rule).
These tests pin the leak and the fix on pools with no class signal at all.
"""

import ast
import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

from itasorl.eval_protocols import readout_from_states  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _twins(k: int, seed: int, T: int = 6, hid: int = 8):
    """k near-identical twin pairs: Hs[j] is Ha[j] plus tiny noise, so labels carry no signal."""
    rng = np.random.default_rng(seed)
    Ha = rng.normal(size=(k, T, hid)).astype(np.float32)
    Hs = Ha + 0.01 * rng.normal(size=Ha.shape).astype(np.float32)
    return Ha, Hs


def _pairs(k: int) -> np.ndarray:
    return np.r_[np.arange(k), np.arange(k)]


@pytest.mark.parametrize("k", [43, 44, 46])     # k % 5 != 0: the default split separates twins
def test_pair_groups_read_near_chance_on_twin_pools(k):
    Ha, Hs = _twins(k, seed=k)
    grouped = readout_from_states(Ha, Hs, seed=0, groups=_pairs(k))["target"]
    assert 0.35 <= grouped <= 0.65, f"pair-grouped twins should read near 0.5, got {grouped}"


@pytest.mark.parametrize("k", [43, 44, 46])
def test_ungrouped_twin_pools_leak_far_below_chance(k):
    """The leak the pair groups remove: without them the twin pools read far below 0.5."""
    Ha, Hs = _twins(k, seed=k)
    ungrouped = readout_from_states(Ha, Hs, seed=0)["target"]
    grouped = readout_from_states(Ha, Hs, seed=0, groups=_pairs(k))["target"]
    assert ungrouped < 0.2, f"expected the split-twin leak (AUROC << 0.5), got {ungrouped}"
    assert grouped - ungrouped > 0.25


def test_pair_count_multiple_of_folds_keeps_twins_together_either_way():
    """With k % 5 == 0 the default split already keeps each pair in one fold, so pair groups
    change nothing there."""
    k = 45
    Ha, Hs = _twins(k, seed=k)
    assert readout_from_states(Ha, Hs, seed=0)["target"] == pytest.approx(
        readout_from_states(Ha, Hs, seed=0, groups=_pairs(k))["target"])


def test_pair_groups_keep_a_real_signal():
    """Twins that differ by a consistent class offset still read well above 0.5 when grouped."""
    k = 44
    Ha, Hs = _twins(k, seed=7)
    Hs = Hs.copy()
    Hs[:, :, 0] += 1.5
    grouped = readout_from_states(Ha, Hs, seed=0, groups=_pairs(k))["target"]
    assert grouped > 0.8


def test_erasure_runner_passes_pair_groups_to_the_refit():
    """scripts/run_erasure_readout.run_cell must refit on the erased twin pools with one CV
    group per pair. Reverting to the default one-group-per-episode split would pass every
    synthetic test above, so this pins the call site itself."""
    with open(os.path.join(ROOT, "scripts", "run_erasure_readout.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_cell")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and ast.unparse(c.func).split(".")[-1] == "readout_from_states"]
    assert calls, "run_cell no longer calls readout_from_states"
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("groups") == "np.r_[np.arange(len(common)), np.arange(len(common))]", kw
