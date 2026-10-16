"""Readout 8(a) surprise AUROC on same-world twins needs pair groups (correction 2026-10-16).

scripts/run_mechanism_readouts.py rolls the authentic and the surrogate episodes from one seed
base (WORLD_BASE + i), and common_rows keeps row j of both on the same world, so the two pools
are twins. With one CV group per episode a twin can land in the train fold while its partner is
tested, and the probe reads the train twin's label off the near-identical test twin: the AUROC
falls below 0.5 on pools with no class signal, so a real signal reads low. The 18.3 surprise
AUROCs stored before this fix used one group per episode and are lower bounds
(docs/CORRECTIONS.md, 2026-10-16). These tests pin the leak, the fix and the call site.
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

from itasorl import mechanism_readouts as mr  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _twins(k: int, seed: int, T: int = 12):
    """k twin pairs of prediction-error traces: Es[j] is Ea[j] plus tiny noise, so the class
    label carries no signal and only the twin pairing links the two pools."""
    rng = np.random.default_rng(seed)
    Ea = rng.uniform(0.0, 1.0, size=(k, T)) * rng.uniform(0.2, 3.0, size=(k, 1))
    Es = Ea + 1e-3 * rng.normal(size=Ea.shape)
    return Ea, Es


def _pairs(k: int) -> np.ndarray:
    return np.r_[np.arange(k), np.arange(k)]


@pytest.mark.parametrize("k", [43, 44, 46])     # k % 5 != 0: the default split separates twins
def test_ungrouped_twins_leak_below_chance(k):
    Ea, Es = _twins(k, seed=k)
    ungrouped = mr.surprise_auroc(Ea, Es)
    grouped = mr.surprise_auroc(Ea, Es, groups=_pairs(k))
    assert ungrouped < 0.35, f"expected the split-twin leak (AUROC < 0.5), got {ungrouped}"
    assert 0.4 <= grouped <= 0.6, f"pair-grouped twins should read near 0.5, got {grouped}"
    assert grouped - ungrouped > 0.15


def test_default_groups_unchanged():
    """groups=None keeps one group per episode, the behavior for disjoint seed bases."""
    Ea, Es = _twins(44, seed=3)
    n = 2 * len(Ea)
    assert mr.surprise_auroc(Ea, Es) == pytest.approx(mr.surprise_auroc(Ea, Es, groups=np.arange(n)))


def test_groups_follow_the_finite_row_mask():
    """Rows dropped for non-finite summaries take their groups with them."""
    k = 44
    Ea, Es = _twins(k, seed=11)
    Es = Es.copy()
    Es[:3] = np.nan
    v = mr.surprise_auroc(Ea, Es, groups=_pairs(k))
    assert np.isfinite(v)
    with pytest.raises(ValueError):
        mr.surprise_auroc(Ea, Es, groups=np.arange(k))


def test_grouped_twins_keep_a_real_signal():
    k = 44
    Ea, Es = _twins(k, seed=7)
    assert mr.surprise_auroc(Ea, Es + 0.5, groups=_pairs(k)) > 0.8


def _calls_in(path: str, func_name: str, callee: str) -> list[ast.Call]:
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name)
    return [c for c in ast.walk(fn) if isinstance(c, ast.Call)
            and ast.unparse(c.func).split(".")[-1] == callee]


def test_mechanism_runner_passes_twin_pair_groups():
    """run_mechanism_readouts.run_seed must hand surprise_auroc one group per twin pair."""
    calls = _calls_in(os.path.join(ROOT, "scripts", "run_mechanism_readouts.py"),
                      "run_seed", "surprise_auroc")
    assert calls, "run_seed no longer calls surprise_auroc"
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("groups") == "np.r_[np.arange(k_e), np.arange(k_e)]", kw
