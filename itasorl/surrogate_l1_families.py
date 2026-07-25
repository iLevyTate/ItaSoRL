"""L1 surrogate families for H2 ablations.

These are off-ladder observation-level perturbations used as structured comparators
in the A3 L1 substrate-grounding tests. They wrap a World and expose the identical
World protocol so the B-v2 readout pipeline can use them unchanged.
"""

from __future__ import annotations

import numpy as np

from .world import SurrogateWrapper, World


class LObsNoise(SurrogateWrapper):
    """Observation-level iid Gaussian noise surrogate.

    Adds N(0, sigma^2) to every observation dimension. The noise is white in time and
    independent of state/action; the only tell is its magnitude. Deterministic given
    a Philox seed; reseed(seed) resets the stream for reproducible per-pool collection.

    At sigma=0 the wrapper is a no-op (byte-identical to the base world).
    """

    def __init__(self, base: World, sigma: float, seed: int = 0) -> None:
        super().__init__(base)
        self.sigma = float(sigma)
        self._seed = int(seed)
        self._rng = np.random.Generator(np.random.Philox(self._seed))

    def reseed(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng = np.random.Generator(np.random.Philox(self._seed))

    def _obs_transform(self, obs: np.ndarray) -> np.ndarray:
        if self.sigma == 0.0:
            return obs
        return obs + self._rng.normal(0.0, self.sigma, size=obs.shape)


class L1Offset(SurrogateWrapper):
    """L1 discretization grid with a phase offset.

    Quantizes observations to a grid shifted by `offset`: each dimension is snapped to
    the nearest value of {k*delta + offset}. Same spacing as the headline L1 grid, but
    a different phase. Used as a same-recipe-different-parameter transfer test for H2.

    At offset=0 the wrapper is byte-identical to plain L1 quantization (provided the
    base world already applies the same delta and sensor noise).
    """

    def __init__(self, base: World, delta: float, offset: float) -> None:
        super().__init__(base)
        self.delta = float(delta)
        self.offset = float(offset)

    def _obs_transform(self, obs: np.ndarray) -> np.ndarray:
        return np.round((obs + self.offset) / self.delta) * self.delta - self.offset
