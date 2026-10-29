"""Strong pairwise-interaction survival DGP."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import DataGenerator


class DataGeneratorStrongFeatureInteractions(DataGenerator):
    def __init__(self, dim: int, *, k: float = 2.5, seed: int) -> None:
        if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 2:
            raise ValueError("dim must be an integer of at least 2")
        self.dim = int(dim)
        super().__init__(k=k, seed=seed)
        upper = self._structural_rng.uniform(0.0, 1.0, size=(self.dim, self.dim))
        self._W = np.triu(upper, k=1)
        self._W += self._W.T

    def _signal(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.einsum("bi,ij,bj->b", X, np.triu(self._W, k=1), X).astype(np.float64, copy=False)

    def _generate_features_and_signal(self, n_samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = self._rng.random((n_samples, self.dim), dtype=np.float64)
        return X, self._signal(X)
