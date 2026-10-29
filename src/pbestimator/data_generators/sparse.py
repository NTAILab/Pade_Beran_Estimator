"""Sparse linear survival DGP."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import DataGenerator


class DataGeneratorSparseFeatures(DataGenerator):
    def __init__(self, dim: int, sparsity: float, *, k: float = 2.5, seed: int) -> None:
        if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not np.isfinite(sparsity) or not 0.0 < sparsity < 1.0:
            raise ValueError("sparsity must be finite and in (0, 1)")
        self.dim = int(dim)
        self.sparsity = float(sparsity)
        super().__init__(k=k, seed=seed)
        self._weights = self._structural_rng.random(self.dim, dtype=np.float64)

    def _signal(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return (X @ self._weights).astype(np.float64, copy=False)

    def _generate_features_and_signal(self, n_samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = self._rng.random((n_samples, self.dim), dtype=np.float64)
        X[self._rng.random((n_samples, self.dim)) >= self.sparsity] = 0.0
        return X, self._signal(X)
