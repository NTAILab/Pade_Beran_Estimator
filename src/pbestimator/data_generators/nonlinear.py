"""Cyclic nonlinear-dependency survival DGP."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import DataGenerator


class DataGeneratorNonlinearDependency(DataGenerator):
    def __init__(self, dim: int, *, k: float = 2.5, seed: int) -> None:
        if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 5:
            raise ValueError("dim must be an integer of at least 5")
        self.dim = int(dim)
        super().__init__(k=k, seed=seed)

    def _signal(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        result = (
            4.0 * np.sin(X[:, 0])
            + np.log(np.abs(X[:, 1]) + 1.0)
            + X[:, 2] ** 2
            + np.exp(0.5 * X[:, 3])
            + np.tanh(X[:, 4])
        )
        for index in range(5, self.dim):
            cycle = (index - 5) % 3
            if cycle == 0:
                result += np.sin(X[:, index]) * np.sqrt(np.abs(X[:, index] - 1.0) + 1.0)
            elif cycle == 1:
                result += np.log(np.abs(X[:, index]) + 1.0) * np.tanh(X[:, index - 2])
            else:
                result += X[:, index] ** 2 * np.cos(X[:, index - 3])
        return result.astype(np.float64, copy=False)

    def _generate_features_and_signal(self, n_samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = self._rng.random((n_samples, self.dim), dtype=np.float64)
        return X, self._signal(X)
