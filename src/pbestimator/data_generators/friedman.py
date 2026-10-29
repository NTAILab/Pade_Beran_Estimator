"""Friedman survival data-generating processes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import DataGenerator


def _generate_friedman23_features(
    rng: np.random.Generator, n_samples: int
) -> NDArray[np.float64]:
    raw = rng.random((n_samples, 4))
    features = np.empty((n_samples, 4), dtype=np.float64)
    features[:, 0] = 100.0 * raw[:, 0]
    features[:, 1] = 520.0 * np.pi * raw[:, 1] + 40.0 * np.pi
    features[:, 2] = raw[:, 2]
    features[:, 3] = 10.0 * raw[:, 3] + 1.0
    return features


class DataGeneratorFriedman1(DataGenerator):
    def __init__(self, dim: int = 5, *, k: float = 2.5, seed: int) -> None:
        if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 5:
            raise ValueError("dim must be an integer of at least 5")
        self.dim = int(dim)
        super().__init__(k=k, seed=seed)

    def _signal(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20.0 * (X[:, 2] - 0.5) ** 2
            + 10.0 * X[:, 3]
            + 5.0 * X[:, 4]
        ).astype(np.float64, copy=False)

    def _generate_features_and_signal(self, n_samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = self._rng.random((n_samples, self.dim), dtype=np.float64)
        return X, self._signal(X)


class DataGeneratorFriedman2(DataGenerator):
    def __init__(self, *, k: float = 2.5, seed: int) -> None:
        super().__init__(k=k, seed=seed)

    def _signal(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        middle = X[:, 1] * X[:, 2] - 1.0 / (X[:, 1] * X[:, 3])
        return np.sqrt(X[:, 0] ** 2 + middle**2).astype(np.float64, copy=False)

    def _generate_features_and_signal(self, n_samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X = _generate_friedman23_features(self._rng, n_samples)
        return X, self._signal(X)


class DataGeneratorFriedman3(DataGenerator):
    def __init__(self, *, k: float = 2.5, seed: int) -> None:
        super().__init__(k=k, seed=seed)

    def _ratio_and_signal(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (X[:, 1] * X[:, 2] - 1.0 / (X[:, 1] * X[:, 3])) / X[:, 0]
            signal = np.arctan(ratio).astype(np.float64, copy=False)
        return ratio, signal

    def _signal(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        _, signal = self._ratio_and_signal(X)
        return signal

    def _generate_features_and_signal(self, n_samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        accepted_features: list[NDArray[np.float64]] = []
        remaining = n_samples
        while remaining:
            candidates = _generate_friedman23_features(self._rng, remaining)
            ratio, candidate_signal = self._ratio_and_signal(candidates)
            accepted = (
                (candidates[:, 0] != 0.0)
                & np.isfinite(ratio)
                & np.isfinite(candidate_signal)
                & (candidate_signal >= 0.0)
            )
            if np.any(accepted):
                accepted_features.append(candidates[accepted])
            remaining = n_samples - sum(batch.shape[0] for batch in accepted_features)
        X = np.concatenate(accepted_features, axis=0)
        return X, self._signal(X)
