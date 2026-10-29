"""Shared machinery for synthetic survival data-generating processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import numpy as np
from numpy.typing import NDArray

from pbestimator.synthetic import SyntheticSurvivalDataset


def weibull_event_times(
    signal: NDArray[np.float64], *, k: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Transform non-negative mean signals into Weibull event times."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("signal must be a finite one-dimensional non-negative array")
    uniforms = np.maximum(rng.random(values.shape[0]), np.nextafter(0.0, 1.0))
    gamma_factor = math.gamma(1.0 + 1.0 / k)
    return ((values / gamma_factor) * np.power(-np.log(uniforms), 1.0 / k)).astype(
        np.float64, copy=False
    )


class DataGenerator(ABC):
    """Base class owning deterministic sampling and legacy censoring."""

    def __init__(self, *, k: float, seed: int) -> None:
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("k must be positive and finite")
        self.k = float(k)
        seed_sequence = np.random.SeedSequence(seed)
        structural_seed, sample_seed = seed_sequence.spawn(2)
        self._structural_rng = np.random.default_rng(structural_seed)
        self._rng = np.random.default_rng(sample_seed)

    @abstractmethod
    def _generate_features_and_signal(
        self, n_samples: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Produce one population's covariates and non-negative signal."""

    def generate(
        self, n_samples: int, *, censor_rate: float = 0.0
    ) -> SyntheticSurvivalDataset:
        if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if not np.isfinite(censor_rate) or not 0.0 <= censor_rate <= 1.0:
            raise ValueError("censor_rate must be finite and in [0, 1]")

        X, signal = self._generate_features_and_signal(int(n_samples))
        features = np.asarray(X, dtype=np.float64)
        values = np.asarray(signal, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != n_samples or not np.all(np.isfinite(features)):
            raise ValueError("X must be a finite two-dimensional array with n_samples rows")
        if values.ndim != 1 or values.shape[0] != n_samples or not np.all(np.isfinite(values)):
            raise ValueError("signal must be a finite one-dimensional array with n_samples values")
        if np.any(values < 0.0):
            raise ValueError("signal must be non-negative")

        event_time = weibull_event_times(values, k=self.k, rng=self._rng)
        observed_time, event = self._apply_legacy_censoring(event_time, float(censor_rate))
        return SyntheticSurvivalDataset(
            X=features,
            event_time=event_time,
            observed_time=observed_time,
            event=event,
        )

    def _apply_legacy_censoring(
        self, event_time: NDArray[np.float64], censor_rate: float
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        n_samples = event_time.shape[0]
        n_censored = math.floor(n_samples * censor_rate)
        event = np.ones(n_samples, dtype=np.bool_)
        observed_time = event_time.copy()
        if n_censored:
            indices = self._rng.choice(n_samples, size=n_censored, replace=False)
            factors = self._rng.uniform(0.05, 0.95, size=n_censored)
            event[indices] = False
            observed_time[indices] = event_time[indices] * factors
        return observed_time.astype(np.float64, copy=False), event
