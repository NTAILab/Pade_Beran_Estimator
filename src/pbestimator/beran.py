"""Gaussian-kernel Beran estimators with exact and approximate log increments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _BeranEstimatorBase:
    """Shared non-parametric Beran core; subclasses define log increments."""

    def __init__(self, tau: float) -> None:
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be positive and finite")
        self.tau = float(tau)

    def fit(self, X: ArrayLike, time: ArrayLike, event: ArrayLike) -> "_BeranEstimatorBase":
        """Validate and retain observations used by the non-trainable estimator."""
        features = self._as_feature_matrix(X, "X")
        observed_time = np.asarray(time, dtype=np.float64)
        event_indicator = np.asarray(event)

        if observed_time.ndim != 1 or event_indicator.ndim != 1:
            raise ValueError("time and event must be one-dimensional")
        if len(features) == 0:
            raise ValueError("X must contain at least one observation")
        if len(features) != len(observed_time) or len(features) != len(event_indicator):
            raise ValueError("X, time, and event must have the same length")
        if not np.all(np.isfinite(observed_time)):
            raise ValueError("time must contain only finite values")
        if not np.all(np.isin(event_indicator, [0, 1])):
            raise ValueError("event must contain only 0 and 1")

        self.X_ = features
        self.time_ = observed_time
        self.event_ = event_indicator.astype(bool, copy=False)
        self._time_order = np.argsort(observed_time, kind="stable")
        self._sorted_time_ = observed_time[self._time_order]
        self._sorted_event_ = self.event_[self._time_order]
        self.unique_time_, self._group_starts_ = np.unique(self._sorted_time_, return_index=True)
        self._event_group_ = np.add.reduceat(self._sorted_event_.astype(np.intp), self._group_starts_) > 0
        return self

    def predict_survival_function(self, X: ArrayLike, times: ArrayLike) -> NDArray[np.float64]:
        """Return ``S(t | x)`` for every query row and requested time."""
        if not hasattr(self, "X_"):
            raise RuntimeError("BeranEstimator must be fitted before prediction")

        queries = self._as_feature_matrix(X, "X")
        if queries.shape[1] != self.X_.shape[1]:
            raise ValueError("X has a different number of features than the fitted data")
        requested_times = np.asarray(times, dtype=np.float64)
        if requested_times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if not np.all(np.isfinite(requested_times)):
            raise ValueError("times must contain only finite values")

        logits = self._kernel_logits(queries)
        survival_after_time = self._predict_survival_after_observed_times_vectorized(logits)

        positions = np.searchsorted(self.unique_time_, requested_times, side="right") - 1
        prediction = np.ones((len(queries), len(requested_times)), dtype=np.float64)
        covered = positions >= 0
        prediction[:, covered] = survival_after_time[:, positions[covered]]
        return prediction

    def _predict_survival_after_observed_times_vectorized(
        self, logits: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate grouped hazards for all query rows in the log domain."""
        return self._survival_from_hazards(self._hazards_from_logits(logits))

    def _hazards_from_logits(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return grouped hazards at ``unique_time_`` for all estimators."""
        sorted_logits = logits[:, self._time_order]
        reverse_log_risk = np.logaddexp.accumulate(sorted_logits[:, ::-1], axis=1)[:, ::-1]
        log_risk = reverse_log_risk[:, self._group_starts_]
        event_logits = np.where(self._sorted_event_[np.newaxis, :], sorted_logits, -np.inf)
        log_event = np.logaddexp.reduceat(event_logits, self._group_starts_, axis=1)
        hazards = np.zeros_like(log_risk)
        event_groups = self._event_group_
        with np.errstate(over="ignore", invalid="ignore"):
            hazards[:, event_groups] = np.exp(log_event[:, event_groups] - log_risk[:, event_groups])
        return hazards

    def _survival_from_hazards(self, hazards: NDArray[np.float64]) -> NDArray[np.float64]:
        """Accumulate hazards into survival after each observed time; never mutates ``hazards``."""
        with np.errstate(divide="ignore"):
            cumulative_log_survival = np.cumsum(self._log_increment(hazards), axis=1, dtype=np.float64)
        return np.exp(cumulative_log_survival)

    def _log_increment(self, hazard: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        raise NotImplementedError

    def _kernel_logits(self, queries: NDArray[np.float64]) -> NDArray[np.float64]:
        with np.errstate(over="ignore"):
            squared_distance = np.sum((queries[:, np.newaxis, :] - self.X_[np.newaxis, :, :]) ** 2, axis=2)
            logits = -squared_distance / self.tau
        if np.any(np.all(np.isneginf(logits), axis=1)):
            raise FloatingPointError(
                "all Gaussian kernel logits underflowed; this query cannot be represented in float64"
            )
        return logits

    @staticmethod
    def _as_feature_matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional feature matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite values")
        return matrix


class BeranEstimator(_BeranEstimatorBase):
    """Exact Beran estimator with log accumulation."""

    def _log_increment(self, hazard: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        with np.errstate(divide="ignore"):
            return np.log1p(-hazard)


class TaylorBeranEstimator(_BeranEstimatorBase):
    """Beran estimator using the first-order log increment ``-hazard``."""

    def _log_increment(self, hazard: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        return -hazard


class PadeBeranEstimator(_BeranEstimatorBase):
    """Beran estimator using the [1/1] Padé log increment ``-2h / (2-h)``."""

    def _log_increment(self, hazard: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        return -2.0 * hazard / (2.0 - hazard)
