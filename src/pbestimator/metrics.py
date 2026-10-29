"""Evaluation primitives for survival predictions."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score as sksurv_integrated_brier_score,
)
from sksurv.util import Surv


class _SurvivalPredictor(Protocol):
    unique_time_: NDArray[np.float64]

    def predict_survival_function(self, X: ArrayLike, times: ArrayLike) -> NDArray[np.float64]: ...


def event_time_horizon(time: ArrayLike, event: ArrayLike) -> float:
    """Return the largest observed event time from training observations."""
    observed_time, event_indicator = _validated_outcomes(time, event)
    event_times = observed_time[event_indicator]
    if event_times.size == 0:
        raise ValueError("at least one observed event is required")
    return float(np.max(event_times))


def restricted_mean_survival_time(
    estimator: _SurvivalPredictor, X: ArrayLike, t_max: float
) -> NDArray[np.float64]:
    """Integrate a fitted Beran right-continuous step function exactly to ``t_max``."""
    horizon = float(t_max)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("t_max must be finite and positive")

    jump_times = np.asarray(estimator.unique_time_, dtype=np.float64)
    interior_jumps = jump_times[(jump_times > 0.0) & (jump_times < horizon)]
    interval_starts = np.concatenate(([0.0], interior_jumps))
    interval_ends = np.concatenate((interior_jumps, [horizon]))
    survival = np.asarray(estimator.predict_survival_function(X, interval_starts), dtype=np.float64)
    return np.sum(survival * (interval_ends - interval_starts), axis=1, dtype=np.float64)


def negative_rmst_risk(
    estimator: _SurvivalPredictor, X: ArrayLike, t_max: float
) -> NDArray[np.float64]:
    """Return the higher-is-riskier score ``-RMST`` at a train-derived horizon."""
    return -restricted_mean_survival_time(estimator, X, t_max)


def concordance_index(event: ArrayLike, time: ArrayLike, risk_score: ArrayLike) -> float:
    """Return Harrell's scalar C-index from scikit-survival for higher-is-riskier scores."""
    observed_time, event_indicator = _validated_outcomes(time, event)
    risk = np.asarray(risk_score, dtype=np.float64)
    if risk.ndim != 1 or risk.shape != observed_time.shape:
        raise ValueError("risk_score must be one-dimensional and match time")
    return float(concordance_index_censored(event_indicator, observed_time, risk)[0])


def integrated_brier_score(
    train_time: ArrayLike,
    train_event: ArrayLike,
    test_time: ArrayLike,
    test_event: ArrayLike,
    survival_probabilities: ArrayLike,
    evaluation_times: ArrayLike,
) -> float:
    """Return censoring-aware IBS using train outcomes for IPCW estimation."""
    train_observed_time, train_event_indicator = _validated_outcomes(train_time, train_event)
    test_observed_time, test_event_indicator = _validated_outcomes(test_time, test_event)
    probabilities = np.asarray(survival_probabilities, dtype=np.float64)
    times = np.asarray(evaluation_times, dtype=np.float64)

    return float(
        sksurv_integrated_brier_score(
            Surv.from_arrays(event=train_event_indicator, time=train_observed_time),
            Surv.from_arrays(event=test_event_indicator, time=test_observed_time),
            probabilities,
            times,
        )
    )


def _validated_outcomes(time: ArrayLike, event: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    observed_time = np.asarray(time, dtype=np.float64)
    event_indicator = np.asarray(event)
    if observed_time.ndim != 1 or event_indicator.ndim != 1 or observed_time.shape != event_indicator.shape:
        raise ValueError("time and event must be matching one-dimensional arrays")
    if observed_time.size == 0:
        raise ValueError("time and event must be non-empty")
    if not np.all(np.isfinite(observed_time)):
        raise ValueError("time must contain only finite values")
    if not np.all(np.isin(event_indicator, [0, 1])):
        raise ValueError("event must contain only 0 and 1")
    return observed_time, event_indicator.astype(bool, copy=False)
