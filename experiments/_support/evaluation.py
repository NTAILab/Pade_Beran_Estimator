"""Train-only horizons and administrative censoring for series metrics."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pbestimator.metrics import event_time_horizon


def ibs_reporting_horizon(train_time: ArrayLike, train_event: ArrayLike) -> float:
    """Return the train-only administrative-censoring horizon for outer-fold IBS."""
    event_indicator, observed_time = _validated_outcomes(train_event, train_time)
    event_horizon = event_time_horizon(observed_time, event_indicator)
    followup_limit = float(np.nextafter(np.max(observed_time), -np.inf))
    return min(event_horizon, followup_limit)


def administratively_censor_test_outcomes(
    test_time: ArrayLike, test_event: ArrayLike, *, horizon: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Copy test outcomes and censor observations at or after a reporting horizon."""
    event_indicator, observed_time = _validated_outcomes(test_event, test_time)
    reporting_horizon = float(horizon)
    if not np.isfinite(reporting_horizon) or reporting_horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    censored_time = np.minimum(observed_time, reporting_horizon)
    censored_event = event_indicator & (observed_time < reporting_horizon)
    return censored_time, censored_event


def _validated_outcomes(event: ArrayLike, time: ArrayLike) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    event_indicator = np.asarray(event, dtype=bool)
    observed_time = np.asarray(time, dtype=np.float64)
    if event_indicator.ndim != 1 or observed_time.ndim != 1 or event_indicator.shape != observed_time.shape:
        raise ValueError("event and time must be matching one-dimensional arrays")
    if event_indicator.size == 0 or not np.all(np.isfinite(observed_time)):
        raise ValueError("time must be a non-empty finite array")
    return event_indicator, observed_time



def evaluation_time_grid(
    train_time: ArrayLike, test_time: ArrayLike, *, t_max: float, n_times: int = 50
) -> NDArray[np.float64]:
    """Create a deterministic grid strictly within both holdout follow-up supports."""
    train = _finite_time_array(train_time, "train_time")
    test = _finite_time_array(test_time, "test_time")
    horizon = float(t_max)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("t_max must be finite and positive")
    if n_times < 2:
        raise ValueError("n_times must be at least 2")

    lower = np.nextafter(max(float(train.min()), float(test.min())), np.inf)
    upper = np.nextafter(min(horizon, float(train.max()), float(test.max())), -np.inf)
    if not lower < upper:
        raise ValueError("no non-empty evaluation interval exists inside train and test follow-up support")

    grid = np.linspace(lower, upper, num=n_times, dtype=np.float64)
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0.0):
        raise ValueError("evaluation grid must be finite and strictly increasing")
    return grid


def _finite_time_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{name} must contain only finite values")
    return times
