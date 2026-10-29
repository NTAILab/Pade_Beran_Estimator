"""NumPy gradient-descent training for the Beran estimator's bandwidth."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from time import perf_counter_ns

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv


TAU_EPS = 1e-12


@dataclass(frozen=True)
class TauTrainingResult:
    """Final bandwidth and its per-epoch training history."""

    tau: float
    losses: NDArray[np.float64]
    taus: NDArray[np.float64]
    step_times_ms: NDArray[np.float64]
    forward_times_ms: NDArray[np.float64]
    loss_times_ms: NDArray[np.float64]
    backward_times_ms: NDArray[np.float64]


@dataclass(frozen=True)
class _SupportGrouping:
    """Time-based support structure shared by every training epoch."""

    order: NDArray[np.intp]
    unique_times: NDArray[np.float64]
    group_starts: NDArray[np.intp]
    event_mask: NDArray[np.bool_]
    group_ids: NDArray[np.intp]


def _prepare_support_grouping(
    support_time: NDArray[np.float64], support_event: NDArray[np.bool_]
) -> _SupportGrouping:
    order = np.argsort(support_time, kind="stable")
    sorted_time = support_time[order]
    unique_times, group_starts = np.unique(sorted_time, return_index=True)
    group_ids = np.repeat(
        np.arange(len(group_starts)), np.diff(np.append(group_starts, len(order)))
    )
    return _SupportGrouping(
        order=order,
        unique_times=unique_times,
        group_starts=group_starts,
        event_mask=support_event[order],
        group_ids=group_ids,
    )


def _forward_survival(
    tau: float,
    support_x: NDArray[np.float64],
    grouping: _SupportGrouping,
    query_x: NDArray[np.float64],
    grid: NDArray[np.float64],
    increment_kind: str,
    leave_one_out: bool,
) -> tuple[NDArray[np.float64], dict[str, NDArray]]:
    """Compute survival and retain forward values needed by backpropagation."""
    squared_distance = np.sum((query_x[:, None, :] - support_x[None, :, :]) ** 2, axis=2)
    logits = -squared_distance / tau
    if leave_one_out:
        if query_x.shape[0] != support_x.shape[0] or not np.array_equal(query_x, support_x):
            raise ValueError("leave_one_out requires query_x to align exactly with support_x")
        diagonal = np.arange(len(support_x))
        logits[diagonal, diagonal] = -np.inf
    sorted_logits = logits[:, grouping.order]
    suffix_log_risk = np.logaddexp.accumulate(sorted_logits[:, ::-1], axis=1)[:, ::-1]
    log_risk = suffix_log_risk[:, grouping.group_starts]
    event_logits = np.where(grouping.event_mask[None, :], sorted_logits, -np.inf)
    group_max = np.maximum.reduceat(event_logits, grouping.group_starts, axis=1)
    safe_group_max = np.where(np.isfinite(group_max), group_max, 0.0)
    event_weights = np.exp(event_logits - safe_group_max[:, grouping.group_ids])
    event_sums = np.add.reduceat(event_weights, grouping.group_starts, axis=1)
    log_event = safe_group_max + np.log(np.where(event_sums > 0, event_sums, 1.0))
    valid = np.isfinite(group_max) & np.isfinite(log_risk) & (event_sums > 0)
    hazards = np.where(valid, np.exp(log_event - log_risk), 0.0)

    # Match the exact-log estimator's terminal branch: survival becomes zero,
    # and the singular terminal increment contributes zero derivative.
    if increment_kind == "exact_log":
        terminal = hazards >= 1.0
        safe_hazard = np.where(terminal, 0.0, hazards)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_increment = np.log1p(-safe_hazard)
        log_increment[terminal] = -np.inf
    elif increment_kind == "taylor_log":
        log_increment = -hazards
    else:  # pade_log
        denominator = 2.0 - hazards
        log_increment = -2.0 * hazards / denominator

    survival_at_unique = np.exp(np.cumsum(log_increment, axis=1))
    positions = np.searchsorted(grouping.unique_times, grid, side="right") - 1
    survival = np.ones((len(query_x), len(grid)), dtype=np.float64)
    covered = positions >= 0
    survival[:, covered] = survival_at_unique[:, positions[covered]]
    cache = {
        "squared_distance": squared_distance,
        "logits": logits,
        "hazards": hazards,
        "survival": survival,
        "survival_at_unique": survival_at_unique,
        "positions": positions,
        "unique_times": grouping.unique_times,
        "order": grouping.order,
        "group_starts": grouping.group_starts,
        "event_mask": grouping.event_mask,
        "sorted_logits": sorted_logits,
        "event_weights": event_weights,
        "event_sums": event_sums,
    }
    return survival, cache


def _backward_tau_gradient(
    tau: float,
    cache: dict[str, NDArray],
    support_time: NDArray[np.float64],
    increment_kind: str,
    leave_one_out: bool,
) -> NDArray[np.float64]:
    """Backpropagate survival sensitivities to tau for the fixed support."""
    logits = cache["logits"]
    hazards = cache["hazards"]
    survival = cache["survival"]
    dlogits = cache["squared_distance"] / (tau * tau)
    if leave_one_out:
        diagonal = np.arange(len(support_time))
        dlogits[diagonal, diagonal] = 0.0
    sorted_logits = cache["sorted_logits"]
    sorted_dlogits = dlogits[:, cache["order"]]
    risk_logsum = np.full(len(logits), -np.inf, dtype=np.float64)
    risk_mean = np.zeros(len(logits), dtype=np.float64)
    risk_means = np.zeros_like(hazards)
    start_to_group = {int(start): group for group, start in enumerate(cache["group_starts"])}
    for index in range(len(support_time) - 1, -1, -1):
        next_logsum = np.logaddexp(risk_logsum, sorted_logits[:, index])
        previous_weight = np.zeros_like(risk_logsum)
        current_weight = np.zeros_like(risk_logsum)
        previous_valid = np.isfinite(risk_logsum)
        current_valid = np.isfinite(sorted_logits[:, index])
        previous_delta = np.zeros_like(risk_logsum)
        current_delta = np.zeros_like(risk_logsum)
        np.subtract(risk_logsum, next_logsum, out=previous_delta, where=previous_valid)
        np.subtract(sorted_logits[:, index], next_logsum, out=current_delta, where=current_valid)
        np.exp(
            previous_delta,
            out=previous_weight,
            where=previous_valid,
        )
        np.exp(
            current_delta,
            out=current_weight,
            where=current_valid,
        )
        risk_mean = previous_weight * risk_mean + current_weight * sorted_dlogits[:, index]
        risk_logsum = next_logsum
        group = start_to_group.get(index)
        if group is not None:
            risk_means[:, group] = risk_mean

    event_moments = np.add.reduceat(
        cache["event_weights"] * sorted_dlogits * cache["event_mask"][None, :],
        cache["group_starts"],
        axis=1,
    )
    dlog_event = np.divide(
        event_moments,
        cache["event_sums"],
        out=np.zeros_like(event_moments),
        where=cache["event_sums"] > 0,
    )
    dhazards = hazards * (dlog_event - risk_means)

    if increment_kind == "exact_log":
        terminal = hazards >= 1.0
        safe_hazard = np.where(terminal, 0.0, hazards)
        with np.errstate(divide="ignore", invalid="ignore"):
            dlog_increment = -dhazards / (1.0 - safe_hazard)
        dlog_increment[terminal] = 0.0
    elif increment_kind == "taylor_log":
        dlog_increment = -dhazards
    else:
        dlog_increment = -4.0 * dhazards / (2.0 - hazards) ** 2

    dsurvival_at_unique = cache["survival_at_unique"] * np.cumsum(dlog_increment, axis=1)
    dsurvival = np.zeros_like(survival)
    positions = cache["positions"]
    covered = positions >= 0
    dsurvival[:, covered] = dsurvival_at_unique[:, positions[covered]]
    return dsurvival


def train_tau_numpy(
    initial_tau: float,
    support_x: ArrayLike,
    support_time: ArrayLike,
    support_event: ArrayLike,
    eval_x: ArrayLike,
    eval_time: ArrayLike,
    eval_event: ArrayLike,
    grid: ArrayLike,
    *,
    learning_rate: float = 0.01,
    epochs: int = 100,
    increment_kind: str = "exact_log",
    leave_one_out: bool = False,
    on_step: Callable[[int, float, float, float, float, float, float], None] | None = None,
) -> TauTrainingResult:
    """Train only Beran's ``tau`` by analytic-gradient descent on IPCW IBS.

    The Gaussian-kernel support and all observations stay fixed. ``learning_rate``
    is the gradient-descent step size. A squared internal parameterization keeps
    tau positive while the update follows the chain rule. ``increment_kind``
    selects the exact-log, Taylor-log, or Pade-log survival accumulation.
    """
    if not np.isfinite(initial_tau) or initial_tau <= TAU_EPS:
        raise ValueError(f"initial_tau must be finite and greater than {TAU_EPS}")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if increment_kind not in ("exact_log", "taylor_log", "pade_log"):
        raise ValueError("increment_kind must be exact_log, taylor_log, or pade_log")

    x = np.asarray(support_x, dtype=np.float64)
    time = np.asarray(support_time, dtype=np.float64)
    event = np.asarray(support_event)
    qx = np.asarray(eval_x, dtype=np.float64)
    etime = np.asarray(eval_time, dtype=np.float64)
    eevent = np.asarray(eval_event)
    times = np.asarray(grid, dtype=np.float64)
    if x.ndim != 2 or qx.ndim != 2 or x.shape[1] != qx.shape[1] or x.shape[0] == 0:
        raise ValueError("support_x and eval_x must be non-empty compatible feature matrices")
    if time.shape != (len(x),) or event.shape != (len(x),):
        raise ValueError("support_time and support_event must match support_x")
    if etime.shape != (len(qx),) or eevent.shape != (len(qx),):
        raise ValueError("eval_time and eval_event must match eval_x")
    if not np.all(np.isin(event, [0, 1])) or not np.all(np.isin(eevent, [0, 1])):
        raise ValueError("event arrays must contain only 0 and 1")
    if not all(np.all(np.isfinite(a)) for a in (x, time, qx, etime, times)):
        raise ValueError("features, observed times, and grid must be finite")
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("grid must contain at least two strictly increasing times")

    censor = CensoringDistributionEstimator().fit(
        Surv.from_arrays(event=event.astype(bool), time=time)
    )
    ipcw = censor.predict_ipcw(Surv.from_arrays(event=eevent.astype(bool), time=etime))
    censor_survival = censor.predict_proba(times)
    if np.any(censor_survival <= 0):
        raise ValueError("zero censoring survival on grid")
    event_before = (etime[:, None] <= times[None, :]) & eevent.astype(bool)[:, None]
    survived = etime[:, None] > times[None, :]
    event_weight = ipcw[:, None]
    survived_weight = 1.0 / censor_survival[None, :]
    duration = times[-1] - times[0]

    # Match the Torch model's raw_tau**2 + epsilon parameterization.
    raw_tau = np.sqrt(initial_tau - TAU_EPS)
    support_event = event.astype(bool)
    grouping = _prepare_support_grouping(time, support_event)
    tau_history = np.empty(epochs, dtype=np.float64)
    loss_history = np.empty(epochs, dtype=np.float64)
    step_times_ms = np.empty(epochs, dtype=np.float64)
    forward_times_ms = np.empty(epochs, dtype=np.float64)
    loss_times_ms = np.empty(epochs, dtype=np.float64)
    backward_times_ms = np.empty(epochs, dtype=np.float64)

    def loss_from_survival(survival: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        score = np.mean(
            survival**2 * event_before * event_weight
            + (1.0 - survival) ** 2 * survived * survived_weight,
            axis=0,
        )
        loss = float(np.trapezoid(score, times) / duration)
        loss_sensitivity = (
            2.0 * survival * event_before * event_weight
            + 2.0 * (survival - 1.0) * survived * survived_weight
        )
        return loss, loss_sensitivity

    for epoch in range(epochs):
        tau = raw_tau * raw_tau + TAU_EPS
        forward_start = perf_counter_ns()
        survival, cache = _forward_survival(
            tau, x, grouping, qx, times, increment_kind, leave_one_out
        )
        forward_times_ms[epoch] = (perf_counter_ns() - forward_start) / 1e6

        loss_start = perf_counter_ns()
        loss, loss_sensitivity = loss_from_survival(survival)
        loss_times_ms[epoch] = (perf_counter_ns() - loss_start) / 1e6

        backward_start = perf_counter_ns()
        dsurvival = _backward_tau_gradient(
            tau, cache, time, increment_kind, leave_one_out
        )
        dscore = np.mean(loss_sensitivity * dsurvival, axis=0)
        dloss_dtau = float(np.trapezoid(dscore, times) / duration)
        raw_tau -= learning_rate * dloss_dtau * (2.0 * raw_tau)
        if not np.isfinite(raw_tau):
            raise FloatingPointError("gradient update produced a non-finite tau")
        backward_times_ms[epoch] = (perf_counter_ns() - backward_start) / 1e6
        step_times_ms[epoch] = (
            forward_times_ms[epoch] + loss_times_ms[epoch] + backward_times_ms[epoch]
        )
        loss_history[epoch] = loss
        tau_history[epoch] = raw_tau * raw_tau + TAU_EPS
        if on_step is not None:
            on_step(
                epoch + 1,
                float(loss),
                float(tau_history[epoch]),
                float(forward_times_ms[epoch]),
                float(loss_times_ms[epoch]),
                float(backward_times_ms[epoch]),
                float(step_times_ms[epoch]),
            )

    return TauTrainingResult(
        tau=float(tau_history[-1]),
        losses=loss_history,
        taus=tau_history,
        step_times_ms=step_times_ms,
        forward_times_ms=forward_times_ms,
        loss_times_ms=loss_times_ms,
        backward_times_ms=backward_times_ms,
    )
