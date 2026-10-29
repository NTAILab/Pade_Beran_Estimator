"""Fidelity of Taylor/Padé Beran curves to the Exact Log reference.

These diagnostics measure how closely an approximation reproduces the Exact Log
Beran survival ``S_ref(t|x)``.  They are NOT errors against analytical DGP truth
and are unrelated to predictive C-index/IBS.  Beran curves are right-continuous
step functions with jumps only at the fitted observed times, so all integrals are
evaluated exactly per interval over ``[0, T]``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

SURVIVAL_LOG_EPS = 1e-12
METRIC_NAMES = ("survival_ise", "survival_iae", "survival_sup", "cumulative_hazard_iae")
REFERENCE_MODEL = "exact_log"


@dataclass(frozen=True)
class ApproximationErrors:
    """Per-test-object errors; ``fold_means`` averages over objects."""

    survival_ise: NDArray[np.float64]
    survival_iae: NDArray[np.float64]
    survival_sup: NDArray[np.float64]
    cumulative_hazard_iae: NDArray[np.float64]

    def fold_means(self) -> dict[str, float]:
        return {name: float(np.mean(getattr(self, name))) for name in METRIC_NAMES}


def _validate_survival(values: NDArray[np.float64], name: str) -> None:
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be finite and lie in [0, 1]")


def approximation_error_metrics(
    reference_survival: NDArray[np.float64],
    model_survival: NDArray[np.float64],
    interval_widths: NDArray[np.float64],
    horizon: float,
    *,
    epsilon: float = SURVIVAL_LOG_EPS,
    reference_at_horizon: NDArray[np.float64] | None = None,
    model_at_horizon: NDArray[np.float64] | None = None,
) -> ApproximationErrors:
    """Exact step-function ISE/IAE/sup/cumulative-hazard IAE per test object.

    Column ``j`` of both survival matrices is the constant value on the ``j``-th
    integration interval of width ``interval_widths[j]``; the widths tile ``[0, T]``.
    ``*_at_horizon`` are the right-continuous values ``S(T|x)``: the sup is over the
    closed ``[0, T]`` and must see a jump exactly at ``T``.  A single point has zero
    measure, so they never affect ISE/IAE/E_H.
    """
    reference = np.asarray(reference_survival, dtype=np.float64)
    model = np.asarray(model_survival, dtype=np.float64)
    widths = np.asarray(interval_widths, dtype=np.float64)
    _validate_survival(reference, "reference_survival")
    _validate_survival(model, "model_survival")
    if reference.shape != model.shape:
        raise ValueError("reference and model survival must have the same shape")
    if widths.ndim != 1 or len(widths) != reference.shape[1]:
        raise ValueError("interval_widths must have one entry per survival column")
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("interval widths must be finite and positive")
    if isinstance(horizon, bool) or not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isclose(math.fsum(widths), horizon, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("interval widths must tile [0, horizon]")
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be finite and in (0, 1)")
    if (reference_at_horizon is None) != (model_at_horizon is None):
        raise ValueError("reference and model survival at the horizon must be given together")
    if reference_at_horizon is not None:
        endpoints = np.column_stack((np.asarray(reference_at_horizon, dtype=np.float64), np.asarray(model_at_horizon, dtype=np.float64)))             if np.ndim(reference_at_horizon) == 1 and np.ndim(model_at_horizon) == 1 and len(reference_at_horizon) == len(model_at_horizon) == reference.shape[0] else None
        if endpoints is None:
            raise ValueError("survival at the horizon must have one value per object")
        _validate_survival(endpoints, "survival at the horizon")

    difference = reference - model
    absolute = np.abs(difference)
    # epsilon is diagnostic-only: np.maximum returns new arrays, predictions stay untouched.
    hazard_difference = np.abs(np.log(np.maximum(model, epsilon)) - np.log(np.maximum(reference, epsilon)))
    return ApproximationErrors(
        survival_ise=(difference**2) @ widths / horizon,
        survival_iae=absolute @ widths / horizon,
        survival_sup=np.max(absolute, axis=1) if reference_at_horizon is None
        else np.maximum(np.max(absolute, axis=1), np.abs(endpoints[:, 0] - endpoints[:, 1])),
        cumulative_hazard_iae=hazard_difference @ widths / horizon,
    )


def step_integration_intervals(unique_times: NDArray[np.float64], horizon: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return interval starts and widths tiling ``[0, T]`` at the observed jump times."""
    if isinstance(horizon, bool) or not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    times = np.unique(np.asarray(unique_times, dtype=np.float64))
    interior = times[(times > 0.0) & (times < horizon)]
    starts = np.concatenate(([0.0], interior))
    widths = np.concatenate((interior, [horizon])) - starts
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("integration interval widths must be finite and positive")
    return starts, widths


def fold_approximation_errors(
    models: Mapping[str, object],
    X_test: NDArray[np.float64],
    horizon: float,
    *,
    hazards: NDArray[np.float64] | None = None,
) -> dict[str, dict[str, float]]:
    """Fold-level (mean-over-objects) errors of every model relative to Exact Log.

    All models share one fitted fold, so they share hazards and jump locations;
    each survival curve is accumulated from the same hazard matrix.  The reference
    model's own errors are computed against the identical array and are exactly 0.
    """
    reference_model = models[REFERENCE_MODEL]
    if hazards is None:
        hazards = reference_model._hazards_from_logits(reference_model._kernel_logits(np.asarray(X_test, dtype=np.float64)))
    starts, widths = step_integration_intervals(reference_model.unique_time_, horizon)
    # interval starts plus the closed endpoint T (right-continuous, used only by the sup)
    points = np.append(starts, float(horizon))
    positions = np.searchsorted(reference_model.unique_time_, points, side="right") - 1
    covered = positions >= 0

    def at_points(model) -> NDArray[np.float64]:
        after = model._survival_from_hazards(hazards)
        curve = np.ones((after.shape[0], len(points)), dtype=np.float64)
        curve[:, covered] = after[:, positions[covered]]
        return curve

    reference = at_points(reference_model)
    result = {}
    for name, model in models.items():
        curve = reference if name == REFERENCE_MODEL else at_points(model)
        result[name] = approximation_error_metrics(
            reference[:, :-1], curve[:, :-1], widths, horizon,
            reference_at_horizon=reference[:, -1], model_at_horizon=curve[:, -1],
        ).fold_means()
        del curve
    if any(value != 0.0 for value in result[REFERENCE_MODEL].values()):
        raise ValueError("Exact Log self-error must be exactly zero")
    return result
