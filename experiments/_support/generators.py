"""Generator configuration and prediction timing used by the series."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
import numpy as np
from pbestimator import data_generators
from . import timing
from pbestimator.beran import BeranEstimator


MODEL_NAMES = ("exact_log", "taylor_log", "pade_log")


_VARIABLE_DIMENSION_DATASETS = {
    "friedman1",
    "strong_feature_interactions",
    "sparse_features",
    "nonlinear_dependency",
}




@dataclass(frozen=True)
class GeneratorConfig:
    """One generator fallback setting, ordered by its search level."""

    dataset: str
    level: int
    tau: float
    dim: int | None
    k: float
    sparsity: float | None = None
    parameters: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.parameters is not None:
            object.__setattr__(self, "parameters", _freeze_identity_value(
                _normalize_identity_value(self.parameters),
            ))

def _require_integer(value: int, name: str, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def make_generator(config: GeneratorConfig, *, seed: int) -> data_generators.DataGenerator:
    """Construct exactly the configured DGP, rejecting ignored/ambiguous fields."""
    classes = {
        "friedman1": data_generators.DataGeneratorFriedman1,
        "friedman2": data_generators.DataGeneratorFriedman2,
        "friedman3": data_generators.DataGeneratorFriedman3,
        "strong_feature_interactions": data_generators.DataGeneratorStrongFeatureInteractions,
        "sparse_features": data_generators.DataGeneratorSparseFeatures,
        "nonlinear_dependency": data_generators.DataGeneratorNonlinearDependency,
    }
    if config.dataset not in classes:
        raise ValueError(f"unknown dataset: {config.dataset}")
    _require_integer(seed, "seed", 0)
    _require_integer(config.level, "level", 0)
    _require_positive_finite(config.tau, "tau")
    _require_positive_finite(config.k, "k")
    if config.parameters is not None and (not isinstance(config.parameters, Mapping) or config.parameters):
        raise ValueError("unsupported generator parameters; use explicit configuration fields")
    kwargs = {"k": config.k, "seed": seed}
    if config.dataset in _VARIABLE_DIMENSION_DATASETS:
        _require_integer(config.dim, "dim", 5)
        kwargs["dim"] = config.dim
    elif config.dim is not None:
        raise ValueError("Friedman2/3 require dim=None (fixed 4D)")
    if config.dataset == "sparse_features":
        _require_positive_finite(config.sparsity, "sparsity")
        if config.sparsity >= 1:
            raise ValueError("sparsity must be in (0, 1)")
        kwargs["sparsity"] = config.sparsity
    elif config.sparsity is not None:
        raise ValueError("sparsity applies only to sparse_features")
    return classes[config.dataset](**kwargs)


def _validate_prediction(prediction: np.ndarray, *, expected_shape: tuple[int, int], model: str) -> None:
    """Validate one of the three search curves without clipping or repairing it."""
    curve = np.asarray(prediction)
    if curve.ndim != 2 or curve.shape != expected_shape or curve.size == 0:
        raise ValueError(f"{model} prediction must have expected shape {expected_shape}")
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"{model} prediction must be finite")
    if np.any(curve < 0) or np.any(curve > 1):
        raise ValueError(f"{model} prediction must lie in [0, 1]")
    if np.any(np.diff(curve, axis=1) > 1e-14):
        raise ValueError(f"{model} prediction must be non-increasing")


def time_predictors(
    models: Mapping[str, BeranEstimator], X_test: np.ndarray, grid: np.ndarray,
    *, protocol, seed: int,
) -> dict[str, tuple[int, ...]]:
    """Return raw prediction-only timings from sequential interleaved repeats.

    ``timed_call`` owns perf-counter/GC handling. All result validation occurs
    after its clock has stopped, including validation of warmup predictions.
    """
    if set(models) != set(MODEL_NAMES):
        raise ValueError(f"models must contain exactly {MODEL_NAMES!r}")
    _require_integer(seed, "seed", 0)
    _require_integer(protocol.timing_warmups, "timing_warmups", 0)
    _require_integer(protocol.timing_repeats, "timing_repeats", 1)
    expected_shape = (len(X_test), len(grid))
    predictors = {
        name: lambda model=models[name]: model.predict_survival_function(X_test, grid)
        for name in MODEL_NAMES
    }
    for _ in range(protocol.timing_warmups):
        for name in MODEL_NAMES:
            _validate_prediction(predictors[name](), expected_shape=expected_shape, model=name)
    samples: dict[str, list[int]] = {name: [] for name in MODEL_NAMES}
    rng = np.random.default_rng(20260916 + seed)
    for _ in range(protocol.timing_repeats):
        for name in rng.permutation(MODEL_NAMES):
            elapsed, prediction = timing.timed_call(predictors[name])
            _require_integer(elapsed, "timing sample", 1)
            _validate_prediction(prediction, expected_shape=expected_shape, model=name)
            samples[name].append(elapsed)
    return {name: tuple(values) for name, values in samples.items()}


def _freeze_identity_value(value: object) -> object:
    """Freeze a normalized snapshot, including nested parameter containers."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_identity_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_identity_value(item) for item in value)
    return value


def _normalize_identity_value(value: object) -> object:
    """Convert supported parameter values to deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration identity values must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("configuration identity mapping keys must be strings")
        return {key: _normalize_identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_identity_value(item) for item in value]
    raise TypeError(f"unsupported configuration identity value: {type(value).__name__}")
