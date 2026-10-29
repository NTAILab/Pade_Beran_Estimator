"""Censor-balanced nested NumPy sample-size series.

The latent test population is shared within each fold. Each subset fits its own
train-only scaler; consequently transformed test features may differ by size.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields, replace
import csv
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from . import series_experiments as ordinary
from .generators import GeneratorConfig, make_generator
from pbestimator.synthetic import SyntheticSurvivalDataset

SAMPLE_SIZE_SERIES_MASTER_SEED = 19000
N_TEST = 2000
FOLDS = 20
SCHEMA_VERSION = 1
_CENSOR_STREAM = 2026100101
_SUBSAMPLING_STREAM = 2026100102
MODEL_NAMES = ordinary.MODEL_NAMES
SERIES_CSV_HEADER = ("n_train", *ordinary.SERIES_CSV_HEADER[1:])
_METADATA_FIELDS = ("n_train", "n_censored", "n_observed", "realized_censor_rate", "subset_identity")
_OBSERVATION_FIELDS = tuple(f.name for f in fields(ordinary.SeriesFoldResult) if f.name != "parameter_value")
FOLDS_CSV_HEADER = (*_METADATA_FIELDS, *_OBSERVATION_FIELDS)


def _rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("censor_rate must be finite and in [0, 1)")
    return float(value)


def _sizes(values: Sequence[int]) -> tuple[int, ...]:
    values = tuple(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 1 for v in values):
        raise ValueError("n_train values must be integers greater than one")
    if len(set(values)) != len(values):
        raise ValueError("n_train values must be unique")
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class DatasetSampleSizeConfig:
    dataset: str
    tau: float
    censor_rate: float
    output_dir: Path

    def __post_init__(self):
        if self.dataset not in ordinary.STARTING_STATES:
            raise ValueError("unknown dataset")
        if isinstance(self.tau, bool) or not isinstance(self.tau, Real) or not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be finite and positive")
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "censor_rate", _rate(self.censor_rate))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


def derive_sample_size_seeds(folds: int = FOLDS) -> tuple[int, ...]:
    return ordinary.derive_series_seeds(SAMPLE_SIZE_SERIES_MASTER_SEED, folds)


def build_nested_train_indices(event: np.ndarray, n_train_values: Sequence[int], censor_rate: float, *, seed: int) -> dict[int, np.ndarray]:
    """Randomize each stratum once, then take monotonically growing pool prefixes."""
    sizes, rate = _sizes(n_train_values), _rate(censor_rate)
    event = np.asarray(event)
    if event.ndim != 1 or event.dtype != np.bool_ or max(sizes) > len(event):
        raise ValueError("invalid event population or n_train exceeds maximum population")
    rng = np.random.default_rng(np.random.SeedSequence([seed, _SUBSAMPLING_STREAM]))
    censored = rng.permutation(np.flatnonzero(~event))
    observed = rng.permutation(np.flatnonzero(event))
    subsets = {}
    for n in sorted(sizes):
        c = math.floor(n * rate)
        if c > len(censored) or n - c > len(observed):
            raise ValueError(f"insufficient stratum members for n_train={n}")
        # Canonical original-row order avoids introducing a censor-status ordering.
        subsets[n] = np.sort(np.concatenate((censored[:c], observed[:n-c])))
    return subsets


def subset_identity(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _generator_config(config: DatasetSampleSizeConfig) -> GeneratorConfig:
    state = ordinary.STARTING_STATES[config.dataset]
    return GeneratorConfig(dataset=config.dataset, level=0, tau=config.tau,
                           dim=state.dim, k=state.k, sparsity=state.sparsity)


def _censor(data: SyntheticSurvivalDataset, rate: float, seed: int) -> SyntheticSurvivalDataset:
    # Same random assignment/factor protocol as ordinary series, dedicated stream.
    rng = np.random.default_rng(np.random.SeedSequence([seed, _CENSOR_STREAM]))
    event = np.ones(len(data.event), dtype=bool)
    indices = rng.permutation(len(event))[:math.floor(len(event) * rate)]
    factors = rng.uniform(.05, .95, len(event))
    observed = data.event_time.copy()
    event[indices] = False
    observed[indices] *= factors[indices]
    return SyntheticSurvivalDataset(data.X.copy(), data.event_time.copy(), observed, event)


def prepare_subset(train: SyntheticSurvivalDataset, test: SyntheticSurvivalDataset, indices: np.ndarray) -> ordinary.PreparedSeriesFold:
    selected = SyntheticSurvivalDataset(*(getattr(train, name)[indices].copy() for name in ("X", "event_time", "observed_time", "event")))
    scaler = StandardScaler().fit(selected.X)
    return ordinary.PreparedSeriesFold(selected, test, scaler.transform(selected.X), scaler.transform(test.X))


def build_sample_size_manifest(config: DatasetSampleSizeConfig, n_train_values: Sequence[int], *, n_test: int, folds: int, sha: str) -> dict:
    sizes = _sizes(n_train_values)
    if type(n_test) is not int or n_test <= 1:
        raise ValueError("n_test must be an integer greater than one")
    # Reuse the canonical timing and approximation declarations unchanged.
    template = ordinary.ExperimentSeries("sample_size", config.dataset, "tau", (config.tau,), config.output_dir)
    base = ordinary.build_series_manifest(template, folds=folds, sha=sha)
    keys = ("implementation_fingerprint", "folds", "timing_warmups", "timing_repeats", "timing_kinds", "approximation_reference", "approximation_metrics", "survival_log_eps", "approximation_integration", "models")
    return {
        **{key: base[key] for key in keys},
        "schema_version": SCHEMA_VERSION,
        "experiment_family": "sample_size_series",
        "dataset": config.dataset,
        "configured_tau": config.tau,
        "configured_censor_rate": config.censor_rate,
        "generator_config": asdict(_generator_config(config)),
        "requested_n_train_values": list(sizes),
        "nesting_order": sorted(sizes),
        "max_train_size": max(sizes),
        "n_test": n_test,
        "test_censor_rate": 0.,
        "master_seed": SAMPLE_SIZE_SERIES_MASTER_SEED,
        "derived_generator_seeds": list(derive_sample_size_seeds(folds)),
        "censor_rng_protocol": {"stream": _CENSOR_STREAM, "rng": "default_rng(SeedSequence([fold_seed, stream]))", "assignment": "permutation prefix floor(max_train_size * rate)", "factors": "uniform(.05, .95, max_train_size)"},
        "subsampling_rng_protocol": {"stream": _SUBSAMPLING_STREAM, "rng": "default_rng(SeedSequence([fold_seed, stream]))", "pools": "censored then observed permutations", "selection": "pool prefixes floor(N*rate), N-floor(N*rate); sort original indices"},
        "subset_identity_protocol": "sha256 of sorted original indices, little-endian int64",
        "pairing": "one uncensored max train then one uncensored test from same generator per fold; censor max train once",
        "preprocessing": "separate StandardScaler fit on each subset only; shared latent test transformed by each scaler",
        "timing_protocol": "2 warmups, 7 randomized/interleaved repeats; fold median; mean/sample std across folds; stages use shared precomputed hazards",
        "sup_endpoint": "closed endpoint T included",
    }


@dataclass(frozen=True)
class SampleSizeFoldResult:
    n_train: int
    n_censored: int
    n_observed: int
    realized_censor_rate: float
    subset_identity: str
    observation: ordinary.SeriesFoldResult


def _validation_manifest(manifest: dict) -> dict:
    return {**manifest, "values": manifest["requested_n_train_values"]}


def _validate_rows(rows: Sequence[SampleSizeFoldResult], manifest: dict) -> None:
    ordinary._validate_rows([row.observation for row in rows], _validation_manifest(manifest))
    for row in rows:
        if type(row.n_train) is not int or row.n_train not in manifest["requested_n_train_values"] or row.observation.parameter_value != row.n_train:
            raise ValueError("invalid persisted n_train")
        c = math.floor(row.n_train * manifest["configured_censor_rate"])
        if type(row.n_censored) is not int or type(row.n_observed) is not int or row.n_censored != c or row.n_observed != row.n_train - c:
            raise ValueError("invalid persisted censor counts")
        if row.realized_censor_rate != c / row.n_train:
            raise ValueError("invalid persisted realized censor rate")
        if not isinstance(row.subset_identity, str) or len(row.subset_identity) != 64 or any(ch not in "0123456789abcdef" for ch in row.subset_identity):
            raise ValueError("invalid persisted subset identity")


def _parse_row(record: dict[str, str]) -> SampleSizeFoldResult:
    if None in record or any(v is None for v in record.values()):
        raise ValueError("invalid folds.csv row width")
    n = int(record["n_train"])
    observation = ordinary._parse_row({"parameter_value": str(n), **record})
    return SampleSizeFoldResult(n, int(record["n_censored"]), int(record["n_observed"]), float(record["realized_censor_rate"]), record["subset_identity"], observation)


class _SampleSizeStore:
    def __init__(self, root: Path, manifest: dict, rows: list[SampleSizeFoldResult]):
        self.root, self.manifest, self.rows = root, manifest, rows

    @classmethod
    def open(cls, root: Path, manifest: dict, resume: bool):
        if root.exists() and any(root.iterdir()) and not resume:
            raise ValueError("output directory is not empty")
        root.mkdir(parents=True, exist_ok=True)
        target = root / "manifest.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if json.dumps(existing, sort_keys=True) != json.dumps(manifest, sort_keys=True):
                raise ValueError("incompatible resume manifest")
        elif any(root.iterdir()):
            raise ValueError("incompatible resume: missing manifest")
        else:
            ordinary._atomic(target, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        rows = []
        path = root / "folds.csv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=";")
                if tuple(reader.fieldnames or ()) != FOLDS_CSV_HEADER:
                    raise ValueError("invalid folds.csv header")
                try:
                    rows = [_parse_row(record) for record in reader]
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError("invalid folds.csv row") from error
        _validate_rows(rows, manifest)
        return cls(root, manifest, rows)

    def save_rows(self, additions: Sequence[SampleSizeFoldResult]) -> None:
        candidate = [*self.rows, *additions]
        _validate_rows(candidate, self.manifest)
        candidate.sort(key=lambda row: (self.manifest["requested_n_train_values"].index(row.n_train), row.observation.fold_index, MODEL_NAMES.index(row.observation.model)))
        lines = [";".join(FOLDS_CSV_HEADER)]
        for row in candidate:
            record = {**asdict(row), **asdict(row.observation)}
            lines.append(";".join(json.dumps(list(record[key])) if key in ordinary._SAMPLE_FIELDS else str(record[key]) for key in FOLDS_CSV_HEADER))
        ordinary._atomic(self.root / "folds.csv", "\n".join(lines) + "\n")
        self.rows = candidate

    def save_failure(self, *, n_train: int | None, fold: int | None, seed: int | None, stage: str, error: BaseException) -> None:
        ordinary._atomic(self.root / "failure.json", json.dumps({"dataset": self.manifest["dataset"], "n_train": n_train, "fold_index": fold, "derived_seed": seed, "stage": stage, "exception_type": type(error).__name__, "message": str(error)}, sort_keys=True) + "\n")


def _run_sample_size_series(config: DatasetSampleSizeConfig, *, n_train_values: Sequence[int], n_test: int, folds: int, resume: bool, sha: str) -> Path:
    """Private smoke seam; the public API fixes the test population to 2000."""
    manifest = build_sample_size_manifest(config, n_train_values, n_test=n_test, folds=folds, sha=sha)
    store = _SampleSizeStore.open(config.output_dir, manifest, resume)
    n, index, seed, stage = None, None, None, "generate"
    try:
        for index, seed in enumerate(manifest["derived_generator_seeds"]):
            n, stage = None, "generate"
            generator = make_generator(_generator_config(config), seed=seed)
            latent = generator.generate(manifest["max_train_size"], censor_rate=0.)
            test = generator.generate(n_test, censor_rate=0.)
            if not np.all(test.event) or not np.array_equal(test.event_time, test.observed_time):
                raise ValueError("test population must be uncensored")
            stage = "censor"
            train = _censor(latent, config.censor_rate, seed)
            stage = "subset"
            subsets = build_nested_train_indices(train.event, manifest["nesting_order"], config.censor_rate, seed=seed)
            # Recompute identities from source data even for fully completed folds.
            for row in store.rows:
                if row.observation.fold_index == index and row.subset_identity != subset_identity(subsets[row.n_train]):
                    n = row.n_train
                    raise ValueError("persisted subset identity mismatch")
            for n in manifest["requested_n_train_values"]:
                if any(row.n_train == n and row.observation.fold_index == index for row in store.rows):
                    continue
                stage = "preprocess"
                data = prepare_subset(train, test, subsets[n])
                stage = "evaluate"
                observations = ordinary.evaluate_prepared_fold(data, tau=config.tau, fold_index=index, seed=seed)
                if len(observations) != 3 or {row.model for row in observations} != set(MODEL_NAMES):
                    raise ValueError("evaluator must return all three models")
                c = int((~data.train.event).sum())
                rows = [SampleSizeFoldResult(n, c, n-c, c/n, subset_identity(subsets[n]), replace(row, parameter_value=n)) for row in observations]
                stage = "persist"
                store.save_rows(rows)
                print(f"dataset={config.dataset} n_train={n} fold={index+1}/{folds} completed", flush=True)
        n, stage = None, "persist"
        aggregates = ordinary.aggregate_series_rows([row.observation for row in store.rows], values=manifest["requested_n_train_values"], folds=folds)
        lines = [";".join(SERIES_CSV_HEADER)]
        for n, aggregate in zip(manifest["requested_n_train_values"], aggregates):
            lines.append(";".join([str(n), *(repr(aggregate[key]) for key in SERIES_CSV_HEADER[1:])]))
        result = config.output_dir / "series.csv"
        ordinary._atomic(result, "\n".join(lines) + "\n")
        (config.output_dir / "failure.json").unlink(missing_ok=True)
        return result
    except BaseException as error:
        # Failure diagnostics must never mask the original failure.
        try:
            store.save_failure(n_train=n, fold=index, seed=seed, stage=stage, error=error)
        except Exception:
            pass
        raise


def run_sample_size_series(config: DatasetSampleSizeConfig, *, n_train_values: Sequence[int] = (100, 200, 500, 1000), n_test: int = N_TEST, folds: int = FOLDS, resume: bool = True) -> Path:
    if type(n_test) is not int or n_test != N_TEST:
        raise ValueError("production n_test must be 2000")
    return _run_sample_size_series(config, n_train_values=n_train_values, n_test=n_test, folds=folds, resume=resume, sha=ordinary.implementation_fingerprint())


def run_sample_size_batch(datasets: Iterable[DatasetSampleSizeConfig], *, n_train_values: Sequence[int] = (100, 200, 500, 1000), n_test: int = N_TEST, folds: int = FOLDS, resume: bool = True) -> list[Path]:
    configs = tuple(datasets)
    directories = [config.output_dir.resolve() for config in configs]
    if len(directories) != len(set(directories)):
        raise ValueError("duplicate output directory in batch")
    sizes = _sizes(n_train_values)
    # Validate all batch settings before any dataset writes or expensive work.
    derive_sample_size_seeds(folds)
    if type(n_test) is not int or n_test != N_TEST:
        raise ValueError("production n_test must be 2000")
    return [run_sample_size_series(config, n_train_values=sizes, n_test=n_test, folds=folds, resume=resume) for config in configs]
