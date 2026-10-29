"""Shared container for generated synthetic survival samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticSurvivalDataset:
    """A synthetic survival sample with latent and observed event times."""

    X: NDArray[np.float64]
    event_time: NDArray[np.float64]
    observed_time: NDArray[np.float64]
    event: NDArray[np.bool_]
