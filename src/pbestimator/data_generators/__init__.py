"""Synthetic survival data generators."""

from .base import DataGenerator
from .friedman import DataGeneratorFriedman1, DataGeneratorFriedman2, DataGeneratorFriedman3
from .interactions import DataGeneratorStrongFeatureInteractions
from .nonlinear import DataGeneratorNonlinearDependency
from .sparse import DataGeneratorSparseFeatures

__all__ = [
    "DataGenerator",
    "DataGeneratorFriedman1",
    "DataGeneratorFriedman2",
    "DataGeneratorFriedman3",
    "DataGeneratorStrongFeatureInteractions",
    "DataGeneratorSparseFeatures",
    "DataGeneratorNonlinearDependency",
]
