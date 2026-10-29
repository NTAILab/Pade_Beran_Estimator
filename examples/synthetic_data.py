"""Run the NumPy estimators on six reproducible synthetic datasets."""
from sklearn.preprocessing import StandardScaler

from pbestimator.data_generators import (
    DataGeneratorFriedman1,
    DataGeneratorFriedman2,
    DataGeneratorFriedman3,
    DataGeneratorNonlinearDependency,
    DataGeneratorSparseFeatures,
    DataGeneratorStrongFeatureInteractions,
)
from _models import predict_all


def main():
    generators = [
        DataGeneratorFriedman1(seed=42), DataGeneratorFriedman2(seed=42),
        DataGeneratorFriedman3(seed=42),
        DataGeneratorStrongFeatureInteractions(dim=5, seed=42),
        DataGeneratorSparseFeatures(dim=5, sparsity=.2, seed=42),
        DataGeneratorNonlinearDependency(dim=5, seed=42),
    ]
    for generator in generators:
        train = generator.generate(48, censor_rate=.25)
        test = generator.generate(16, censor_rate=0.)
        scaler = StandardScaler().fit(train.X)
        predict_all(scaler.transform(train.X), train.observed_time, train.event,
                    scaler.transform(test.X), label=type(generator).__name__)


if __name__ == "__main__":
    main()
