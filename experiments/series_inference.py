from pathlib import Path
import argparse
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _support import series_experiments
from _support.series_experiments import ExperimentSeries, run_series_batch

CENSOR_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DATASETS = {
    "friedman1": {"censor_rate": CENSOR_RATES},
    "friedman2": {"censor_rate": CENSOR_RATES},
    "friedman3": {"censor_rate": CENSOR_RATES},
    "strong_feature_interactions": {"censor_rate": CENSOR_RATES},
    "sparse_features": {"censor_rate": CENSOR_RATES},
    "nonlinear_dependency": {"censor_rate": CENSOR_RATES},
}

N_TRAIN = 2000
N_TEST = 2000
FOLDS = 20
RESUME = True
OUTPUT_ROOT = ROOT / "results" / "series"


def build_series() -> tuple[ExperimentSeries, ...]:
    return tuple(
        ExperimentSeries(
            name=f"{dataset}_{parameter}",
            dataset=dataset,
            parameter=parameter,
            values=values,
            output_dir=OUTPUT_ROOT / f"{dataset}_{parameter}",
        )
        for dataset, parameters in DATASETS.items()
        for parameter, values in parameters.items()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a tiny end-to-end persistence check")
    args = parser.parse_args()
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="pbestimator-inference-smoke-") as temp:
            series_experiments.N_TRAIN = 40
            series_experiments.N_TEST = 32
            smoke = ExperimentSeries(
                "smoke", "friedman1", "censor_rate", (0.0, 0.3), Path(temp) / "friedman1"
            )
            run_series_batch((smoke,), resume=False, folds=1)
        print("series_inference smoke passed", flush=True)
        raise SystemExit(0)
    series_experiments.N_TRAIN = N_TRAIN
    series_experiments.N_TEST = N_TEST
    run_series_batch(build_series(), resume=RESUME, folds=FOLDS)
