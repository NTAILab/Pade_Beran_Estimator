from pathlib import Path
import argparse
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _support import sample_size_series
from _support.sample_size_series import DatasetSampleSizeConfig, run_sample_size_batch

DATASETS = {
    "friedman1": {"tau": 0.8, "censor_rate": .3},
}

N_TRAIN_VALUES = (100, 200, 500, 1000, 2000, 5000, 10000)
N_TEST = 2000
FOLDS = 20
RESUME = True
OUTPUT_ROOT = ROOT / "results" / "sample_size_series"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a tiny end-to-end persistence check")
    args = parser.parse_args()
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="pbestimator-size-smoke-") as temp:
            config = DatasetSampleSizeConfig("friedman1", .8, .3, Path(temp) / "friedman1")
            sample_size_series._run_sample_size_series(
                config,
                n_train_values=(24, 40),
                n_test=32,
                folds=1,
                resume=False,
                sha=sample_size_series.ordinary.implementation_fingerprint(),
            )
        print("series_size smoke passed", flush=True)
        raise SystemExit(0)
    configs = tuple(DatasetSampleSizeConfig(dataset=name, output_dir=OUTPUT_ROOT / name, **settings) for name, settings in DATASETS.items())
    run_sample_size_batch(configs, n_train_values=N_TRAIN_VALUES, n_test=N_TEST, folds=FOLDS, resume=RESUME)
