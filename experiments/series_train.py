"""Train Beran bandwidths with NumPy on paired synthetic censor-rate series."""
from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _support.generators import GeneratorConfig, make_generator
from _support.evaluation import (
    administratively_censor_test_outcomes,
    evaluation_time_grid,
    ibs_reporting_horizon,
)
from _support.generators import GeneratorConfig, make_generator
from _support.series_experiments import STARTING_STATES
from pbestimator import BeranEstimator, PadeBeranEstimator, TaylorBeranEstimator, train_tau_numpy
from pbestimator.metrics import integrated_brier_score


DATASET = "friedman1"
CENSOR_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
FOLDS = 20
N_TRAIN = 2000
N_TEST = 2000
EPOCHS = 100
LEARNING_RATE = 0.025
GRID_SIZE = 100
MASTER_SEED = 18000
OUTPUT_DIR = ROOT / "results" / "series_train" / DATASET
MODELS = ("exact_log", "taylor_log", "pade_log")
FIELDS = (
    "dataset", "censor_rate", "fold", "model", "seed", "epoch",
    "tau_before", "tau_after", "train_loss", "forward_time_ms",
    "loss_time_ms", "backward_time_ms", "step_time_ms",
    "final_test_ibs",
)
ESTIMATORS = {
    "exact_log": BeranEstimator,
    "taylor_log": TaylorBeranEstimator,
    "pade_log": PadeBeranEstimator,
}


def _generator_config(dataset: str) -> GeneratorConfig:
    state = STARTING_STATES[dataset]
    return GeneratorConfig(dataset, 0, state.tau, state.dim, state.k, state.sparsity)


def _paired_censor(data, rate: float, seed: int):
    event = np.ones(len(data.event_time), dtype=bool)
    observed_time = data.event_time.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260920]))
    indices = rng.permutation(len(event))[: int(np.floor(len(event) * rate))]
    factors = rng.uniform(0.05, 0.95, len(event))
    event[indices] = False
    observed_time[indices] *= factors[indices]
    return observed_time, event


def run() -> Path:
    if DATASET not in STARTING_STATES:
        raise ValueError(f"unknown dataset: {DATASET}")
    if not CENSOR_RATES or any(not 0 <= rate < 1 for rate in CENSOR_RATES):
        raise ValueError("censor rates must be in [0, 1)")
    if FOLDS < 1 or N_TRAIN < 4 or N_TEST < 2 or EPOCHS < 1 or GRID_SIZE < 2:
        raise ValueError("invalid training dimensions")

    output = OUTPUT_DIR / "steps.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing training output: {output}")

    seeds = tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(MASTER_SEED).spawn(FOLDS)
    )
    generator_config = _generator_config(DATASET)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for fold, seed in enumerate(seeds):
            generator = make_generator(generator_config, seed=seed)
            latent = generator.generate(N_TRAIN, censor_rate=0.0)
            test = generator.generate(N_TEST, censor_rate=0.0)
            scaler = StandardScaler().fit(latent.X)
            support_x = scaler.transform(latent.X)
            test_x = scaler.transform(test.X)
            for rate in CENSOR_RATES:
                support_time, support_event = _paired_censor(latent, rate, seed)
                if np.count_nonzero(support_event) < 2:
                    raise ValueError("training requires at least two observed events")
                train_horizon = ibs_reporting_horizon(support_time, support_event)
                grid = evaluation_time_grid(
                    support_time, support_time, t_max=train_horizon, n_times=GRID_SIZE
                )
                for model in MODELS:
                    initial_tau = STARTING_STATES[DATASET].tau

                    def report_epoch(epoch, loss, tau, forward_ms, loss_ms, backward_ms, step_ms):
                        print(
                            f"fold={fold + 1}/{FOLDS} censor={rate:g} model={model} "
                            f"epoch={epoch}/{EPOCHS} tau={tau:.8g} loss={loss:.8g} "
                            f"step_ms={step_ms:.3f}",
                            flush=True,
                        )

                    result = train_tau_numpy(
                        initial_tau=initial_tau,
                        support_x=support_x,
                        support_time=support_time,
                        support_event=support_event,
                        eval_x=support_x,
                        eval_time=support_time,
                        eval_event=support_event,
                        grid=grid,
                        learning_rate=LEARNING_RATE,
                        epochs=EPOCHS,
                        increment_kind=model,
                        leave_one_out=True,
                        on_step=report_epoch,
                    )
                    estimator = ESTIMATORS[model](result.tau).fit(
                        support_x, support_time, support_event
                    )
                    eval_horizon = ibs_reporting_horizon(support_time, support_event)
                    test_time, test_event = administratively_censor_test_outcomes(
                        test.event_time, test.event, horizon=eval_horizon
                    )
                    eval_grid = evaluation_time_grid(
                        support_time, test_time, t_max=eval_horizon, n_times=GRID_SIZE
                    )
                    test_prediction = estimator.predict_survival_function(test_x, eval_grid)
                    test_ibs = integrated_brier_score(
                        support_time, support_event, test_time, test_event,
                        test_prediction, eval_grid,
                    )
                    for epoch, (tau, loss, forward, loss_ms, backward, step) in enumerate(
                        zip(result.taus, result.losses, result.forward_times_ms,
                            result.loss_times_ms, result.backward_times_ms,
                            result.step_times_ms, strict=True), start=1
                    ):
                        writer.writerow({
                            "dataset": DATASET, "censor_rate": rate, "fold": fold,
                            "model": model, "seed": seed, "epoch": epoch,
                            "tau_before": initial_tau if epoch == 1 else float(result.taus[epoch - 2]),
                            "tau_after": float(tau), "train_loss": float(loss),
                            "forward_time_ms": float(forward), "loss_time_ms": float(loss_ms),
                            "backward_time_ms": float(backward), "step_time_ms": float(step),
                            "final_test_ibs": test_ibs if epoch == EPOCHS else "",
                        })
                    stream.flush()
                    print(
                        f"fold={fold + 1}/{FOLDS} censor={rate:g} model={model} "
                        f"tau={result.tau:.8g} loss={result.losses[-1]:.8g} "
                        f"test_ibs={test_ibs:.8g}",
                        flush=True,
                    )
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a tiny NumPy training and holdout check")
    args = parser.parse_args()
    if args.smoke:
        CENSOR_RATES = (0.0, 0.3)
        FOLDS, N_TRAIN, N_TEST, EPOCHS, GRID_SIZE = 1, 32, 24, 2, 4
        with tempfile.TemporaryDirectory(prefix="pbestimator-training-smoke-") as temp:
            OUTPUT_DIR = Path(temp)
            print(run())
        print("series_train smoke passed", flush=True)
    else:
        print(run())
