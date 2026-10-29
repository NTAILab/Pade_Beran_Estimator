# PBEstimator

A compact NumPy research implementation of Gaussian-kernel conditional survival estimation. The package provides exact, Taylor-log and Padé-log Beran estimators, six synthetic data generators, and runnable examples. Full reproduction of a paper is outside this repository's scope.

## Estimators

| Class | Log increment |
| --- | --- |
| BeranEstimator | log(1 - h) |
| TaylorBeranEstimator | -h |
| PadeBeranEstimator | -2h / (2 - h) |

The estimators vectorize kernel logits, grouped risk-set sums and survival accumulation with NumPy. Fit retains the reference observations; predict_survival_function returns one survival curve per query row.

```python
from pbestimator import BeranEstimator

model = BeranEstimator(tau=1.).fit(X_train, observed_time, event)
survival = model.predict_survival_function(X_query, times)
```

## Install

Python 3.13 or later is required by the current package metadata. Install from the repository root:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install .
```

The only numerical backend is NumPy. Other dependencies support data handling, preprocessing, and the real-data example.

## Examples

```sh
python examples/real_data.py
python examples/synthetic_data.py
```

The real-data example uses the bundled veterans' lung cancer dataset from scikit-survival. It uses a seeded 48/16 holdout and fits numeric scaling and categorical encoding on training rows only. These examples demonstrate prediction, not clinical validation or model selection.

The six synthetic generators are Friedman 1, 2 and 3, Strong Feature Interactions, Sparse Features, and Nonlinear Dependency:

```python
from pbestimator.data_generators import DataGeneratorFriedman1

sample = DataGeneratorFriedman1(seed=42).generate(48, censor_rate=.25)
# sample.X, sample.event_time (latent), sample.observed_time, sample.event
```

Generator instances retain their random state, so repeated calls draw successive samples. Recreate an instance with the same seed to reproduce its first sample. The generators' built-in censoring rule is based on latent event times and is therefore not independent censoring.

## Experiments

The checkout includes inference, sample-size, and NumPy tau-training experiment entry points. Edit the configuration near the top of each script before starting a full run:

```sh
python experiments/series_size.py
python experiments/series_inference.py
python experiments/series_train.py
```

Pairwise kernel calculations require memory proportional to query size times support size and feature count; use small batches for large data.
