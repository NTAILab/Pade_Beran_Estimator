"""Shared NumPy prediction example helper."""
import numpy as np

from pbestimator import BeranEstimator, PadeBeranEstimator, TaylorBeranEstimator


def predict_all(X_train, time, event, X_test, *, label):
    grid = np.linspace(0, np.quantile(time, .8), 8)
    print(f"{label}: train={X_train.shape}, test={X_test.shape}, times={grid.round(3)}")
    for cls in (BeranEstimator, TaylorBeranEstimator, PadeBeranEstimator):
        model = cls(tau=1.).fit(X_train, time, event)
        prediction = model.predict_survival_function(X_test, grid)
        print(f"  {cls.__name__} {prediction.shape}: {prediction[0, :4].round(6)}")
