"""Veterans' lung cancer: train-only preprocessing, then all estimator forms."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.datasets import load_veterans_lung_cancer
from _models import predict_all


def main():
    X, y = load_veterans_lung_cancer()  # Bundled with scikit-survival; no download.
    train, test = train_test_split(np.arange(len(X)), train_size=48, test_size=16,
                                   random_state=42, stratify=y["Status"])
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], sparse_threshold=0.)
    X_train = preprocessing.fit_transform(X.iloc[train])
    X_test = preprocessing.transform(X.iloc[test])
    predict_all(X_train, y["Survival_in_days"][train], y["Status"][train], X_test,
                label="veterans_lung_cancer (time in days)")


if __name__ == "__main__":
    main()
