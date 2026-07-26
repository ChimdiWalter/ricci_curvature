from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_classifier(seed: int = 1729, max_iter: int = 2000, class_weight: str = "balanced"):
    base = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=seed,
    )
    clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logistic", base),
        ]
    )
    return clf


def positive_class_scores(model, x: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(x)
    if prob.ndim == 2 and prob.shape[1] > 1:
        return prob[:, 1]
    return prob.ravel()
