"""Post-hoc probability calibration wrapper.

The calibrator is fitted on *out-of-fold* predictions from the temporal CV
folds (never on the test seasons), then applied on top of the final model.
It lives in its own module so pickled models can be loaded by the app.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method: str, raw_prob: np.ndarray, y: np.ndarray):
    """Fit a 'sigmoid' (Platt scaling on the logit) or 'isotonic' calibrator."""
    if method == "sigmoid":
        return LogisticRegression(C=1e6).fit(_logit(raw_prob), y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_prob, y)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator, raw_prob: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(raw_prob), _EPS, 1 - _EPS)
    return calibrator.predict_proba(_logit(raw_prob))[:, 1]


class CalibratedModel:
    """A fitted pipeline plus a probability calibrator, exposing ``predict_proba``."""

    def __init__(self, base_model, calibrator, method: str):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        p = apply_calibrator(self.calibrator, self.base_model.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
