"""Lightweight surrogate model abstractions used by the ensemble."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PolynomialSurrogate:
    """One-dimensional polynomial surrogate storing its coefficients."""

    coefficients: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at the provided coordinates."""
        X = np.asarray(X, dtype=float).reshape(-1)
        values = np.polyval(self.coefficients, X)
        return values.reshape(-1)
