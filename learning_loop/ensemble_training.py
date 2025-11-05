"""Utilities for training and evaluating polynomial surrogate ensembles."""

from __future__ import annotations

import numpy as np

from .surrogate_models import PolynomialSurrogate


def _fit_polynomial(X: np.ndarray, Y: np.ndarray, degree: int) -> PolynomialSurrogate:
    """Fit a polynomial surrogate of the specified degree."""
    x_flat = np.asarray(X, dtype=float).reshape(-1)
    y_flat = np.asarray(Y, dtype=float).reshape(-1)
    max_degree = max(0, len(x_flat) - 1)
    deg = min(degree, max_degree)
    coeffs = np.polyfit(x_flat, y_flat, deg=deg)
    return PolynomialSurrogate(coefficients=coeffs)


def train_ensemble(
    X_obs,
    Y_obs,
    M: int = 8,
    degree: int = 5,
    bootstrap: bool = False,
) -> list[PolynomialSurrogate]:
    """Train an ensemble of polynomial surrogate models."""
    X_obs = np.asarray(X_obs, dtype=float)
    Y_obs = np.asarray(Y_obs, dtype=float)

    n = len(X_obs)
    models: list[PolynomialSurrogate] = []
    for _ in range(M):
        if bootstrap and n > 1:
            idx = np.random.randint(0, n, size=n)
        else:
            idx = np.arange(n)
        models.append(_fit_polynomial(X_obs[idx], Y_obs[idx], degree))
    return models


def ensemble_predict(models: list[PolynomialSurrogate], X_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate each surrogate in the ensemble and return mean and standard deviation."""
    X_grid = np.asarray(X_grid, dtype=float)
    preds = np.stack([model.predict(X_grid) for model in models], axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0, ddof=1) if len(models) > 1 else np.zeros_like(mu)
    return mu, sigma
