"""Acquisition function utilities for active learning."""

from __future__ import annotations

import math

import numpy as np

_erf = np.vectorize(math.erf)

def lcb(mu, sigma, kappa=1.6):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.

    Parameters
    ----------
    mu : np.ndarray
        Predicted mean values from ensemble.

    sigma : np.ndarray
        Predicted standard deviation (uncertainty).

    kappa : float, optional (default=1.6)
        Exploration-exploitation trade-off parameter.
        Higher kappa encourages more exploration.

    Returns
    -------
    acq : np.ndarray
        Lower Confidence Bound scores (lower = better for minimization).

    Notes
    -----
    LCB(x) = mu(x) - kappa * sigma(x)
    """
    return mu - kappa * sigma


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x ** 2)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def expected_improvement(mu, sigma, y_best, xi=0.01):
    """
    Expected Improvement (EI) acquisition function for minimization.

    Parameters
    ----------
    mu : np.ndarray
        Predicted mean values from ensemble.

    sigma : np.ndarray
        Predicted standard deviation (uncertainty).

    y_best : float
        Best observed (minimum) objective value so far.

    xi : float, optional (default=0.01)
        Small positive number encouraging exploration near known minima.

    Returns
    -------
    ei : np.ndarray
        Expected Improvement values (higher = better).

    Notes
    -----
    EI(x) = (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (y_best - mu - xi) / sigma
    """
    sigma = np.maximum(sigma, 1e-8)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * _norm_cdf(Z) + sigma * _norm_pdf(Z)
    return ei
