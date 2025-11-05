import numpy as np
from scipy.stats import norm

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
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei
