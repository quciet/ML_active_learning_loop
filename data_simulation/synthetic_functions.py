"""Piecewise synthetic functions and sampling helpers for simulations."""

from __future__ import annotations

import numpy as np


def f_seg1(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defined on the first interval of the piecewise nonlinear curve."""
    return 2 * (x - 0.15) ** 2 + 3.0


def f_seg2(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defining the first valley of the piecewise nonlinear curve."""
    return -5 * (x - 0.18) ** 2 + 2.0


def f_seg3(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defining the second valley of the piecewise nonlinear curve."""
    return 6 * (x - 0.65) ** 2 + 1.0


def f_seg4(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defining the final rise of the piecewise nonlinear curve."""
    return 1.5 * (x - 0.50) ** 2 + 3.0


cuts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
functions = [f_seg1, f_seg2, f_seg3, f_seg4]


def f(x: np.ndarray | float) -> np.ndarray | float:
    """Evaluate the piecewise nonlinear function at the provided coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    mask1 = (x >= cuts[0]) & (x < cuts[1])
    mask2 = (x >= cuts[1]) & (x < cuts[2])
    mask3 = (x >= cuts[2]) & (x < cuts[3])
    mask4 = (x >= cuts[3]) & (x <= cuts[4])

    y[mask1] = f_seg1(x[mask1])
    y[mask2] = f_seg2(x[mask2])
    y[mask3] = f_seg3(x[mask3])
    y[mask4] = f_seg4(x[mask4])
    return y.item() if y.size == 1 else y


def generate_piecewise_data(n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample evenly spaced observations from a synthetic piecewise nonlinear function.
    Returns arrays X, Y suitable for active learning demonstrations.
    """
    segments = [
        np.linspace(cuts[i], cuts[i + 1], n_points // 4, endpoint=False)
        for i in range(len(cuts) - 1)
    ]
    segments[-1] = np.linspace(cuts[-2], cuts[-1], n_points // 4, endpoint=True)
    x_all = np.concatenate(segments)
    y_all = f(x_all)
    return x_all.reshape(-1, 1), y_all.reshape(-1, 1)


def f2(X: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Two-dimensional nonlinear test function.

    Parameters
    ----------
    X : np.ndarray or tuple of arrays
        Input coordinates. If np.ndarray, should be shape (n_samples, 2).

    Returns
    -------
    y : np.ndarray
        Scalar output computed as nonlinear combination of both inputs.
    """
    if isinstance(X, tuple):
        x1, x2 = X
    else:
        x1, x2 = X[:, 0], X[:, 1]

    y = (
        2.0 * (x1 - 0.3) ** 2
        + 1.5 * (x2 - 0.7) ** 2
        - 1.2 * np.sin(3 * np.pi * x1) * np.cos(2 * np.pi * x2)
        + 3.0
    )
    return y


def generate_2d_data(n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a grid of 2D input points and evaluate f2 on it.

    Parameters
    ----------
    n_points : int
        Total number of sample points (approximately forms a sqrt(n_points) × sqrt(n_points) grid).

    Returns
    -------
    X : np.ndarray of shape (n_points, 2)
        Flattened coordinates.
    Y : np.ndarray of shape (n_points,)
        Corresponding function evaluations.
    """
    n_side = int(np.sqrt(n_points))
    x1 = np.linspace(0, 1, n_side)
    x2 = np.linspace(0, 1, n_side)
    X1, X2 = np.meshgrid(x1, x2)
    Y = f2((X1, X2))
    return np.stack([X1.ravel(), X2.ravel()], axis=1), Y.ravel()
