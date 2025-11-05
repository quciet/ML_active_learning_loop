"""Utility functions for constructing candidate grids."""

from __future__ import annotations

import numpy as np


def make_uniform_grid(n_points: int = 1000, x_min: float = 0.0, x_max: float = 1.0) -> np.ndarray:
    """Generate an evenly spaced grid over ``[x_min, x_max]``."""
    X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
    return X


def make_random_grid(
    n_points: int = 1000, x_min: float = 0.0, x_max: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """Generate a sorted grid of random samples drawn uniformly from the interval."""
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(x_min, x_max, size=(n_points, 1))
    return np.sort(X, axis=0)
