"""Piecewise synthetic functions and sampling helpers for simulations."""

from __future__ import annotations

import numpy as np


def f_seg1(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defined on the first interval of the W-shaped curve."""
    return 2 * (x - 0.15) ** 2 + 3.0


def f_seg2(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defining the first valley of the W-shaped curve."""
    return -5 * (x - 0.18) ** 2 + 2.0


def f_seg3(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defining the second valley of the W-shaped curve."""
    return 6 * (x - 0.65) ** 2 + 1.0


def f_seg4(x: np.ndarray | float) -> np.ndarray | float:
    """Quadratic segment defining the final rise of the W-shaped curve."""
    return 1.5 * (x - 0.50) ** 2 + 3.0


cuts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
functions = [f_seg1, f_seg2, f_seg3, f_seg4]


def f(x: np.ndarray | float) -> np.ndarray | float:
    """Evaluate the piecewise W-shaped function at the provided coordinates."""
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


def generate_w_shape_data(n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Sample evenly spaced observations from the synthetic W-shaped function."""
    segments = [
        np.linspace(cuts[i], cuts[i + 1], n_points // 4, endpoint=False)
        for i in range(len(cuts) - 1)
    ]
    segments[-1] = np.linspace(cuts[-2], cuts[-1], n_points // 4, endpoint=True)
    x_all = np.concatenate(segments)
    y_all = f(x_all)
    return x_all.reshape(-1, 1), y_all.reshape(-1, 1)
