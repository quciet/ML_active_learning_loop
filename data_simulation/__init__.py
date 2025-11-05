"""Synthetic data generation utilities for active learning simulations."""

from .synthetic_functions import f, f2, generate_piecewise_data, generate_2d_data
from .grid_utils import make_uniform_grid, make_random_grid

__all__ = [
    "f",
    "f2",
    "generate_piecewise_data",
    "generate_2d_data",
    "make_uniform_grid",
    "make_random_grid",
]
