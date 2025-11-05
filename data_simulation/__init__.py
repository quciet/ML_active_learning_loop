"""Synthetic data generation utilities for active learning simulations."""

from .synthetic_functions import f, generate_w_shape_data
from .grid_utils import make_uniform_grid, make_random_grid

__all__ = [
    "f",
    "generate_w_shape_data",
    "make_uniform_grid",
    "make_random_grid",
]
