"""Lightweight surrogate model abstractions used by the ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from torch import nn, optim


@dataclass
class PolynomialSurrogate:
    """One-dimensional polynomial surrogate storing its coefficients."""

    coefficients: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at the provided coordinates."""
        X = np.asarray(X, dtype=float).reshape(-1)
        values = np.polyval(self.coefficients, X)
        return values.reshape(-1)

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, degree: int = 5) -> "PolynomialSurrogate":
        """Fit a polynomial of the specified degree to the provided data."""
        coeffs = np.polyfit(np.asarray(X).flatten(), np.asarray(Y).flatten(), deg=degree)
        return cls(coefficients=coeffs)


class TreeSurrogate:
    """Decision-tree based surrogate model wrapper."""

    def __init__(self, model: DecisionTreeRegressor):
        self.model = model

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        max_depth: int = 5,
        random_state: int | None = None,
    ) -> "TreeSurrogate":
        reg = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        reg.fit(np.asarray(X).reshape(-1, 1), np.asarray(Y).reshape(-1))
        return cls(reg)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X).reshape(-1, 1))


class NNSurrogate:
    """Small neural-network surrogate model implemented with PyTorch."""

    def __init__(self, model: nn.Module):
        self.model = model

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> "NNSurrogate":
        X_t = torch.tensor(np.asarray(X).reshape(-1, 1), dtype=torch.float32)
        Y_t = torch.tensor(np.asarray(Y).reshape(-1, 1), dtype=torch.float32)

        model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(X_t)
            loss = criterion(preds, Y_t)
            loss.backward()
            optimizer.step()

        return cls(model.eval())

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(np.asarray(X).reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            preds = self.model(X_t)
        return preds.cpu().numpy().flatten()
