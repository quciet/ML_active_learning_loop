import torch
import torch.optim as optim
import numpy as np
import copy
from torch import nn
from .surrogate_models import SurrogateNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ensemble(
    X_obs, Y_obs, M=8, epochs=200, lr=1e-3, weight_decay=1e-4,
    bootstrap=False
):
    """
    Train an ensemble of M neural networks as surrogate models.

    Parameters
    ----------
    X_obs : array-like of shape (n_samples,)
        Observed input points.

    Y_obs : array-like of shape (n_samples,)
        Observed target values corresponding to X_obs.

    M : int, optional (default=8)
        Number of neural networks in the ensemble.

    epochs : int, optional (default=200)
        Number of training epochs for each model.

    lr : float, optional (default=1e-3)
        Learning rate for the Adam optimizer.

    weight_decay : float, optional (default=1e-4)
        L2 regularization coefficient for optimizer.

    bootstrap : bool, optional (default=False)
        If True, each model trains on a bootstrap-resampled dataset.
        If False, all models train on the full dataset (recommended for small data).

    Returns
    -------
    models : list of trained SurrogateNN models

    Notes
    -----
    - Bootstrap (bagging) adds diversity but can drop important boundary points
      in small datasets. Keeping it off ensures every model sees all data.
    - Diversity is still introduced via random weight initialization and
      stochastic gradient updates.
    """
    models = []
    n = len(X_obs)
    loss_fn = nn.MSELoss()

    for m in range(M):
        # Choose indices: bootstrap sample or full dataset
        if bootstrap:
            idx = np.random.randint(0, n, size=n)
        else:
            idx = np.arange(n)

        Xb = X_obs[idx]
        Yb = Y_obs[idx]

        X_t = torch.tensor(Xb.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
        Y_t = torch.tensor(Yb.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

        model = SurrogateNN().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            y_pred = model(X_t)
            loss = loss_fn(y_pred, Y_t)
            loss.backward()
            opt.step()

        models.append(copy.deepcopy(model))

    return models




def ensemble_predict(models, X_grid):
    """
    Make ensemble predictions at candidate grid points.

    Parameters
    ----------
    models : list of SurrogateNN
        Ensemble of trained surrogate models.

    X_grid : array-like of shape (n_candidates, 1)
        Candidate input values to evaluate.

    Returns
    -------
    mu : np.ndarray
        Mean predicted value across ensemble models.

    sigma : np.ndarray
        Standard deviation (uncertainty estimate) across models.
    """
    Xg_t = torch.tensor(X_grid, dtype=torch.float32).to(DEVICE)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds.append(model(Xg_t).cpu().numpy().reshape(-1))
    preds = np.stack(preds, axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0, ddof=1)
    return mu, sigma
