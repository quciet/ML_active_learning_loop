import torch
import torch.nn as nn

class SurrogateNN(nn.Module):
    """
    Simple feedforward neural network used as the surrogate model
    for approximating the underlying objective function f(x).

    Parameters
    ----------
    hidden_sizes : tuple of int, optional (default=(32, 32))
        Number of neurons in each hidden layer.

    activation : torch.nn.Module, optional (default=torch.nn.ReLU)
        Activation function applied between layers.

    Notes
    -----
    - Input and output dimensions are both 1 by default, assuming scalar x and y.
    - Extendable to multivariate problems by changing input_dim/output_dim.
    - Used within the ensemble of surrogate models in train_ensemble().

    Example
    -------
    >>> model = SurrogateNN(hidden_sizes=(64, 64), activation=nn.Tanh)
    >>> x = torch.rand(10, 1)
    >>> y = model(x)
    """
    def __init__(self, hidden_sizes=(32, 32), activation=nn.ReLU):
        super().__init__()
        layers = []
        input_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
