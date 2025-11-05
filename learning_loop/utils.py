import numpy as np
import torch
import random

def set_global_seed(seed=0):
    """
    Set global random seed across NumPy, PyTorch, and Python's built-in RNG.
    Ensures reproducibility across runs.

    Parameters
    ----------
    seed : int, optional (default=0)
        Random seed value to apply globally.

    Notes
    -----
    - This function affects NumPy, PyTorch, and Python random generators.
    - Also sets deterministic cuDNN behavior for reproducibility on GPU.
    - For stochastic ensemble diversity, you can still rely on random
      weight initialization per model within a fixed global seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global seed set to: {seed}")
