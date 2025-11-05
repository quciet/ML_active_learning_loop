import numpy as np
from .ensemble_training import train_ensemble, ensemble_predict
from .acquisition_functions import lcb, expected_improvement

def active_learning_loop(
    f, X_obs, Y_obs, X_grid,
    n_iters=30, M=8, epochs=200, lr=1e-3, weight_decay=1e-4,
    kappa_start=1.6, kappa_end=0.8, acquisition="LCB",
    patience=10, min_improve_pct=0.01
):
    """
    Active-learning optimization loop using ensemble-based surrogates.

    This version maintains cumulative behavior:
    - Each call continues from the provided X_obs, Y_obs dataset.
    - Surrogate retrains on the expanded dataset each iteration.
    - No automatic reset between runs — cumulative learning is expected.
    - To start fresh, manually reinitialize X_obs and Y_obs before calling.

    Parameters
    ----------
    f : callable
        True objective function f(x) to evaluate.

    X_obs, Y_obs : np.ndarray
        Observed (x, y) data arrays. Passed in directly to allow continuation.

    X_grid : np.ndarray
        Candidate grid for acquisition evaluation.

    n_iters : int
        Maximum number of iterations to run the loop.

    M, epochs, lr, weight_decay : int/float
        Training hyperparameters for ensemble models.

    kappa_start, kappa_end : float
        Range of LCB kappa values (annealed linearly).

    acquisition : {'LCB', 'EI'}
        Acquisition function type.

    patience : int
        Number of consecutive non-improving iterations before stopping.

    min_improve_pct : float
        Minimum relative (%-based) improvement threshold per iteration.

    Returns
    -------
    X_obs, Y_obs : np.ndarray
        Updated datasets including new points (expanded cumulatively).

    history : np.ndarray
        Iteration history: (t, x_next, y_next, best_y).

    results_cache : dict
        Cached evaluations {x: y}.
    """

    X_obs = np.copy(X_obs)
    Y_obs = np.copy(Y_obs)

    results_cache = {round(float(x), 6): float(y) for x, y in zip(X_obs, Y_obs)}
    kappas = np.linspace(kappa_start, kappa_end, n_iters)
    history = []
    no_improve_counter = 0

    best_y_prev = float(np.min(Y_obs))
    print(f"Starting active learning run — initial best_y = {best_y_prev:.4f}")

    for t in range(n_iters):
        models = train_ensemble(X_obs, Y_obs, M=M, epochs=epochs, lr=lr, weight_decay=weight_decay)
        mu, sigma = ensemble_predict(models, X_grid)
        y_best = np.min(Y_obs)

        if acquisition.upper() == "LCB":
            acq = lcb(mu, sigma, kappa=kappas[t])
            sort_order = np.argsort(acq)
        elif acquisition.upper() == "EI":
            acq = expected_improvement(mu, sigma, y_best)
            sort_order = np.argsort(-acq)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

        x_next = None
        for idx in sort_order:
            candidate = float(X_grid[idx, 0])
            key = round(candidate, 6)
            if key not in results_cache:
                x_next = candidate
                break
        if x_next is None:
            print("All candidates evaluated. Stopping early.")
            break

        key = round(x_next, 6)
        if key in results_cache:
            y_next = results_cache[key]
            reused = True
        else:
            y_next = f(x_next)
            results_cache[key] = y_next
            reused = False

        X_obs = np.append(X_obs, x_next)
        Y_obs = np.append(Y_obs, y_next)
        best_y_curr = float(np.min(Y_obs))
        history.append((t, x_next, y_next, best_y_curr))

        print(f"[{t+1:03d}/{n_iters}] acq={acquisition}, x_next={x_next:.4f}, y_next={y_next:.4f} "
              f"({'reused' if reused else 'new'}), best_y={best_y_curr:.4f}")

        rel_change = abs(best_y_curr - best_y_prev) / (abs(best_y_prev) + 1e-8)
        if rel_change < min_improve_pct:
            no_improve_counter += 1
        else:
            no_improve_counter = 0

        if no_improve_counter >= patience:
            print(f"Stopping early: no stepwise improvement > {min_improve_pct*100:.2f}% "
                  f"for {patience} consecutive iterations.")
            break

        best_y_prev = best_y_curr

    return X_obs, Y_obs, np.array(history), results_cache
