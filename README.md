# Active Learning Simulation Loop

This repository contains reusable components for running an active learning loop
with lightweight surrogate models and synthetic test functions.

## Repository Structure

- `acquisition_functions.py` — Acquisition heuristics such as LCB and expected improvement.
- `active_learning.py` — Orchestrates the active learning optimization loop.
- `ensemble_training.py` — Training helpers for the surrogate ensemble.
- `surrogate_models.py` — Lightweight polynomial surrogate abstraction.
- `utils.py` — Utility helpers such as deterministic seeding.
- `data_simulation/` — Synthetic data generators and grid utilities.
- `notebooks/testing.ipynb` — Example notebook exercising the loop end-to-end.
