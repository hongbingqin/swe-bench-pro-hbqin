Replace GridSearchCV in examples/word_language_model/train.py with bounded 1-D search over learning rate with explicit bounds and iteration budget.

Keep single-config training under __main__ guard, module import-safe. Optimizer logic as a class constructed with bounds and history-aware.

Preserve original skorch training wiring under main guard.

Objective should evaluate via cross-validation over limited training data and train best configuration saving via save_params.

Remove GridSearchCV, define iteration cap constant and add flags for iteration budget and LR bounds with sensible defaults.

Trust-region behavior: few random seed points uniform in bounds using seed for determinism, initial radius as fraction of bound width, shrink factor in (0,1) on non-improvement, deterministic.
