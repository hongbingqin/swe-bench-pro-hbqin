Replace GridSearchCV in examples/word_language_model/train.py with bounded 1-D search over learning rate with explicit bounds and iteration budget.

Keep single-config training path under __main__ guard, module import-safe. Optimizer class constructed with bounds (lower, upper), with history-aware methods to predict score and propose next config within trust region around current best, shrinking radius on non-improvement, deterministic given seed, respecting bounds and budget.

Preserve original skorch training wiring (train_split handling validation split, data loaders with bptt and device, corpus loading, seeding, callbacks) under main guard.

Objective should set learning rate and evaluate via cross-validation over limited training data converted for sklearn compatibility, using cross_val_score.

After search, train best configuration and save via save_params with provided save path, print best and evaluation count.

Remove GridSearchCV import/instantiation, define iteration cap constant used as default for new CLI flag that controls budget, and add LR bounds flags with sensible defaults.

Trust-region behavior: start with few random seed points uniform in bounds using seed for determinism, initial radius as fraction of bound width, shrink factor in (0,1) on non-improvement.
