Replace GridSearchCV in examples/word_language_model/train.py with bounded 1-D search over learning rate with explicit bounds (lower, upper) and iteration budget.

Keep single-config training under __main__ guard, module import-safe. Optimizer class constructed with bounds, with history-aware predict and propose methods, deterministic optimize respecting bounds and budget.

CLI must support --search-iter for iteration cap (default from constant DEFAULT_SEARCH_ITERATIONS), --lr-low and --lr-high for LR bounds with sensible defaults, plus existing --seed, --data, --bptt, --batch_size, --epochs, --data-limit, --save — all under main guard.

Preserve original skorch training wiring under main: train_split handling validation split, data loaders with bptt, corpus loading, callbacks.

Objective must set lr via net.set_params(lr=float(x)) and evaluate via cross_val_score averaged over limited training data converted with .numpy() for sklearn, and save best via save_params.

Remove GridSearchCV import/instantiation, define DEFAULT_SEARCH_ITERATIONS constant (default 16) used as default for --search-iter.

Trust-region: start with few random seed points uniform in bounds using seed for determinism, initial radius as fraction of bound width (fraction in (0,1)), shrink factor in (0,1) on non-improvement, same seed gives same result.
