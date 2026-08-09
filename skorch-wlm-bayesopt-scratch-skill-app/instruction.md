Replace GridSearchCV in examples/word_language_model/train.py with bounded 1-D search over learning rate with explicit bounds and iteration cap.

Keep single-config training under __main__ guard, module import-safe. Optimizer class constructed with bounds.

Preserve original skorch training wiring.
