Right now the training code uses `GridSearchCV` for hyperparameter tuning. I want a bounded 1‑D search over learning rate within explicit bounds `(lower, upper)`, with a fixed iteration cap so the run stays bounded.

Don’t break the normal training path: single-config training should still work the same way under the `__main__` guard, and importing `train.py` must be safe (no training on import). The new optimizer must be a class defined in `train.py` (not a separate module), and it should replace `GridSearchCV` in the tuning path.

For testing, the optimizer needs an inspectable surface: (1) a way to fit/evaluate a surrogate at a given point from an explicit history, (2) a way to propose the next point from an explicit history plus an explicit trust radius, and (3) a way to run the bounded search loop. These calls must be deterministic given the same history (and radius/seed), with no hidden mutable trust-region state required to reproduce a proposal.

The surrogate is a deterministic least-squares degree‑2 polynomial fit: given a history of `(x, score)` pairs, fit `a·x² + b·x + c` (to the chosen history points, e.g. all history) and evaluate it at `x` when needed. The proposal step uses the quadratic vertex `x* = −b/(2a)`, clipped to the feasible interval given by trust-region∩bounds:
`[ max(lower, best_x − radius), min(upper, best_x + radius) ]`, where `best_x` is the history point with the highest observed score. If `a ≥ 0`, use a pinned fallback: evaluate the fitted quadratic at the two feasible endpoints and choose the endpoint with the higher predicted value; if they tie, choose the lower endpoint.

In the loop, start from a modest trust radius and shrink it when an iteration doesn’t improve the best observed score. Seed the search with a small number of initial random points drawn uniformly from within the bounds using the provided seed; after that, propose points using the vertex rule above.
