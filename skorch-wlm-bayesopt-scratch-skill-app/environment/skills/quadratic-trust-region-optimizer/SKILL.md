---
name: quadratic-trust-region-optimizer
description: From-scratch bounded 1-D trust-region optimizer for hyperparameter search in skorch word_language_model. Use when replacing GridSearchCV with an explicit iteration budget and a deterministic quadratic surrogate over (x, score) history.
---

# Quadratic Trust-Region Optimizer for Bounded 1-D Search
I want a bounded 1‑D search over learning rate with explicit bounds `(lower, upper)`, where I set an iteration cap and that cap is what keeps the run bounded.

## When to use this

examples/word_language_model/train.py still uses GridSearchCV.
need iteration cap not full grid
need history-driven predict/propose callable deterministically

## How to apply it

Construct optimizer with bounds tuple (lo, hi), e.g., bounds handling for 1-D search.

predict(history, x): fit a surrogate to history — use least-squares degree-2 polynomial to all observed points, return fitted value at x (deterministic, no GP/kernel/linear, no snapping to grid).

propose(history, radius): identify current best (argmax score), build feasible interval as trust region around best intersected with bounds, if surrogate suggests an interior optimum (concave) propose it clipped to feasible, otherwise propose a feasible endpoint chosen by predicted value.

optimize(objective, n_iter, seed): begin with few seed evaluations sampled uniformly from bounds using seed for determinism, start with trust radius as fraction of bound width, shrink radius when iteration does not improve, return best found and full history, deterministic given seed.

Keep training module import-safe under __main__ guard and remove GridSearchCV import.


## Common mistakes

Mutable hidden radius, snapping to grid, non-deterministic seeding, defining optimizer in new module.



## See also

sklearn GridSearchCV -> bounded search migration pattern in skorch examples
