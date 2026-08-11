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

Class: QuadraticTrustRegionOptimizer, constructed with bounds tuple (lo, hi)
predict(history, x): least-squares degree-2 polynomial fit to all points in history, return value of fitted quadratic at x (deterministic, no GP/kernel/linear)
propose(history, radius): current best is argmax score over history, feasible interval is trust region around best intersected with bounds ([max(lo, best-radius), min(hi, best+radius)]), if fitted quadratic is concave (has interior maximum) propose its vertex clipped to feasible, else propose feasible endpoint with higher predicted value (tie -> lower)
optimize(objective, n_iter, seed): start with few seed points drawn uniformly at random from bounds using provided seed for determinism, initial trust radius is fraction of bound width, shrink radius on non-improving iteration, return best config and history, same seed gives same result, propose continuous values not snapped to grid

Module import-safe, __main__ guard, remove GridSearchCV import, optimizer class defined inside train.py not in separate module


## Common mistakes

Mutable hidden radius, snapping to grid, non-deterministic seeding, defining optimizer in new module.



## See also

sklearn GridSearchCV -> bounded search migration pattern in skorch examples
