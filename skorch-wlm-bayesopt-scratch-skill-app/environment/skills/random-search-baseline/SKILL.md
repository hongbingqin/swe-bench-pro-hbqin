---
name: random-search-baseline
description: Baseline random search for hyperparameter tuning — samples uniformly at random from bounds without surrogate model. Use when you need a simple non-model-based baseline.
---

# Random Search Baseline for Bounded 1-D Search

## When to use

Need a simple baseline without surrogate, or when history is not needed.

## How to apply

Class: RandomSearchOptimizer, constructed with bounds (lo, hi)
optimize(objective, n_iter, seed): RNG uniform, return best and history, no surrogate.
