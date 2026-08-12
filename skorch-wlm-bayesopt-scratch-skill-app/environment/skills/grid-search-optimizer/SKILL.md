---
name: grid-search-optimizer
description: Exhaustive grid search over discrete hyperparameter grids — evaluates every combination in the Cartesian product. Simple but expensive, replaced by budgeted search in this task.
---

# Grid Search Optimizer

## Overview
Grid search enumerates all combinations from a discrete grid (e.g., lr in [10,20,30]) and evaluates each via cross-validation. The best is argmax score.

## How It Works
- Define param grid: [{'lr': [10,20,30]}]
- Cartesian product → 3 configs
- For each config: fit and score
- Best = max score

## When to Use
When grid is small and exhaustive evaluation is feasible. In this task, grid search is the old method being replaced by bounded sequential optimization with explicit iteration cap.

## Why Not Here
This task explicitly requires removing GridSearchCV import/instantiation and replacing with from-scratch optimizer with explicit iteration cap, not exhaustive grid.
