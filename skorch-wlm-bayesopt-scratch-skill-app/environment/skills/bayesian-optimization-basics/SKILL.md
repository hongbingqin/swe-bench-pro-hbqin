---
name: bayesian-optimization-basics
description: Foundational concepts for Bayesian optimization — surrogate models, acquisition functions, and exploration-exploitation tradeoff. Useful background for understanding surrogate-based search.
---

# Bayesian Optimization Basics

## Overview
Bayesian optimization builds a probabilistic surrogate model (e.g., Gaussian Process) of the objective function and uses an acquisition function (EI, UCB, PI) to balance exploration and exploitation when proposing next points.

## Key Concepts
- Surrogate: GP with kernel (RBF, Matern) predicts mean and uncertainty
- Acquisition: Expected Improvement, Upper Confidence Bound, Probability of Improvement
- Tradeoff: Explore high uncertainty vs exploit high predicted mean

## When to Use
When you need to optimize expensive black-box functions with few evaluations and can afford GP overhead. Not suitable when you must implement from-scratch with only numpy and no BO library.

## Limitations
Requires GP library (GPy, scikit-optimize, botorch), not allowed in this task which explicitly bans BO libraries and requires numpy-only from-scratch implementation.
