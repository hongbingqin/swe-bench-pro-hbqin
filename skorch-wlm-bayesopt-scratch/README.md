# codimango/skorch-wlm-bayesopt-scratch

> Draft — filled in during QA once the bespoke algorithm + calibration are set.

## Description

Harder sibling of `skorch-wlm-bayesopt`. Instead of swapping in a library
(`BayesSearchCV`), the solver must implement a **from-scratch** sequential
model-based optimizer for the `examples/word_language_model` hyperparameter
search: a hand-written **surrogate** over the observed `(config, score)` history
plus an **acquisition rule** that proposes the next configuration, capped by an
explicit iteration budget — using only numpy/scipy (no BO library is installed).

This both raises difficulty (real algorithm, not a one-line swap) and closes the
library version's leniency: the optimizer can be tested to **beat random search**
on a deterministic objective, discriminating real history-guided search from
random sampling.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python

## Reference solution (oracle)

_TBD — once the bespoke surrogate/acquisition is chosen and authored._

## Completion Rates

_TBD — measured during validation._

## Model Analysis

_TBD._

## Anti-Cheating Analysis

_TBD._
