# codimango/skorch-wlm-bayesopt

> Draft — filled in during QA (Step 6) once calibration numbers are known.

## Description

Replaces the exhaustive `GridSearchCV(net, params)` hyperparameter search in
skorch's `examples/word_language_model/train.py` with **Bayesian optimization**
that samples a bounded number of configurations (a hard iteration cap) instead
of evaluating the full Cartesian product, while keeping the existing
single-config training path intact.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python

## Reference solution (oracle)

_TBD — library-based (`scikit-optimize` `BayesSearchCV`) in v1._

## Completion Rates

_TBD — measured during validation._

## Model Analysis

_TBD._

## Anti-Cheating Analysis

_TBD._
