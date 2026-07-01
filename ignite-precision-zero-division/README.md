# codimango/ignite-precision-zero-division

> Draft — filled in during QA once solution/tests + calibration are set.

## Description

Adds a scikit-learn-style **`zero_division`** option to PyTorch-Ignite's
classification **Precision** / **Recall** (and **Fbeta**) metrics. Today the
metrics mask division-by-zero with an `eps` hack, so a class with no
predicted/actual samples always contributes **0** with no way to change it. This
feature makes that configurable — **0, 1, or NaN** — matching
`sklearn.metrics.precision_score(..., zero_division=...)`, and it must behave
correctly across every `average` mode (`False`/`None`/`macro`/`weighted`/
`micro`/`samples`) and for per-class array outputs.

- **Repo:** `pytorch/ignite` @ `d5208644ab5646ee250c3b5db08f5c5e9f128354`
- **Language:** Python
- **Files touched by the reference solution:** `ignite/metrics/precision.py`
  (shared `_BasePrecisionRecall`), `ignite/metrics/recall.py`,
  `ignite/metrics/fbeta.py` (as needed)

## Reference solution (oracle)

_TBD._

## Completion Rates

_TBD — measured during validation._

## Model Analysis

_TBD._

## Anti-Cheating Analysis

_TBD._
