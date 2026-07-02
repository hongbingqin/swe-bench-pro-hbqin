# codimango/ignite-precision-labels

> Draft — filled in during QA.

## Description

Adds a scikit-learn-style **`labels`** parameter to PyTorch-Ignite's
classification **Precision** / **Recall** (and **Fbeta**) metrics. It restricts
(and reorders) the set of classes/labels included in the per-class output and in
the `macro`/`weighted`/`None` aggregations — matching
`sklearn.metrics.precision_score(..., labels=[...])` semantics. ignite currently
has no such parameter (confirmed absent at HEAD).

- **Repo:** `pytorch/ignite` @ `d5208644ab5646ee250c3b5db08f5c5e9f128354`
- **Language:** Python
- **Files (reference):** `ignite/metrics/precision.py` (shared `_BasePrecisionRecall`),
  `ignite/metrics/recall.py`, `ignite/metrics/fbeta.py`

## Reference solution (oracle)

- `ignite/metrics/precision.py` (`_BasePrecisionRecall`): new `labels` kwarg
  (validated — rejects empty/duplicate/negative/out-of-range → `ValueError`;
  default `None`). `compute()` `index_select`s `_numerator`/`_denominator`/`_weight`
  along the class dim into `labels` order **before** aggregation; `micro` sums the
  subset only, `weighted` divides by subset supports, `macro`/`None`/`False`
  return per-class values in `labels` order.
- `ignite/metrics/recall.py`: `update()` accumulates per-class (for subset `micro`)
  and restricts `samples` to the label columns; init/compute inherited.
- `ignite/metrics/fbeta.py`: threads `labels` into the internal Precision/Recall.

## Completion Rates

_TBD — measured during validation._

## Model Analysis

_TBD (calibration pending)._ sklearn-parity feature in real library code; the
difficulty is correct subset restriction across `average` modes (esp. micro /
weighted / samples restricted to the subset, and per-class ordering). Expected to
behave like the `zero_division` sibling: too-easy-for-avocado but flaky on
code-gen → a re-roll lands balance.

## Anti-Cheating Analysis

- **Tests are verifier-only** (`test_patch` adds `tests/ignite/metrics/test_precision_labels.py`).
- **fail_to_pass execute the patched code** — import the patched classes, call update/compute.
- **No hardcoded oracle** — expected values computed at runtime from **sklearn**
  (`precision_score`/`recall_score`/`fbeta_score` with matching `average`/`labels`),
  compared via `assert_allclose` (order-respecting).
- **Backward-compat enforced** — `pass_to_pass` includes `labels=None` tests, so a
  solution that changes default behavior is caught.
- **No oracle/ground-truth in agent-readable paths** — Dockerfile clones + pip only.

## Notes

- `tests/run_script.sh` forces `--color=no -o addopts=` (ignite pins
  `addopts="--color=yes"`, whose ANSI codes break `parser.py`).
- `fail_to_pass` (18 curated) is a config-only calibration lever — the full
  186-case file runs; expand the scored subset if too-easy.
