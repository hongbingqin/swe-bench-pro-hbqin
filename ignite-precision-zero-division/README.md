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

- `ignite/metrics/precision.py` (`_BasePrecisionRecall`): new `zero_division` kwarg
  (validated to `0`/`1`/`nan`, default `0`). `compute()` replaces the `eps`
  div-by-zero mask with explicit per-class substitution where the denominator is
  0, applied **before** aggregation; `macro` uses nan-aware mean, `weighted`
  drops zero-weight/nan classes, `micro` substitutes on total-zero, `samples` is
  handled per-sample via a shared helper.
- `ignite/metrics/recall.py`: `Recall` inherits the new `__init__`; its `samples`
  update uses the shared helper. No duplication.
- `ignite/metrics/fbeta.py`: `Fbeta` gains a validated `zero_division` threaded
  into its internal Precision/Recall, plus substitution where `beta^2*P + R == 0`.

## Completion Rates

_TBD — measured during validation._

## Model Analysis

_TBD (calibration pending)._ Design intent: a real sklearn-parity feature in
library code (not an `examples/` script). Difficulty is spread across the six
`average` modes × {0, 1, nan} × per-class arrays, matching sklearn exactly
(incl. nan-masking and per-sample `samples`) — the subtle modes are where a
partial implementation fails.

## Anti-Cheating Analysis

- **Tests are verifier-only** (`test_patch` adds `tests/ignite/metrics/test_precision_zero_division.py`).
- **fail_to_pass execute the patched code** — import the patched `ignite.metrics`
  classes and call `update`/`compute`.
- **No hardcoded oracle** — expected values are computed at runtime from
  **sklearn** (`precision_score`/`recall_score`/`fbeta_score`) with matching
  `average`/`labels`/`zero_division`, compared via `assert_allclose(equal_nan=True)`.
- **Backward-compat enforced** — `pass_to_pass` includes `test_default_*`
  (default → 0) so a solution that changes the default behavior is caught.
- **No oracle/ground-truth in agent-readable paths** — Dockerfile only clones +
  pip-installs pinned deps (sklearn is a test-time reference, not the answer).

## Notes

- `tests/run_script.sh` forces `--color=no -o addopts=` because the repo pins
  `addopts = "--color=yes"`, whose ANSI codes otherwise break `parser.py`'s regex.
- `fail_to_pass` (32 curated) is a config-only calibration lever — the full
  123-case test file runs; the scored subset can be expanded/contracted from the
  first cloud balance read.
