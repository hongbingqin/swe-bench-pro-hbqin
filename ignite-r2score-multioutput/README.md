# codimango/ignite-r2score-multioutput

> Draft — filled during QA.

## Description

Adds scikit-learn-style **multi-output** support to PyTorch-Ignite's **R2Score**.
Today R2Score only accepts `(N,)`/`(N,1)` and returns a scalar. This adds:
`(N, K)` inputs, per-target accumulation, and a **`multioutput`** parameter —
`'raw_values'` (per-target array), `'uniform_average'` (mean, default),
`'variance_weighted'` — matching `sklearn.metrics.r2_score`. Single-output
behavior is unchanged (backward compatible).

- **Repo:** `pytorch/ignite` @ `d5208644ab5646ee250c3b5db08f5c5e9f128354`
- **Files (reference):** `ignite/metrics/regression/r2_score.py` (+ shape handling)

## Reference solution (oracle)

`ignite/metrics/regression/r2_score.py`: new `multioutput` kwarg
(`raw_values`/`uniform_average`(default)/`variance_weighted`, validated →
`ValueError` for invalid values, matching sklearn). An R2Score-local shape check
allows `(N, K)` (without loosening the shared `_BaseRegression` check);
accumulation becomes per-target `(K,)` (`_sum_of_errors`/`_y_sum`/`_y_sq_sum`,
computed in float64 to preserve the original scalar precision). `compute()`
derives per-target SS_res/SS_tot, replicates sklearn's `force_finite`/constant-
target handling, then aggregates per mode; single-output + default returns the
same scalar float as before.

## Completion Rates

_TBD — measured during validation._

## Model Analysis

_TBD (calibration pending)._ Meatier than the parity kwargs — the difficulty is
the scalar→per-target restructure, the `(N,K)` shape handling, and matching
sklearn's `variance_weighted` + constant-target edge exactly. Likely moderate
(not trivially too-easy); if agents miss `variance_weighted` or the shape relax,
that's genuine difficulty.

## Anti-Cheating Analysis

- **Tests verifier-only** (`test_patch` adds `test_r2_score_multioutput.py`).
- **fail_to_pass execute the patched code** — import/call the patched R2Score.
- **No hardcoded oracle** — expected values from **sklearn** `r2_score(...)` at
  runtime, `assert_allclose` (elementwise for `raw_values`).
- **Backward-compat enforced** — `pass_to_pass` includes single-output default
  tests + existing `test_r2_score[cpu]`/`test_zero_sample`, so the scalar path
  can't silently regress.
- **No oracle/ground-truth in agent paths** — Dockerfile clones + pip only.

## Notes

- `tests/run_script.sh` forces `--color=no -o addopts=` (ignite pins
  `--color=yes`, which breaks `parser.py`).
- `fail_to_pass` (20 curated) is a config-only calibration lever.
