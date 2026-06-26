# codimango/skorch-wlm-bayesopt-scratch

## Description

Harder sibling of `skorch-wlm-bayesopt`. Instead of swapping in a library
(`BayesSearchCV`), the solver implements a **from-scratch bounded 1-D optimizer**
for the `examples/word_language_model` learning-rate search: a **local quadratic
trust-region** method (least-squares degree-2 surrogate + vertex step with a
convex/linear fallback + a shrinking trust radius), capped by an explicit
iteration budget, using only numpy/scipy.

A first attempt used a Nadaraya–Watson kernel surrogate; with a fully-specified
formula it calibrated **too easy** (avocado 5/5). This quadratic trust-region
version keeps the algorithm fully specified (so the discriminator tests are
fair) but adds **branchy, error-prone procedure** — the `a ≥ 0` fallback, the
trust-region clipping, the shrink-on-no-improvement loop — to separate a careful
solution from a quick one.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python
- **Files touched by the reference solution:** `examples/word_language_model/train.py`

## Reference solution (oracle)

`train.py`: a `QuadraticTrustRegionOptimizer` class constructed with the search
`bounds`, with deterministic, history-driven methods:
- `predict(history, x)` — value at `x` of the least-squares degree-2 polynomial
  fit (`a·x²+b·x+c`) to all history points.
- `propose(history, radius)` — the vertex `x* = −b/(2a)` clipped to the feasible
  interval `[max(lo, best_x−radius), min(hi, best_x+radius)]` (`best_x` = max-score
  point); if `a ≥ 0` (no usable max), the feasible endpoint with the higher
  predicted value (tie → lower endpoint).
- `optimize(objective, n_iter, seed)` — 3 seeded uniform points, then vertex
  proposals; trust radius starts at `0.25·(upper−lower)` and shrinks `×0.5` on a
  non-improving iteration; returns `(best_config, history)`.

All training (argparse, corpus, `Net`, `.fit`) is wrapped in `main()` under
`if __name__ == '__main__':` (import-safe); the optimizer replaces `GridSearchCV`.
`net.py` / `model.py` are untouched.

## Completion Rates

_TBD — measured by the platform during validation._

## Model Analysis

_TBD (calibration pending)._ Design intent: escape the "too easy" outcome of the
NW version by making the *procedure* error-prone (vertex vs. convex-fallback
branch, trust-region clipping, shrink loop) rather than a single straight
formula — while keeping it fully specified, deterministic, and exactly testable.

## Anti-Cheating Analysis

- **Tests are verifier-only** (`test_patch` at verify time); fail_to_pass import
  the patched `train` and call the optimizer's methods.
- **Strong discrimination (verified locally):** GP-surrogate and linear-surrogate
  cheats both **fail all four `predict`/`propose` tests** — `predict` pins the LS
  quadratic value, and `propose` pins the vertex, the trust-region clip, and the
  `a ≥ 0` endpoint fallback. Only a faithful quadratic-trust-region implementation
  passes.
- **Deterministic / RNG-robust:** `predict`/`propose` tests use fixed histories
  (no randomness, explicit radius); the end-to-end test checks loop mechanics +
  determinism with a loose convergence band.
- **No BO library in the image** (skopt/optuna/hyperopt/bayes_opt/GPy/botorch all
  absent); from-scratch is enforced behaviorally by the discriminator tests, so
  internet stays at the platform default `true`.

## Known caveats

- **Integration leniency:** no test runs `train.py`'s `main`, so an agent could
  add the optimizer class without actually removing `GridSearchCV` from `main`.
  The hard part (the optimizer) is fully tested; the replacement is not.
