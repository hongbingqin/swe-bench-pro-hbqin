# codimango/skorch-wlm-bayesopt-scratch

## Description

Harder sibling of `skorch-wlm-bayesopt`. Instead of swapping in a library
(`BayesSearchCV`), the solver must implement a **from-scratch** sequential
model-based optimizer for the `examples/word_language_model` hyperparameter
search: a hand-written **Nadaraya–Watson kernel-weighted surrogate** over the
observed `(config, score)` history plus a **distance-based exploration**
acquisition, capped by an explicit iteration budget — using only numpy/scipy
(no BO library is installed, and internet is off).

This raises difficulty (a real algorithm, not a one-line swap) **and** closes the
library version's leniency: the exact surrogate/acquisition behavior is checked
on fixed histories, so a recalled textbook GP-EI and a random sampler both fail.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python
- **Files touched by the reference solution:** `examples/word_language_model/train.py`

## Reference solution (oracle)

`train.py`: a `KernelWeightedOptimizer` class constructed with the search
`bounds`, exposing `predict` / `propose` / `optimize` methods:
- `predict(history, x)` — Nadaraya–Watson weighted mean with Gaussian kernel,
  bandwidth `h = 0.1·(upper−lower)` derived from `self.bounds`; fallback to the
  mean of observed scores when `Σw < 1e-12`.
- `propose(history, candidates)` — argmax of `mean + 0.2·(distance to nearest
  sampled point)`, smallest-index tie-break.
- `optimize(objective, n_iter, seed)` — 3 seeded uniform-random points, then
  acquisition-driven proposals over a 101-point grid until the cap; returns
  `(best_config, history)`.

All module-level training (argparse, corpus, `Net`, `.fit`) is wrapped in
`main()` under `if __name__ == '__main__':` so the module is import-safe; the
optimizer replaces `GridSearchCV` in the training path. `net.py` / `model.py`
are untouched.

## Completion Rates

_TBD — measured by the platform during validation._

## Model Analysis

_TBD (calibration pending)._ Design intent: harder than the library sibling
(real from-scratch algorithm + class contract + integration) without being
unsolvable — the exact formula/constants are specified (Necessary Specification,
like ALiBi's slopes), so the work is correct implementation, not algorithm
invention. Internet-off + no BO library in the image enforce "from scratch."

## Anti-Cheating Analysis

- **Tests are verifier-only**, applied via `test_patch` at verify time.
- **fail_to_pass execute the patched code** — import the patched `train`,
  construct the optimizer, call its methods.
- **Strong discrimination (verified locally):** a GP-surrogate cheat and a
  random-search cheat both **fail all three exact-formula tests**
  (`test_surrogate_is_nadaraya_watson`, `test_surrogate_zero_weight_fallback`,
  `test_propose_maximizes_acquisition`) — they pass only the trivial
  import/RNN/loop-mechanics tests, so neither earns reward. This rejects both
  the textbook-GP recall path and the random shortcut.
- **Deterministic, RNG-robust:** the formula tests use fixed histories (no
  randomness); the end-to-end test checks loop mechanics + determinism with a
  loose convergence band, so a correct optimizer passes regardless of its
  seeding RNG.
- **No oracle/ground-truth in agent-readable paths**; Dockerfile only clones +
  pip-installs pinned deps; **no BO library present** in the image (verified:
  skopt/optuna/hyperopt/bayes_opt/GPy/botorch all absent).
- **From-scratch is enforced behaviorally, not by the network.** Even with
  internet on, a `pip install`ed BO library / GP doesn't help: a GP surrogate
  fails the exact Nadaraya–Watson `predict` and acquisition `propose` tests
  (verified locally).

## Known caveats

- **Integration leniency (review M1):** no test runs `train.py`'s `main`, so an
  agent could (in principle) leave `GridSearchCV` in `main` and merely *add* the
  optimizer class and still pass. The hard part (the optimizer) is fully tested;
  the replacement itself is not behaviorally verified.
- **`allow_internet`:** originally set `false` to block pip-installing a BO
  library, but that starved the in-sandbox model agents (opus produced no
  trials). Reverted to the platform default `true`; from-scratch is enforced by
  the discriminator tests instead.
