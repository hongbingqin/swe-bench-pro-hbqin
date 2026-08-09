# codimango/skorch-wlm-bayesopt-scratch

## Description

Harder sibling of `skorch-wlm-bayesopt`. Instead of swapping in a library
(`BayesSearchCV`), the solver implements a **from-scratch bounded 1-D optimizer**
for the `examples/word_language_model` learning-rate search — a surrogate-guided
search over the observed `(config, score)` history with a shrinking trust region,
capped by an explicit iteration budget, using only numpy/scipy.

The instruction states the *goal and the surrogate family* (a quadratic surrogate
over the history) but deliberately **does not spell out the proposal formula, the
clipping convention, the degenerate-case handling, or the trust-region constants**.
The agent must derive those; the tests pin the correct behavior via computed
oracles. This is the difficulty lever — deriving a correct, deterministic
surrogate optimizer rather than transcribing a fully-specified recipe.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python
- **Files touched by the reference solution:** `examples/word_language_model/train.py`

> Exact reference-solution details (formulas, constants, tie-breaks) intentionally
> live in `solution/` only — not in this README or in `instruction.md` — to keep
> the task from being one-pass reconstructible from author-facing files.

## Reference solution (oracle)

`train.py` gains a small from-scratch optimizer class constructed with the search
`bounds`, exposing deterministic, history-driven methods for (a) evaluating the
fitted quadratic surrogate at a point, (b) proposing the next point within a
trust region around the current best, and (c) an end-to-end `optimize(...)` loop
that seeds a few points, proposes subsequent ones, shrinks the trust radius on
non-improvement, and returns the best config found. All training (argparse,
corpus, `Net`, `.fit`) is wrapped in `main()` under `if __name__ == '__main__':`
(import-safe); the optimizer replaces `GridSearchCV`. `net.py` / `model.py` are
untouched.

(See `solution/` for the exact surrogate/proposal/constants — omitted here by
design.)

## Completion Rates

_Pending recalibration after the Shape-A redesign (recipe withheld from
instruction + hardened test plumbing). Will be regenerated from platform job
logs once the balance gate passes._

| Agent | Model | Pass rate |
|-------|-------|-----------|
| oracle | oracle | 3/3 (1.000) |
| metacode | avocado_dvsc_tester | _pending_ |
| claude-code | claude-opus-4-6 | _pending_ |
| (aux) | gpt-5.5 | _pending_ |

## Model Analysis

_Pending recalibration._ With the proposal recipe withheld, observed failures are
genuine derivation errors (e.g. mishandling the no-interior-maximum degenerate
case, non-deterministic seeding, off-by-one budget/loop bugs) rather than
return-shape/plumbing artifacts. To be filled from real job logs after a passing
run.

## Anti-Cheating Analysis

- **Tests are verifier-only** (`test_patch` applied at verify time); fail_to_pass
  import the patched `train` and call the optimizer's methods.
- **Strong discrimination (verified locally):** GP-surrogate and linear-surrogate
  substitutes both fail the `predict`/`propose` discriminator tests, which assert
  the true least-squares-quadratic values at multiple points and the exact
  proposal behavior via computed oracles. Only a faithful quadratic surrogate
  passes — but the *instruction does not disclose those values*, so passing
  requires deriving the method, not recalling a stated one.
- **Return-shape robust:** the harness accepts a scalar, a `(best, history)`
  tuple, or a dict for `optimize()`'s return, and multiple method-name/arg-order
  variants — so a valid solution is judged on behavior, not on matching a pinned
  return shape.
- **Deterministic / RNG-robust:** `predict`/`propose` tests use fixed histories
  (no randomness, explicit radius); the end-to-end test checks loop mechanics +
  determinism with a bounded-history and convergence band.
- **No BO library in the image** (skopt/optuna/hyperopt/bayes_opt/GPy/botorch all
  absent); from-scratch is enforced behaviorally by the discriminator tests.

## Calibration notes

- **Shape-A redesign:** the instruction was changed to withhold the proposal
  formula/clip/fallback/constants (previously fully specified → recall-adjacent).
  Test plumbing was hardened (return-shape/signature tolerant) so failures reflect
  genuine derivation errors, not artifacts.
- **Integration tested:** `test_gridsearchcv_replaced_in_train` checks the
  exhaustive `GridSearchCV(` is no longer instantiated/imported in `train.py`.
- Difficulty and completion numbers to be refreshed from platform logs after the
  next passing validation.
