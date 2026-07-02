Right now the training code uses GridSearch for hyperparameter tuning, and I want to move away from that. I don’t want it to try every single point in a big grid. I want a bounded 1‑D search over learning rate with explicit bounds `(lower, upper)`, where I set an iteration cap and that cap is what keeps the run bounded.

I also don’t want to break the normal training path. Training a single config still works the same way under the `__main__` guard, and the module is safe to import (importing it doesn’t kick off training).

For testing, the search logic needs to be importable and inspectable, so I’m putting it into an optimizer class that’s constructed with the bounds. The key design decision is that surrogate fitting and next-point proposal must be callable deterministically on a given history, without relying on hidden mutable trust-region state. I’m choosing the contract where `propose(history, radius)` takes the trust radius.  optimize() returns the best config found.

The surrogate is a least-squares quadratic fit. Given a history list of `(x, score)` pairs, I fit a degree‑2 polynomial `a·x² + b·x + c` deterministically to all points in the history. The quadratic evaluation entry point is `predict(history, x)`, which returns the value of that fitted quadratic at `x` (so tests can verify it’s a true quadratic LS fit and not some other smoother).

Propose where the surrogate looks best, within a trust region around the current best.

Here `best_x` means the `x` value in the history with the maximum observed score. If the quadratic is not a usable maximum for a maximization problem, handle the degenerate case.

The trust region behavior is pinned in the loop. Trust region shrinks when there's no improvement.

For seeding, I start with a few seed points drawn uniformly at random from within the bounds using the provided seed, then I propose subsequent points. Same seed gives the same result. The proposed next point is a continuous value (not snapped to a grid). The agent must define the optimizer class inside train.py, not in a new/separate module.
