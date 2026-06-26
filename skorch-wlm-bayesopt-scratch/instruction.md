Right now the training code uses GridSearch for hyperparameter tuning, and I want to move away from that. I don’t want it to try every single point in a big grid. I want a bounded 1‑D search over learning rate with explicit bounds `(lower, upper)`, where I set an iteration cap and that cap is what keeps the run bounded.

I also don’t want to break the normal training path. Training a single config still works the same way under the `__main__` guard, and the module is safe to import (importing it doesn’t kick off training).

For testing, the search logic needs to be importable and inspectable, so I’m putting it into an optimizer class that’s constructed with the bounds. The key design decision is that surrogate fitting and next-point proposal must be callable deterministically on a given history, without relying on hidden mutable trust-region state. I’m choosing the contract where `propose(history, radius)` takes the trust radius explicitly, and only `optimize(...)` runs the loop and updates the radius.

The surrogate is a least-squares quadratic fit. Given a history list of `(x, score)` pairs, I fit a degree‑2 polynomial `a·x² + b·x + c` deterministically to all points in the history. The quadratic evaluation entry point is `predict(history, x)`, which returns the value of that fitted quadratic at `x` (so tests can verify it’s a true quadratic LS fit and not some other smoother).

The next-point proposal uses the quadratic vertex. From the fitted coefficients I compute `x* = −b/(2a)`, then clip it to the feasible interval (trust region intersected with bounds):
`[ max(lower, best_x − radius),  min(upper, best_x + radius) ]`.
Here `best_x` means the `x` value in the history with the maximum observed score. If the quadratic is not a usable maximum for a maximization problem (specifically if `a ≥ 0`, so it’s convex or effectively linear), I take a pinned fallback branch: I evaluate the fitted quadratic at the two feasible endpoints of that interval and choose the endpoint with the higher predicted value; if the predictions tie exactly, I pick the lower endpoint.

The trust region behavior is pinned in the loop. I start with an initial radius of `0.25 × (upper − lower)`. On any iteration with no improvement in the best observed score, I shrink the radius by a factor of `0.5` (and I don’t rely on hidden state to do this; it’s only updated inside `optimize`). The loop is budgeted by an explicit iteration cap.

For seeding, I start with 3 initial points drawn uniformly at random from within the bounds using the provided seed, then I propose subsequent points using the vertex rule above. Same seed gives the same result. The proposed next point is a continuous value (not snapped to a grid), so the tests can directly check the vertex formula and the clipping behavior. The agent must define the optimizer class inside train.py, not in a new/separate module.

Finally, I’ll replace `GridSearchCV` in `train.py` with this optimizer, while keeping the single-config training path intact under the `__main__` guard so imports stay safe.
