I'm adding multi-output support to `R2Score`. Right now it only accepts shapes like `(N,)` or `(N,1)` — I'm making it handle `(N, K)` so you can pass multiple targets at once and get R² computed per target.

There's a new `multioutput` parameter that controls what comes back. Set it to `'raw_values'` and you get the full per-target R² array. `'uniform_average'` (which is the default) just takes the mean across targets. And `'variance_weighted'` weights each target's R² by that target's variance before averaging — so noisier targets count more.

I'm matching `sklearn.metrics.r2_score`'s `multioutput` semantics exactly. If you only pass single-output data with the default setting, you get the same scalar as before — nothing breaks. Getting all three modes correct and matching sklearn is really the hard part here, but that's the point — the tests will enforce the full matrix against sklearn as ground truth.
