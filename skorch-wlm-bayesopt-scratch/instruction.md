Right now the training code uses GridSearch for hyperparameter tuning, and I want to move away from that. I don’t want it to try every single point in the whole grid. I want a Bayesian-style search that only evaluates a fixed number of configs. The cap is explicit, and it is the thing that keeps the run bounded.

I also don’t want to break the normal training path. Training a single config still works the same way, behind the guard, and the module is safe to import.

For testing, the search logic is importable and inspectable, not buried in the script’s main path. There are three entry points the test can call directly. The surrogate prediction entry point takes a given history (a list of (config, score)) plus a point `x`, and returns a predicted score. The next-pick proposal entry point takes a given history plus a candidate pool and returns the chosen candidate. The end-to-end optimize entry point replaces GridSearch, takes the estimator or objective, the bounds or search space, the iteration cap, and a seed, and returns the best config (and the full history). All three live at module scope behind the guard, and nothing runs real training just because the module got imported.

The surrogate predicts with a Nadaraya–Watson weighted average. It uses a Gaussian kernel with weights `w_i = exp(-(x - x_i)^2 / (2*h^2))`. The bandwidth is **h = 0.1 × (upper_bound − lower_bound)**. The predicted score is `sum(w_i * s_i) / sum(w_i)`. When `sum(w_i) < 1e-12`, the prediction returns the **mean of the observed scores** in the history.

The next pick maximizes an acquisition score. The exploration term is the distance from the candidate to the nearest sampled point in the history. The acquisition is `mean + β * exploration` with **β = 0.2**. We are maximizing and higher score is better. The candidate pool is a fixed grid of **N = 101** evenly spaced points across the bounds, inclusive. If two candidates tie on acquisition, the tie-break is **smallest index in the candidate list**.

The loop is budgeted by the iteration cap. For end-to-end seeding, it starts with **3** initial points chosen by drawing uniformly at random from the bounds using the provided seed. After that it proposes points using the acquisition rule. Same seed gives the same result.

I want the optimizer to be a class that gets constructed with the bounds, and predict, propose, and optimize are methods on that class, so values like h are derived from self.bounds.

Finally, GridSearch is removed from the training path and replaced with this bounded optimizer. The single-config training flow stays intact behind the guard so imports are safe.
