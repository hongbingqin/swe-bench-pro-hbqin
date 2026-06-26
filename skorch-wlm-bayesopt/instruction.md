Right now the training code uses GridSearch to tune hyperparameters. I want to change that. I don’t want it to try every single point in the whole grid anymore. I want a Bayesian-style search instead, and I want it to try only a limited number of configs. The limit needs to be explicit. An iteration cap.

I also don’t want to break the existing training flow. The “train one config” path should still work the same way. Keep the guard so normal training still runs.

The test needs to be able to poke at the search logic without running a real training job. That’s the point. So I need the search to be something I can import and call on its own, not something buried in the script’s main path. I want a standalone function I can import from a unit test. It should take an estimator, a search space, and the iteration cap (a seed too, if we can). And it should just return a configured search object. No training run. No side effects on import. factory defined in train.py.
